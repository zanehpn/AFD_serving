"""CPU-safe contracts for independently replicated native AFD service groups.

Expert DP means separate vLLM instances, not vLLM's DP flag: native MoE
flattens its DP ranks into the expert sharding domain. Each replica uses
either EP or intra-expert TP. Combined EP x TP inside one instance requires
a different backend and is rejected explicitly.
"""
from itertools import product

FIELDS = ('attention_dp', 'attention_tp', 'expert_dp', 'expert_ep', 'expert_tp')


def validate(c):
    for key in (*FIELDS, 'microbatches'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    na, ne = len(c['attention_gpus']), len(c['expert_gpus'])
    if len(set(c['attention_gpus'] + c['expert_gpus'])) != na + ne:
        raise ValueError('GPU roles overlap')
    ad, at, ed, ep, et = (c[k] for k in FIELDS)
    if na != ad * at or ne != ed * ep * et:
        raise ValueError('Physical allocation differs from parallel layout')
    if ad % ed:
        raise ValueError('native_replica_attention_dp_must_divide_evenly')
    if ep > 1 and et > 1:
        raise ValueError('native_combined_expert_ep_tp_requires_new_backend')
    ga, ge = na // ed, ne // ed
    if ga < ge or ga % ge:
        raise ValueError('native_replica_p2p_rank_ratio')
    # Each F TP lane must receive identical tensors. The existing contiguous
    # P2P grouping guarantees this when the replica has one Attention TP group.
    # Multiple A DP groups need a lane-aware connector, not a flag change.
    if et > 1 and ad // ed != 1:
        raise ValueError('native_ffn_tp_requires_single_attention_group_per_replica')
    if c.get('execution_mode', 'eager') != 'eager':
        raise ValueError('native_joint_layout_requires_eager')
    return c


def deployment(c):
    validate(c)
    return {**{k: c[k] for k in FIELDS}, 'ffn_ep': c['expert_ep'],
            'attention_ranks': len(c['attention_gpus']),
            'expert_ranks': len(c['expert_gpus']), 'microbatches': c['microbatches'],
            'layout_contract': 'independent_afd_replicas_v1'}


def replicas(c, api_port=18000, expert_api_port=18001, afd_port=16239, rpc_port=29550):
    validate(c)
    ed = c['expert_dp']
    na, ne = len(c['attention_gpus']) // ed, len(c['expert_gpus']) // ed
    output = []
    for i in range(ed):
        # Preserve the single-replica public API. Multiple replicas expose one
        # deterministic proxy; backend ports live beyond that public pair.
        offset = (i + 1) * 64 if ed > 1 else 0
        output.append({'replica': i, 'attention_gpus': c['attention_gpus'][i * na:(i + 1) * na],
                       'expert_gpus': c['expert_gpus'][i * ne:(i + 1) * ne],
                       'attention_rank_offset': i * na, 'expert_rank_offset': i * ne,
                       'attention_ranks': na, 'expert_ranks': ne,
                       'attention_tp': c['attention_tp'], 'expert_tp': c['expert_tp'],
                       'attention_dp': c['attention_dp'] // ed,
                       'expert_native_dp': c['expert_ep'],
                       'expert_enable_ep': c['expert_ep'] > 1,
                       'api_port': api_port + offset, 'expert_api_port': expert_api_port + offset,
                       'afd_port': afd_port + offset, 'dp_rpc_base_port': rpc_port + offset})
    ports = [p for r in output for p in (r['api_port'], r['expert_api_port'],
             *range(r['afd_port'], r['afd_port'] + r['expert_ranks'] + 1),
             r['dp_rpc_base_port'], r['dp_rpc_base_port'] + 1)]
    if ed > 1:
        ports.append(api_port)
    if len(ports) != len(set(ports)) or any(not 1024 <= p <= 65535 for p in ports):
        raise ValueError('Replica rendezvous/API/RPC ports overlap or are out of range')
    return output


def enumerate_layouts(gpus, experts, microbatches=(1, 2, 4), degrees=(1, 2, 4, 8), model=None):
    """Return mapped structural candidates and explicit backend exclusions."""
    if len(set(gpus)) != len(gpus) or not gpus:
        raise ValueError('Distinct physical GPUs required')
    if any(type(n) is not int or n < 1 for n in (*microbatches, *degrees)):
        raise ValueError('Positive candidate degrees and microbatch counts required')
    accepted, rejected = [], []
    for ad, at, ed, ep, et, m in product(degrees, degrees, degrees, degrees, degrees, microbatches):
        na, ne = ad * at, ed * ep * et
        if na + ne > len(gpus) or experts % ep:
            continue
        c = dict(zip(FIELDS, (ad, at, ed, ep, et)))
        c.update(attention_gpus=gpus[:na], expert_gpus=gpus[na:na + ne],
                 microbatches=m, execution_mode='eager')
        try:
            validate(c)
            if model:
                for field in ('num_attention_heads', 'linear_num_key_heads', 'linear_num_value_heads'):
                    if model.get(field) and model[field] % at:
                        raise ValueError(f'model_{field}_attention_tp_divisibility')
                kv = model.get('num_key_value_heads')
                if kv and kv % at and at % kv:
                    raise ValueError('model_kv_heads_attention_tp_divisibility')
                width = model.get('moe_intermediate_size', model.get('intermediate_size'))
                if width and width % et:
                    raise ValueError('model_expert_intermediate_tp_divisibility')
        except ValueError as error:
            rejected.append({'configuration': c, 'reason': str(error)})
        else:
            accepted.append(c)
    return accepted, rejected


def verify_launch(c, launch):
    """Validate a launch receipt against every requested structural control."""
    expected = deployment(c)
    for key in (*FIELDS, 'attention_ranks', 'expert_ranks', 'microbatches', 'layout_contract'):
        if launch.get(key) != expected[key]:
            raise ValueError(f'Actual launch {key} differs from candidate')
    for role in ('attention', 'expert'):
        if launch.get(role + '_gpus') != ','.join(map(str, c[role + '_gpus'])):
            raise ValueError(f'Actual {role} GPU mapping differs from candidate')
    if launch.get('dbo_enabled') != (c['microbatches'] == 2):
        raise ValueError('Actual DBO mode differs from candidate')
    return True

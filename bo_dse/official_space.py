"""Public vLLM CLI mappings for the pinned upstream CUDA P2P AFD backend.

EP spans DP*TP ranks. expert_dp/TP here are *vLLM launch degrees*, not
independent expert replicas or additional factors of the EP world size.
"""
import json

SEMANTICS = 'vllm_dp_tp_ep_world'


def structures(gpus, model):
    if len(gpus) not in (4, 8) or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError('Official validation requires four or eight distinct physical GPU indices')
    if model not in ('qwen36', 'deepseek-v2-lite'):
        raise ValueError('Unmapped model')
    rows = []
    # Enumerate resources and factorisations, never model-name/example allowlists.
    # See pinned upstream afd_plugin/distributed/topology.py. Model divisibility
    # and physical execution evidence are separate filters in official.py.
    for na in range(1, len(gpus)):
        for nf in range(1, len(gpus)-na+1):
            if na < nf or na % nf:
                continue
            for at in range(1, na+1):
                for ft in range(1, nf+1):
                    if na % at or nf % ft:
                        continue
                    rows.append(dict(attention_gpus=gpus[:na], expert_gpus=gpus[na:na+nf],
                                     attention_dp=na//at, attention_tp=at,
                                     expert_dp=nf//ft, expert_tp=ft, expert_ep=nf,
                                     parallelism_semantics=SEMANTICS))
    return rows


def validate(c):
    a, f = len(c['attention_gpus']), len(c['expert_gpus'])
    if c.get('parallelism_semantics') != SEMANTICS:
        raise ValueError('Official AFD parallelism semantics missing')
    for key in ('attention_dp', 'attention_tp', 'expert_dp', 'expert_tp', 'expert_ep'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (not a or not f or a+f > 8 or a < f or a % f
            or len(set(c['attention_gpus'] + c['expert_gpus'])) != a+f):
        raise ValueError('Invalid official P2P allocation (maximum eight GPUs)')
    if (a != c['attention_dp'] * c['attention_tp']
            or f != c['expert_dp'] * c['expert_tp'] or c['expert_ep'] != f):
        raise ValueError('Official EP world must equal FFN DP*TP, not DP*EP*TP')
    if c['microbatches'] not in (1, 2) or c['execution_mode'] != 'eager':
        raise ValueError('Initial official space supports eager with DBO disabled/enabled (two ubatches)')
    for key in ('dbo_decode_token_threshold', 'dbo_prefill_token_threshold'):
        if key in c and (type(c[key]) is not int or c[key] < 1):
            raise ValueError('DBO thresholds must be positive integers')


def commands(config, c):
    validate(c)
    output = {}
    for role, prefix in (('attention', 'attention'), ('ffn', 'expert')):
        afd = dict(role=role, connector='P2pNcclAFDConnector', host='127.0.0.1',
                   port=config['afd_port'], num_attention_ranks=len(c['attention_gpus']),
                   num_ffn_ranks=len(c['expert_gpus']), compute_gate_on_attention=False)
        cmd = [config['native_python'], '-m', 'vllm.entrypoints.cli.main', 'serve', config['model_path'],
               '--served-model-name', 'official-afd', '--host', '127.0.0.1',
               '--port', str(config['api_port'] + (0 if role == 'attention' else 1)),
               '--data-parallel-size', str(c[prefix + '_dp']),
               '--tensor-parallel-size', str(c[prefix + '_tp']), '--enable-expert-parallel',
               '--data-parallel-rpc-port', str(config['dp_rpc_port'] + (0 if role == 'attention' else 100)),
               '--additional-config', json.dumps({'afd': afd}), '--enforce-eager',
               '--no-enable-prefix-caching', '--max-model-len', str(config['max_model_len']),
               '--max-num-seqs', '32', '--max-num-batched-tokens', '3072',
               '--gpu-memory-utilization', '0.85', '--generation-config', 'vllm',
               '--seed', '0', '--trust-remote-code']
        if config['model'] == 'qwen36':
            cmd.append('--language-model-only')
        if c['microbatches'] == 2:
            cmd += ['--enable-dbo', '--dbo-decode-token-threshold', str(c.get('dbo_decode_token_threshold', 2)),
                    '--dbo-prefill-token-threshold', str(c.get('dbo_prefill_token_threshold', 12))]
        else:
            cmd.append('--no-enable-dbo')
        output[role] = {'command': cmd, 'gpus': c[prefix + '_gpus']}
    return output

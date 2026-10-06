"""Require actual worker state, including effective MoE shards, for every GPU."""
import argparse
import json
from pathlib import Path
from bo_layout import replicas, verify_launch


def verify(configuration, run, expected_groups=None, thresholds=None):
    receipt = json.loads((run / 'joint-launch.json').read_text())
    verify_launch(configuration, receipt)
    thresholds = thresholds or {'dbo_decode_token_threshold': 2, 'dbo_prefill_token_threshold': 12}
    if any(receipt.get(k) != v for k, v in thresholds.items()):
        raise ValueError('Actual microbatch thresholds differ from frozen protocol')
    actual_groups = receipt['replicas']
    structural_groups = replicas(configuration)
    port_keys = {'api_port', 'expert_api_port', 'afd_port', 'dp_rpc_base_port'}
    strip_ports = lambda groups: [{k: v for k, v in group.items() if k not in port_keys} for group in groups]
    if strip_ports(actual_groups) != strip_ports(structural_groups):
        raise ValueError('Actual replica GPU groups differ from candidate')
    if expected_groups is not None and actual_groups != expected_groups:
        raise ValueError('Actual replica mapping or ports differ from frozen plan')
    records = []
    for group in receipt['replicas']:
        for role, key in (('attention', 'attention'), ('ffn', 'expert')):
            for local in range(group[key + '_ranks']):
                rank = local + group[key + '_rank_offset']
                path = run / f'worker-layout-{role}-{rank}.json'
                r = json.loads(path.read_text())
                expected = {'native_tp': group[key + '_tp'],
                            'native_dp': group['attention_dp' if role == 'attention' else 'expert_native_dp'],
                            'enable_expert_parallel': group['expert_enable_ep'] if role == 'ffn' else False,
                            'microbatches': configuration['microbatches'], **thresholds}
                if (r.get('actual') != expected or r.get('verified') is not True or r.get('rank') != rank
                        or r.get('replica') != group['replica'] or r.get('local_role_rank') != local or r.get('role') != role):
                    raise ValueError(f'Worker layout mismatch: {path}')
                if role == 'ffn':
                    expected_shard = {'tp': group['expert_tp'], 'ep': group['expert_native_dp'] if group['expert_enable_ep'] else 1}
                    if not r.get('expert_shards') or any(s != expected_shard for s in r['expert_shards']):
                        raise ValueError(f'Actual expert shards mismatch: {path}')
                records.append(str(path))
    return receipt, records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    layout = json.loads(args.layout.read_text())
    receipt, paths = verify(layout['configuration'], args.run, thresholds=layout['microbatch_thresholds'])
    launch = args.run / 'launch_config.json'
    data = json.loads(launch.read_text())
    data.update(receipt, worker_layout_files=paths)
    launch.write_text(json.dumps(data, indent=2) + '\n')

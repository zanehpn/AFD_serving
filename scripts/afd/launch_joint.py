"""Launch independent AFD replicas with explicit expert sharding semantics."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'migration'))
from bo_layout import replicas, deployment
from launch_pair import command


def specifications(args):
    frozen = json.loads(args.layout_json.read_text())
    c = frozen['configuration']
    thresholds = frozen.get('microbatch_thresholds', {
        'dbo_decode_token_threshold': 2, 'dbo_prefill_token_threshold': 12})
    if any(getattr(args, key) != value or type(value) is not int or value <= 0
           for key, value in thresholds.items()):
        raise ValueError('Launcher thresholds differ from frozen joint layout')
    groups = replicas(c, args.api_port, args.expert_api_port, args.afd_port, args.dp_rpc_base_port)
    output = []
    for group in groups:
        for role in ('expert', 'attention'):
            child = copy.copy(args)
            for key in ('attention_ranks', 'expert_ranks', 'attention_tp', 'expert_tp',
                        'api_port', 'expert_api_port', 'afd_port', 'dp_rpc_base_port'):
                setattr(child, key, group[key])
            child.microbatches = c['microbatches']
            child.enable_dbo = int(c['microbatches'] == 2)
            # Attention owns no experts. Enabling EP there would trigger SP MoE
            # for TP>1/DP>1 and change the remote-proxy execution semantics.
            child.enable_expert_parallel = role == 'expert' and group['expert_enable_ep']
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, group[role + '_gpus'])),
                       ECODEP_AFD_REPLICA=str(group['replica']),
                       ECODEP_STAGE_RANK_OFFSET=str(group[role + '_rank_offset']),
                       ECODEP_EXPECTED_LAYOUT=json.dumps({**group, 'microbatches': c['microbatches'],
                                                         'dbo_decode_token_threshold': args.dbo_decode_token_threshold,
                                                         'dbo_prefill_token_threshold': args.dbo_prefill_token_threshold,
                                                         'role': role, 'directory': str(args.results.resolve())}))
            argv = command(child, role) + ['--generation-config', 'vllm', '--seed', '0']
            output.append({'role': role, 'replica': group['replica'], 'argv': argv, 'env': env})
    return c, groups, output


def run(args):
    c, groups, specs = specifications(args)
    args.results.mkdir(parents=True, exist_ok=True)
    children, logs = [], {}
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # Append to one role log so existing readiness and protocol checks also
        # observe every replica. Dedicated worker receipts prove each layout.
        for role in ('attention', 'expert'):
            logs[role] = (args.results / (role + '.log')).open('w')
        for item in specs:
            children.append(subprocess.Popen(item['argv'], env=item['env'], stdout=logs[item['role']],
                                             stderr=subprocess.STDOUT))
        if len(groups) > 1:
            proxy = [sys.executable, str(Path(__file__).with_name('replica_proxy.py')),
                     '--port', str(args.api_port), '--backends', *[str(g['api_port']) for g in groups]]
            children.append(subprocess.Popen(proxy, stdout=logs['attention'], stderr=subprocess.STDOUT))
        receipt = {**deployment(c), 'dbo_enabled': c['microbatches'] == 2,
                   'dbo_decode_token_threshold': args.dbo_decode_token_threshold,
                   'dbo_prefill_token_threshold': args.dbo_prefill_token_threshold,
                   'attention_gpus': ','.join(map(str, c['attention_gpus'])),
                   'expert_gpus': ','.join(map(str, c['expert_gpus'])),
                   'replicas': groups, 'request_dispatch': 'round_robin_no_retry'}
        (args.results / 'joint-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
        while not stopping:
            for child in children:
                if child.poll() is not None:
                    return child.returncode or 1
            time.sleep(.5)
    finally:
        stop()
        deadline = time.monotonic() + 30
        for child in children:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs.values():
            log.close()
    return 0

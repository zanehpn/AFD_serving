#!/usr/bin/env python3
"""Run complete independent mathematical static + dynamic campaigns on eight A100 GPUs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def plan(root, run_id, target='fixed'):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id) or run_id in {'.', '..'}:
        raise ValueError('run-id must be a simple directory name')
    layouts = [('deepseek-v2-lite', '0,1', '2,3', 18000, 18001, 16239, 29550, 9097),
               ('qwen36', '4,5', '6,7', 18100, 18101, 16339, 29650, 9098)]
    rows = []
    for model, attention, expert, api, expert_api, afd, dp, clock in layouts:
        rows.append({'model': model, 'workspace': str(Path(root) / 'parallel_runs' / run_id / model),
                     'environment': {'ECODEP_MODELS': model, 'ECODEP_STATIC_TARGET': target, 'ECODEP_ATTENTION_GPUS': attention,
                                     'ECODEP_EXPERT_GPUS': expert, 'ECODEP_API_PORT': str(api),
                                     'ECODEP_EXPERT_API_PORT': str(expert_api), 'ECODEP_AFD_PORT': str(afd),
                                     'ECODEP_DP_RPC_BASE_PORT': str(dp),
                                     'ECODEP_CLOCK_URL': f'http://127.0.0.1:{clock}',
                                     'ECODEP_EXECUTION_LAYOUT': 'parallel_8gpu_two_four_gpu_allocations'},
                     'ports': [api, expert_api, afd, dp, dp + 1, clock], 'clock_port': clock})
    return rows


def validate_dse(root, rows):
    # This describes the new search space; old one-shot decisions do not fix topology.
    for row in rows:
        row['static_dse'] = {'candidate_topologies': ['2A1E', '2A2E'],
                             'scope': 'measured_per_model_per_workload',
                             'allocation_includes_inactive_gpu': True,
                             'eight_gpu_global_optimality_claimed': False}


def git(*args):
    return subprocess.check_output(['git', *map(str, args)], text=True).strip()


def prepare_workspace(root, workspace, revision):
    root, workspace = Path(root), Path(workspace)
    if not workspace.exists():
        workspace.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(root), str(workspace)], check=True)
        subprocess.run(['git', '-C', str(workspace), 'checkout', '--detach', revision], check=True)
    if git('-C', workspace, 'rev-parse', 'HEAD') != revision or git('-C', workspace, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Existing parallel workspace revision differs; use a new --run-id.')
    for name in ['.venv', 'third_party', 'artifacts/models']:
        source, target = root / name, workspace / name
        if not source.exists():
            raise FileNotFoundError(f'{source}: finish setup/model downloads in the main checkout first')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f'Unexpected shared dependency target: {target}')
        elif target.exists():
            raise RuntimeError(f'Refusing to replace existing dependency: {target}')
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)


def probe_ports(rows):
    sockets = []
    try:
        for port in [p for row in rows for p in row['ports']]:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(('127.0.0.1', port))
    finally:
        for sock in sockets:
            sock.close()


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', default='a100-8gpu-math-full')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--hardware-profile', type=Path, help='Existing portable profile; default creates/reuses results/primitives/HARDWARE.json')
    mode.add_argument('--legacy-topology-probes', action='store_true', help='Explicitly use the old per-topology stage-equivalent protocol')
    parser.add_argument('--target', choices=('fixed', 'per-rate'), default='fixed')
    parser.add_argument('--dry-run', action='store_true', help='show assignments without creating files, processes or GPU contexts')
    args = parser.parse_args()
    rows = plan(ROOT, args.run_id, args.target)
    validate_dse(ROOT, rows)
    profile = (args.hardware_profile or ROOT / 'results/primitives/HARDWARE.json').resolve()
    for row in rows:
        if args.legacy_topology_probes:
            row['environment']['ECODEP_LEGACY_TOPOLOGY_PROBES'] = '1'
        else:
            row['environment']['ECODEP_HARDWARE_PROFILE'] = str(profile)
            row['primitive_profile_preparation'] = 'reuse_verified' if profile.exists() else 'generate_once_before_model_launches'
    if args.dry_run:
        print(json.dumps(rows, indent=2))
        return
    if os.geteuid() != 0:
        raise SystemExit('Run on the destination as root, after setup and model downloads.')
    base = ROOT / 'parallel_runs' / args.run_id
    base.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'parallel_runs/launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if git('-C', ROOT, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Commit or restore tracked changes before starting immutable parallel workspaces.')
    revision = git('-C', ROOT, 'rev-parse', 'HEAD')
    probe_ports(rows)
    if not args.legacy_topology_probes:
        if args.hardware_profile and not profile.exists():
            raise FileNotFoundError('Explicit --hardware-profile must already exist; omit it for automatic preparation')
        (base / 'STATUS.json').write_text(json.dumps({'status': 'preparing_shared_primitives'}) + '\n')
        try:
            subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'migration/ensure_hardware.py'),
                            '--profile', str(profile), '--gpus', '0,1,2,3,4,5,6,7'], check=True)
        except Exception as error:
            (base / 'STATUS.json').write_text(json.dumps({'status': 'failed', 'phase': 'shared_primitives', 'error': str(error)}) + '\n')
            raise
    for row in rows:
        prepare_workspace(ROOT, row['workspace'], revision)
    (base / 'PLAN.json').write_text(json.dumps({'revision': revision, 'models': rows}, indent=2) + '\n')
    clocks, campaigns, logs = [], {}, []
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for row in rows:
            workspace = Path(row['workspace'])
            env = {key: value for key, value in os.environ.items() if not key.startswith('ECODEP_')}
            env.update(row['environment'])
            env.update(PYTHIA_ALLOWED_GPUS=env['ECODEP_ATTENTION_GPUS'] + ',' + env['ECODEP_EXPERT_GPUS'],
                       PYTHIA_NVCTL_PORT=str(row['clock_port']))
            log = (base / f"{row['model']}-clock.log").open('ab'); logs.append(log)
            clock = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(workspace / 'services/nvcontrold.py')],
                                     env=env, stdout=log, stderr=log, start_new_session=True)
            clocks.append(clock)
            deadline = time.monotonic() + 20
            while True:
                if clock.poll() is not None:
                    raise RuntimeError('Clock service exited; inspect its log')
                try:
                    with urllib.request.urlopen(env['ECODEP_CLOCK_URL'] + '/health', timeout=1) as response:
                        health = json.load(response)
                    if set(health['allowed']) != set(map(int, env['PYTHIA_ALLOWED_GPUS'].split(','))):
                        raise RuntimeError('Clock-service GPU allowlist mismatch')
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Clock service startup timed out')
                    time.sleep(.2)
            log = (base / f"{row['model']}-campaign.log").open('ab'); logs.append(log)
            campaigns[row['model']] = subprocess.Popen(['bash', 'migration/run.sh'], cwd=workspace, env=env,
                                                       stdout=log, stderr=log, start_new_session=True)
        while any(p.poll() is None for p in campaigns.values()):
            states = {model: process.poll() for model, process in campaigns.items()}
            (base / 'STATUS.json').write_text(json.dumps({'status': 'running', 'returncodes': states}, indent=2) + '\n')
            time.sleep(2)
        states = {model: process.returncode for model, process in campaigns.items()}
        complete = all(code == 0 for code in states.values())
        (base / 'STATUS.json').write_text(json.dumps({'status': 'collecting_results' if complete else 'failed', 'returncodes': states}, indent=2) + '\n')
        if not complete:
            raise SystemExit('One or more model campaigns failed; inspect each model log and calibration report.')
        results = []
        for row in rows:
            path = Path(row['workspace']) / f"results/math-full/math-full-v1-{row['model']}/RESULTS.json"
            results.append(json.loads(path.read_text()))
        (base / 'RESULTS.json').write_text(json.dumps({'execution_layout': 'parallel_8gpu_two_four_gpu_allocations', 'phase': 'static_and_dynamic_complete', 'dynamic_or_heldout_started': True, 'models': results}, indent=2) + '\n')
        (base / 'STATUS.json').write_text(json.dumps({'status': 'complete', 'returncodes': states}) + '\n')
        print(f'Both models complete: {base / "RESULTS.json"}')
    except KeyboardInterrupt:
        (base / 'STATUS.json').write_text(json.dumps({'status': 'interrupted'}) + '\n')
        raise SystemExit(130)
    except Exception as error:
        (base / 'STATUS.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}) + '\n')
        raise
    finally:
        # Let experiment runners clean up their native services before stopping clocks.
        for process in campaigns.values():
            stop(process)
        for process in clocks:
            stop(process)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()

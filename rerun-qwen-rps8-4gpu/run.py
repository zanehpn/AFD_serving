"""Own the clock service and one bounded, fresh four-GPU Qwen V2 search."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'results/a6000-4gpu-qwen-rps8-20260917-latest-v2'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(p)

def progress(limits):
    p = RUN / 'search/qwen-rps8/comparison/v2-seed0/state.json'
    if not p.exists():
        return dict(attempts=0, failures=0, feasible=0, best=None, pending=False)
    state = read(p)
    observations = state['observations']
    passing = []
    for obs in observations:
        metrics = obs.get('metrics', {})
        if obs.get('status') != 'ok':
            continue
        if all((metrics.get('output_tps', -1) >= limit if key == 'min_output_tps'
                else metrics.get(key, float('inf')) <= limit) for key, limit in limits.items()):
            passing.append(obs)
    best = min(passing, key=lambda o: o['metrics']['energy_j']) if passing else None
    return dict(attempts=len(observations), failures=sum(o.get('status') != 'ok' for o in observations),
                feasible=len(passing), best=best, pending=bool(state.get('pending')), frozen=state.get('frozen'))

def main():
    owner = (HERE / 'owner.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / 'STATUS.json').exists():
        raise RuntimeError('Run already started; inspect its state before any restart')
    plan = read(RUN / 'PLAN.json')
    plan_hash = sha(RUN / 'PLAN.json')
    manifest = read(RUN / 'SOURCE_MANIFEST.json')
    for name, expected in manifest['files_sha256'].items():
        assert sha(RUN / name) == expected, name
    assert read(RUN / 'VALIDATION.json')['passed']
    record = dict(status='starting', pid=os.getpid(), plan_sha256=plan_hash,
                  driver_sha256=sha(__file__), started_at=datetime.now(timezone.utc).isoformat(),
                  scene='qwen-rps8', gpus=plan['gpus'], method='legal_contractions_v2',
                  heldout_evaluated=False, confirmation_evaluated=False)
    def save(status):
        assert sha(RUN / 'PLAN.json') == plan_hash
        record.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(),
                      progress=progress(plan['jobs'][0]['expected_limits']))
        write(RUN / 'STATUS.json', record)
    children = []
    env = dict(os.environ, PATH=os.environ.get('PATH', ''),
               PYTHIA_ALLOWED_GPUS='0,1,2,3', PYTHIA_NVCTL_PORT='19097',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               **plan['runtime_environment'])
    save('starting')
    def launch(command, name):
        with (RUN / name).open('ab') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child
    def wait(child, stage):
        while child.poll() is None:
            if clock.poll() is not None:
                raise RuntimeError('Clock service exited during ' + stage)
            save(stage)
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(stage + ' exited ' + str(child.returncode))
    try:
        # Do not stop or interfere with any unrelated GPU process.
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        uuids = subprocess.check_output(['nvidia-smi', '-i', '0,1,2,3', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
        assert not any(uuid in output for uuid in uuids), 'Selected GPUs are occupied'
        clock = launch([str(ROOT / '.venv-a6000-p2p/bin/python'),
                        str(RUN / 'source/services/nvcontrold.py')], 'clock.log')
        record['clock_pid'] = clock.pid
        for _ in range(30):
            if clock.poll() is not None:
                raise RuntimeError('Clock service startup failed')
            try:
                with urllib.request.urlopen('http://127.0.0.1:19097/health', timeout=1) as response:
                    health = json.load(response)
                assert health['allowed'] == [0, 1, 2, 3]
                assert health['protocol'] == 'nvcontrold.applied_ack.v2'
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('Clock service health timeout')
        scene = RUN / 'search/qwen-rps8'
        if not (scene / 'READY.json').exists():
            prep = launch(plan['jobs'][0]['command'], 'prepare.log')
            record['prepare_pid'] = prep.pid
            wait(prep, 'preparing')
        config = read(scene / 'official-config.json')
        settings = read(scene / 'campaign-settings.json')
        resolved_settings = read(scene / 'comparison/v2-seed0/bundle.json')['settings']
        assert config['gpus'] == [0, 1, 2, 3] and not config['resident_search']
        assert config['limits'] == plan['jobs'][0]['expected_limits'] == settings['limits']
        assert resolved_settings['bo']['capacity_policy_revision'] == 'legal_contractions_v2'
        assert config['comparison_methods'] == ['v2'] and config['evaluation_requests'] == 200
        assert config['budget']['evaluations'] == 17 and config['warmup_requests'] == 8
        assert not read(scene / 'comparison/v2-seed0/state.json')['observations']
        write(RUN / 'PREPARATION_VERIFIED.json', dict(passed=True, limits=config['limits'],
              policy=resolved_settings['bo']['capacity_policy_revision'], fresh_optimizer=True,
              current_hardware=config['hardware']['host'], inherited_max=True,
              source_hashes_verified=True, cold_start_per_trial=True))
        search = launch([sys.executable, str(RUN / 'source/bo_dse/official.py'), 'run', str(scene)], 'search.log')
        record['search_pid'] = search.pid
        write(ROOT / 'results/A6000_ACTIVE_RUN.json', dict(directory=str(RUN)))
        wait(search, 'searching')
        state = read(scene / 'comparison/v2-seed0/state.json')
        assert len(state['observations']) == 16 and not state.get('pending')
        save('finishing_cleanup')
    except BaseException as error:
        record['error'] = repr(error)
        save('needs_attention')
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    save('complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.detach:
        with (HERE / 'supervisor.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'supervisor_pid': child.pid}))
    else:
        main()

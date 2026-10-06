"""Supervise the fresh eight-card calibration queue and its clock service."""
import argparse
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
RUN = ROOT/'results/a6000-8gpu-rps8-16-20260916'
QUEUE_SCRIPT = None


def main():
    lock = (RUN/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN/'STATUS.json').exists():
        raise RuntimeError('Existing queue status requires inspection before restart')
    for model in json.loads((ROOT/'environment/models.lock.json').read_text()):
        directory = ROOT/'artifacts/models'/model['name']
        assert hashlib.sha256((directory/'config.json').read_bytes()).hexdigest() == model['config_sha256']
        index = json.loads((directory/'model.safetensors.index.json').read_text())
        for name in set(index['weight_map'].values()):
            assert (directory/name).is_file(), f'Missing model shard: {directory/name}'
    env = dict(os.environ, PATH=os.environ.get('PATH', ''),
               PYTHIA_ALLOWED_GPUS='0,1,2,3,4,5,6,7', PYTHIA_NVCTL_PORT='19096',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    children = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    record = dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  state='starting', heldout_evaluated=False)
    def save():
        temporary = RUN/'SUPERVISOR.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(RUN/'SUPERVISOR.json')
    save()
    try:
        with (RUN/'clock.log').open('ab') as log:
            clock = subprocess.Popen([str(ROOT/'.venv-a6000-p2p/bin/python'),
                str(RUN/'source/services/nvcontrold.py')], env=env, cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(clock)
        record['clock_pid'] = clock.pid
        for _ in range(60):
            if clock.poll() is not None:
                raise RuntimeError('Clock service exited; inspect clock.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:19096/health', timeout=1) as response:
                    health = json.load(response)
                assert health['allowed'] == list(range(8))
                assert health['protocol'] == 'nvcontrold.applied_ack.v2'
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('Clock service startup timed out')
        with (RUN/'queue.log').open('ab') as log:
            queue = subprocess.Popen([str(ROOT/'bo_dse/.venv/bin/python'),
                str(QUEUE_SCRIPT or RUN/'run_queue.py'), str(RUN)], env=env, cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(queue)
        record.update(state='running', queue_pid=queue.pid)
        save()
        while queue.poll() is None:
            if clock.poll() is not None:
                raise RuntimeError('Clock service stopped during the experiment')
            time.sleep(5)
        if queue.returncode:
            raise RuntimeError(f'Queue exited {queue.returncode}; inspect STATUS.json and queue.log')
        record['state'] = 'search_finished'
    except BaseException as error:
        record.update(state='needs_attention', error=repr(error))
        raise
    finally:
        for child in reversed(children):
            # Each process group was created by this supervisor.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=RUN)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--queue-script', type=Path)
    args = parser.parse_args()
    RUN = args.directory.resolve()
    QUEUE_SCRIPT = args.queue_script.resolve() if args.queue_script else None
    if args.detach:
        with (RUN/'supervisor.log').open('ab') as log:
            command = [sys.executable, str(Path(__file__).resolve()), '--directory', str(RUN)]
            if QUEUE_SCRIPT:
                command += ['--queue-script', str(QUEUE_SCRIPT)]
            process = subprocess.Popen(command,
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(3)
        print(json.dumps(dict(supervisor_pid=process.pid, exit_code=process.poll())))
        if process.poll() is not None:
            raise SystemExit(process.returncode or 1)
    else:
        main()

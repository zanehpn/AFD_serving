"""Start the prepared latest-V2 run after all baseline arms finish and shut down."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('generic_bo', 'random', 'ga')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed(previous, scenes, attempts):
    if read(previous / 'STATUS.json')['status'] != 'active_methods_finished_v2_paused':
        return False
    if read(previous / 'SUPERVISOR.json')['state'] != 'search_finished':
        return False
    for scene in scenes:
        for method in METHODS:
            state = read(previous / 'search' / scene / 'comparison' / f'{method}-seed0/state.json')
            if len(state['observations']) != attempts or state.get('pending'):
                return False
    return True


def run(schedule_path):
    schedule = read(schedule_path)
    previous, following = (Path(schedule[k]) for k in ('previous', 'following'))
    state_path = schedule_path.with_name('CHAIN_STATUS.json')
    lock = schedule_path.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state_path.exists() and read(state_path).get('state') == 'started':
        return
    record = dict(pid=os.getpid(), previous=str(previous), following=str(following))

    def save(state, **extra):
        record.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write(state_path, record)

    def verify():
        for name, expected in schedule['frozen_files'].items():
            if sha(Path(name)) != expected:
                raise RuntimeError(f'Scheduled plan or driver changed: {name}')

    try:
        verify()
        save('waiting_for_baselines')
        while not completed(previous, schedule['scenes'], schedule['attempts_per_arm']):
            current = read(previous / 'STATUS.json')
            save('waiting_for_baselines', previous_status=current['status'],
                 current_job=current.get('current_job'))
            time.sleep(15)
        verify()
        # Both supervisor locks prove no competing supervisor owns either run.
        locks = []
        try:
            for directory in (previous, following):
                handle = (directory / 'supervisor.lock').open('a')
                locks.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not completed(previous, schedule['scenes'], schedule['attempts_per_arm']):
                raise RuntimeError('Baseline completion changed at handoff')
            retained = {}
            for scene in schedule['scenes']:
                directory = previous / 'search' / scene
                retained[scene] = {}
                for method in METHODS:
                    path = directory / 'comparison' / f'{method}-seed0/state.json'
                    state = read(path)
                    retained[scene][method] = dict(directory=str(path.parent),
                        state_sha256=sha(path), attempts=len(state['observations']),
                        failed=sum(o['status'] != 'ok' for o in state['observations']),
                        frozen=state.get('frozen', False),
                        limits=read(directory / 'official-config.json')['limits'])
            transition = following / 'execution-transitions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-auto-start')
            transition.mkdir(parents=True)
            for name in ('STATUS.json', 'SUPERVISOR.json', 'PAUSED.json', 'RETAINED_BASELINE_METHODS.json'):
                path = following / name
                if path.exists():
                    path.rename(transition / name)
            write(following / 'RETAINED_BASELINE_METHODS.json', dict(
                source_directory=str(previous), methods=retained,
                rule='Completed original baseline arms retained read-only; new V2 uses matching original MAX and SLOs.'))
            write(transition / 'HANDOFF.json', dict(schedule=str(schedule_path),
                previous_finished=True, baseline_attempts=schedule['attempts_per_arm'],
                active_methods=['v2'], previous_results_preserved=True))
        finally:
            for handle in reversed(locks):
                handle.close()
        save('starting_latest_v2')
        subprocess.run([sys.executable, str(ROOT / 'tools/launch_a6000_8gpu.py'),
            '--directory', str(following), '--queue-script',
            str(ROOT / 'tools/run_a6000_resident_queue.py'), '--detach'], check=True)
        for _ in range(60):
            sup = read(following / 'SUPERVISOR.json')
            if sup['state'] == 'needs_attention':
                raise RuntimeError(f'V2 startup failed: {sup.get("error")}')
            if sup['state'] in ('running', 'search_finished') and (following / 'STATUS.json').exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError('V2 startup status timed out')
        write(ROOT / 'results/A6000_ACTIVE_RUN.json', dict(directory=str(following)))
        save('started', supervisor_pid=sup['pid'])
    except BaseException as error:
        save('needs_attention', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('schedule', type=Path)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.detach:
        with args.schedule.with_suffix('.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                str(args.schedule.resolve())], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2)
        print(json.dumps(dict(pid=process.pid, exit_code=process.poll())))
        if process.poll() is not None:
            raise SystemExit(process.returncode or 1)
    else:
        run(args.schedule.resolve())

"""Run a frozen four-method PLAN serially; retain failures and stop on errors."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def progress(directory):
    result = {}
    for path in sorted((directory/'comparison').glob('*/state.json')):
        state = read(path)
        observations = state['observations']
        result[path.parent.name] = dict(attempts=len(observations),
            failed=sum(o.get('status') != 'ok' for o in observations),
            pending=bool(state.get('pending')), frozen=bool(state.get('frozen')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    lock = (root/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(root/'PLAN.json')
    plan_hash = hashlib.sha256((root/'PLAN.json').read_bytes()).hexdigest()
    env = dict(os.environ, PATH=os.environ.get('PATH', ''),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    record = dict(status='running', plan_sha256=plan_hash, pid=os.getpid(),
                  started_at=datetime.now(timezone.utc).isoformat(), jobs={})

    def save():
        record['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(root/'STATUS.json', record)

    def execute(command, logfile):
        if hashlib.sha256((root/'PLAN.json').read_bytes()).hexdigest() != plan_hash:
            raise ValueError('PLAN changed after queue launch')
        with logfile.open('ab') as stream:
            process = subprocess.Popen(command, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            record['child_pid'] = process.pid
            record['current_log'] = str(logfile)
            save()
            code = process.wait()
        if code:
            raise RuntimeError(f'Command exited {code}; inspect {logfile}')

    save()
    try:
        if plan.get('baseline_first'):
            baselines = {}
            for job in plan['jobs']:
                name = job['id']
                directory = root/'search'/name
                record['current_job'] = name
                record['jobs'][name] = dict(status='measuring_max', arms=progress(directory))
                save()
                if not (directory/'READY.json').exists():
                    if directory.exists():
                        raise RuntimeError(f'Incomplete MAX preparation: {directory}; inspect before resuming')
                    execute(job['command'], root/f'{name}-prepare.log')
                config = read(directory/'official-config.json')
                reference = read(directory/'inputs/slo-reference.json')
                baselines[name] = dict(metrics=reference['metrics'], limits=reference['limits'],
                    configuration=reference['configuration'], receipt=reference['receipt'],
                    original_a100_limits=job['original_a100_limits'])
                write(root/'BASELINES.json', baselines)
                record['jobs'][name].update(status='max_measured', local_limits=config['limits'])
                save()
        for job in plan['jobs']:
            name = job['id']
            directory = root/'search'/name
            record['current_job'] = name
            row = record['jobs'].setdefault(name, dict(status='preparing', arms=progress(directory)))
            save()
            command = job['command']
            if not (directory/'READY.json').exists():
                if directory.exists():
                    raise RuntimeError(f'Incomplete preparation already exists: {directory}; inspect before resuming')
                execute(command, root/f'{name}-prepare.log')
            row['status'] = 'searching'
            for round_number in range(plan['search_attempts_per_arm']+2):
                before = progress(directory)
                row.update(round=round_number+1, arms=before)
                save()
                execute(command[:2]+['run', str(directory), '--one'], root/f'{name}-search.log')
                after = progress(directory)
                row['arms'] = after
                save()
                if before == after:
                    row['status'] = 'search_finished'
                    row['completion_scope'] = 'Search stopped by configured budget/eligibility; consult per-arm feasible results'
                    break
            else:
                raise RuntimeError('Search did not stop within its finite attempt budget')
            save()
        record['status'] = 'search_finished'
        record['heldout_evaluated'] = False
        record['confirmation_evaluated'] = False
        save()
    except BaseException as error:
        record.update(status='stopped_for_inspection', error=str(error))
        if record.get('current_job'):
            record['jobs'][record['current_job']]['arms'] = progress(root/'search'/record['current_job'])
        save()
        raise


if __name__ == '__main__':
    main()

"""Finish frozen baselines, then keep one resident session per search scene.

The execution overlay changes service lifetime only. Frozen source, requests,
model configurations, SLOs, optimizer order and search budgets stay intact.
"""
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
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def progress(directory):
    arms = {}
    for path in sorted((directory/'comparison').glob('*/state.json')):
        state = read(path)
        observations = state['observations']
        arms[path.parent.name] = dict(attempts=len(observations),
            failed=sum(o.get('status') != 'ok' for o in observations),
            reused=sum(o.get('resident_service', {}).get('reused', False) for o in observations),
            pending=bool(state.get('pending')), frozen=bool(state.get('frozen')))
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory.resolve()
    lock = (root/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = root/'PLAN.json'
    plan = read(plan_path)
    plan_hash = sha(plan_path)
    overlay_path = root/'RESIDENT_EXECUTION.json'
    overlay = read(overlay_path)
    assert overlay['plan_sha256'] == plan_hash
    assert overlay['driver_sha256'] == sha(__file__)
    sys.path.insert(0, str(Path(plan['source'])/'bo_dse'))
    import official as o
    record = dict(status='running', pid=os.getpid(), plan_sha256=plan_hash,
                  execution_overlay=o.artifact(overlay_path), resident_search=True,
                  started_at=datetime.now(timezone.utc).isoformat(), jobs={})
    def save():
        record['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(root/'STATUS.json', record)
    def check():
        if sha(plan_path) != plan_hash or read(overlay_path) != overlay:
            raise RuntimeError('Frozen plan/execution overlay changed')
    save()
    try:
        baselines = read(root/'BASELINES.json') if (root/'BASELINES.json').exists() else {}
        for job in plan['jobs']:
            check()
            name = job['id']
            directory = root/'search'/name
            record.update(current_job=name)
            record['jobs'][name] = dict(status='measuring_max', arms=progress(directory))
            save()
            if not (directory/'READY.json').exists():
                if directory.exists():
                    raise RuntimeError(f'Incomplete preparation: {directory}; inspect before resuming')
                logfile = root/f'{name}-prepare.log'
                with logfile.open('ab') as stream:
                    process = subprocess.Popen(job['command'], cwd=root,
                        stdout=stream, stderr=subprocess.STDOUT)
                    record.update(child_pid=process.pid, current_log=str(logfile))
                    save()
                    if process.wait():
                        raise RuntimeError(f'Preparation failed; inspect {logfile}')
            config = o.read(directory/'official-config.json')
            o.validate_context(config)
            reference = o.read(directory/'inputs/slo-reference.json')
            baselines[name] = dict(metrics=reference['metrics'], limits=reference['limits'],
                configuration=reference['configuration'], receipt=reference['receipt'],
                original_a100_limits=job['original_a100_limits'])
            write(root/'BASELINES.json', baselines)
            record['jobs'][name].update(status='max_measured', local_limits=config['limits'])
            save()
        for job in plan['jobs']:
            check()
            name = job['id']
            directory = root/'search'/name
            config_path = directory/'official-config.json'
            original_hash = sha(config_path)
            config = dict(o.read(config_path), resident_search=True)
            # This explicit overlay leaves official-config.json byte-identical;
            # existing campaign request hashes and source checks remain valid.
            record.update(current_job=name, child_pid=None, current_log=str(root/'queue.log'))
            record['jobs'][name].update(status='searching', resident_search=True)
            save()
            original_evaluate, original_tell = o.evaluate, o.tell
            def evaluate(runtime_config, request, trial):
                check()
                if sha(config_path) != original_hash:
                    raise RuntimeError('Original official configuration changed')
                result = original_evaluate(runtime_config, request, trial)
                result['artifacts'].append(o.artifact(overlay_path))
                result['execution_overlay'] = dict(resident_search=True,
                    original_config_sha256=original_hash, manifest=o.artifact(overlay_path))
                o.write(Path(trial)/'result.json', result)
                return result
            def tell(path, result):
                value = original_tell(path, result)
                record['jobs'][name].update(arms=progress(directory),
                    last_status=result['status'],
                    last_service_reused=result.get('resident_service', {}).get('reused', False))
                save()
                return value
            o.evaluate, o.tell = evaluate, tell
            try:
                # One run covers every round and method in this scene. --one
                # would close Session and reload at each round boundary.
                report = o.run(config)
            finally:
                o.evaluate, o.tell = original_evaluate, original_tell
            write(root/f'{name}-summary.json', report)
            record['jobs'][name].update(status='search_finished', arms=progress(directory))
            save()
        record.update(status='search_finished', heldout_evaluated=False, confirmation_evaluated=False)
        save()
    except BaseException as error:
        record.update(status='stopped_for_inspection', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()

"""Continue existing RPS8 searches on GPUs 0,1,2,4 without replaying receipts."""
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

BASE = Path(__file__).resolve().parent
PRIMARY = '--primary' in sys.argv
OLD = BASE.with_name('dbo_on_1256_v3' if PRIMARY else 'dbo_on_0347_v3')
sys.path.insert(0, str(OLD))
import run_experiment as r
from mapping import GPU_MAP, PRIMARY_MAP, PHYSICAL_GPUS, load_manifest
if PRIMARY:GPU_MAP=PRIMARY_MAP

original_worker = r.o.worker
original_locks = r.o.gpu_locks

def worker(config, action, directory, candidate=None):
    root = r.o.ROOT
    r.o.ROOT = BASE/'worker-runtime'
    try:
        return original_worker(config, action, directory, candidate)
    finally:
        r.o.ROOT = root

def wait_idle():
    import pynvml as nv
    nv.nvmlInit()
    try:
        while True:
            busy = [{ 'gpu':g, 'pids':[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByIndex(g))] } for g in PHYSICAL_GPUS]
            busy = [v for v in busy if v['pids']]
            if not busy: return
            r.status('waiting_for_gpus', gpus=PHYSICAL_GPUS, busy=busy, migration=str(BASE/'MIGRATION.json'))
            time.sleep(30)
    finally:
        nv.nvmlShutdown()

r.o.worker = worker
r.o.gpu_locks = lambda gpus: original_locks([GPU_MAP[g] for g in gpus])
r.wait_for_available_gpus = wait_idle

def main():
    load_manifest()
    plan = r.o.read(OLD/'PLAN.json')
    for path, digest in plan['files_sha256'].items():
        assert r.o.sha(path) == digest, path
    assert r.o.sha(r.o.read(OLD/'container.json')['path']) == r.o.read(OLD/'container.json')['sha256']
    with (OLD/'RUNNING.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_idle()
        reports = []
        for job in (plan['jobs'][:2] if PRIMARY else plan['jobs']):
            cfg = r.initialize_job(job)
            r.status('searching', job=job['id'], physical_gpus=PHYSICAL_GPUS)
            r.o.run(cfg)
            reports.append(r.confirm(job, cfg))
            r.o.write(OLD/'RESULTS.json', dict(complete=False, groups=reports, migration=str(BASE/'MIGRATION.json')))
        r.o.write(OLD/'RESULTS.json', dict(complete=True, groups=reports, migration=str(BASE/'MIGRATION.json')))
        r.status('complete', physical_gpus=PHYSICAL_GPUS)

if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try: main()
    except BaseException as error:
        r.status('failed', error=repr(error), physical_gpus=PHYSICAL_GPUS)
        raise

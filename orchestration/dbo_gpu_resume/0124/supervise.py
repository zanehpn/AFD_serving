"""Monitor the migrated lane; serialize the overlapping original lane after it."""
import json
import os
import sys
import time
from pathlib import Path
BASE = Path(__file__).resolve().parent
COMBINED = BASE.with_name('dbo_on_parallel_v3')
sys.path.insert(0,str(COMBINED))
import coordinate as c
import inspect_progress as progress
import resume_after_contention as recovery

def main():
    import pynvml as nv
    nv.nvmlInit()
    started_primary = False
    while True:
        launch = c.read(COMBINED/'LAUNCH.json')
        extra = c.read(c.EXTRA/'STATUS.json')
        extra_live = c.alive(launch['extra'])
        if not extra_live and extra.get('phase') == 'complete' and not started_primary:
            busy = recovery.busy([0,1,2,4])
            if not busy:
                launch['primary'] = c.launch('primary-after-0124', BASE/'run_resume.py', ['--primary'])
                c.write(COMBINED/'LAUNCH.json', launch)
                started_primary = True
        primary = c.read(c.OLD/'STATUS.json')
        primary_live = c.alive(launch['primary'])
        lanes = {'extra':{'process':launch['extra'],'live':extra_live,'status':extra,'physical_gpus':[0,1,2,4]},
                 'primary':{'process':launch['primary'],'live':primary_live,'status':primary,
                            'physical_gpus':[0,1,2,4], 'scheduling':'All remaining experiments serialized on the user-selected pool'}}
        groups=[]
        for root,names in [(c.OLD,['deepseek-rps4','qwen-rps4']),(c.EXTRA,['deepseek-rps8','qwen-rps8'])]:
            for name in names:
                path=root/'confirmation'/name/'RESULTS.json'
                if path.exists():groups.append(c.read(path))
        done = not extra_live and not primary_live and extra.get('phase') == primary.get('phase') == 'complete'
        c.status('searches_complete_validation_mapping_pending' if done else 'running_on_0124' if extra_live else 'primary_pending_or_running',
                 lanes=lanes, physical_migration=str(BASE/'MIGRATION.json'), measurements_rerun=False)
        c.write(COMBINED/'RESULTS.json',dict(complete=False, searches_complete=done,groups=groups,lanes=lanes,
                  split='calibration_confirmation',validation_gate='Freeze and audit physical mapping before heldout launch'))
        original = progress.collect
        def collect():
            report = original();report['runner_live']=True;report['physical_migration']=str(BASE/'MIGRATION.json');return report
        progress.collect=collect
        try:progress.main()
        finally:progress.collect=original
        if done or (not extra_live and extra.get('phase')=='failed'):break
        time.sleep(30)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        c.status('migration_supervisor_needs_inspection',error=repr(error))
        raise

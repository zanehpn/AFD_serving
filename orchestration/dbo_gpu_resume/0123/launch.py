import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path

BASE=Path(__file__).resolve().parent
COMBINED=BASE.with_name('dbo_on_parallel_v3')
sys.path.insert(0,str(COMBINED))
import coordinate as c
from mapping import load_manifest
import pynvml as nv

def request(path, body=None):
    req=urllib.request.Request('http://127.0.0.1:19099'+path,
        data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)

def launch(name, script):
    with (BASE/(name+'.log')).open('ab') as log:
        import subprocess
        process=subprocess.Popen([sys.executable,str(BASE/script)],env=c.ENV,stdout=log,stderr=log,start_new_session=True)
    return dict(pid=process.pid,start_ticks=int((Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]),command=str(BASE/script),started_utc=c.utc())

load_manifest()
record=c.read(COMBINED/'LAUNCH.json')
assert not c.alive(record['extra']) and not c.alive(record['primary'])
assert not (BASE/'LAUNCH.json').exists(), 'Inspect previous launch before starting another'
nv.nvmlInit()
assert all(not nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByIndex(g)) for g in [0,1,2,3])
nv.nvmlShutdown()
controller=launch('dcgm-control','dcgm_control.py')
c.write(BASE/'LAUNCH.json',dict(controller=controller,phase='control_preflight'))
for _ in range(40):
    try:
        health=request('/health');assert health['allowed']==[0,1,2,3];break
    except OSError:time.sleep(.5)
else:raise RuntimeError('Mapped clock controller unavailable')
checks=[]
for gpu in [0,1,2,3]:
    try:
        power=request('/set_power_limit',dict(gpu=gpu,watts=200))
        clock=request('/set_clock',dict(gpu=gpu,sm_mhz=1050))
        assert power['applied_w']==200 and clock['applied_mhz']==1050
        checks.append(dict(gpu=gpu,power=power,clock=clock))
    finally:
        request('/reset',dict(gpu=gpu))
        request('/set_power_limit',dict(gpu=gpu,reset=True))
c.write(BASE/'CLOCK_PREFLIGHT.json',dict(passed=True,checks=checks,restored=True))
runner=launch('runner','run_resume.py')
record['extra']=runner
record['physical_migration']=str(BASE/'MIGRATION.json')
record['primary_scheduling']='Resume after RPS8 completes; shared GPUs 1,2 cannot run concurrently'
c.write(COMBINED/'LAUNCH.json',record)
supervisor=launch('supervisor','supervise.py')
record['coordinator_pid']=supervisor['pid'];record['supervisor_command']=supervisor['command']
c.write(COMBINED/'LAUNCH.json',record)
c.write(BASE/'LAUNCH.json',dict(controller=controller,extra=runner,supervisor=supervisor,physical_gpus=[0,1,2,3]))
c.write(COMBINED/'validation/STATUS.json',dict(phase='waiting_for_migrated_searches_and_physical_freeze',heldout_evaluated=False,
            reason='Physical GPU pool changed by user; freeze mapping consistently before any heldout evaluation'))
print(json.dumps(dict(runner=runner,physical_gpus=[0,1,2,3],prior_measurements_rerun=False)))

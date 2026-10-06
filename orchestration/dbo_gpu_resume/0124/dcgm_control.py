"""DCGM application-clock/power controller restricted to the extra GPU pool."""
import atexit
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pynvml as nv

ALLOWED = [0,1,2,4]
GROUPS = {}
INITIAL = {}
TOUCHED = set()
UNLOCKED = set()
LOCK = threading.RLock()
BASE = Path(__file__).resolve().parent

def command(*args):
    p=subprocess.run(['dcgmi',*(format(x, 'g') if isinstance(x, float) else str(x) for x in args)],capture_output=True,text=True,timeout=20)
    if p.returncode:
        raise RuntimeError(p.stdout+p.stderr)
    return p.stdout

def h(gpu):return nv.nvmlDeviceGetHandleByIndex(gpu)
def state(gpu):
    handle=h(gpu)
    return dict(gpu=gpu,uuid=nv.nvmlDeviceGetUUID(handle),
        application_memory_mhz=nv.nvmlDeviceGetApplicationsClock(handle,nv.NVML_CLOCK_MEM),
        application_sm_mhz=nv.nvmlDeviceGetApplicationsClock(handle,nv.NVML_CLOCK_SM),
        sm_clock_mhz=nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_SM),
        power_limit_w=nv.nvmlDeviceGetEnforcedPowerLimit(handle)/1000,
        clock_control='application',control_backend='dcgm_config')

def readback(gpu, expected):
    deadline=time.monotonic()+3
    while True:
        actual=state(gpu)
        if all(actual[k]==v for k,v in expected.items()):return actual
        if time.monotonic()>=deadline:raise RuntimeError(f'DCGM readback mismatch: expected {expected}, actual {actual}')
        time.sleep(.05)

def unlock_if_needed(gpu):
    if gpu in UNLOCKED:return
    if nv.nvmlDeviceGetComputeRunningProcesses(h(gpu)):
        raise RuntimeError('GPU has compute processes before first control')
    TOUCHED.add(gpu)
    # Existing NVML controllers can clear old locked clocks on these devices.
    port={0:9095,1:9095,2:9095,4:9096}.get(gpu)
    if port:
        req=urllib.request.Request(f'http://127.0.0.1:{port}/reset',data=json.dumps({'gpu':gpu}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=10) as r:
            assert json.load(r)['clock_reset'] is True
    UNLOCKED.add(gpu)

def set_value(gpu,body,clock=False):
    unlock_if_needed(gpu)
    TOUCHED.add(gpu)
    start=time.monotonic()
    if clock:
        value=int(body['sm_mhz']);mem=nv.nvmlDeviceGetDefaultApplicationsClock(h(gpu),nv.NVML_CLOCK_MEM)
        command('config','-g',GROUPS[gpu],'--set','-a',f'{mem},{value}')
        actual=readback(gpu,{'application_sm_mhz':value,'application_memory_mhz':mem})
        return dict(gpu=gpu,requested_mhz=value,applied_mhz=value,memory_mhz=mem,clock_control='application',control_backend='dcgm_config',readback=actual,set_ms=1000*(time.monotonic()-start))
    value=nv.nvmlDeviceGetPowerManagementDefaultLimit(h(gpu))/1000 if body.get('reset') else float(body['watts'])
    command('config','-g',GROUPS[gpu],'--set','-P',value)
    actual=readback(gpu,{'power_limit_w':value})
    return dict(gpu=gpu,requested_w=value,applied_w=value,power_limit_w=value,power_control='default' if body.get('reset') else 'limited',control_backend='dcgm_config',readback=actual)

def reset(gpu):
    if gpu not in TOUCHED:return dict(gpu=gpu,clock_reset=True,clock_control='reset')
    original=INITIAL[gpu]
    command('config','-g',GROUPS[gpu],'--set','-a',f"{original['application_memory_mhz']},{original['application_sm_mhz']}")
    actual=readback(gpu,{k:original[k] for k in ['application_memory_mhz','application_sm_mhz']})
    return dict(gpu=gpu,clock_reset=True,clock_control='reset',restored_application_clocks=True,readback=actual)

class Handler(BaseHTTPRequestHandler):
    def reply(self,code,body):
        data=json.dumps(body).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path=='/health':return self.reply(200,dict(ok=True,allowed=ALLOWED,protocol='nvcontrold.applied_ack.v2',clock_control='application',control_backend='dcgm_config'))
        if self.path=='/state':return self.reply(200,dict(gpus=[state(g) for g in ALLOWED]))
        self.reply(404,{'error':'unsupported endpoint'})
    def do_POST(self):
        try:
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));gpu=body.get('gpu')
            if type(gpu) is not int or gpu not in ALLOWED:return self.reply(403,{'error':'GPU outside allocation'})
            with LOCK:
                if self.path=='/set_clock':out=set_value(gpu,body,True)
                elif self.path=='/set_power_limit':out=set_value(gpu,body)
                elif self.path=='/reset':out=reset(gpu)
                else:return self.reply(404,{'error':'unsupported endpoint'})
            self.reply(200,out)
        except Exception as e:self.reply(500,{'error':repr(e)})

def cleanup():
    with LOCK:
        errors=[]
        for gpu in list(GROUPS):
            try:
                if gpu in TOUCHED:
                    reset(gpu)
                    command('config','-g',GROUPS[gpu],'--set','-P',INITIAL[gpu]['power_limit_w'])
                command('group','-d',GROUPS[gpu]);del GROUPS[gpu]
            except Exception as e:errors.append({'gpu':gpu,'error':repr(e)})
        (BASE/'DCGM_CONTROL_SHUTDOWN.json').write_text(json.dumps({'errors':errors},indent=2)+'\n')

def main():
    nv.nvmlInit();atexit.register(cleanup)
    for gpu in ALLOWED:
        INITIAL[gpu]=state(gpu)
        out=command('group','-c',f'dbo_parallel_{os.getpid()}_gpu{gpu}','-a',gpu)
        match=re.search(r'group ID of (\d+)',out)
        if not match:raise RuntimeError(out)
        GROUPS[gpu]=int(match.group(1))
    (BASE/'DCGM_CONTROL_START.json').write_text(json.dumps({'pid':os.getpid(),'initial':INITIAL,'groups':GROUPS,'mode':'application'},indent=2)+'\n')
    def stop(signum,frame):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    ThreadingHTTPServer(('127.0.0.1',19100),Handler).serve_forever()
if __name__=='__main__':main()

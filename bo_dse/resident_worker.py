"""Reuse owned AFD processes across serialized, fully drained measurements."""
import fcntl
import hashlib
import json
import time
from pathlib import Path

import official_worker as w


def service_key(config, c):
    # The real launch argv covers model, allocation, DP/TP/EP, DBO, ports,
    # memory/scheduler limits and backend. Frequency/cap are runtime controls.
    payload = {'launch': w.commands(config, c), 'model_config': w.artifact(Path(config['model_path'])/'config.json'),
               'worker_code': w.sha(Path(w.__file__)), 'resident_code': w.sha(Path(__file__))}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def healthy(config, session):
    records = [session/f'service-{role}.json' for role in ('attention','ffn')]
    if not all(p.exists() and w.group_members(w.read(p)) for p in records):
        return False
    try:
        w.http(f"http://127.0.0.1:{config['api_port']}/health")
        return True
    except (OSError, ValueError):
        return False


def check_identity_and_owners(config, monitor, session, reuse):
    owned = set()
    if reuse:
        for p in session.glob('service-*.json'):
            owned.update(pid for pid, _ in w.group_members(w.read(p)))
    for gpu,handle in zip(monitor.gpus,monitor.handles):
        if monitor.nv.nvmlDeviceGetUUID(handle) != config['hardware']['devices'][str(gpu)]['uuid']:
            raise ValueError('Physical GPU identity changed')
        processes = monitor.nv.nvmlDeviceGetComputeRunningProcesses(handle)
        if any(p.pid not in owned for p in processes):
            raise ValueError(f'GPU {gpu} has a compute process outside the resident service')


def invalidate(config, session):
    w.cleanup(config, session)
    w.write(session/'resident-state.json', {'valid':False})


def resident_trial(config, c, directory):
    directory=Path(directory);session=Path(config['resident_session_directory'])
    session.mkdir(parents=True,exist_ok=True)
    with (session/'.trial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _trial(config,c,directory,session)


def _trial(config,c,directory,session):
    started=time.monotonic();monitor=None;result={};reuse=False;phases={}
    try:
        w.verify_runtime()
        key=service_key(config,c)
        cache=w.read(session/'resident-state.json') if (session/'resident-state.json').exists() else {}
        reuse=cache.get('valid') is True and cache.get('key')==key and healthy(config,session)
        t=time.monotonic()
        if not reuse and (list(session.glob('service-*.json')) or (session/'control-ownership.json').exists()):
            invalidate(config,session)
        phases['previous_service_cleanup_s']=time.monotonic()-t
        if not reuse:w.check_ports(config)
        monitor=w.Monitor(c['attention_gpus']+c['expert_gpus'])
        check_identity_and_owners(config,monitor,session,reuse)
        monitor.start()
        w.write(session/'control-ownership.json',{'gpus':monitor.gpus})
        t=time.monotonic();w.clocks(config,c,directory);phases['apply_controls_s']=time.monotonic()-t
        t=time.monotonic()
        if not reuse:
            w.start_services(config,c,session)
            w.write(session/'resident-state.json',{'valid':True,'key':key})
        phases['service_start_s']=time.monotonic()-t
        # A separate warmup and complete replay occur on EVERY trial. The API
        # replay waits for all requests; prefix caching stays disabled.
        w.write(directory/'launch.json',w.commands(config,c))
        w.write(directory/'resident-services.json',{
            'session_directory':str(session),'key':key,'reused':reuse,
            'services':{p.name:w.read(p) for p in session.glob('service-*.json')}})
        t=time.monotonic();result=w.measure_requests(config,c,directory,monitor)
        phases['warmup_and_replay_s']=time.monotonic()-t
        if not healthy(config,session):raise RuntimeError('Resident service failed post-replay health check')
        result['resident_service']={'reused':reuse,'key':key,'cleanup_deferred':True}
        w.write(directory/'resident-handoff.json',{'healthy':True,'all_replay_requests_returned':True,
                                                 'prefix_caching':False,'services_retained':True})
    except BaseException as error:
        result={'status':'failed','failure_reason':f'{type(error).__name__}: {error}'}
        try:invalidate(config,session)
        except Exception as cleanup_error:
            result.update(cleanup_required=True,failure_reason=result['failure_reason']+'; cleanup: '+str(cleanup_error))
        if isinstance(error,(KeyboardInterrupt,SystemExit)):result['interrupted']=True
    finally:
        if monitor is not None:
            if monitor.thread.ident is not None:monitor.stop(directory)
            else:monitor.nv.nvmlShutdown()
        elapsed=time.monotonic()-started
        energy=0.
        if monitor is not None:
            energy=sum((b['timestamp_ns']-a['timestamp_ns'])/1e9*(sum(a['power_w'])+sum(b['power_w']))/2
                       for a,b in zip(monitor.rows,monitor.rows[1:]))
        result['cost']={'wall_seconds':elapsed,'gpu_hours':elapsed*len(config['gpus'])/3600,
                        'tuning_energy_j':max(energy,result.get('metrics',{}).get('energy_j',0))}
        # Driver-level reserved-GPU accounting also covers loading/teardown and
        # idle optimizer time outside this per-trial NVML window.
        result['cost_energy_complete']=False
        result['execution_protocol']='resident_afd_v1'
        result['phase_seconds']=phases
        names=('STARTED.json','launch.json','clock-receipts.json','nvml.jsonl','replay.jsonl','warmup.jsonl',
               'correctness.json','configuration.json','resident-services.json','resident-handoff.json')
        result['artifacts']=[w.artifact(directory/n) for n in names if (directory/n).exists()]
        w.write(directory/'worker-result.json',result)
    return result

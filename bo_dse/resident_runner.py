"""One allocation owner; all methods share serialized resident AFD services."""
import argparse
import json
from pathlib import Path
import signal
import time

import official as o
import official_worker as w


def window_cost(rows, gpu_count):
    elapsed=max(0.,(rows[-1]['timestamp_ns']-rows[0]['timestamp_ns'])/1e9) if len(rows)>1 else 0.
    energy=sum((b['timestamp_ns']-a['timestamp_ns'])/1e9*(sum(a['power_w'])+sum(b['power_w']))/2
               for a,b in zip(rows,rows[1:]))
    return {'wall_seconds':elapsed,'gpu_hours':elapsed*gpu_count/3600,'tuning_energy_j':energy}


def run(plan_path):
    plan=o.read(plan_path);session=Path(plan['session']);session.mkdir(parents=True,exist_ok=True)
    jobs=[dict(name=a['name'],base=Path(a['directory']),config=o.read(Path(a['directory'])/'official-config.json')) for a in plan['arms']]
    cfg=jobs[0]['config'];gpus=cfg['gpus']
    if any(j['config']['gpus']!=gpus or j['config']['resident_session_directory']!=str(session) for j in jobs):
        raise ValueError('Resident jobs must share an allocation and session')
    state={'done':{},'rounds':0,'current':'starting','plan':str(plan_path)};meter=None;last=0
    def save():o.write(session/'driver-status.json',state)
    with o.gpu_locks(gpus):
        try:
            # Recover only services owned by this session before taking over.
            if list(session.glob('service-*.json')):o.worker(cfg,'cleanup',session)
            meter=w.Monitor(gpus);meter.start();last=len(meter.rows)-1
            while not all(state['done'].get(j['name'],False) for j in jobs):
                for job in jobs:
                    name=job['name'];base=job['base'];config=job['config'];campaign=base/'campaign'
                    if state['done'].get(name):continue
                    state.update(current=name,updated=time.time());save()
                    if o.status(campaign)['frozen']:
                        state['done'][name]=True;save();continue
                    req=o.ask(campaign)
                    if req.get('stopped'):
                        if o.status(campaign)['best']:o.freeze(campaign)
                        state['done'][name]=True;save();continue
                    trial=campaign/'trials'/req['trial_id']
                    result=o.evaluate(config,req,trial)
                    meter.sample();end=len(meter.rows)-1;rows=list(meter.rows[last:end+1]);last=end
                    # Record all reserved-GPU power, including idle residence
                    # and optimizer/launch/switching work between measurements.
                    cost_path=trial/'resident-cost-window.jsonl'
                    cost_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    result['worker_cost']=result['cost'];result['cost']=window_cost(rows,len(gpus))
                    result['cost_energy_complete']=bool(len(rows)>1 and not meter.errors and all(
                        b['timestamp_ns']-a['timestamp_ns']<=1e9 for a,b in zip(rows,rows[1:])))
                    result['cost_protocol']='reserved_gpu_resident_interval_v1'
                    result['artifacts'].append(w.artifact(cost_path))
                    o.write(trial/'result.json',result)
                    o.tell(campaign,result)
                    state[name]={'last_trial':req['trial_id'],'status':result['status'],
                                 'reused':result.get('resident_service',{}).get('reused',False),
                                 'wall_seconds':result['cost']['wall_seconds']}
                    save()
                state['rounds']+=1
            state['current']='complete'
        except BaseException as error:
            state.update(current='failed',error=f'{type(error).__name__}: {error}');raise
        finally:
            try:o.worker(cfg,'cleanup',session)
            except BaseException as error:
                state['cleanup_error']=str(error)
            if meter is not None:
                meter.stop(session)
                o.write(session/'session-cost.json',{
                    'total':window_cost(meter.rows,len(gpus)),
                    'unassigned_shutdown_overhead':window_cost(meter.rows[last:],len(gpus)),
                    'sample_errors':meter.errors,
                    'rule':'Per-trial intervals cover resident idle/optimizer/switching costs; final shutdown overhead is reported separately.'})
            save()
    return state


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Interrupted by {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path)
    run(parser.parse_args().plan)

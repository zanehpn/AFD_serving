import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import official_worker as w
import resident_worker as resident
from resident_runner import window_cost


@pytest.fixture
def host(tmp_path,monkeypatch):
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    cfg={'model':'deepseek-v2-lite','model_path':str(model),'native_python':sys.executable,
         'gpus':[0,1,2,3],'api_port':18000,'afd_port':16000,'dp_rpc_port':28000,
         'max_model_len':8192,'resident_session_directory':str(tmp_path/'session'),
         'hardware':{'devices':{str(i):{'uuid':f'gpu-{i}'} for i in range(4)}}}
    c={'attention_gpus':[0,1],'expert_gpus':[2,3],'attention_dp':2,'attention_tp':1,
       'expert_dp':2,'expert_tp':1,'expert_ep':2,'parallelism_semantics':'vllm_dp_tp_ep_world',
       'attention_mhz':1410,'expert_mhz':1410,'attention_power_w':400,'expert_power_w':400,
       'microbatches':1,'execution_mode':'eager'}
    calls=[];live={'value':False,'fail_replay':False,'foreign':False,'health':True}
    class Monitor:
        def __init__(self,gpus):
            self.gpus=gpus;self.handles=gpus;self.rows=[];self.errors=[];self.thread=SimpleNamespace(ident=None)
            self.nv=SimpleNamespace(nvmlDeviceGetUUID=lambda h:f'gpu-{h}',nvmlShutdown=lambda:None,
                nvmlDeviceGetComputeRunningProcesses=lambda h:([SimpleNamespace(pid=999)] if live['foreign'] else [SimpleNamespace(pid=100)] if live['value'] else []))
        def start(self):self.thread.ident=1;self.rows=[{'timestamp_ns':0,'power_w':[10]*len(self.gpus)}]
        def stop(self,d):self.rows.append({'timestamp_ns':1000000000,'power_w':[10]*len(self.gpus)})
    def start(config,c,d):
        calls.append('start');live['value']=True
        for role in ['attention','ffn']:w.write(d/f'service-{role}.json',{'pid':100,'start_ticks':'1','boot_id':'test'})
    def cleanup(config,d):
        calls.append('cleanup');live['value']=False
        for p in d.glob('service-*.json'):w.write(p,{'stopped':True})
    def measure(config,c,d,monitor):
        calls.append('warmup+replay')
        if live['fail_replay']:raise RuntimeError('replay failed')
        return {'status':'ok','metrics':{'energy_j':10}}
    def health(*args):
        if not live['health']:raise OSError('unhealthy')
        return {}
    monkeypatch.setattr(w,'Monitor',Monitor);monkeypatch.setattr(w,'verify_runtime',lambda:None)
    monkeypatch.setattr(w,'check_ports',lambda c:calls.append('ports'))
    monkeypatch.setattr(w,'group_members',lambda s:[] if s.get('stopped') or not live['value'] else [(100,'1')])
    monkeypatch.setattr(w,'http',health);monkeypatch.setattr(w,'clocks',lambda *a:calls.append('clocks'))
    monkeypatch.setattr(w,'start_services',start);monkeypatch.setattr(w,'cleanup',cleanup);monkeypatch.setattr(w,'measure_requests',measure)
    def run(c):
        d=tmp_path/f'trial-{len(list(tmp_path.glob("trial-*")))}';d.mkdir();return resident.resident_trial(cfg,c,d)
    return cfg,c,calls,live,run


def test_frequency_and_power_only_reuse_with_fresh_controls_and_warmup(host):
    cfg,c,calls,live,run=host
    assert run(c)['resident_service']['reused'] is False
    changed={**c,'expert_mhz':1050,'attention_power_w':250}
    assert run(changed)['resident_service']['reused'] is True
    assert calls.count('start')==1 and calls.count('clocks')==2 and calls.count('warmup+replay')==2
    assert 'cleanup' not in calls


@pytest.mark.parametrize('change',[{'microbatches':2},{'attention_dp':1,'attention_tp':2},
    {'expert_dp':1,'expert_tp':2},{'expert_gpus':[2],'expert_dp':1,'expert_ep':1}])
def test_structural_changes_restart_services(host,change):
    cfg,c,calls,live,run=host;run(c)
    result=run({**c,**change})
    assert result['status']=='ok' and result['resident_service']['reused'] is False
    assert calls.count('start')==2 and calls.count('cleanup')==1


def test_failed_replay_invalidates_service_before_next_trial(host):
    cfg,c,calls,live,run=host;run(c);live['fail_replay']=True
    assert run(c)['status']=='failed'
    assert json.loads((Path(cfg['resident_session_directory'])/'resident-state.json').read_text())['valid'] is False
    live['fail_replay']=False
    assert run(c)['resident_service']['reused'] is False


def test_model_and_launch_limits_are_part_of_key(host):
    cfg,c,*_=host;original=resident.service_key(cfg,c)
    assert resident.service_key({**cfg,'max_model_len':4096},c)!=original
    assert resident.service_key({**cfg,'api_port':19000},c)!=original
    (Path(cfg['model_path'])/'config.json').write_text('{"changed":true}')
    assert resident.service_key(cfg,c)!=original


def test_foreign_process_is_never_treated_as_resident(host):
    cfg,c,calls,live,run=host;run(c);live['foreign']=True
    result=run(c)
    assert result['status']=='failed' and 'outside the resident service' in result['failure_reason']


def test_cost_window_includes_reserved_idle_gpus():
    rows=[{'timestamp_ns':0,'power_w':[100,100,40,40]}, {'timestamp_ns':2000000000,'power_w':[100,100,40,40]}]
    cost=window_cost(rows,4)
    assert cost['tuning_energy_j']==560 and cost['gpu_hours']==pytest.approx(8/3600)

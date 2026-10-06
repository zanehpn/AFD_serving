"""Synthetic mechanism tests, never A100 performance evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/afd'));sys.path.insert(0,str(ROOT))
from general_dse import engine,model_ir,primitives
from migration import general_dse as cli


def model(tmp_path,kind='mixtral',name='model',hidden=64):
    cfg=dict(model_type=kind,hidden_size=hidden,num_hidden_layers=2,vocab_size=128,
        num_attention_heads=8,num_key_value_heads=2,intermediate_size=128,
        num_local_experts=8,num_experts_per_tok=2,tie_word_embeddings=False)
    path=tmp_path/name/'config.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(cfg))
    return model_ir.from_config(path)


def hardware(n=8):
    values=dict(idle_w=30.,active_w=180.,memory_bytes_per_s=1e11,
        gemm=[dict(m=m,k=k,n=k,flops_per_s=1e10) for m in (1,32,512) for k in (64,4096)])
    return dict(schema='hardware_primitives_v1',selection_split='calibration',dtype_bytes=2,
        gpus=[dict(id=i,memory_bytes=80*1024**3) for i in range(n)],
        device=dict(power_limit_w=400,idle_reference_mhz=1410,by_mhz={'1410':copy.deepcopy(values),'1050':copy.deepcopy(values)}),
        links={f'{a}:{b}':dict(latency_s=1e-6,bandwidth_bytes_per_s=1e10) for a in range(n) for b in range(n) if a!=b},
        fabric_bandwidth_bytes_per_s=1e11)


def scheduler():
    return dict(max_num_seqs=16,max_num_batched_tokens=64,max_output_tokens=8,microbatches=2,
        workspace_bytes_per_gpu=1024,memory_fraction=.9)


def trace(n=6):
    return [dict(source_index=i,source_timestamp=str(i),evaluation_split='calibration',arrival_s=float(i),input_tokens=16,output_tokens=8) for i in range(n)]


def point(h,dp=2,ep=2,tp=1,etp=1,placement='disaggregated'):
    return engine.enumerate_points(h,[g['id'] for g in h['gpus']],[dp],[tp],[ep],[etp],[1410],[placement])[0]


def test_architecture_adapters_and_explicit_unknown_operator(tmp_path):
    dense=model(tmp_path,'llama','dense');moe=model(tmp_path,'mixtral','moe')
    assert dense['blocks'][0]['ffn']['experts']==1
    assert moe['blocks'][0]['ffn']['experts']==8
    assert len(moe['blocks'])==2
    assert moe['blocks'][0]['ffn']['expert_flops']==6*64*128
    with pytest.raises(ValueError,match='adapter'):
        model(tmp_path,'unknown_arch')
    bad=copy.deepcopy(moe);bad['blocks'][0]['ffn']['routing_probabilities']=[1]*8
    with pytest.raises(ValueError,match='marginals'):model_ir.validate(bad)


def test_more_than_four_gpus_and_unmeasured_topologies_need_no_stage_anchors(tmp_path):
    h=hardware(16);m=model(tmp_path)
    c=point(h,dp=8,ep=8)
    pred=engine.simulate(m,h,c,trace(),scheduler(),4)
    assert pred['status']=='predicted_requires_validation'
    assert pred['active_gpus_count']==pred['allocation_gpus_count']==16
    assert not pred['execution']['native_case_shape_supported']
    assert not pred['execution']['automatic_launch_authorized']
    assert 'topologies' not in h


def test_dp_does_not_divide_single_request_latency(tmp_path):
    h=hardware();m=model(tmp_path,'llama')
    one=engine.simulate(m,h,point(h,dp=1,placement='colocated'),trace(2),scheduler(),.01)
    four=engine.simulate(m,h,point(h,dp=4,placement='colocated'),trace(2),scheduler(),.01)
    assert four['p90_ttft_ms']==pytest.approx(one['p90_ttft_ms'])
    assert four['p90_tpot_ms']==pytest.approx(one['p90_tpot_ms'])


def test_tp_shards_weights_but_dp_replicates_and_kv_head_replication_is_explicit(tmp_path):
    h=hardware();m=model(tmp_path,'llama')
    s=scheduler();s['max_num_seqs']=1
    dp1=engine.Physics(m,h,point(h,dp=1,placement='colocated')).memory(s,trace())
    dp2=engine.Physics(m,h,point(h,dp=2,placement='colocated')).memory(s,trace())
    tp2=engine.Physics(m,h,point(h,dp=1,tp=2,placement='colocated')).memory(s,trace())
    assert dp1['required_bytes_by_gpu'][0]==dp2['required_bytes_by_gpu'][0]
    assert tp2['required_bytes_by_gpu'][0]<dp1['required_bytes_by_gpu'][0]
    # TP beyond KV-head count replicates KV rather than dividing by arbitrarily many GPUs.
    tp4=engine.Physics(m,h,point(h,dp=1,tp=4,placement='colocated')).memory(s,trace())
    assert tp4['required_bytes_by_gpu'][0]>tp2['required_bytes_by_gpu'][0]/2


def test_physical_placement_and_missing_links_change_prediction(tmp_path):
    m=model(tmp_path);h=hardware();c=point(h)
    first=engine.simulate(m,h,c,trace(),scheduler(),2)
    for key in h['links']:h['links'][key]['latency_s']=.01
    slow=engine.simulate(m,h,c,trace(),scheduler(),2)
    assert slow['p90_ttft_ms']>first['p90_ttft_ms']
    h['links'].pop('0:2')
    missing=engine.simulate(m,h,c,trace(),scheduler(),2)
    assert missing['status']=='needs_primitive_measurement'
    assert '0:2' in missing['reason']


def test_allocation_idle_energy_and_memory_gate(tmp_path):
    m=model(tmp_path);h=hardware();c=point(h,dp=1,ep=1)
    a=engine.simulate(m,h,c,trace(),scheduler(),1)
    small=copy.deepcopy(c);small['allocation']=[0,1]
    b=engine.simulate(m,h,small,trace(),scheduler(),1)
    assert a['energy_j_per_request']-b['energy_j_per_request']==pytest.approx(6*30*a['makespan_s']/len(trace()))
    h['gpus'][0]['memory_bytes']=100
    assert engine.simulate(m,h,c,trace(),scheduler(),1)['status']=='analytically_infeasible'


def test_workload_shape_and_model_structure_change_predictions_from_same_hardware(tmp_path):
    h=hardware();c=point(h);s=scheduler()
    m1=model(tmp_path,name='small');m2=model(tmp_path,name='large',hidden=128)
    a=engine.simulate(m1,h,c,trace(),s,1);b=engine.simulate(m2,h,c,trace(),s,1)
    long=trace();long[0]['input_tokens']=256
    d=engine.simulate(m1,h,c,long,s,1)
    assert b['p90_tpot_ms']>a['p90_tpot_ms']
    assert d['p90_ttft_ms']>a['p90_ttft_ms']
    high=engine.simulate(m1,h,c,trace(),s,50000)
    assert high['mean_batch_tokens']>a['mean_batch_tokens']


def test_heldout_and_unmeasured_frequency_rejected(tmp_path):
    h=hardware();m=model(tmp_path);c=point(h)
    requests=trace();requests[0]['evaluation_split']='heldout'
    with pytest.raises(ValueError,match='calibration'):engine.simulate(m,h,c,requests,scheduler(),1)
    c['attention_mhz']=900
    assert engine.simulate(m,h,c,trace(),scheduler(),1)['status']=='needs_primitive_measurement'
    h['hardware_by_topology']={}
    with pytest.raises(ValueError,match='stage equivalents'):engine.validate_hardware(h)


def test_search_baseline_is_explicit_and_predictions_never_authorize_launch(tmp_path):
    m=model(tmp_path);h=hardware()
    points=engine.enumerate_points(h,list(range(8)),[1,2,4],[1],[1,2,4],[1],[1410],['disaggregated'])
    baseline=point(h)['id']
    pred=engine.search(m,h,points,trace(),scheduler(),[1,4],baseline)
    assert pred['baseline_id']==baseline
    assert len(pred['rows'])==2*len(points)
    assert pred['topology_specific_calibration_required'] is False
    assert pred['deployment_authorized'] is False
    assert pred['predicted_fixed_configuration'] in {p['id'] for p in points}


def test_independent_primitive_identification(tmp_path):
    h=hardware(2)
    raw=dict(kind='independent_primitive_measurements',selection_split='calibration',complete=True,dtype_bytes=2,
        gpus=h['gpus'],power_limit_w=400,provenance={},device_measurements={'1410':dict(idle_w=30,
        gemm=[dict(m=2,k=4,n=8,seconds=1e-6,power_w=100)],copy=[dict(bytes=1000,seconds=1e-6,power_w=90)])},
        links={'0:1':[dict(bytes=n,seconds=1e-6+n/1e9) for n in (4096,1048576)]})
    source=tmp_path/'raw.json';primitives.write(source,raw)
    built=primitives.build(raw,source)
    assert built['device']['by_mhz']['1410']['gemm'][0]['flops_per_s']==pytest.approx(128e6)
    assert built['device']['by_mhz']['1410']['memory_bytes_per_s']==pytest.approx(2e9)
    assert built['links']['0:1']['latency_s']==pytest.approx(1e-6)
    assert built['links']['0:1']['bandwidth_bytes_per_s']==pytest.approx(1e9)
    primitives.verify_sources(built)
    source.write_text('{}')
    with pytest.raises(ValueError,match='changed'):primitives.verify_sources(built)


def test_cli_cpu_plan_freezes_inputs_and_uses_same_hardware_for_new_models(tmp_path):
    h=hardware();source=tmp_path/'primitive-source.json';primitives.write(source,{'synthetic_test_only':True})
    h['sources']={str(source):primitives.sha(source)}
    hp=tmp_path/'hardware.json';primitives.write(hp,h)
    for name in ('one','two'):
        m=model(tmp_path,name=name,hidden=64 if name=='one' else 128)
        mp=tmp_path/(name+'.json');primitives.write(mp,m)
        baseline=point(h,dp=2,ep=2)['id'];plan=tmp_path/(name+'-plan.json')
        cli.make_search_plan(mp,hp,ROOT/'inputs/traces/calibration-200.jsonl',ROOT/'inputs/traces/heldout-400.jsonl',
            plan,list(range(8)),[1],[2],[1],[2],[1],[1410],['disaggregated'],baseline)
        frozen=json.loads(plan.read_text());assert frozen['GPU_execution_requested'] is False
        assert frozen['evaluation_identity_audit']['status']=='PASS'
    source.write_text('changed')
    with pytest.raises(ValueError,match='changed'):cli.execute_search(plan,tmp_path/'blocked.json')
    assert not (tmp_path/'blocked.json').exists()


def test_probe_plan_has_no_model_or_trace_specific_measurements():
    p=primitives.probe_plan(ROOT,list(range(8)),[1050,1410])
    assert p['model_weights_required'] is False and p['request_trace_access'] is False
    assert len(p['gemm_shapes'])==6
    assert p['physical_gpus']==list(range(8))


def test_portable_physics_is_used_by_the_bound_dynamic_controller(tmp_path):
    from math_dynamic.controller import Controller,load_config
    from test_full_math_workflow import controller_fixture
    c=controller_fixture();c['parameters']=dict(model_kind='general_operator_physics',general_model=model(tmp_path),
        general_hardware=hardware(4),scheduler=scheduler())
    path=tmp_path/'controller.json';primitives.write(path,c);load_config(path)
    controller=Controller(c)
    controller.consume(dict(event='submit',request_id='a',wall_ns=1_000_000_000,input_tokens=16,requested_output_tokens=8))
    controller.consume(dict(event='first_token',request_id='a',wall_ns=2_000_000_001,output_chunks=1))
    chosen,reason=controller.choose(2_000_000_002)
    assert reason=='portable_operator_batch_energy'
    assert chosen['topology']=='2a1e'
    assert all(p.t['used']=={0,1,2} for p in controller.general_physics.values())


def test_case_workflow_consumes_shared_primitives_without_topology_probes(tmp_path,monkeypatch):
    from migration import workflow,static_dse as static,dynamic_dse as dynamic
    from general_dse import case_adapter
    from test_full_math_workflow import write_mock_measurements
    h=hardware(4);source=tmp_path/'primitive-raw.json';primitives.write(source,{'synthetic_test_only':True})
    h['sources']={str(source):primitives.sha(source)}
    hp=tmp_path/'hardware.json';primitives.write(hp,h)
    ledger=model(tmp_path)
    monkeypatch.setattr(case_adapter.model_ir,'from_config',lambda _:ledger)
    phases=[]
    def search(m,h,points,requests,scheduler,rates,baseline):
        return dict(predicted_best_by_rate={str(r):'2a1e-max' for r in rates},predicted_fixed_configuration='2a1e-max')
    monkeypatch.setattr(case_adapter.engine,'search',search)
    monkeypatch.setattr(static,'paths',lambda p,e:(tmp_path/'suites'/e['suite_id'],tmp_path/'runs'/e['suite_id']))
    def execute(path):
        p=static.verify_plan(path);phases.append(p['phase'])
        if p.get('evaluation_split')=='heldout':dynamic.verify_heldout_gate(p)
        for e in p['schedule']:
            if not (static.paths(p,e)[0]/'COMPLETE').exists():write_mock_measurements(p,e,tmp_path)
        static.collect(p)
    monkeypatch.setattr(static,'run',execute)
    monkeypatch.setattr(workflow.parameters,'build',lambda *_:pytest.fail('Portable mode ran legacy topology probes'))
    base=tmp_path/'portable-workflow'
    workflow.run('qwen36',base,[0,1,2,3],hardware_profile=hp)
    assert phases==['candidate_validation','dynamic_calibration','formal_heldout']
    assert not (base/'probes').exists()
    controller=json.loads((base/'fixed/dynamic-calibration/controller.json').read_text())
    assert controller['parameters']['model_kind']=='general_operator_physics'
    assert json.loads((base/'RESULTS.json').read_text())['formal_request_count']==7200


def test_frozen_prediction_validation_reports_errors_without_selecting(tmp_path):
    m=model(tmp_path);h=hardware();c=point(h)
    r=engine.simulate(m,h,c,trace(),scheduler(),1)
    plan=tmp_path/'plan.json';primitives.write(plan,{'synthetic_test_only':True})
    pred=tmp_path/'prediction.json';primitives.write(pred,dict(plan=str(plan),plan_sha256=primitives.sha(plan),
        sources={},rows=[dict(candidate_id=c['id'],rate=1,prediction=r)]))
    source=tmp_path/'measurements.json';primitives.write(source,{'synthetic_test_only':True})
    measured={k:r[k]*1.1 for k in ('p90_ttft_ms','p90_tpot_ms','output_tps','energy_j_per_request')}
    obs=tmp_path/'observations.json';primitives.write(obs,dict(prediction_sha256=primitives.sha(pred),
        purpose='prediction_validation_only',used_for_parameter_selection=False,sources={str(source):primitives.sha(source)},
        rows=[dict(candidate_id=c['id'],rate=1,repetition=1,metrics=measured)]))
    out=cli.validate_predictions(pred,obs,tmp_path/'errors.json')
    assert not out['selection_performed'] and not out['formal_generalization_claim_authorized']
    assert out['cells'][0]['errors']['output_tps']['relative_error']==pytest.approx(-1/11)


@pytest.mark.parametrize('failure', [None,'occupied','kernel'])
def test_native_primitive_runner_ownership_and_clock_restoration_on_cpu(tmp_path,monkeypatch,failure):
    import os
    from types import SimpleNamespace as NS
    import subprocess
    ids=[4,6];changes=[];clocks={i:1410 for i in ids};power={i:250000 for i in ids}
    nv=NS(nvmlInit=lambda:None,nvmlShutdown=lambda:None,nvmlDeviceGetHandleByIndex=lambda i:i,
        nvmlDeviceGetComputeRunningProcesses=lambda h:[NS(pid=999999)] if failure=='occupied' else [],
        nvmlDeviceGetGraphicsRunningProcesses=lambda h:[],nvmlDeviceGetName=lambda h:'MOCK GPU',
        nvmlDeviceGetMaxClockInfo=lambda h,k:1000,nvmlDeviceGetSupportedGraphicsClocks=lambda h,m:[1050,1410],
        nvmlDeviceGetUUID=lambda h:f'GPU-{h}',nvmlDeviceGetMemoryInfo=lambda h:NS(total=80*1024**3),
        nvmlDeviceGetPowerManagementDefaultLimit=lambda h:400000,
        nvmlDeviceGetPowerManagementLimit=lambda h:power[h],nvmlDeviceGetPowerUsage=lambda h:30000,
        nvmlDeviceGetClockInfo=lambda h,k:clocks[h],NVML_CLOCK_MEM=0,NVML_CLOCK_SM=1)
    def set_clock(h,a,b):clocks[h]=a;changes.append(('clock',h,a))
    def reset_clock(h):changes.append(('reset',h))
    def set_power(h,w):power[h]=w;changes.append(('power',h,w))
    nv.nvmlDeviceSetGpuLockedClocks=set_clock;nv.nvmlDeviceResetGpuLockedClocks=reset_clock
    nv.nvmlDeviceSetPowerManagementLimit=set_power
    class Tensor:
        def copy_(self,other,**kw):return self
    def mm(a,b,out):
        if failure=='kernel':raise RuntimeError('mock kernel failed')
        return out
    torch=NS(__version__='mock',version=NS(cuda='mock'),bfloat16='bf16',uint8='uint8',
        ones=lambda *a,**k:Tensor(),empty=lambda *a,**k:Tensor(),empty_like=lambda a:Tensor(),mm=mm,
        cuda=NS(device_count=lambda:len(ids),get_device_properties=lambda i:NS(uuid=f'GPU-{ids[i]}'),
                set_device=lambda i:None,synchronize=lambda i:None))
    monkeypatch.setitem(sys.modules,'pynvml',nv);monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setattr(os,'geteuid',lambda:0)
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'mock topology')
    # Restore the environment even though the runner sets it for its own process.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','before-test')
    p=primitives.probe_plan(ROOT,ids,[1050,1410]);p['minimum_measurement_seconds']=0;p['idle_measurement_seconds']=0
    pp=tmp_path/'plan.json';primitives.write(pp,p);out=tmp_path/'raw.json'
    if failure:
        with pytest.raises(RuntimeError,match='occupied' if failure=='occupied' else 'kernel failed'):primitives.run(pp,out)
    else:
        raw=primitives.run(pp,out)
        assert raw['complete'] and raw['restored']
        assert set(raw['links'])=={'4:6','6:4'}
    if failure=='occupied':
        assert not changes and not out.exists()
    else:
        assert {x[1] for x in changes if x[0]=='reset'}==set(ids)
        assert all(power[i]==250000 for i in ids)


def test_shared_profile_reuse_checks_uuid_and_never_reprobes(tmp_path,monkeypatch):
    from migration import ensure_hardware
    h=hardware(4)
    for g in h['gpus']:g['uuid']=f"GPU-{g['id']}"
    raw=tmp_path/'raw.json';primitives.write(raw,{'synthetic_test_only':True})
    h['sources']={str(raw):primitives.sha(raw)}
    profile=tmp_path/'HARDWARE.json';primitives.write(profile,h)
    monkeypatch.setattr(ensure_hardware.subprocess,'run',lambda *a,**k:pytest.fail('Reused profile started probes'))
    monkeypatch.setattr(ensure_hardware.subprocess,'check_output',lambda *a,**k:'0,GPU-0\n1,GPU-1\n2,GPU-2\n3,GPU-3\n')
    assert ensure_hardware.ensure(profile,[0,1,2,3])==profile
    monkeypatch.setattr(ensure_hardware.subprocess,'check_output',lambda *a,**k:'0,GPU-other\n1,GPU-1\n2,GPU-2\n3,GPU-3\n')
    with pytest.raises(ValueError,match='UUIDs differ'):ensure_hardware.ensure(profile,[0,1,2,3])


def test_default_parallel_dry_run_selects_shared_primitives_without_gpu_or_files(tmp_path):
    import subprocess
    result=subprocess.run([sys.executable,str(ROOT/'migration/parallel_8gpu.py'),'--dry-run','--run-id','general-dry-test'],
        capture_output=True,text=True,check=True)
    rows=json.loads(result.stdout)
    assert all('ECODEP_HARDWARE_PROFILE' in r['environment'] for r in rows)
    assert not (ROOT/'parallel_runs/general-dry-test').exists()
    legacy=subprocess.run([sys.executable,str(ROOT/'migration/parallel_8gpu.py'),'--dry-run','--legacy-topology-probes'],
        capture_output=True,text=True,check=True)
    assert all(r['environment']['ECODEP_LEGACY_TOPOLOGY_PROBES']=='1' for r in json.loads(legacy.stdout))


def test_default_workflow_dry_run_uses_portable_plan(tmp_path):
    import subprocess
    base=tmp_path/'not-started'
    result=subprocess.run([sys.executable,str(ROOT/'migration/workflow.py'),'--model','qwen36',
        '--campaign',str(base),'--dry-run'],capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['stages'][0]=='portable_primitive_mathematical_search'
    assert 'two_topology_probes' not in plan['stages']
    assert not base.exists()

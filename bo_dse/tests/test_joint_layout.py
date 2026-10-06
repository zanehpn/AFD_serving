"""Joint-layout contracts: physical indices, argv semantics, and measured receipts."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import fcntl
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'migration'))
sys.path.insert(0, str(ROOT / 'scripts/afd'))
sys.path.insert(0, str(ROOT / 'bo_dse'))
from bo_layout import enumerate_layouts, replicas, deployment, validate, verify_launch
from gpu_selection import inspect
from launch_pair import parse_args
from launch_joint import specifications
from verify_joint_launch import verify


def candidate(ad=1, at=2, ed=1, ep=1, et=2, m=4):
    gpus = [7, 3, 5, 1, 6, 2, 4, 0]
    return dict(attention_dp=ad, attention_tp=at, expert_dp=ed, expert_ep=ep, expert_tp=et,
                attention_gpus=gpus[:ad*at], expert_gpus=gpus[ad*at:ad*at+ed*ep*et],
                microbatches=m, execution_mode='eager')


def test_joint_space_explores_degrees_without_inventing_expert_replicas():
    rows, excluded = enumerate_layouts([7, 3, 5, 1, 6, 2, 4, 0], 256)
    for key in ('attention_tp', 'expert_tp', 'expert_dp', 'expert_ep', 'microbatches'):
        assert len({r[key] for r in rows}) > 1
    for r in rows:
        validate(r)
        groups = replicas(r)
        assert [gpu for g in groups for gpu in g['attention_gpus']] == r['attention_gpus']
        assert [gpu for g in groups for gpu in g['expert_gpus']] == r['expert_gpus']
    assert candidate(ad=2, at=1, ed=2, ep=1, et=1, m=2) in rows
    assert any('combined_expert_ep_tp' in r['reason'] for r in excluded)


@pytest.mark.parametrize('c', [candidate(), candidate(ad=2, at=1, ed=2, ep=1, et=1, m=1),
                              candidate(ad=4, at=1, ed=2, ep=2, et=1, m=2)])
def test_joint_argv_matches_candidate_and_gpu_groups(tmp_path, monkeypatch, c):
    path = tmp_path / 'layout.json'
    path.write_text(json.dumps({'configuration': c}))
    monkeypatch.setattr(sys, 'argv', ['launch_pair', '--layout-json', str(path), '--results', str(tmp_path)])
    _, groups, launches = specifications(parse_args())
    assert len(launches) == 2*c['expert_dp']
    for item in launches:
        role, argv = item['role'], item['argv']
        group = groups[item['replica']]
        assert int(argv[argv.index('--tensor-parallel-size')+1]) == c[role+'_tp']
        assert int(argv[argv.index('--data-parallel-size')+1]) == (c['attention_dp']//c['expert_dp'] if role == 'attention' else c['expert_ep'])
        assert ('--enable-expert-parallel' in argv) == (role == 'expert' and c['expert_ep'] > 1)
        assert ('--enable-dbo' in argv) == (c['microbatches'] == 2)
        assert ('--ubatch-size' in argv) == (c['microbatches'] > 2)
        assert argv[argv.index('--dbo-decode-token-threshold')+1] == '2'
        assert argv[argv.index('--dbo-prefill-token-threshold')+1] == '12'
        assert item['env']['CUDA_VISIBLE_DEVICES'] == ','.join(map(str, group[role+'_gpus']))
        if c['microbatches'] > 2:
            assert argv[argv.index('--ubatch-size')+1] == str(c['microbatches'])


def test_worker_receipts_reject_silent_tp_and_sharding_changes(tmp_path):
    c = candidate()
    groups = replicas(c)
    launch = {**deployment(c), 'dbo_enabled': False, 'replicas': groups,
              'dbo_decode_token_threshold': 2, 'dbo_prefill_token_threshold': 12,
              **{r+'_gpus': ','.join(map(str,c[r+'_gpus'])) for r in ('attention','expert')}}
    (tmp_path/'joint-launch.json').write_text(json.dumps(launch))
    for g in groups:
        for role, key in [('attention','attention'), ('ffn','expert')]:
            for i in range(g[key+'_ranks']):
                r = dict(role=role, rank=i, local_role_rank=i, replica=0, verified=True,
                         actual=dict(native_tp=g[key+'_tp'], native_dp=1, enable_expert_parallel=False, microbatches=4,
                                     dbo_decode_token_threshold=2, dbo_prefill_token_threshold=12),
                         expert_shards=[dict(tp=2,ep=1)] if role=='ffn' else [])
                (tmp_path/f'worker-layout-{role}-{i}.json').write_text(json.dumps(r))
    verify(c, tmp_path)
    with pytest.raises(ValueError, match='thresholds'):
        verify(c, tmp_path, thresholds={'dbo_decode_token_threshold':32, 'dbo_prefill_token_threshold':512})
    altered_groups = copy.deepcopy(launch)
    altered_groups['replicas'][0]['attention_gpus'].reverse()
    (tmp_path/'joint-launch.json').write_text(json.dumps(altered_groups))
    with pytest.raises(ValueError, match='replica GPU groups'):
        verify(c, tmp_path)
    (tmp_path/'joint-launch.json').write_text(json.dumps(launch))
    wrong_ports = copy.deepcopy(groups)
    wrong_ports[0]['api_port'] += 1
    with pytest.raises(ValueError, match='frozen plan'):
        verify(c, tmp_path, wrong_ports)
    p=tmp_path/'worker-layout-ffn-0.json'
    r=json.loads(p.read_text()); r['expert_shards']=[dict(tp=1,ep=2)]; p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='shards mismatch'):
        verify(c, tmp_path)
    altered=copy.deepcopy(launch); altered['attention_tp']=1
    with pytest.raises(ValueError, match='attention_tp'):
        verify_launch(c, altered)


class NVML:
    def __init__(self, busy=()): self.busy=set(busy); self.closed=False
    def nvmlInit(self): pass
    def nvmlShutdown(self): self.closed=True
    def nvmlDeviceGetCount(self): return 8
    def nvmlDeviceGetHandleByIndex(self, i): return i
    def nvmlDeviceGetMemoryInfo(self, i): return SimpleNamespace(free=79*1024**3)
    def nvmlDeviceGetName(self, i): return 'NVIDIA A100-SXM4-80GB'
    def nvmlDeviceGetUUID(self, i): return f'GPU-{i}'
    def nvmlDeviceGetComputeRunningProcesses(self, i): return [SimpleNamespace(pid=900+i)] if i in self.busy else []
    def nvmlDeviceGetGraphicsRunningProcesses(self, i): return []


def test_gpu_selection_preserves_physical_order_and_excludes_busy_locked(tmp_path, monkeypatch):
    nvml=NVML([0,1])
    with (tmp_path/'moe-bo-gpu-2.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r=inspect(nvml, count=4, lock_directory=tmp_path)
        assert r['gpus']==[3,4,5,6]
        assert r['devices'][2]['reasons']==['reserved_by_other_bo_campaign']
    assert nvml.closed
    assert not (tmp_path/'moe-bo-gpu-3.lock').exists()
    r=inspect(nvml, [7,3,5,4], lock_directory=tmp_path)
    assert r['gpus']==[7,3,5,4]
    r=inspect(nvml, [7,3,0,4], lock_directory=tmp_path)
    assert r['status']=='insufficient_available_gpus' and not r['gpus']
    with pytest.raises(ValueError): inspect(nvml,[0,0,1,2],lock_directory=tmp_path)
    original_open = Path.open
    def inaccessible_lock(path, *args, **kwargs):
        if path.name == 'moe-bo-gpu-7.lock':
            raise PermissionError('another user owns the reservation lock')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', inaccessible_lock)
    r=inspect(nvml, [7,3,5,4], lock_directory=tmp_path)
    assert r['status']=='insufficient_available_gpus'
    assert r['devices'][7]['reasons']==['reservation_lock_not_accessible']


def test_executor_can_lock_existing_read_only_files_and_rejects_competitors(tmp_path, monkeypatch):
    import native
    import builtins
    path = tmp_path/'moe-bo-gpu-7.lock'
    path.touch(mode=0o444)
    def test_open(name, *args, **kwargs):
        return builtins.open(tmp_path/Path(name).name, *args, **kwargs)
    monkeypatch.setattr(native, 'open', test_open, raising=False)
    with native.gpu_locks([7]):
        with path.open('r') as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with native.gpu_locks([6,7]):
        assert (tmp_path/'moe-bo-gpu-6.lock').exists()


def test_tp_workload_deduplicates_queries_but_keeps_slowest_shard():
    sys.path.insert(0,str(ROOT/'bo_dse/scripts/afd'))
    from build_deepseek_v6_four_stage_profile import deduplicate_attention_tp
    rows={}
    for rank in range(4):
        n=3 if rank<2 else 5
        rows[str(rank)]={'parallel_layout':{'replica':0,'local_role_rank':rank,'tp_size':2},
                         'prefill_tokens':0,'decode_tokens':n,'context':{'decode_context_tokens':n*100},
                         'decode_request_spans':[],'event_layers':[0,1],
                         'layer_ms':{0:float(rank+1),1:float(rank+2)}, 'duration_ms':2*rank+3.}
    logical=deduplicate_attention_tp(rows,2)
    assert sum(r['decode_tokens'] for r in logical.values())==8
    assert sum(r['context']['decode_context_tokens'] for r in logical.values())==800
    assert logical['replica-0-dp-0']['duration_ms']==5
    damaged=copy.deepcopy(rows);damaged['1']['decode_tokens']=9
    with pytest.raises(ValueError,match='TP peers disagree'):
        deduplicate_attention_tp(damaged,2)
    with pytest.raises(ValueError,match='Incomplete'):
        deduplicate_attention_tp({k:v for k,v in rows.items() if k!='1'},2)


def test_independent_replicas_cannot_receive_single_group_structural_prior(monkeypatch):
    import static_dse.mechanism as mechanism
    c = {'topology': candidate(ad=2, at=1, ed=2, ep=1, et=1, m=1), 'microbatches': 1,
         'knobs': {'attention_mhz': 1410, 'expert_mhz': 1410,
                   'attention_power_w': 400, 'expert_power_w': 400},
         'prior': {'energy_sd': .1, 'log_energy': 10.}}
    def invalid_interpolation(*args):
        raise AssertionError('Independent replicas must not use the single-group pipeline model')
    monkeypatch.setattr(mechanism, 'structural_predict', invalid_interpolation)
    mechanism.inform_candidates([c], {'groups': {}}, {}, 12)
    assert c['prior']['energy_sd'] >= 1.
    assert c['prior']['source'] == 'broad_empirical_bo_fallback'
    assert c['mechanism']['status'] == 'analytical_model_unavailable'


def test_gpu_selection_receipt_flows_into_native_dry_run(tmp_path):
    selected=inspect(NVML(),[7,3,5,1],lock_directory=tmp_path)
    path=tmp_path/'selection.json';path.write_text(json.dumps(selected))
    cmd=[sys.executable,str(ROOT/'bo_dse/native.py'),'--backend','customized','start','--directory',str(tmp_path/'campaign'),
         '--gpu-selection',str(path),'--dry-run']
    result=subprocess.run(cmd,check=True,capture_output=True,text=True)
    plan=json.loads(result.stdout)
    assert plan['gpus']==[7,3,5,1] and plan['space_mode']=='joint'
    assert any(c['attention_tp']==2 and c['expert_tp']==2 for c in plan['supported_structures'])
    assert not plan['gpu_actions_performed'] and not (tmp_path/'campaign').exists()
    selected['status']='insufficient_available_gpus';path.write_text(json.dumps(selected))
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode!=0 and 'not ready' in result.stderr

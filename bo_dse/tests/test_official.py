"""Upstream contracts and CPU integration; fixtures are not GPU evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

BO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BO))
sys.path.insert(0, str(BO/'scripts/afd'))
import official
import official_worker
from official_space import SEMANTICS, structures, commands, validate
from static_dse.space import configuration, enumerate_candidates, hard_filter, layout, KNOBS
from static_dse.external_model import inform, validate_observable
from static_dse.campaign import ask, read_json, write_json, status


def space(model='qwen36'):
    return dict(selection_split='calibration', topologies=structures([0, 1, 2, 3], model), enumerate_parallelism=False,
                attention_frequencies_mhz=[1050, 1410], expert_frequencies_mhz=[1050, 1410],
                attention_power_caps_w=[250, 400], expert_power_caps_w=[250, 400], microbatches=[1, 2])


def observation(c, duration=10, ap=100, ep=200):
    return dict(candidate_id=c['id'], status='ok', selection_split='calibration',
                metrics=dict(energy_j=duration*(ap+ep), ttft_ms=10, tpot_ms=10, output_tps=10),
                external_observables=dict(provenance='public_replay_and_nvml', duration_s=duration,
                                          role_mean_power_w=dict(attention=ap, expert=ep)))


@pytest.mark.parametrize('model,count', [('qwen36', 18), ('deepseek-v2-lite', 18)])
def test_official_space_and_cli(model, count):
    candidates = enumerate_candidates(space(model))
    assert len({json.dumps(layout(c), sort_keys=True) for c in candidates}) == count
    for candidate in candidates:
        c = configuration(candidate)
        validate(c)
        cfg = dict(native_python='/env/bin/python', model=model, model_path='/models/model', api_port=18000,
                   afd_port=16000, dp_rpc_port=28000, max_model_len=8192)
        launch = commands(cfg, c)
        for role, item in launch.items():
            cmd = item['command']
            assert '--no-enable-prefix-caching' in cmd
            assert ('--enable-dbo' in cmd) == (c['microbatches'] == 2)
            assert not any('ecodep' in v or 'stage_trace' in v or 'ubatch-size' in v for v in cmd)
            assert ('--language-model-only' in cmd) == (model == 'qwen36')
            afd = json.loads(cmd[cmd.index('--additional-config')+1])['afd']
            assert afd['role'] == role and afd['compute_gate_on_attention'] is False
        if c['expert_ep'] == 2 and c['expert_dp'] == 2:
            assert c['expert_tp'] == 1  # EP overlaps DP; this is not four FFN GPUs.


def test_model_names_do_not_prune_parallelism_and_missing_evidence_stays_pending():
    qwen = structures([0, 1, 2, 3], 'qwen36')
    assert qwen == structures([0, 1, 2, 3], 'deepseek-v2-lite')
    # Explicit resource/degree mappings include Qwen TP2 and the unspecial-cased
    # 3A1F TP3 candidate; EP is the FFN world, not a third multiplicative axis.
    assert {(len(c['attention_gpus']), len(c['expert_gpus']), c['attention_dp'], c['attention_tp'],
             c['expert_dp'], c['expert_tp'], c['expert_ep']) for c in qwen} == {
        (1, 1, 1, 1, 1, 1, 1), (2, 1, 2, 1, 1, 1, 1), (2, 1, 1, 2, 1, 1, 1),
        (3, 1, 3, 1, 1, 1, 1), (3, 1, 1, 3, 1, 1, 1),
        (2, 2, 2, 1, 2, 1, 2), (2, 2, 1, 2, 2, 1, 2),
        (2, 2, 2, 1, 1, 2, 2), (2, 2, 1, 2, 1, 2, 2)}
    hardware = dict(devices={str(i): dict(memory_mib=81920, min_power_w=100, max_power_w=400,
                                         clock_pairs=[dict(memory_mhz=1593, graphics_mhz=1050)]) for i in range(4)})
    runtime = dict(adapter='official_v026', gpu_budget=4, allowed_gpus=list(range(4)), connector='p2p_nccl',
                   execution_modes=['eager'], memory_clock_mhz=1593,
                   divisibility=dict(attention_tp=32, expert_tp=32, expert_ep=256), structures=[])
    candidates = enumerate_candidates(space())
    tp2 = next(c for c in candidates if c['topology']['attention_tp'] == 2)
    assert hard_filter(tp2, hardware, runtime)['status'] == 'pending_validation'
    tp3 = next(c for c in candidates if c['topology']['attention_tp'] == 3)
    rejected = hard_filter(tp3, hardware, runtime)
    assert rejected['status'] == 'hard_rejected' and 'model_divisibility:attention_tp' in rejected['reasons']
    runtime['divisibility']['attention_tp'] = 24
    assert hard_filter(tp3, hardware, runtime)['status'] == 'pending_validation'


def test_search_threshold_profiles_reach_both_role_commands():
    spec = space()
    spec.update(microbatches=[2], dbo_threshold_profiles=[[2,12],[8,128],[32,512]])
    cfg = dict(native_python='/env/bin/python', model='qwen36', model_path='/models/model',
               api_port=18000, afd_port=16000, dp_rpc_port=28000, max_model_len=8192)
    candidates = enumerate_candidates(spec)
    assert len({c['id'] for c in candidates}) == len(candidates)
    for candidate in candidates:
        c = configuration(candidate)
        for launch in commands(cfg, c).values():
            command = launch['command']
            for key in ('dbo_decode_token_threshold', 'dbo_prefill_token_threshold'):
                assert command[command.index('--'+key.replace('_','-'))+1] == str(c[key])


def test_historical_observable_backend_requires_explicit_option(tmp_path):
    command = [sys.executable, str(BO/'native.py'), '--backend', 'legacy-observable',
               'start', '--directory', str(tmp_path/'none'), '--dry-run']
    report = json.loads(subprocess.check_output(command, text=True))
    assert report['backend'] == 'official' and not report['execution_verified']
    assert report['structures'] == 18 and report['raw_candidates'] == 11250
    assert not report['model_filter_applied'] and not any('Qwen TP' in r for r in report['excluded'])
    assert not (tmp_path/'none').exists()
    bad = subprocess.run(command + ['--microbatches', '4'], capture_output=True)
    assert bad.returncode != 0


def test_observable_model_identifies_only_separate_interventions():
    candidates = enumerate_candidates(space())[:32]
    same = [c for c in candidates if layout(c) == layout(candidates[0])]
    ref = same[0]
    changed = next(c for c in same if c['knobs']['attention_mhz'] != ref['knobs']['attention_mhz'] and
                   all(c['knobs'][k] == ref['knobs'][k] for k in KNOBS if k != 'attention_mhz'))
    predicted, report = inform(same, [observation(ref), observation(changed, duration=8)], 3000)
    assert next(iter(report['structures'].values()))['identified_knobs'] == ['attention_mhz']
    assert all('mechanism' not in c for c in predicted)
    assert all(c['prior']['energy_sd'] >= .35 for c in predicted)
    correlated = next(c for c in same if c['knobs']['attention_mhz'] != ref['knobs']['attention_mhz'] and
                      c['knobs']['attention_power_w'] != ref['knobs']['attention_power_w'] and
                      all(c['knobs'][k] == ref['knobs'][k] for k in ('expert_mhz', 'expert_power_w')))
    _, report = inform(same, [observation(ref), observation(correlated)], 3000)
    assert next(iter(report['structures'].values()))['identified_knobs'] == []


@pytest.mark.parametrize('bad', ['heldout', 'stages', 'energy'])
def test_external_labels_fail_closed(bad):
    row = observation(enumerate_candidates(space())[0])
    if bad == 'heldout':
        row['selection_split'] = 'heldout'
    elif bad == 'stages':
        row['four_stage'] = {}
    else:
        row['metrics']['energy_j'] = 1
    with pytest.raises(ValueError):
        validate_observable(row)


def test_failure_before_ownership_does_not_reset_any_gpu(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(official_worker, 'http', lambda *a, **kw: calls.append(a))
    official_worker.cleanup({'gpus': [0, 1, 2, 3]}, tmp_path)
    assert calls == []


def test_runtime_rejection_is_charged_without_touching_gpus(tmp_path, monkeypatch):
    def reject():
        raise ValueError('Runtime source differs from locked wheel')
    calls = []
    monkeypatch.setattr(official_worker, 'verify_runtime', reject)
    monkeypatch.setattr(official_worker, 'http', lambda *a: calls.append(a))
    result = official_worker.trial({'gpus': [0, 1, 2, 3]}, {}, tmp_path)
    assert result['status'] == 'failed' and result['cost']['gpu_hours'] > 0
    assert not result['cost_energy_complete'] and calls == []


def test_interrupted_attempt_retains_failure_cost_and_does_not_relaunch(tmp_path, monkeypatch):
    config = dict(gpus=[0, 1, 2, 3])
    write_json(tmp_path/'STARTED.json', dict(started_wall_ns=official.time.time_ns()-2_000_000_000))
    calls = []
    monkeypatch.setattr(official, 'worker', lambda *a: calls.append(a[1]))
    official.recover_attempt(config, tmp_path)
    result = official.attempt(config, {}, tmp_path)
    assert calls == ['cleanup'] and result['status'] == 'failed'
    assert result['cost']['gpu_hours'] >= 8/3600 and not result['cost_energy_complete']


def test_cleanup_resolution_preserves_original_failed_receipt(tmp_path):
    receipt = dict(status='failed', cleanup_required=True)
    write_json(tmp_path/'worker-result.json', receipt)
    with pytest.raises(RuntimeError, match='Cleanup required'):
        official.attempt({}, {}, tmp_path)
    write_json(tmp_path/'CLEANUP_RESOLVED.json', dict(verified=True, original_receipt=official.artifact(tmp_path/'worker-result.json')))
    assert official.attempt({}, {}, tmp_path) == receipt
    assert read_json(tmp_path/'worker-result.json') == receipt


@pytest.mark.parametrize('power_ack,accepted', [
    ({'applied_w': 400, 'power_control': 'limited'}, True),
    ({'power_limit_w': 400, 'power_control': 'limited'}, False),
    ({'applied_w': 350, 'power_control': 'limited'}, False),
    ({'applied_w': 400, 'power_control': 'unknown'}, False),
])
def test_official_clock_power_ack_contract(tmp_path, monkeypatch, power_ack, accepted):
    def http(url, body):
        if url.endswith('/set_power_limit'):
            return dict(gpu=body['gpu'], **power_ack)
        return dict(gpu=body['gpu'], applied_mhz=1410, clock_control='locked')
    monkeypatch.setattr(official_worker, 'http', http)
    c = dict(attention_gpus=[0], expert_gpus=[1], attention_power_w=400,
             expert_power_w=400, attention_mhz=1410, expert_mhz=1410)
    if accepted:
        official_worker.clocks({'clock_url': 'http://localhost'}, c, tmp_path)
    else:
        with pytest.raises(ValueError, match='acknowledgement mismatch'):
            official_worker.clocks({'clock_url': 'http://localhost'}, c, tmp_path)


def test_inherit_failed_reference_cost_and_integrity(tmp_path):
    prior, current = tmp_path/'old', tmp_path/'new'
    for p in (prior/'inputs', current/'inputs', prior/'validation/stock-reference'):
        p.mkdir(parents=True)
    keys = ('model', 'model_path', 'gpus', 'rps', 'limits', 'budget', 'comparison', 'seeds',
            'max_model_len', 'max_output_tokens', 'warmup_requests', 'time_scale')
    config = dict.fromkeys(keys, None)
    config.update(trace={'sha256': 'cal'}, heldout_trace={'sha256': 'held'})
    write_json(prior/'official-config.json', config)
    for p in (prior, current):
        write_json(p/'inputs/specification.json', {'grid': 'frozen'})
    path = prior/'validation/stock-reference/worker-result.json'
    receipt = dict(status='failed', cost=dict(gpu_hours=.01, wall_seconds=9, tuning_energy_j=12))
    write_json(path, receipt)
    original = path.read_bytes()
    official.inherit_failed_reference(config, prior, current/'inputs')
    assert official.setup_receipts(current) == [receipt]
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match='budget exhausted'):
        official.check_setup_budget({'budget': {'evaluations': 1, 'gpu_hours': 8}}, current)
    write_json(path, dict(receipt, status='ok'))
    with pytest.raises(ValueError, match='receipt changed'):
        official.setup_receipts(current)


@pytest.mark.parametrize('tbt', [False, True])
@pytest.mark.parametrize('actual_tokens,accepted', [(2, True), (1, False)])
def test_request_completion_policy_never_claims_token_correctness(tmp_path, monkeypatch, actual_tokens, accepted, tbt):
    trace = tmp_path/'trace.jsonl'
    trace.write_text(json.dumps(dict(source_index=1, output_tokens=2))+'\n')
    row = dict(source_index=1, http_status=200, error=None, actual_output_tokens=actual_tokens,
               submit_wall_ns=1000000000, finish_wall_ns=2000000000, ttft_ms=10, tpot_ms=1)
    if tbt:
        row.update(token_arrival_s=[.01, .04], tbt_ms=[30.])
    (tmp_path/'replay.jsonl').write_text(json.dumps(row)+'\n')
    def no_compare(*args):
        raise AssertionError('Token comparison must not run under request_completion')
    monkeypatch.setattr(official_worker, 'compare', no_compare)
    monitor = SimpleNamespace(gpus=[0, 1], errors=[], rows=[
        dict(timestamp_ns=t, power_w=[100, 100], operating_state=[
            dict(power_limit_w=400, memory_clock_mhz=1593)]*2)
        for t in (500000000, 1000000000, 1500000000, 2000000000, 2500000000)])
    config = dict(output_validation='request_completion', trace={'path': str(trace)},
                  max_output_tokens=128, memory_clock_mhz=1593, tbt_slo=tbt)
    c = dict(attention_gpus=[0], expert_gpus=[1], attention_power_w=400, expert_power_w=400, microbatches=1)
    if accepted:
        result = official_worker.metrics(config, c, tmp_path, monitor)
        assert result['status'] == 'ok' and result['metrics']['energy_j'] == pytest.approx(200)
        assert result['output_correctness']['verified'] is False
        assert result['output_correctness']['exact_tokens_checked'] is False
        assert result['output_correctness']['request_completion_verified'] is True
        if tbt:
            assert result['metrics']['tbt_ms'] == 30.
            row.pop('tbt_ms')
            (tmp_path/'replay.jsonl').write_text(json.dumps(row)+'\n')
            with pytest.raises(ValueError, match='token-arrival evidence'):
                official_worker.metrics(config, c, tmp_path, monitor)
    else:
        with pytest.raises(ValueError, match='correctness failed'):
            official_worker.metrics(config, c, tmp_path, monitor)


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    root = tmp_path/'repo'
    for folder in ('environment', 'bo_dse', 'migration', 'scripts/afd', 'services', 'model'):
        (root/folder).mkdir(parents=True)
    for name in ('official-runtime.lock.json', 'OFFICIAL_INSTALLED.json', 'native-requirements.lock.txt'):
        (root/'environment'/name).write_text('{}')
    (root/'bo_dse/official_worker.py').write_text('# external launcher fixture\n')
    write_json(root/'model/config.json', dict(num_experts=256, num_attention_heads=32))
    (root/'model/weights.safetensors').write_bytes(b'cpu fixture; never execute on GPU')
    for label, offset in (('calibration', 0), ('heldout', 100)):
        rows = [dict(source_index=i+offset, source_timestamp=f'{label}-{i}', request_id=i+offset,
                     evaluation_split=label, arrival_s=i, input_tokens=8, output_tokens=8) for i in range(8)]
        (root/f'{label}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(official, 'ROOT', root)
    monkeypatch.setattr(official, 'gpu_locks', lambda g: nullcontext())
    hardware = dict(devices={str(i): dict(uuid=f'cpu-{i}', name='A100-SXM4-80GB', memory_mib=81920,
                                         min_power_w=100, max_power_w=400,
                                         clock_pairs=[dict(memory_mhz=1215, graphics_mhz=f) for f in (1050, 1410)]) for i in range(8 if getattr(request, 'param', None) == 'capacity8' else 4)})
    launches = []
    def worker(config, action, path, c=None):
        path.mkdir(exist_ok=True, parents=True)
        launches.append(action)
        if action == 'preflight':
            write_json(path/'preflight.json', dict(hardware=hardware))
            return
        if action == 'reference':
            write_json(config['correctness_reference_path'], {'fixture': True})
            result = dict(status='ok')
        else:
            candidate = next(x for x in enumerate_candidates(read_json(Path(config['directory'])/'inputs/specification.json')) if configuration(x) == c)
            result = observation(candidate)
            if config.get("tbt_slo"):
                result["metrics"]["tbt_ms"] = 12.
            result.update(requests=config['trace']['requests'], completed_requests=config['trace']['requests'],
                          failed_requests=0, execution_verified=True, telemetry_valid=True,
                          output_correctness=dict(protocol='stock_vllm_exact_tokens_v1', verified=True,
                                                  reference_sha256=(official.sha(config['correctness_reference_path'])
                                                                    if Path(config['correctness_reference_path']).exists() else 'no-stock-gate'),
                                                  requests=8, mismatched_requests=0, output_tokens=64, selection_split='calibration'))
            if getattr(request, 'param', None) == 'fail_tp2' and c['attention_tp'] == 2:
                result = dict(status='failed', failure_reason='fixture: output regression failed')
        result.update(cost=dict(wall_seconds=1, gpu_hours=.001, tuning_energy_j=3000),
                      cost_energy_complete=True, artifacts=[official.artifact(path/'STARTED.json')])
        write_json(path/'worker-result.json', result)
    monkeypatch.setattr(official, 'worker', worker)
    args = SimpleNamespace(directory=tmp_path/'campaign', model='qwen36', model_path=root/'model', gpus=list(range(len(hardware['devices']))),
                           microbatches=[1, 2], frequencies=[1050, 1410], power_caps=[250, 400], evaluations=20,
                           gpu_hours=8, rps=1, trial_timeout_seconds=10, ttft_ms=20, tpot_ms=20, min_output_tps=1,
                           dry_run=False, seeds=[0], calibration=root/'calibration.jsonl', heldout=root/'heldout.jsonl',
                           native_python=Path(sys.executable), clock_url='http://localhost:9096',
                           api_port=18000, afd_port=16000, dp_rpc_port=28000, comparison=True)
    if getattr(request, 'param', None) in ('direct', 'capacity', 'capacity8'):
        args.direct_search = True
        args.output_validation = 'request_completion'
    if getattr(request, 'param', None) in ('capacity', 'capacity8'):
        args.exploration_policy = 'capacity_v2'
        args.microbatches = [2]
        args.slo_mode = 'relative_max'
        args.tbt_slo = True
    config = official.initialize(args)
    ready = official.prepare(config)
    return args, read_json(args.directory/'official-config.json'), ready, launches


@pytest.mark.parametrize('prepared', ['capacity', 'capacity8'], indirect=True)
def test_capacity_comparison_freezes_profile_and_keeps_baselines_independent(prepared):
    args, config, ready, launches = prepared
    from static_dse.comparison import create_comparison
    assert len(config['reference_configuration']['attention_gpus']) == len(args.gpus)//2
    assert len(config['reference_configuration']['expert_gpus']) == len(args.gpus)//2
    assert read_json(args.directory/'inputs/runtime.json')['gpu_budget'] == len(args.gpus)
    manifest = read_json(args.directory/'comparison/comparison.json')
    assert {a['method'] for a in manifest['campaigns']} == {'v2', 'generic_bo', 'random', 'ga'}
    official.run(config, one=True)
    for arm in manifest['campaigns']:
        directory = Path(arm['directory'])
        bundle = read_json(directory/'bundle.json')
        settings = bundle['settings']
        assert settings['setup_cost']['evaluations'] == 1
        assert settings['capacity_profile']['model']['status'] == 'unknown'
        assert 'capacity_profile' in settings['context_files']
        assert settings['limits'] == config['limits']
        state = read_json(directory/'state.json')
        observation = state['observations'][0]
        if arm['method'] == 'v2':
            assert settings['bo']['exploration_policy'] == 'capacity_v2'
            assert observation['proposal_reason'] == 'capacity_structure_probe'
            candidate = next(c for c in bundle['candidates'] if c['id'] == observation['candidate_id'])
            assert len(candidate['topology']['attention_gpus']) == len(candidate['topology']['expert_gpus']) == 1
            assert candidate['microbatches'] == 2
        else:
            assert settings['bo']['exploration_policy'] == 'legacy'
            assert observation['proposal_reason'] == 'measure_reference'
    assert create_comparison(args.directory/'campaign-settings.json', args.directory/'comparison', seeds=[0], resume=True) == manifest


@pytest.mark.parametrize('prepared', ['capacity8'], indirect=True)
def test_inherited_max_does_not_execute_or_claim_local_validation(prepared, monkeypatch):
    import copy
    args, old, _, launches = prepared
    args = copy.copy(args)
    args.reuse_max_directory = args.directory
    args.directory = args.directory.parent/'inherited-campaign'
    args.comparison = False
    previous_trials = launches.count('trial')
    config = official.initialize(args)
    def no_measurement(*args, **kwargs):
        pytest.fail('Reference reuse must never call attempt')
    monkeypatch.setattr(official, 'attempt', no_measurement)
    ready = official.prepare(config)
    assert launches.count('trial') == previous_trials
    assert ready['verified_structures'] == ready['attempted_structures'] == 0
    assert ready['inherited_max'] is True
    assert ready['limits'] == old['limits']
    assert ready['setup_cost_per_arm']['evaluations'] == 1
    bundle = read_json(args.directory/'campaign/bundle.json')
    assert bundle['settings']['bo']['exploration_policy'] == 'capacity_v2'
    assert read_json(args.directory/'campaign/state.json')['observations'] == []
    reference = read_json(args.directory/'inputs/slo-reference.json')
    assert reference['origin'] == 'historical_max_reuse'
    assert reference['local_measurement_performed'] is False
    assert not (args.directory/'validation').exists()
    from inherited_reference import inherit
    incompatible = copy.deepcopy(config)
    incompatible['rps'] = 16
    with pytest.raises(ValueError, match='protocol mismatch: rps'):
        inherit(incompatible, args.reuse_max_directory, args.directory/'inputs')


@pytest.mark.parametrize('prepared', ['direct'], indirect=True)
def test_direct_search_keeps_pending_structures_and_runs_comparison(prepared):
    args, config, ready, launches = prepared
    assert launches.count('reference') == 0 and launches.count('trial') == 1
    assert ready['verified_structures'] == 1 and ready['pending_structures'] == 15
    assert ready['setup_cost_per_arm']['evaluations'] == 1
    settings = read_json(args.directory/'campaign-settings.json')
    assert settings['allow_structure_probes'] and not settings['require_output_correctness']
    assert 'correctness_reference' not in settings['context_files']
    official.run(config, one=True)
    official.run(config, one=True)
    assert launches.count('trial') == 9  # one shared reference, two rounds of four methods


@pytest.mark.parametrize('prepared', ['direct'], indirect=True)
@pytest.mark.parametrize("tbt", [False, True])
def test_relative_max_two_requests_and_matching_arm_costs(prepared, tbt):
    args, old, _, launches = prepared
    official.run(old, one=True)
    full = args.directory/'comparison/v2-seed0'
    pending = official.ask(full)
    result = official.evaluate(old, pending, full/'trials'/pending['trial_id'])
    official.tell(full, result)
    old_state = (full/'state.json').read_bytes()
    changed = copy.copy(args)
    changed.directory = args.directory.parent/'relative-two'
    changed.prior_run_directory = args.directory
    changed.evaluation_requests = 2
    changed.slo_mode = 'relative_max'
    changed.tbt_slo = tbt
    config = official.initialize(changed)
    assert config['limits'] is None and config['trace']['requests'] == 2
    assert read_json(changed.directory/'inputs/source-isolation.json')['calibration']['requests'] == 8
    assert len((changed.directory/'inputs/warmup.jsonl').read_text().splitlines()) == 8
    ready = official.prepare(config)
    assert ready['limits'] == pytest.approx(dict(ttft_ms=10.5, tpot_ms=10.5, min_output_tps=9.5, **({'tbt_ms': 12.6} if tbt else {})))
    assert ready['setup_cost_per_arm']['evaluations'] == 2  # old common + new matched Max
    for method, expected in [('v2', 4), ('generic_bo', 3), ('random', 3), ('ga', 3)]:
        report = official.status(changed.directory/f'comparison/{method}-seed0')
        assert report['cost']['evaluations'] == expected
        assert report['cost']['gpu_hours'] == pytest.approx(expected*.001)
    assert (full/'state.json').read_bytes() == old_state
    frozen = read_json(changed.directory/'official-config.json')
    official.validate_context(frozen)
    assert official.prepare(frozen) == ready  # never remeasure or adjust Max after search
    official.run(frozen, one=True)


def test_official_prepare_search_comparison_and_freeze(prepared):
    args, config, ready, launches = prepared
    assert ready['verified_structures'] == 16 and ready['setup_cost_per_arm']['evaluations'] == 17
    assert launches.count('reference') == 1 and launches.count('trial') == 16
    manifest = read_json(Path(ready['target'])/'comparison.json')
    for arm in manifest['campaigns']:
        bundle = read_json(Path(arm['directory'])/'bundle.json')
        assert not bundle['settings']['require_four_stage']
        assert bundle['settings']['mechanism_model'] == 'external_power_duration_v1'
    official.run(config)
    for arm in manifest['campaigns']:
        report = status(arm['directory'])
        assert report['frozen'] and report['cost']['evaluations'] == 20
        assert report['four_stage_feedback_count'] == 0
    assert official.prepare(config) == ready  # no repeated calibration


def test_filtered_budget_and_rejection_evidence(prepared):
    args, config, ready, launches = prepared
    audit = read_json(args.directory/'inputs/space-audit.json')
    assert audit['structural_layouts'] == 18
    assert audit['reasons']['model_divisibility:attention_tp'] > 0
    assert audit['counts']['pending_validation'] > 0 and 'eligible' not in audit['counts']
    args.directory = args.directory.with_name('exact-filtered-budget')
    args.evaluations = 19  # 16 actually valid structures + stock reference + 2 BO
    fresh = official.initialize(args)
    assert len(fresh['validation_candidates']) == 16


@pytest.mark.parametrize('prepared', ['fail_tp2'], indirect=True)
def test_failed_measurements_exclude_only_the_measured_structure(prepared):
    args, config, ready, launches = prepared
    rules = read_json(args.directory/'inputs/runtime.json')['structures']
    for rule in rules:
        if rule['layout']['attention_tp'] == 2:
            assert rule['status'] == 'validation_failed' and rule['failure_reason']
        else:
            assert rule['status'] == 'verified'
    assert ready['setup_cost_per_arm']['evaluations'] == 17  # failures stay charged
    assert ready['verified_structures'] < ready['attempted_structures']
    manifest = read_json(Path(ready['target'])/'comparison.json')
    for arm in manifest['campaigns']:
        bundle = read_json(Path(arm['directory'])/'bundle.json')
        checks = {r['id']: r for r in bundle['audit']['candidates']}
        for candidate in bundle['candidates']:
            if candidate['topology']['attention_tp'] == 2:
                assert 'measured_structure_validation_failed' in checks[candidate['id']]['reasons']
        assert ask(arm['directory'])['configuration']['attention_tp'] == 1


def test_heldout_mutation_blocks_launch(prepared):
    args, config, ready, launches = prepared
    count = len(launches)
    Path(config['heldout_trace']['path']).write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        official.run(config)
    assert len(launches) == count


def test_tbt_constraint_rejects_stalls_despite_good_tpot():
    from static_dse.optimizer import best_measured
    limits = dict(ttft_ms=20, tpot_ms=20, tbt_ms=30, min_output_tps=5)
    rows = [dict(candidate_id='stalls', status='ok', metrics=dict(energy_j=10, ttft_ms=10, tpot_ms=10, tbt_ms=40, output_tps=10)),
            dict(candidate_id='smooth', status='ok', metrics=dict(energy_j=20, ttft_ms=10, tpot_ms=10, tbt_ms=25, output_tps=10))]
    assert best_measured(rows, limits)['candidate_id'] == 'smooth'
    assert best_measured(rows[:1], limits) is None

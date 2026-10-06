"""CPU-only native integration fixtures. No GPU measurements or model launches."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

BO_ROOT = Path(__file__).resolve().parents[1]
REPO = BO_ROOT.parent
sys.path.insert(0, str(BO_ROOT))
spec = importlib.util.spec_from_file_location('bo_native', BO_ROOT / 'native.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
import capture_hardware
from static_dse.campaign import ask, read_json, status, tell, write_json
from static_dse.space import digest
from four_stage_dse_v6.model import STAGES


@pytest.fixture
def host(tmp_path, monkeypatch, request):
    root = tmp_path / 'host'
    root.mkdir()
    for name in ('bo_dse/configs', 'bo_dse/scripts/afd', 'scripts/afd/static_dse'):
        shutil.copytree(REPO / name, root / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('migration/output_correctness.py', 'migration/static_dse.py', 'migration/bo_topology.py', 'migration/bo_layout.py', 'migration/verify_joint_launch.py', 'migration/native_service.py', 'scripts/afd/summarize_replay.py', 'bo_dse/native_backend.py',
                 'bo_dse/correctness_reference.py', 'bo_dse/stock_reference.py',
                 'bo_dse/native.py', 'bo_dse/entrypoint.py', 'environment/plugins.lock.json', 'inputs/protocols/qwen36/calibration-max-deployment.json'):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    for directory in ('environment', 'services', 'third_party/afd-plugin-ecodep-v026-dvfs-v6-four-stage/afd_plugin',
                      'artifacts/models/Qwen3.6-35B-A3B'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    write_json(root / 'artifacts/models/Qwen3.6-35B-A3B/config.json', {'num_hidden_layers': 28, 'num_experts': 256})
    for name in ('native-runtime.json', 'native-platform.json'):
        write_json(root / 'environment' / name, {'synthetic_fixture': True})
    traces = tmp_path / 'traces'
    traces.mkdir()
    for split, offset in (('calibration', 0), ('heldout', 100)):
        rows = [{'source_index': i + offset, 'source_timestamp': f'{split}-{i}', 'evaluation_split': split,
                 'request_id': i, 'arrival_s': i, 'input_tokens': 8, 'output_tokens': 16} for i in range(12)]
        (traces / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    monkeypatch.setattr(native, 'ROOT', root)
    monkeypatch.setattr(native, 'BO', root / 'bo_dse')
    monkeypatch.setattr(native.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(native, 'preflight', lambda config: None)
    # Exercise real flock semantics without reserving the host's physical GPUs
    # or contending with another independent CPU test process.
    def test_open(path, *args, **kwargs):
        if Path(path).name.startswith('moe-bo-gpu-'):
            path = tmp_path / Path(path).name
        return builtins.open(path, *args, **kwargs)
    monkeypatch.setattr(native, 'open', test_open, raising=False)
    hardware = {'host': 'synthetic-fixture', 'devices': {str(i): {
        'uuid': f'fixture-{i}', 'memory_mib': 81920, 'min_power_w': 100, 'max_power_w': 400,
        'clock_pairs': [{'memory_mhz': 1593, 'graphics_mhz': f} for f in (1050, 1170, 1290, 1350, 1410)]}
        for i in range(12)}}
    monkeypatch.setattr(capture_hardware, 'capture', lambda ids, nvml: hardware)
    original_invoke = native.invoke
    def invoke(config, args, **kwargs):
        if str(args[0]).endswith('native_backend.py') and str(args[1]) == 'validate':
            return original_invoke(config, args, **kwargs)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(native, 'invoke', invoke)
    gpus = getattr(request, 'param', [4, 5, 6, 7])
    joint = isinstance(gpus, dict)
    if joint:
        gpus = gpus['gpus']
    args = SimpleNamespace(directory=tmp_path / 'experiment', gpus=gpus, evaluations=len(native.topologies(gpus, 256)) + 4,
        gpu_hours=8, trial_timeout_seconds=60, calibration=traces / 'calibration.jsonl',
        heldout=traces / 'heldout.jsonl', model='qwen36', rps=1, native_python=Path(sys.executable),
        clock_url='http://127.0.0.1:9096', api_port=18000, afd_port=16239, dp_rpc_port=29550,
        comparison=False, seeds=[0], ttft_ms=None, tpot_ms=None, min_output_tps=None)
    if joint:
        args.space_mode='joint'
        args.microbatches=[1,2,4]
        args.parallel_degrees=[1,2,4]
        args.gpu_selection = tmp_path / 'gpu-selection.json'
        write_json(args.gpu_selection, {'status': 'ready', 'gpus': gpus,
                   'devices': [{'index': i, 'uuid': f'fixture-{i}', 'eligible_now': True} for i in gpus]})
    config = native.initial_inputs(args)
    if joint:
        selection_copy = args.directory / 'inputs/gpu-selection.json'
        assert read_json(selection_copy)['gpus'] == gpus
        assert str(selection_copy) in read_json(config['context_manifest'])['files_sha256']
        import correctness_reference
        def reference_measurement(config, directory, native):
            rows = [json.loads(s) for s in Path(config['trace']['path']).read_text().splitlines()]
            write_json(directory / 'reference.json', {'contract': config['correctness'], 'runtime': {
                'backend': 'stock_vllm_colocated', 'plugins': [], 'tensor_parallel_size': 2,
                'microbatches': 1, 'afd_enabled': False, 'vllm_version': '0.26.0'},
                'outputs': [{**r, 'output_token_ids': [10] * 16} for r in rows]})
            write_json(directory / 'tuning-telemetry.json', {'sample_error_count': 0, 'sample_time_coverage': 1.,
                'energy_j': 1000., 'gpu_ids': config['gpus'][:2], 'fixture': True})
        monkeypatch.setattr(correctness_reference, 'measurement', reference_measurement)
    launches = []
    def measured(config, plan, request, directory):
        launches.append(copy.deepcopy(request))
        write_cell(config, plan, request, directory)
    monkeypatch.setattr(native, 'measured_suite', measured)
    return args, config, launches


def write_cell(config, plan, request, directory):
    """Emit native file schemas; real collectors/importers validate these fixtures."""
    suite, run = native.suite_paths(plan)
    suite.mkdir(parents=True)
    run.mkdir(parents=True)
    cell = run / f'rps-{config["rps"]}'
    (cell / 'traces').mkdir(parents=True)
    entry = plan['schedule'][0]
    deployment = directory / 'deployments' / (request['candidate_id'] + '.json')
    deploy = read_json(deployment)
    cfg = request['configuration']
    ids = plan['measurement_gpus']
    energy = len(ids) * 200.
    start, end = 10_000_000_000, 12_000_000_000
    rows = [json.loads(r) for r in Path(config['trace']['path']).read_text().splitlines()]
    replay = [{**r, 'error': None, 'actual_output_tokens': 16, 'ttft_ms': 100., 'tpot_ms': 10.,
               'actual_output_token_ids': [10] * 16,
               'submit_wall_ns': start, 'finish_wall_ns': end} for r in rows]
    (run / f'replay-ecodep-v026-rps-{config["rps"]}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in replay))
    summary = {'requests': len(rows), 'completed_requests': len(rows), 'failed_requests': 0,
               'energy_j': energy, 'duration_s': 2., 'ttft_ms': {'p90': 100.}, 'tpot_ms': {'p90': 10.},
               'output_token_throughput_tps': len(rows) * 16 / 2}
    write_json(cell / 'summary.json', summary)
    telemetry = {'returncode': 0, 'request_count': len(rows), 'sample_time_coverage': 1., 'sample_error_count': 0,
                 'gpu_ids': ids, 'energy_j': energy, 'duration_s': 2., 'per_gpu_energy_j': [200.] * len(ids),
                 'measurement_window_source': 'request_first_submit_to_last_finish',
                 'started_wall_ns': start, 'finished_wall_ns': end}
    write_json(cell / 'telemetry.json', telemetry)
    write_json(cell / 'measurement-trace-reset.json', {'all_files_empty_after_reset': True, 'reset_wall_ns': start - 100})
    def rank_tokens(total, role, rank):
        na, ne = len(cfg['attention_gpus']), len(cfg['expert_gpus'])
        per_attention = ([max(1, total // cfg['attention_dp'])] * na if config.get('joint_space') else
                         [total // na + int(r < total % na) for r in range(na)])
        if role == 'attention':
            return per_attention[rank]
        ratio = na // ne
        return sum(per_attention[rank * ratio:(rank + 1) * ratio])

    for role in ('attention', 'ffn'):
        n = len(cfg['attention_gpus' if role == 'attention' else 'expert_gpus'])
        for rank in range(n):
            stages = [s for s in STAGES if (s == 'ffn_compute') == (role == 'ffn')]
            events = [{'event': stage, 'transaction_id': f'txn-{tokens}', 'stage_idx': mb,
                       'start_wall_ns': start + 100, 'end_wall_ns': end - 100,
                       'duration_us': 1000 + tokens * 10, 'prefill_tokens': 0,
                       'decode_tokens': rank_tokens(tokens, role, rank)}
                      for tokens in (2, 4, 6) for mb in range(cfg['microbatches']) for stage in stages]
            if config.get('joint_space'):
                per_replica = n // cfg['expert_dp']
                for event in events:
                    event['transaction_id'] = f"replica-{rank // per_replica}:" + event['transaction_id']
                    event['parallel_layout'] = {'replica': rank // per_replica, 'local_role_rank': rank % per_replica,
                                                'tp_size': cfg['attention_tp' if role == 'attention' else 'expert_tp']}
            (cell / 'traces' / f'stage-{role}-{rank}-123.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in events))
    states = [{'graphics_mhz': cfg[role + '_mhz'], 'power_limit_w': cfg[role + '_power_w'],
               'gpu_utilization': 50, 'throttle_reasons': 0}
              for role in ('attention', 'expert') for gpu in cfg[role + '_gpus']]
    samples = [{'timestamp_ns': t, 'gpu_ids': ids, 'power_w': [100.] * len(ids), 'operating_state': states}
               for t in (start, start + 1_000_000_000, end)]
    (cell / 'power-samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in samples))
    point = deploy['operating_points']['by_rps'][str(config['rps'])]
    write_json(cell / 'case.json', {'evaluation_split': 'calibration', 'offered_rps': config['rps'],
                                   'repetition': 1, 'operating_point': point})
    write_json(cell / 'operating-point-ack.json', {'verified': True, 'verification_errors': [],
        'attention_gpus': cfg['attention_gpus'], 'expert_gpus': cfg['expert_gpus'],
        **{'requested_' + k: point[k] for k in ('attention_mhz', 'expert_mhz', 'attention_power_w', 'expert_power_w')}})
    write_json(cell / 'gpu-contamination-validation.json', {'verified': True})
    write_json(run / 'launch_config.json', {'attention_ranks': len(cfg['attention_gpus']), 'expert_ranks': len(cfg['expert_gpus']),
        'attention_gpus': ','.join(map(str, cfg['attention_gpus'])), 'expert_gpus': ','.join(map(str, cfg['expert_gpus']))})
    if config.get('joint_space'):
        from bo_layout import deployment as joint_deployment, replicas
        groups = replicas(cfg)
        actual = {**joint_deployment(cfg), 'replicas': groups, 'dbo_enabled': cfg['microbatches'] == 2,
                  **config['microbatch_thresholds'],
                  **{r+'_gpus': ','.join(map(str,cfg[r+'_gpus'])) for r in ('attention','expert')}}
        write_json(run/'joint-launch.json', actual)
        write_json(run/'launch_config.json', actual)
        for group in groups:
            for role, key in [('attention','attention'),('ffn','expert')]:
                for local in range(group[key+'_ranks']):
                    rank = local + group[key+'_rank_offset']
                    write_json(run/f'worker-layout-{role}-{rank}.json', {
                        'role':role,'rank':rank,'local_role_rank':local,'replica':group['replica'],'verified':True,
                        'actual': {'native_tp':group[key+'_tp'],
                                   'native_dp':group['attention_dp' if role=='attention' else 'expert_native_dp'],
                                   'enable_expert_parallel':group['expert_enable_ep'] if role=='ffn' else False,
                                   'microbatches':cfg['microbatches'], **config['microbatch_thresholds']},
                        'expert_shards':[{'tp':cfg['expert_tp'],'ep':cfg['expert_ep']}] if role=='ffn' else []})
    for name in ('allocation-reset-before.json', 'clock-reset.json'):
        write_json(suite / name, {'verified': True, 'requested_gpus': ids})
    write_json(suite / 'manifest.json', {'evaluation_split': 'calibration', 'formal_evaluation_eligible': False,
        'trace_sha256': config['trace']['sha256'], 'deployment_sha256': native.file_hash(deployment),
        'model': plan['model_tag'], 'repetition': 1, 'rates_rps': [config['rps']], 'plugin': deploy['plugin'],
        'schedule': {'position': 1, 'sha256': native.file_hash(directory / 'schedule.json')},
        'comparison_contract': {'measurement': {'gpu_ids': ids}, 'topology': deploy['topology'],
                                'server': config.get('microbatch_thresholds', {}),
                                'model': {'config_sha256': native.file_hash(native.ROOT / 'artifacts/models/Qwen3.6-35B-A3B/config.json')},
                                'generation': {'max_output_tokens': 128,
                                               **config.get('correctness', {}).get('generation', {})}}})
    (suite / 'COMPLETE').touch()
    write_json(directory / 'tuning-telemetry.json', {'sample_error_count': 0, 'sample_time_coverage': 1.,
                                                   'energy_j': len(ids) * 500., 'gpu_ids': ids, 'fixture': True})


def test_bootstrap_to_real_importer_ask_tell_freeze(host):
    args, config, launches = host
    settings = native.prepare(config)
    assert len(launches) == len(config['references'])
    assert settings['mechanism_model'] == 'four_stage_fifo_v1'
    assert settings['require_four_stage'] is True
    assert not settings['workload'].get('allow_model_feedback_fallback')
    deployment = next((args.directory / 'bootstrap-00/native/deployments').glob('*.json'))
    assert read_json(deployment)['plugin']['commit'] == config['plugin_commit']
    assert settings['setup_cost']['evaluations'] == 4
    assert settings['limits']['ttft_ms'] == pytest.approx(105)
    assert settings['workload']['decode_tokens_per_microbatch'] == 4
    assert settings['workload']['four_stage_token_scope'] == 'attention_dp_total'
    campaign = args.directory / 'campaign'
    assert read_json(campaign / 'mechanism.json')['physical_anchor_count'] == 4
    assert status(campaign)['filter_counts'] == {'eligible': 2500}
    for _ in range(4):
        req = ask(campaign)
        path = campaign / 'trials' / req['trial_id'] / 'result.json'
        path.parent.mkdir(parents=True)
        result = native.execute(config, req, path)
        assert result['status'] == 'ok', result.get('failure_reason')
        assert result['four_stage']['provenance'] == 'raw_calibration_stage_traces_and_nvml'
        tell(campaign, result)
    assert ask(campaign)['stopped']
    frozen = native.freeze(campaign)
    assert frozen['cost']['evaluations'] == args.evaluations
    assert frozen['best']['feasible']
    assert frozen['heldout_evaluation_completed'] is False


def test_native_context_calibration_flows_through_bootstrap_feedback_and_bo(host, monkeypatch):
    args, config, _ = host

    def measured_with_context(config, plan, request, directory):
        write_cell(config, plan, request, directory)
        _, run = native.suite_paths(plan)
        for path in (run / f'rps-{config["rps"]}' / 'traces').glob('*.jsonl'):
            source = [json.loads(line) for line in path.read_text().splitlines()]
            rows = []
            for context_length in (100, 200, 300):
                for original in source:
                    row = {**original, 'transaction_id': original['transaction_id'] + f'-ctx{context_length}', 'layer_idx': 0}
                    if row['event'] == 'attention_compute':
                        d = row['decode_tokens']
                        row['attention_workload'] = dict(schema_version=1, source='request_token_spans',
                            active_requests=d, prefill_tokens=0, decode_tokens=d,
                            prefill_context_tokens=0, decode_context_tokens=d * context_length,
                            kv_sequence_tokens=d * context_length)
                        row['duration_us'] = 100 + 20 * d + .5 * d * context_length
                    for layer in range(28):
                        rows.append({**row, 'layer_idx': layer, 'duration_us': row['duration_us'] / 28})
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    monkeypatch.setattr(native, 'measured_suite', measured_with_context)
    settings = native.prepare(config)
    assert settings['workload']['decode_context_tokens_per_microbatch'] == 800
    normalization = read_json(args.directory / 'inputs/attention-workload-normalization.json')
    assert normalization['selection_split'] == 'calibration'
    assert normalization['status'] == 'shared_observed_context_supported'
    campaign = args.directory / 'campaign'
    req = ask(campaign)
    path = campaign / 'trials' / req['trial_id'] / 'result.json'
    path.parent.mkdir(parents=True)
    result = native.execute(config, req, path)
    assert result['status'] == 'ok', result.get('failure_reason')
    diagnostics = result['four_stage']['attention_diagnostics']
    assert diagnostics['context_coverage'] == 1
    assert diagnostics['prediction']['context_enabled']
    assert diagnostics['prediction']['available']
    assert diagnostics['synchronization_wait_measured'] is False
    analytical = result['four_stage']['analytical_provisioning']
    assert analytical['layer_barrier_coverage'] == 1
    assert len(analytical['schedule_model']['layer_weights']) == 28
    assert analytical['coefficient_fits']['attention']['coefficients_identified']
    assert analytical['role_power_w']['attention'] == 100 * len(req['configuration']['attention_gpus'])
    assert analytical['mapping']['ffn_service_groups'] == 1
    assert not analytical['workload_moments']['available']  # Fixture has no request lifetimes.
    tell(campaign, result)
    recommendation = ask(campaign)
    assert recommendation['decision']['reason'] == 'analytical_provisioning_probe'
    assert recommendation['mechanism']['analytical_provisioning']['available']
    assert not recommendation['structure_validation_trial']
    next_path = campaign / 'trials' / recommendation['trial_id'] / 'result.json'
    next_path.parent.mkdir(parents=True)
    next_result = native.execute(config, recommendation, next_path)
    assert next_result['status'] == 'ok', next_result.get('failure_reason')
    tell(campaign, next_result)
    assert status(campaign)['cost']['evaluations'] == len(config['references']) + 2
    mechanism = read_json(campaign / 'current-mechanism.json')
    assert all(row['attention_basis'] == 'context_rank' for g in mechanism['groups'].values() for row in g['anchors'])


def test_native_plan_uses_active_gpu_energy_and_frozen_operating_points(host):
    args, config, launches = host
    cfg = next(c for c in config['references'] if len(c['attention_gpus']) == 2 and len(c['expert_gpus']) == 1)
    req = native.bootstrap_request(config, cfg, 1)
    path = args.directory / 'plan-only'
    path.mkdir()
    plan = native.build_plan(config, req, path)
    cmd, env = native.suite_command(config, plan, req)
    assert plan['measurement_gpus'] == [4, 5, 6]
    assert plan['allocation'] == [4, 5, 6, 7]
    assert env['ECODEP_BO_FEEDBACK'] == '1'
    assert env['ECODEP_MEASUREMENT_GPUS'] == '4,5,6'
    assert env['ECODEP_EXPERT_CLOCKS_MHZ'] == '1410'
    assert cmd[-1] == '1'
    assert not launches
    req['configuration']['expert_mhz'] = 1170
    with pytest.raises(ValueError, match='digest'):
        native.build_plan(config, req, path)


def test_native_complete_decode_lifetimes_enable_renewal_and_gaussian_features(host, monkeypatch):
    args, config, _ = host

    def measured(config, plan, request, directory):
        write_cell(config, plan, request, directory)
        _, run = native.suite_paths(plan)
        na = len(request['configuration']['attention_gpus'])
        for path in (run / f'rps-{config["rps"]}' / 'traces').glob('*.jsonl'):
            originals = [json.loads(line) for line in path.read_text().splitlines()]
            rank = int(path.name.split('-')[2])
            rows = []
            for wave in range(8):
                for original in originals:
                    mb = original['stage_idx']
                    if wave == 7 and mb == 1:
                        continue
                    total = int(original['transaction_id'].split('-')[1])
                    position = 8 + 2 * wave + mb
                    for layer in range(28):
                        row = {**original, 'transaction_id': original['transaction_id'] + f'-wave{wave}', 'layer_idx': layer}
                        row['duration_us'] /= 28
                        if path.name.startswith('stage-attention'):
                            n = row['decode_tokens']
                            row['attention_workload'] = dict(schema_version=1, source='request_token_spans',
                                active_requests=n, prefill_tokens=0, decode_tokens=n, prefill_context_tokens=0,
                                decode_context_tokens=n * (position + 1), kv_sequence_tokens=n * (position + 1))
                            if layer == 0:
                                offset = {2: 0, 4: 2, 6: 6}[total] + sum(total // na + int(r < total % na) for r in range(rank))
                                row['decode_request_spans'] = [dict(request_id=f'r-{offset+i}', prompt_tokens=8,
                                    first_token_position=position, token_count=1) for i in range(n)]
                            if row['event'] == 'attention_compute':
                                row['duration_us'] = (100 + 10 * n + .5 * n * (position + 1)) / 28
                        rows.append(row)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    monkeypatch.setattr(native, 'measured_suite', measured)
    native.prepare(config)
    profile = read_json(args.directory / 'inputs/profile.json')
    for candidate in profile['candidates']:
        model = candidate['analytical_provisioning']
        moments = model['workload_moments']
        assert moments['available']
        assert moments['decode_slot_steps'] == 12 * 15
        assert moments['theta'] == 16.
        assert moments['expected_decode_queries'] == 180
        assert model['evidence']['completed_replay_sha256']
    request = ask(args.directory / 'campaign')
    prediction = request['mechanism']['analytical_provisioning']
    assert prediction['available']
    assert prediction['gaussian_cycle_ms'] >= prediction['mean_field_cycle_ms']


@pytest.mark.parametrize('host', [list(range(6)), [7, 3, 5, 1, 6, 2, 4, 0]], indirect=True)
def test_six_eight_gpu_bootstrap_feedback_and_bo(host):
    args, config, launches = host
    count = len(args.gpus)
    expected = {6: [(4, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (2, 2)],
                8: [(4, 4), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (2, 2), (4, 2), (6, 2)]}[count]
    assert [(len(c['attention_gpus']), len(c['expert_gpus'])) for c in config['references']] == expected
    runtime = read_json(args.directory / 'inputs/runtime.json')
    assert runtime['gpu_budget'] == count
    assert len(config['configurations']) == len(expected) * 625
    settings = native.prepare(config)
    assert settings['setup_cost']['evaluations'] == len(expected)
    campaign = args.directory / 'campaign'
    assert status(campaign)['filter_counts'] == {'eligible': len(expected) * 625}
    for i, (na, ne) in enumerate(expected):
        directory = args.directory / f'bootstrap-{i:02d}' / 'native'
        plan = read_json(directory / 'PLAN.json')
        req = native.bootstrap_request(config, config['references'][i], i)
        _, env = native.suite_command(config, plan, req)
        deploy = read_json(directory / 'deployments' / (req['candidate_id'] + '.json'))
        assert deploy['topology'] == dict(attention_dp=na, ffn_ep=ne, attention_tp=1, expert_tp=1)
        assert plan['allocation'] == args.gpus
        assert plan['measurement_gpus'] == args.gpus[:na + ne]
        assert env['ECODEP_ATTENTION_GPUS'] == ','.join(map(str, args.gpus[:na]))
        assert env['ECODEP_EXPERT_GPUS'] == ','.join(map(str, args.gpus[na:na + ne]))
        assert env['ECODEP_ATTENTION_CLOCKS_MHZ'].split(',') == ['1410'] * na
        assert env['ECODEP_EXPERT_POWER_W'].split(',') == ['400'] * ne
        receipt = read_json(directory.parent / 'result.json')
        assert receipt['metrics']['energy_j'] == (na + ne) * 200.
        assert receipt['cost']['tuning_energy_j'] == (na + ne) * 500.
        assert receipt['cost']['gpu_hours'] == pytest.approx(count * receipt['cost']['wall_seconds'] / 3600)
    for _ in range(args.evaluations - len(expected)):
        req = ask(campaign)
        path = campaign / 'trials' / req['trial_id'] / 'result.json'
        path.parent.mkdir(parents=True)
        result = native.execute(config, req, path)
        assert result['status'] == 'ok', result.get('failure_reason')
        tell(campaign, result)
    assert ask(campaign)['stopped']
    frozen = native.freeze(campaign)
    assert frozen['cost']['evaluations'] == args.evaluations
    assert frozen['best']['feasible']
    assert not frozen['heldout_evaluation_completed']


@pytest.mark.parametrize('host', [list(range(6)), list(range(8))], indirect=True)
@pytest.mark.parametrize('damage', ['launch_mapping', 'stage_rank', 'power_gpu'])
def test_larger_allocations_reject_mismatched_feedback(host, monkeypatch, damage):
    args, config, launches = host
    def damaged(config, plan, req, directory):
        write_cell(config, plan, req, directory)
        _, run = native.suite_paths(plan)
        if damage == 'launch_mapping':
            path = run / 'launch_config.json'
            doc = read_json(path)
            doc['attention_ranks'] = 2
            write_json(path, doc)
        elif damage == 'stage_rank':
            next((run / f'rps-{config["rps"]}' / 'traces').glob('stage-attention-0-*')).unlink()
        else:
            path = run / f'rps-{config["rps"]}' / 'telemetry.json'
            doc = read_json(path)
            doc['gpu_ids'] = doc['gpu_ids'][:-1]
            write_json(path, doc)
    monkeypatch.setattr(native, 'measured_suite', damaged)
    req = native.bootstrap_request(config, config['references'][0], 0)
    result = native.execute(config, req, args.directory / 'damaged/result.json', bootstrap=True)
    assert result['status'] == 'failed'
    assert result['cost']['gpu_hours'] > 0


@pytest.mark.parametrize('count,expected', [(4, 4), (6, 7), (8, 11)])
def test_allocation_dry_run_without_hardware(tmp_path, count, expected):
    destination = tmp_path / 'dry'
    result = subprocess.run([sys.executable, str(BO_ROOT / 'native.py'), '--backend', 'customized', 'start',
                             '--directory', str(destination), '--gpus', *map(str, range(count)), '--space-mode', 'legacy', '--dry-run'],
                            check=True, capture_output=True, text=True)
    doc = json.loads(result.stdout)
    assert doc['bootstrap_evaluations'] == expected
    assert not doc['gpu_actions_performed']
    assert not destination.exists()


@pytest.mark.parametrize('gpus', [[0, 1, 2], list(range(5)), [0, 1, 2, 2, 4, 5], list(range(9))])
def test_unsupported_allocations_fail_before_gpu_access(gpus):
    with pytest.raises(ValueError, match='4, 6, or 8'):
        native.topologies(gpus)


def test_budget_spaces_are_nested_and_model_divisibility_is_applied():
    previous = set()
    for count, size in [(4, 4), (6, 7), (8, 11)]:
        rows = native.topologies(list(range(count)), 256)
        pairs = {(len(c['attention_gpus']), len(c['expert_gpus'])) for c in rows}
        assert len(pairs) == size
        assert previous <= pairs
        assert pairs == {(a, e) for a in range(1, count) for e in range(1, count)
                         if a + e <= count and a >= e and a % e == 0 and 256 % e == 0}
        assert pairs == {(len(c['attention_gpus']), len(c['expert_gpus']))
                         for c in native.topologies(list(range(count)), 64)}
        previous = pairs
    assert any(c['id'] == '3a3e' for c in native.topologies(list(range(8)), 12))


def test_idle_gpus_excluded_from_measured_and_failure_tuning_energy(host, monkeypatch):
    args, config, _ = host
    cfg = next(c for c in config['references'] if len(c['attention_gpus']) == len(c['expert_gpus']) == 1)
    req = native.bootstrap_request(config, cfg, 1)
    directory = args.directory / 'energy-boundary'
    directory.mkdir()
    plan = native.build_plan(config, req, directory)
    active = cfg['attention_gpus'] + cfg['expert_gpus']
    write_json(directory / 'STARTED.json', {'started_wall_seconds': 10.})
    monkeypatch.setattr(native.time, 'time', lambda: 12.)
    # Large limits on idle cards must not inflate even the crash estimate.
    config = copy.deepcopy(config)
    config['maximum_power_w'] = [200, 300, 10000, 20000]
    cost, source = native.receipt_cost(config, directory)
    assert source == 'power_limit_upper_bound_after_meter_failure'
    assert cost['tuning_energy_j'] == 1000.
    assert cost['gpu_hours'] == pytest.approx(4 * 2 / 3600)
    telemetry = {'sample_error_count': 0, 'sample_time_coverage': 1., 'gpu_ids': active, 'energy_j': 123.}
    write_json(directory / 'tuning-telemetry.json', telemetry)
    cost, source = native.receipt_cost(config, directory)
    assert cost['tuning_energy_j'] == 123.
    assert source == 'nvml_full_suite_window'
    telemetry.update(gpu_ids=config['gpus'], energy_j=99999.)
    write_json(directory / 'tuning-telemetry.json', telemetry)
    cost, source = native.receipt_cost(config, directory)
    assert source == 'power_limit_upper_bound_after_meter_failure'
    assert cost['tuning_energy_j'] == 1000.
    # Exercise the production meter command up to process launch, without NVML.
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.suite_command = native.suite_command
    def capture_meter(command, **kwargs):
        assert command[command.index('--gpus') + 1] == ','.join(map(str, active))
        raise RuntimeError('meter captured without GPU access')
    monkeypatch.setattr(native.subprocess, 'Popen', capture_meter)
    with pytest.raises(RuntimeError, match='meter captured'):
        original.measured_suite(config, plan, req, directory)


def test_p2p_and_model_divisibility_filter(host):
    args, config, _ = host
    spec = read_json(args.directory / 'inputs/space.json')
    runtime = read_json(args.directory / 'inputs/runtime.json')
    hardware = read_json(args.directory / 'inputs/hardware.json')
    c = native.enumerate_candidates(spec)[0]
    c['topology'].update(attention_gpus=[4, 5, 6], attention_dp=3, expert_gpus=[7, 8], expert_ep=2)
    runtime.update(gpu_budget=8, allowed_gpus=list(range(12)))
    assert 'p2p_attention_expert_rank_ratio' in native.hard_filter(c, hardware, runtime)['reasons']
    c['topology'].update(attention_gpus=[4, 5, 6], attention_dp=3, expert_gpus=[7, 8, 9], expert_ep=3)
    assert 'model_divisibility:expert_ep' in native.hard_filter(c, hardware, runtime)['reasons']


def test_failed_trial_is_charged_and_cached_without_reexecution(host, monkeypatch):
    args, config, launches = host
    native.prepare(config)
    campaign = args.directory / 'campaign'
    req = ask(campaign)
    path = campaign / 'trials' / req['trial_id'] / 'result.json'
    path.parent.mkdir(parents=True)
    def fail(*args):
        raise RuntimeError('model launch failed')
    monkeypatch.setattr(native, 'measured_suite', fail)
    result = native.execute(config, req, path)
    assert result['status'] == 'failed'
    assert result['cost']['tuning_energy_j'] > 0
    assert result['cost_energy_source'] == 'power_limit_upper_bound_after_meter_failure'
    assert native.execute(config, req, path) == result
    tell(campaign, result)
    assert status(campaign)['cost']['evaluations'] == len(config['references']) + 1
    assert status(campaign)['best'] is None


def test_context_tamper_rejected_before_launch(host):
    args, config, launches = host
    Path(config['warmup_trace']).write_text('{}\n')
    with pytest.raises(ValueError, match='changed'):
        native.prepare(config)
    assert not launches


def test_dry_run_has_no_outputs_or_gpu_dependency(tmp_path):
    directory = tmp_path / 'must-not-exist'
    result = subprocess.run([sys.executable, str(BO_ROOT / 'native.py'), 'start', '--directory', str(directory), '--dry-run'],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['gpu_actions_performed'] is False
    assert not directory.exists()


@pytest.mark.parametrize('host', [{'gpus':[7,3,5,1]}], indirect=True)
@pytest.mark.parametrize('target', [dict(attention_tp=2, expert_tp=2, expert_dp=1, microbatches=4),
                                   dict(attention_tp=1, expert_tp=1, expert_dp=2, microbatches=1)])
def test_joint_gpu_mapping_through_prepare_ask_measure_tell(host, monkeypatch, target):
    args, config, launches = host
    settings = native.prepare(config)
    assert settings['setup_cost']['evaluations'] == 2
    assert settings['require_output_correctness']
    assert settings['allow_structure_probes']
    campaign = args.directory/'campaign'
    req = ask(campaign)
    path = campaign/'trials'/req['trial_id']/'result.json'; path.parent.mkdir(parents=True)
    receipt = native.execute(config, req, path)
    assert receipt['status']=='ok', receipt.get('failure_reason')
    tell(campaign,receipt)
    import static_dse.campaign as campaigns
    def choose(candidates, observations, settings, eligible, remaining):
        c = next(c for c in candidates if c['id'] in eligible and
                 all(native.configuration(c)[k]==v for k,v in target.items()))
        return c, {'reason':'test_nondefault_structure', 'expected_gpu_hours':.01,'base_gpu_hours':.01}
    monkeypatch.setattr(campaigns,'propose',choose)
    req=ask(campaign)
    assert req['structure_validation_trial'] is True
    assert req['configuration']['attention_gpus'] == [7,3]
    assert req['configuration']['expert_gpus'] == [5,1]
    assert all(req['configuration'][k]==v for k,v in target.items())
    path=campaign/'trials'/req['trial_id']/'result.json';path.parent.mkdir(parents=True)
    receipt=native.execute(config,req,path)
    if target['expert_dp'] > 1:
        # Independent replica schedules currently have no identified four-stage
        # prior. Charge the failed trial instead of accepting end-to-end fallback.
        assert receipt['status'] == 'failed'
        assert 'independent_replica' in receipt['failure_reason']
        tell(campaign, receipt)
        observations = read_json(campaign / 'state.json')['observations']
        assert observations[-1]['status'] == 'failed'
        assert observations[-1]['candidate_id'] == req['candidate_id']
        return
    assert receipt['status']=='ok',receipt.get('failure_reason')
    assert receipt['output_correctness']['verified']
    missing_gate = copy.deepcopy(receipt)
    missing_gate.pop('output_correctness')
    with pytest.raises(ValueError, match='output correctness'):
        tell(campaign, missing_gate)
    assert receipt['four_stage']['observed_microbatches']==target['microbatches']
    tell(campaign,receipt)
    assert ask(campaign)['structure_validation_trial'] is False


@pytest.mark.parametrize('host', [{'gpus':[7,3,5,1]}], indirect=True)
def test_joint_corrupt_output_is_failed_and_charged(host, monkeypatch):
    args, config, _ = host
    native.prepare(config)
    campaign = args.directory / 'campaign'
    req = ask(campaign)
    def corrupt(config, plan, request, directory):
        write_cell(config, plan, request, directory)
        _, run = native.suite_paths(plan)
        replay = run / f'replay-ecodep-v026-rps-{config["rps"]}.jsonl'
        rows = [json.loads(s) for s in replay.read_text().splitlines()]
        rows[0]['actual_output_token_ids'][0] = 99
        replay.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    monkeypatch.setattr(native, 'measured_suite', corrupt)
    path = campaign / 'trials' / req['trial_id'] / 'result.json'
    result = native.execute(config, req, path)
    assert result['status'] == 'failed'
    assert result['cost']['gpu_hours'] > 0
    plan = read_json(path.parent / 'native/PLAN.json')
    _, run = native.suite_paths(plan)
    report = read_json(run / f'rps-{config["rps"]}/correctness-report.json')
    assert not report['verified'] and report['mismatched_requests'] == 1
    tell(campaign, result)
    assert status(campaign)['best'] is None


@pytest.mark.parametrize('host', [{'gpus':[7,3,5,1]}], indirect=True)
def test_stock_reference_failure_retry_freeze_and_cost(host, monkeypatch):
    args, config, launches = host
    import correctness_reference
    real_measure = correctness_reference.measurement
    def fail(*args): raise RuntimeError('reference fixture failure')
    monkeypatch.setattr(correctness_reference, 'measurement', fail)
    with pytest.raises(ValueError, match='Stock reference failed'):
        native.prepare(config)
    assert not launches
    failed = read_json(args.directory / 'correctness-attempt-000/result.json')
    assert failed['cost']['tuning_energy_j'] > 0
    assert not (args.directory / 'inputs/correctness-reference.json').exists()
    monkeypatch.setattr(correctness_reference, 'measurement', real_measure)
    settings = native.prepare(config, retry_failed=True)
    assert settings['setup_cost']['evaluations'] == 3
    assert settings['setup_cost']['tuning_energy_j'] > failed['cost']['tuning_energy_j'] + 1000
    reference = args.directory / 'inputs/correctness-reference.json'
    assert str(reference) in read_json(config['context_manifest'])['files_sha256']
    monkeypatch.setattr(correctness_reference, 'measurement', fail)
    native.prepare(config)  # READY resume cannot remeasure the reference.
    reference.write_text('{}')
    with pytest.raises(ValueError, match='Frozen environment'):
        native.validate_context(config)


def test_partial_trial_requires_explicit_recovery(host):
    args, config, launches = host
    native.prepare(config)
    campaign = args.directory / 'campaign'
    req = ask(campaign)
    path = campaign / 'trials' / req['trial_id'] / 'result.json'
    directory = path.parent / 'native'
    directory.mkdir(parents=True)
    native.build_plan(config, req, directory)
    write_json(directory / 'request.json', req)
    import time
    write_json(directory / 'STARTED.json', {'started_wall_seconds': time.time()})
    with pytest.raises(ValueError, match='Interrupted'):
        native.execute(config, req, path)
    result = native.execute(config, req, path, recover=True)
    assert result['status'] == 'failed'
    assert len(launches) == len(config['references'])
    tell(campaign, result)


def test_bootstrap_retry_preserves_failure_cost(host, monkeypatch):
    args, config, launches = host
    measured = native.measured_suite
    def fail(*args):
        raise RuntimeError('transient startup failure')
    monkeypatch.setattr(native, 'measured_suite', fail)
    with pytest.raises(ValueError, match='Bootstrap failed'):
        native.prepare(config)
    failed = read_json(args.directory / 'bootstrap-00/result.json')
    monkeypatch.setattr(native, 'measured_suite', measured)
    with pytest.raises(ValueError, match='Bootstrap failed'):
        native.prepare(config)
    settings = native.prepare(config, retry_failed=True)
    assert settings['setup_cost']['evaluations'] == len(config['references']) + 1
    assert settings['setup_cost']['tuning_energy_j'] == pytest.approx(sum(500 * (len(c['attention_gpus']) + len(c['expert_gpus'])) for c in config['references']) + failed['cost']['tuning_energy_j'])
    assert (args.directory / 'bootstrap-00-retry-001/result.json').exists()
    assert read_json(args.directory / 'bootstrap-00/result.json') == failed


def test_cleanup_failure_blocks_next_launch(host, monkeypatch):
    args, config, launches = host
    native.prepare(config)
    req = ask(args.directory / 'campaign')
    path = args.directory / 'failed/result.json'
    def fail(*args):
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(native, 'measured_suite', fail)
    monkeypatch.setattr(native, 'cleanup', fail)
    result = native.execute(config, req, path)
    assert result['cleanup_error']
    assert (args.directory / 'CLEANUP_REQUIRED.json').exists()
    with pytest.raises(ValueError, match='Previous cleanup failed'):
        native.execute(config, req, args.directory / 'another/result.json')
    assert len(launches) == len(config['references'])


@pytest.mark.parametrize('host', [list(range(4)), list(range(6)), list(range(8))], indirect=True)
def test_comparison_initialization_uses_matched_bootstrap_and_budgets(host):
    args, config, launches = host
    config['comparison'] = True
    write_json(config['config_path'], config)
    native.prepare(config)
    manifest = read_json(args.directory / 'comparison/comparison.json')
    assert len(manifest['campaigns']) == 4
    bootstrap_count = len(config['references'])
    assert manifest['setup_cost_per_arm']['evaluations'] == bootstrap_count
    assert manifest['budget_per_arm']['evaluations'] == args.evaluations
    for arm in manifest['campaigns']:
        assert status(arm['directory'])['four_stage_feedback_count'] == 0
        assert status(arm['directory'])['cost']['evaluations'] == bootstrap_count
    assert len(launches) == bootstrap_count
    # Simulate interruption after campaigns were committed but before READY.
    (args.directory / 'READY.json').unlink()
    native.prepare(config)
    assert len(launches) == bootstrap_count
    assert read_json(args.directory / 'comparison/comparison.json') == manifest


def test_timeout_terminates_owned_process_and_calls_service_cleanup(host, monkeypatch):
    args, config, launches = host
    request = native.bootstrap_request(config, config['references'][0], 0)
    directory = args.directory / 'timeout'
    directory.mkdir()
    plan = native.build_plan(config, request, directory)
    # A real CPU subprocess exercises timeout and process-group termination.
    real_popen = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        process = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        children.append(process)
        return process
    cleaned = []
    monkeypatch.setattr(native.subprocess, 'Popen', spawn)
    monkeypatch.setattr(native, 'cleanup', lambda config, plan: cleaned.append(plan))
    config['trial_timeout_seconds'] = .05
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            # Restore the production function from a fresh module, retaining this fixture's plan/environment.
            original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original)
            original.cleanup = native.cleanup
            original.measured_suite(config, plan, request, directory)
        assert children[0].poll() is not None
        assert cleaned
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.wait()


def test_run_one_invokes_real_evaluator_cli_and_settles_failure(host):
    args, config, launches = host
    native.prepare(config)
    # No process mocks here: the isolated evaluator CLI rejects a non-root user
    # or the fixture's absent runtime fingerprint helper before any GPU access.
    result = native.run(config, one=True)
    assert result['cost']['evaluations'] == len(config['references']) + 1
    campaign = args.directory / 'campaign'
    assert status(campaign)['pending'] is None
    row = read_json(campaign / 'state.json')['observations'][0]
    assert row['status'] == 'failed'
    assert row['cost']['wall_seconds'] > 0
    assert row['cleanup_error'] is None
    assert len(launches) == len(config['references'])

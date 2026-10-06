import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from migration import static_dse as static
from migration import build_math_parameters as builder
from migration import dynamic_dse as dynamic
from migration import workflow
from math_dynamic.controller import Controller, load_config
from test_static_dse_math import parameters, point, trace


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def controller_fixture(topology='2a1e'):
    selected = point(topology)
    selected['id'] = topology + '-static'
    selected['attention_mhz'] = selected['expert_mhz'] = 1050
    config = dict(method='static_bound_mathematical_causal_dvfs', selection_split='calibration', model='fixture',
        parameters=parameters(), static_configuration=selected, points=[point(topology), selected],
        guard_id=topology + '-max', attention_gpus=[0, 1], expert_gpus=[2] if topology == '2a1e' else [2, 3],
        allocation=[0, 1, 2, 3], control_interval_ms=50, downshift_hold_ms=100,
        minimum_transition_interval_ms=100, event_stale_ms=1000, relative_service_budget=1.05,
        guard_by_rps=[dict(rps=r, prefill_age_guard_ms=100, progress_gap_guard_ms=100) for r in [1, 2, 4]],
        files_sha256={})
    return config


def test_direct_parameter_identification_excludes_warmup_and_keeps_topology():
    link = builder.communication([dict(time=.001 + n / 1000, bytes=n * 100, prefill=0, decode=n) for n in [1, 2, 8]])
    assert link['identification'] == 'two_payload_direct_alpha_beta'
    assert link['latency_s'] == pytest.approx(.001)
    assert link['effective_bytes_per_s'] == pytest.approx(100000)
    single = builder.communication([dict(time=.005, bytes=100, prefill=0, decode=1)])
    assert single['identification'] == 'bandwidth_equivalent_zero_alpha_assumption'


def test_stage_events_pairing_and_warmup_exclusion(tmp_path):
    events = []
    for tx, start in [('warmup', 1), ('calibration', 100)]:
        for name, duration in [('attention_layer_total', 50), ('remote_ffn_roundtrip', 30),
                               ('a2f_dispatch', 5), ('ffn_compute', 20), ('f2a_combine', 4)]:
            events.append(dict(event=name, duration_us=duration, start_wall_ns=start, end_wall_ns=start + 10,
                transaction_id=tx, layer_idx=0, stage_idx=0, prefill_tokens=0, decode_tokens=1, bytes=100))
    path = tmp_path / 'stage.jsonl'; path.write_text('\n'.join(json.dumps(r) for r in events))
    rows, coverage = builder.stage_groups([path], 90, 120)
    assert coverage['complete_groups'] == 1 and len(rows) == 4
    assert next(r['time'] for r in rows if r['stage'] == 'attention_compute') == pytest.approx(20e-6)


def test_weight_ledger_reads_headers_not_weight_payloads(tmp_path):
    import struct
    cfg = dict(model_type='deepseek_v2', num_hidden_layers=2, hidden_size=4, n_routed_experts=2,
               num_experts_per_tok=1, num_attention_heads=1, qk_nope_head_dim=2, qk_rope_head_dim=2,
               v_head_dim=2, kv_lora_rank=2)
    put(tmp_path / 'config.json', cfg)
    shapes = {'model.layers.1.self_attn.q_proj.weight': [4, 4],
              'model.layers.1.mlp.experts.0.gate_proj.weight': [4, 4],
              'model.layers.1.mlp.shared_experts.up_proj.weight': [4, 4],
              'model.visual.layers.0.weight': [1000, 1000]}
    header = json.dumps({k: dict(shape=v, dtype='BF16', data_offsets=[0, 0]) for k, v in shapes.items()}).encode()
    (tmp_path / 'weights.safetensors').write_bytes(struct.pack('<Q', len(header)) + header)
    put(tmp_path / 'model.safetensors.index.json', {'weight_map': {k: 'weights.safetensors' for k in shapes}})
    a, ledger = builder.weight_ledger(tmp_path)
    assert a['attention_flops_per_token'] == 16
    assert a['expert_flops_per_routed_token'] == 8
    assert a['kv_bytes_per_context_token'] == 8
    assert ledger['tensor_counts']['attention'] == 1


def test_causal_controller_guards_and_topology_do_not_read_future_trace():
    c = Controller(controller_fixture())
    submit = dict(event='submit', request_id='one', wall_ns=1_000_000_000, input_tokens=10, requested_output_tokens=20)
    c.consume(submit)
    chosen, reason = c.choose(1_000_000_001)
    assert chosen['id'] == '2a1e-max' and reason == 'startup_guard'
    chosen, reason = c.choose(2_100_000_000)
    assert chosen['id'] == '2a1e-max'
    c.consume(dict(event='first_token', request_id='one', wall_ns=2_100_000_001, output_chunks=1))
    chosen, reason = c.choose(2_210_000_001)
    assert reason == 'inherited_progress_guard'
    assert all(p['topology'] == '2a1e' for p in c.config['points'])


def test_controller_rejects_cross_topology_or_changed_dependency(tmp_path):
    config = controller_fixture()
    source = tmp_path / 'static-freeze.json'; source.write_text('frozen')
    config['files_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    path = tmp_path / 'controller.json'; put(path, config)
    load_config(path)
    source.write_text('edited')
    with pytest.raises(ValueError, match='dependency changed'):
        load_config(path)
    config['files_sha256'] = {}; config['points'].append(point('2a2e')); put(path, config)
    with pytest.raises(ValueError, match='cannot change'):
        load_config(path)


def static_fixture(tmp_path, monkeypatch):
    params = tmp_path / 'params.json'; put(params, parameters())
    prediction_path = tmp_path / 'PREDICTION.json'
    prediction = dict(points=[point('2a1e'), point('2a2e')], parameters=str(params), parameters_sha256=static.sha(params))
    put(prediction_path, prediction)
    original = dict(model='qwen36', model_tag='Qwen3.6-35B-A3B', rates=[1, 2, 4], allocation=[0, 1, 2, 3],
                    prediction={'path': str(prediction_path)}, ports={})
    freeze = dict(selected_configuration=point('2a2e'), selected_by_rate={'1': point('2a1e'), '2': point('2a2e'), '4': point('2a2e')},
                  allocation=[0, 1, 2, 3])
    path = tmp_path / 'STATIC_FREEZE.json'; put(path, freeze)
    monkeypatch.setattr(dynamic, 'verify_static', lambda _: (freeze, original, prediction))
    return path


def test_dynamic_plan_consumes_per_workload_static_result_and_keeps_inherited_guards(tmp_path, monkeypatch):
    path = static_fixture(tmp_path, monkeypatch)
    c1, _ = dynamic.controller_config(path, 1)
    c4, _ = dynamic.controller_config(path, 4)
    assert c1['expert_gpus'] == [2] and c4['expert_gpus'] == [2, 3]
    source = static.read(ROOT / 'inputs/calibration-max/qwen36/combined-controller-v10.json')
    assert c1['guard_by_rps'] == source['predictor']['calibration_guard_by_rps']
    p = dynamic.make_plan(path, tmp_path / 'dynamic-cal', rate=1)
    assert len(p['schedule']) == 3 and p['evaluation_split'] == 'calibration'
    assert {e['candidate_id'] for e in p['schedule']} == {'max', 'static', 'dynamic'}
    command, env = static.command_for(p, p['schedule'][-1])
    assert env['ECODEP_EXPERT_GPUS'] == '2' and env['ECODEP_MEASUREMENT_GPUS'] == '0,1,2,3'
    deployment = static.read(Path(command[2]))
    assert deployment['math_dynamic_controller']['sha256'] == static.sha(p['controller_path'])
    assert deployment['runtime_contract']['stage_trace'] is False


def test_failed_calibration_cannot_create_formal_execution(tmp_path, monkeypatch):
    freeze = static_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(dynamic, 'summarize', lambda _: {'all_calibration_gates_pass': False})
    with pytest.raises(ValueError, match='gate failed'):
        dynamic.make_plan(freeze, tmp_path / 'blocked-heldout', calibration_plan=tmp_path / 'cal/PLAN.json', calibration_report=tmp_path / 'report.json')
    assert not (tmp_path / 'blocked-heldout').exists()


def test_workflow_dry_run_is_no_gpu_no_file(tmp_path, capsys):
    path = tmp_path / 'workflow'
    workflow.run('qwen36', path, [0, 1, 2, 3], dry_run=True)
    assert not path.exists()
    assert 'run_heldout_all_targets' in capsys.readouterr().out


def test_real_controller_and_replay_processes_with_mock_http(tmp_path):
    # Exercise CLI flags, event schema, submit FIFO, lifecycle, and clock ACKs on CPU.
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/v1/completions':
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
                for i in range(payload['max_tokens']):
                    event = {'choices': [{'text': 'x'}]}
                    self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode()); self.wfile.flush()
                    time.sleep(.004)
                self.wfile.write(('data: ' + json.dumps({'choices': [], 'usage': {'completion_tokens': payload['max_tokens']}}) + '\n\ndata: [DONE]\n\n').encode())
                return
            calls.append((self.path, payload['gpu']))
            if self.path == '/set_clock':
                out = dict(gpu=payload['gpu'], requested_mhz=payload['sm_mhz'], applied_mhz=payload['sm_mhz'], clock_control='locked')
            else:
                out = dict(gpu=payload['gpu'], requested_w=payload['watts'], applied_w=payload['watts'], power_control='limited')
            raw = json.dumps(out).encode()
            self.send_response(200); self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    config = tmp_path / 'controller.json'; put(config, controller_fixture())
    requests = trace(n=3, input_tokens=5, output_tokens=4)
    for row in requests:
        row['request_id'] = row['source_index']
    request_path = tmp_path / 'requests.jsonl'; request_path.write_text('\n'.join(json.dumps(r) for r in requests))
    case = tmp_path / 'case'
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        done = subprocess.run([sys.executable, str(ROOT / 'scripts/afd/math_dynamic/run_replay.py'),
            '--config', str(config), '--case-dir', str(case), '--clock-url', url, '--',
            str(request_path), '--model', 'mock', '--endpoint', url + '/v1/completions',
            '--output', str(tmp_path / 'replay.jsonl'), '--time-scale', '.01', '--max-output-tokens', '4'],
            capture_output=True, text=True, timeout=30)
        assert done.returncode == 0, done.stderr
    finally:
        server.shutdown(); thread.join()
    summary = static.read(case / 'controller-summary.json')
    assert summary['replay_ended'] and summary['restored_to_guard'] and summary['outstanding'] == 0
    assert {gpu for _, gpu in calls} == {0, 1, 2}  # inactive GPU 3 is never actuated by DVFS
    events = [json.loads(s) for s in (case / 'events.jsonl').read_text().splitlines()]
    assert len([r for r in events if r['event'] == 'submit']) == 3
    assert not (case / 'submit.fifo').exists()


def write_mock_measurements(p, entry, output_root, fail_dynamic=False):
    """GPU boundary fixture; use real collectors, trace identities, gates and freezes."""
    suite, run = static.paths(p, entry)
    suite.mkdir(parents=True); run.mkdir(parents=True)
    cid = entry.get('candidate_id', entry['arm'])
    deploy_path = Path(p['campaign']) / 'deployments' / (cid + '.json')
    deploy = static.read(deploy_path)
    ne = deploy['topology']['ffn_ep']; gpus = p['allocation']
    split = p.get('evaluation_split', 'calibration')
    source = Path(p.get('trace_path', ROOT / 'inputs/traces/calibration-200.jsonl'))
    comparison = dict(measurement={'gpu_ids': gpus},
        topology=dict(attention_dp=2, ffn_ep=ne, attention_tp=1, expert_tp=1), generation={'max_output_tokens': 128})
    digest = hashlib.sha256(json.dumps(comparison, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    put(suite / 'manifest.json', dict(evaluation_split=split, formal_evaluation_eligible=split == 'heldout',
        trace_sha256=static.sha(source), deployment_sha256=static.sha(deploy_path), model=p['model_tag'],
        repetition=entry['repetition'], rates_rps=entry['rates_rps'], plugin=p['template']['plugin'],
        schedule=dict(sha256=static.sha(Path(p['campaign']) / 'schedule.json'), position=entry['position']),
        comparison_contract=comparison, comparison_contract_sha256=digest))
    if split == 'heldout':
        assert deploy['operating_points']['comparison_contract_sha256'] == digest
    for name in ('allocation-reset-before.json', 'clock-reset.json'):
        put(suite / name, dict(verified=True, requested_gpus=gpus))
    put(run / 'launch_config.json', dict(attention_ranks=2, expert_ranks=ne,
        attention_gpus=','.join(map(str, gpus[:2])), expert_gpus=','.join(map(str, gpus[2:2+ne]))))
    for rate in entry['rates_rps']:
        case = run / f'rps-{rate}'; case.mkdir()
        requests = [json.loads(s) for s in source.read_text().splitlines()]
        ttft = 20 if fail_dynamic and cid == 'dynamic' else 10
        for i, row in enumerate(requests):
            row.update(error=None, actual_output_tokens=min(row['output_tokens'], 128),
                submit_wall_ns=int(i / rate * 1e9), finish_wall_ns=int((i / rate + .1) * 1e9), ttft_ms=ttft, tpot_ms=1)
        duration = (requests[-1]['finish_wall_ns'] - requests[0]['submit_wall_ns']) / 1e9
        (run / f'replay-ecodep-v026-rps-{rate}.jsonl').write_text('\n'.join(json.dumps(r) for r in requests))
        energy = len(requests) * (2 if ne == 1 else 3)
        put(case / 'summary.json', dict(requests=len(requests), completed_requests=len(requests), failed_requests=0,
            energy_j=energy, duration_s=duration, ttft_ms={'p90': ttft}, tpot_ms={'p90': 1},
            output_token_throughput_tps=sum(r['actual_output_tokens'] for r in requests) / duration))
        put(case / 'telemetry.json', dict(returncode=0, request_count=len(requests), gpu_ids=gpus,
            sample_time_coverage=1, sample_error_count=0, measurement_window_source='request_first_submit_to_last_finish',
            energy_j=energy, per_gpu_energy_j=[energy/4]*4, duration_s=duration))
        put(case / 'gpu-contamination-validation.json', {'verified': True})
        op = deploy['operating_points']['by_rps'][str(rate)]
        put(case / 'operating-point-ack.json', dict(verified=True, verification_errors=[],
            attention_gpus=gpus[:2], expert_gpus=gpus[2:2+ne],
            **{'requested_'+k: op[k] for k in ('attention_mhz', 'expert_mhz', 'attention_power_w', 'expert_power_w')}))
        put(case / 'case.json', dict(evaluation_split=split, offered_rps=rate, repetition=entry['repetition'], operating_point=op))
        if cid == 'dynamic':
            put(case / 'controller-summary.json', dict(status='complete', replay_ended=True, outstanding=0, restored_to_guard=True))
    (suite / 'COMPLETE').touch()


@pytest.mark.parametrize('fail_calibration', [False, True])
def test_full_workflow_real_artifact_gates_with_mock_gpu_boundary(tmp_path, monkeypatch, fail_calibration):
    base = tmp_path / 'workflow'
    launched = []
    monkeypatch.setattr(static, 'paths', lambda p, e: (tmp_path / 'suites' / e['suite_id'], tmp_path / 'runs' / e['suite_id']))
    def executor(path):
        p = static.verify_plan(path)
        if p.get('evaluation_split') == 'heldout':
            assert (base / 'HELDOUT_STARTED.json').exists()
            marker = static.read(base / 'HELDOUT_STARTED.json')
            assert len(marker['formal_plans_sha256']) == 3
            static.check(all((base / f'rps-{r}/CALIBRATION_REPORT.json').exists() for r in (1, 2, 4)), 'Not all targets calibrated')
            dynamic.verify_heldout_gate(p)
        launched.append(p['phase'])
        for e in p['schedule']:
            if not (static.paths(p, e)[0] / 'COMPLETE').exists():
                write_mock_measurements(p, e, tmp_path, fail_dynamic=fail_calibration)
        static.collect(p)
    def build_parameters(probe, out):
        # Stage algebra/header parsing has separate tests; supply synthetic physics at this boundary.
        p = static.verify_plan(probe)
        source = static.paths(p, p['schedule'][0])[0] / 'manifest.json'
        params = parameters()
        params.update(model=p['model_tag'], measurement_sources=[dict(path=str(source), sha256=static.sha(source), selection_split='calibration')])
        static.write(out, params)
    def search(probe, params, out, rates):
        p = static.verify_plan(probe)
        records = static.collect(p)
        static.write(out, dict(parameters=str(params), parameters_sha256=static.sha(params),
            probe_plan=str(probe), probe_plan_sha256=static.sha(probe),
            probe_evidence_sha256={f: h for r in records for f, h in r['evidence_sha256'].items()},
            points=[point('2a1e'), point('2a2e')], predicted_fixed_configuration='2a1e-max',
            predicted_best_by_rate={str(r): '2a1e-max' for r in rates}))
    monkeypatch.setattr(static, 'run', executor)
    monkeypatch.setattr(builder, 'build', build_parameters)
    monkeypatch.setattr(static, 'mathematical_search', search)
    if fail_calibration:
        with pytest.raises(ValueError, match='calibration failed'):
            workflow.run('qwen36', base, [0, 1, 2, 3], target='per-rate')
        assert 'formal_heldout' not in launched
        assert not (base / 'HELDOUT_STARTED.json').exists()
        return
    workflow.run('qwen36', base, [0, 1, 2, 3], target='per-rate')
    result = static.read(base / 'RESULTS.json')
    assert result['status'] == 'complete' and result['formal_request_count'] == 7200
    assert all(r['topology'] == '2a1e' and len(r['cells']) == 6 for r in result['heldout_results'])
    launched.clear()
    workflow.run('qwen36', base, [0, 1, 2, 3], target='per-rate')
    assert launched == ['formal_heldout'] * 3  # upstream calibration is verification-only on resume
    # Removing calibration evidence after opening heldout cannot trigger recollection.
    p = static.read(base / 'probes/PLAN.json')
    (static.paths(p, p['schedule'][0])[0] / 'COMPLETE').unlink()
    launched.clear()
    with pytest.raises(ValueError, match='Missing complete'):
        workflow.run('qwen36', base, [0, 1, 2, 3], target='per-rate')
    assert not launched


def test_suite_ids_are_unique_across_campaigns_and_rate_targets(tmp_path, monkeypatch):
    freeze = static_fixture(tmp_path, monkeypatch)
    plans = [dynamic.make_plan(freeze, tmp_path / f'rps-{r}/cal', rate=r) for r in (1, 2, 4)]
    ids = [e['suite_id'] for p in plans for e in p['schedule']]
    assert len(ids) == len(set(ids))
    a = static.make_plan('qwen36', tmp_path / 'a/probes', [0, 1, 2, 3])
    b = static.make_plan('qwen36', tmp_path / 'b/probes', [0, 1, 2, 3])
    assert not {e['suite_id'] for e in a['schedule']} & {e['suite_id'] for e in b['schedule']}


def test_prediction_uncertainty_cannot_drop_topology_anchors_and_historical_candidates():
    ids = static.validation_ids(dict(predicted_fixed_configuration='2a2e-max', predicted_best_by_rate={'1': '2a2e-max'}))
    assert {'2a1e-max', '2a2e-max', '2a1e-a810-e1050-p200', '2a1e-a1050-e1290-p200'} <= ids

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/afd'))
from static_dse.analytical import layer_stages, search, simulate
from static_dse.selection import select

spec = importlib.util.spec_from_file_location('static_migration', ROOT / 'migration/static_dse.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def parameters():
    # Deliberately synthetic physics for mechanism tests; never shipped as A100 data.
    kernel_a = dict(effective_flops_per_s=1e12, effective_bytes_per_s=1e12, launch_s=1e-6)
    kernel_e = dict(effective_flops_per_s=1e10, effective_bytes_per_s=1e12, launch_s=1e-6)
    link = dict(messages_per_layer=0, latency_s=0, bytes_per_token=0, effective_bytes_per_s=1e12)
    topologies = {}
    for name in ('2a1e', '2a2e'):
        topologies[name] = dict(routing_load_factor=1, a2f=copy.deepcopy(link), f2a=copy.deepcopy(link), expert_collective=copy.deepcopy(link))
    topologies['2a2e']['expert_collective'].update(messages_per_layer=1, latency_s=.004)
    return dict(model_kind='roofline_alpha_beta_reentrant_pipeline', selection_split='calibration',
                architecture=dict(layers=1, hidden_size=64, experts=2, top_k=1,
                                  attention_flops_per_token=1e6, attention_context_flops_per_pair=0,
                                  attention_weight_bytes_per_layer=1000, attention_activation_bytes_per_token=100,
                                  kv_bytes_per_context_token=0, expert_flops_per_routed_token=6e7,
                                  expert_weight_bytes_per_layer=1000, expert_activation_bytes_per_routed_token=100,
                                  shared_expert_flops_per_token=0, shared_expert_weight_bytes_per_layer=0),
                hardware=dict(attention=dict(reference_mhz=1410, idle_w=30, dynamic_w_at_reference=100,
                                             prefill=kernel_a, decode=kernel_a,
                                             voltage_ratio_by_mhz={'1410': 1, '1050': 1}),
                              expert=dict(reference_mhz=1410, idle_w=30, dynamic_w_at_reference=100,
                                          prefill=kernel_e, decode=kernel_e,
                                          voltage_ratio_by_mhz={'1410': 1, '1050': 1}), inactive_gpu_w=15),
                topologies=topologies,
                scheduler=dict(max_num_seqs=32, max_num_batched_tokens=3072, microbatches=1, max_output_tokens=128))


def point(topology='2a1e'):
    return dict(id=f'{topology}-max', topology=topology, attention_mhz=1410, expert_mhz=1410,
                attention_power_w=400, expert_power_w=400)


def trace(n=12, input_tokens=1, output_tokens=40):
    return [dict(source_index=i, source_timestamp=str(i), evaluation_split='calibration', arrival_s=float(i),
                 input_tokens=input_tokens, output_tokens=output_tokens) for i in range(n)]


def test_physical_communication_compute_crossover():
    p = parameters()
    # With one token, the extra collective exceeds the compute saved by EP2.
    assert layer_stages(p, point('2a1e'), [(0, 1, 1)])[2] < layer_stages(p, point('2a2e'), [(0, 1, 1)])[2]
    # With 16 tokens, compute parallelism pays for that same collective overhead.
    assert layer_stages(p, point('2a1e'), [(0, 16, 16)])[2] > layer_stages(p, point('2a2e'), [(0, 16, 16)])[2]


def test_workload_drives_batching_queue_and_topology_choice():
    p, requests = parameters(), trace(output_tokens=100)
    low = search(p, [point('2a2e'), point('2a1e')], requests, [1, 2, 4])
    assert low['predicted_best_by_rate']['1'] == '2a1e-max'
    assert low['predicted_best_by_rate']['4'] == '2a2e-max'
    assert low['predicted_fixed_configuration'] == '2a2e-max'
    rows = [r['prediction'] for r in low['rows'] if r['candidate_id'] == '2a1e-max']
    assert rows[1]['mean_batch_tokens'] > rows[0]['mean_batch_tokens']
    assert low['deployment_authorized'] is False


def test_request_lengths_matter_at_same_rps():
    p = parameters()
    short = simulate(p, point(), trace(input_tokens=1), 1)
    long = simulate(p, point(), trace(input_tokens=100), 1)
    assert long['p90_ttft_ms'] > short['p90_ttft_ms']
    assert long['energy_j_per_request'] > short['energy_j_per_request']


def test_bandwidth_floor_survives_frequency_changes():
    p = parameters()
    p['architecture']['expert_weight_bytes_per_layer'] = 1e12
    a, b = point(), point()
    b['expert_mhz'] = 1050
    assert layer_stages(p, a, [(0, 1, 1)])[2] == layer_stages(p, b, [(0, 1, 1)])[2]


def test_idle_fourth_gpu_and_power_caps_are_not_free_savings():
    p = parameters()
    before = simulate(p, point(), trace(), 1)
    p['hardware']['inactive_gpu_w'] += 10
    after = simulate(p, point(), trace(), 1)
    assert after['energy_j_per_request'] - before['energy_j_per_request'] == pytest.approx(10 * before['makespan_s'] / 12)
    capped = point(); capped['expert_power_w'] = 50
    assert simulate(p, capped, trace(), 1)['status'] == 'needs_parameter_measurement'


def test_unknown_voltage_and_heldout_are_rejected():
    p, c = parameters(), point()
    c['expert_mhz'] = 810
    assert simulate(p, c, trace(), 1)['status'] == 'needs_parameter_measurement'
    heldout = trace(); heldout[0]['evaluation_split'] = 'heldout'
    with pytest.raises(ValueError, match='calibration'):
        simulate(p, point(), heldout, 1)


def selection_fixture():
    candidates = [point('2a2e'), point('2a1e')]
    plan = dict(candidates=candidates, rates=[1, 2, 4], repetitions=[1, 2], baseline_id='2a2e-max',
                request_weights={'1': 1/3, '2': 1/3, '4': 1/3},
                contract=dict(latency_ratio_max=1.05, throughput_ratio_min=.95))
    records = [dict(candidate_id=c['id'], rate=rate, repetition=rep, selection_split='calibration',
                    metrics=dict(p90_ttft_ms=100, p90_tpot_ms=10,
                                 output_tps=100 if c['topology'] == '2a2e' or rate == 1 else 90,
                                 energy_j_per_request=100 if c['topology'] == '2a2e' else 70))
               for c in candidates for rate in plan['rates'] for rep in plan['repetitions']]
    return plan, records


def test_measured_admission_checks_throughput_all_repetitions_and_target_workload():
    plan, records = selection_fixture()
    result = select(plan, records)
    assert result['by_rate'] == {'1': '2a1e-max', '2': '2a2e-max', '4': '2a2e-max'}
    assert result['fixed_configuration']['candidate_id'] == '2a2e-max'
    records[-1]['metrics']['p90_ttft_ms'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        select(plan, records)


def test_missing_cells_are_unknown_not_failed():
    plan, records = selection_fixture()
    with pytest.raises(ValueError, match='incomplete'):
        select(plan, records[:-1])
    records[0]['selection_split'] = 'heldout'
    with pytest.raises(ValueError, match='calibration'):
        select(plan, records)


def test_sparse_plan_has_only_two_replays_and_true_gpu_mapping(tmp_path):
    campaign = tmp_path / 'probe'
    plan = migration.make_plan('qwen36', campaign, [4, 5, 6, 7])
    assert len(plan['schedule']) == 2
    assert plan['rates'] == [2] and plan['repetitions'] == [1]
    assert plan['workload']['arrival_interval_cv'] > 0
    for entry in plan['schedule']:
        cmd, env = migration.command_for(plan, entry)
        assert env['ECODEP_MEASUREMENT_GPUS'] == '4,5,6,7'
        deployment = json.loads(Path(cmd[2]).read_text())
        assert deployment['math_dse_parameter_probe'] is True
        if '2a1e' in entry['arm']:
            assert env['ECODEP_EXPERT_GPUS'] == '6'
            assert deployment['topology']['ffn_ep'] == 1
    migration.verify_plan(campaign / 'PLAN.json')
    d = campaign / 'deployments/2a1e-max.json'
    d.write_text(d.read_text() + ' ')
    with pytest.raises(ValueError, match='changed'):
        migration.verify_plan(campaign / 'PLAN.json')


def test_dry_run_never_starts_gpu_processes(tmp_path, monkeypatch, capsys):
    campaign = tmp_path / 'dry-probe'
    migration.make_plan('deepseek-v2-lite', campaign, [0, 1, 2, 3])
    monkeypatch.setattr(migration.subprocess, 'run', lambda *a, **k: pytest.fail('dry run launched subprocess'))
    migration.run(campaign / 'PLAN.json', dry_run=True)
    assert '2a1e-max' in capsys.readouterr().out
    assert not (campaign / 'STARTED.json').exists()


def test_collect_audits_real_request_tail_throughput_and_inactive_gpu(tmp_path, monkeypatch):
    campaign = tmp_path / 'audit-probe'
    plan = migration.make_plan('qwen36', campaign, [4, 5, 6, 7])
    entry = next(e for e in plan['schedule'] if e['arm'] == '2a1e-max')
    suite, run = tmp_path / 'suite', tmp_path / 'run'
    suite.mkdir(); run.mkdir()
    case = run / 'rps-2'; case.mkdir()
    monkeypatch.setattr(migration, 'paths', lambda *_: (suite, run))
    def put(path, data):
        path.write_text(json.dumps(data))
    deploy = campaign / 'deployments/2a1e-max.json'
    operating = json.loads(deploy.read_text())['operating_points']['by_rps']['2']
    comparison = dict(measurement={'gpu_ids': [4, 5, 6, 7]},
                      topology={'attention_dp': 2, 'ffn_ep': 1, 'attention_tp': 1, 'expert_tp': 1},
                      generation={'max_output_tokens': 128})
    put(suite / 'manifest.json', dict(evaluation_split='calibration', formal_evaluation_eligible=False,
        trace_sha256=migration.sha(ROOT / 'inputs/traces/calibration-200.jsonl'), deployment_sha256=migration.sha(deploy),
        model=plan['model_tag'], repetition=1, rates_rps=[2], plugin=plan['template']['plugin'],
        schedule=dict(sha256=migration.sha(campaign / 'schedule.json'), position=entry['position']), comparison_contract=comparison))
    (suite / 'COMPLETE').touch()
    for name in ('allocation-reset-before.json', 'clock-reset.json'):
        put(suite / name, dict(verified=True, requested_gpus=[4, 5, 6, 7]))
    put(run / 'launch_config.json', dict(attention_ranks=2, expert_ranks=1, attention_gpus='4,5', expert_gpus='6'))
    requests = [json.loads(s) for s in (ROOT / 'inputs/traces/calibration-200.jsonl').read_text().splitlines()]
    for i, row in enumerate(requests):
        row.update(error=None, actual_output_tokens=min(row['output_tokens'], 128),
                   submit_wall_ns=int(i * 1e9), finish_wall_ns=int((i + .1) * 1e9), ttft_ms=10, tpot_ms=1)
    duration = (requests[-1]['finish_wall_ns'] - requests[0]['submit_wall_ns']) / 1e9
    (run / 'replay-ecodep-v026-rps-2.jsonl').write_text('\n'.join(json.dumps(r) for r in requests))
    summary = dict(requests=200, completed_requests=200, failed_requests=0, energy_j=400,
                   duration_s=duration, ttft_ms={'p90': 10}, tpot_ms={'p90': 1},
                   output_token_throughput_tps=sum(r['actual_output_tokens'] for r in requests) / duration)
    put(case / 'summary.json', summary)
    put(case / 'telemetry.json', dict(returncode=0, request_count=200, gpu_ids=[4, 5, 6, 7],
        sample_time_coverage=1, sample_error_count=0, measurement_window_source='request_first_submit_to_last_finish',
        energy_j=400, per_gpu_energy_j=[100, 100, 100, 100], duration_s=duration))
    put(case / 'gpu-contamination-validation.json', dict(verified=True))
    put(case / 'operating-point-ack.json', dict(verified=True, verification_errors=[], attention_gpus=[4, 5], expert_gpus=[6],
        **{'requested_' + k: operating[k] for k in ('attention_mhz', 'expert_mhz', 'attention_power_w', 'expert_power_w')}))
    put(case / 'case.json', dict(evaluation_split='calibration', offered_rps=2, repetition=1, operating_point=operating))
    observed = migration.collect_entry(plan, entry)
    assert observed[0]['metrics']['energy_j_per_request'] == 2
    summary['output_token_throughput_tps'] *= 1.5
    put(case / 'summary.json', summary)
    with pytest.raises(ValueError, match='throughput differs'):
        migration.collect_entry(plan, entry)
    summary['output_token_throughput_tps'] /= 1.5
    put(case / 'summary.json', summary)
    telemetry = json.loads((case / 'telemetry.json').read_text())
    telemetry['gpu_ids'] = [4, 5, 6]
    put(case / 'telemetry.json', telemetry)
    with pytest.raises(ValueError, match='telemetry'):
        migration.collect_entry(plan, entry)

#!/usr/bin/env python3
"""Plan, run and freeze a measured static DSE on the destination native runtime."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'migration'))
sys.path.insert(0, str(ROOT / 'scripts/afd'))
from static_dse.selection import metrics, select
from static_dse.analytical import search, validate_parameters
from summarize_replay import percentile

MODELS = {'qwen36': 'Qwen3.6-35B-A3B', 'deepseek-v2-lite': 'DeepSeek-V2-Lite-Chat'}
BASELINE = '2a2e-max'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        out.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def candidates():
    return [dict(id=f'{topology}-max', topology=topology, attention_mhz=1410,
                 expert_mhz=1410, attention_power_w=400, expert_power_w=400)
            for topology in ('2a2e', '2a1e')]


def validation_ids(prediction, scope='both'):
    # Mandatory topology anchors and historical hypotheses must not be rejected by prediction alone.
    ids = {BASELINE, '2a1e-max', '2a1e-a810-e1050-p200', '2a1e-a1050-e1290-p200'}
    if scope in ('per-rate', 'both'):
        ids.update(prediction['predicted_best_by_rate'].values())
    if scope in ('fixed', 'both'):
        ids.add(prediction['predicted_fixed_configuration'])
    return ids


def code_files():
    return sorted(p for directory in ('migration', 'scripts', 'services', 'environment')
                  for p in (ROOT / directory).rglob('*')
                  if p.is_file() and (p.suffix in ('.py', '.sh') or p.name.endswith('.lock.json'))
                  and '__pycache__' not in p.parts and 'results' not in p.relative_to(ROOT).parts)


def make_plan(model, campaign, allocation, rates=(2,), weights=None, chosen=None, prediction=None, ports=None):
    campaign = Path(campaign).resolve()
    check(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', campaign.name), 'Invalid campaign name')
    check(len(allocation) == 4 and len(set(allocation)) == 4 and min(allocation) >= 0,
          'Reserve four distinct GPUs, including the idle fourth GPU for 2A1E')
    rates = sorted(set(rates))
    check(rates and set(rates) <= {1, 2, 4}, 'Supported offered rates: 1, 2, 4')
    weights = weights or {str(rate): 1 / len(rates) for rate in rates}
    check(set(weights) == set(map(str, rates)) and all(math.isfinite(v) and v > 0 for v in weights.values())
          and math.isclose(sum(weights.values()), 1), 'Request weights must be positive and sum to one')
    check(not campaign.exists(), 'Campaign exists; use run to resume or a new campaign name')
    trace = ROOT / 'inputs/traces/calibration-200.jsonl'
    trace_rows = [json.loads(s) for s in trace.read_text().splitlines() if s.strip()]
    check(all(r['evaluation_split'] == 'calibration' for r in trace_rows), 'Wrong calibration split')
    audit = json.loads(subprocess.check_output([
        sys.executable, str(ROOT / 'scripts/audit_trace_isolation.py'),
        '--calibration', str(trace), '--evaluation', str(ROOT / 'inputs/traces/heldout-400.jsonl'),
        '--identity-field', 'source_index', '--identity-field', 'source_timestamp'], text=True))
    check(audit['status'] == 'PASS', 'Trace identity isolation failed')
    # Warmup is calibration-derived, as permitted by the trace isolation contract.
    warmup = read(ROOT / 'inputs/traces/warmup-manifest.json')
    check(warmup['selection_split'] == 'calibration' and warmup['source_sha256'] == sha(trace)
          and warmup['output_sha256'] == sha(ROOT / 'inputs/traces/warmup-8.jsonl'), 'Invalid warmup provenance')
    template = read(ROOT / 'inputs/protocols' / model / 'calibration-max-deployment.json')
    template['plugin']['root'] = str(ROOT / 'third_party' / Path(template['plugin']['root']).name)
    chosen = chosen or candidates()
    phase = "candidate_validation" if prediction else "parameter_probe"
    repetitions = (1, 2) if prediction else (1,)
    check(len({c['id'] for c in chosen}) == len(chosen), 'Duplicate candidate id')
    for c in chosen:
        check(c['topology'] in ('2a1e', '2a2e'), 'Unsupported topology')
        check(re.fullmatch(r'[a-zA-Z0-9_.-]+', c['id']), 'Invalid candidate id')
        check(all(isinstance(c[k], int) and lo <= c[k] <= hi for k, lo, hi in (
            ('attention_mhz', 810, 1410), ('expert_mhz', 1050, 1410),
            ('attention_power_w', 200, 400), ('expert_power_w', 200, 400))), 'Unsupported operating point')
    # Reverse candidate and rate order in repetition two, frozen before any replay.
    schedule = []
    tag = f'sdse-{campaign.name}-{hashlib.sha256(str(campaign).encode()).hexdigest()[:10]}-{model}'
    for rep in repetitions:
        for c in (chosen if rep == 1 else list(reversed(chosen))):
            schedule.append(dict(position=len(schedule) + 1, arm=c['id'], repetition=rep,
                                 rates_rps=rates if rep == 1 else list(reversed(rates)),
                                 suite_id=f'{tag}-{c["id"]}-r{rep}'))
    arrivals = sorted(float(r['arrival_s']) for r in trace_rows)
    intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    base_rps = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
    workload = {'trace_sha256': sha(trace), 'request_count': len(trace_rows),
                'input_tokens': {q: percentile([r['input_tokens'] for r in trace_rows], v)
                                 for q, v in [('p50', .5), ('p90', .9), ('p99', .99)]},
                'output_tokens': {q: percentile([min(r['output_tokens'], 128) for r in trace_rows], v)
                                  for q, v in [('p50', .5), ('p90', .9), ('p99', .99)]},
                'arrival_interval_cv': statistics.pstdev(intervals) / statistics.mean(intervals),
                'by_rate': {str(rate): {'offered_rps': rate, 'arrival_time_scale': base_rps / rate}
                            for rate in rates},
                'generalization_scope': 'this_trace_shape_distribution_at_measured_rates_only'}
    plan = dict(schema_version=1, workload=workload, selection_split='calibration', model=model, model_tag=MODELS[model],
                campaign=str(campaign), allocation=allocation, rates=rates, repetitions=list(repetitions), phase=phase,
                candidates=chosen, baseline_id=BASELINE, request_weights=weights,
                request_count=len(trace_rows), template=template, schedule=schedule,
                contract={'latency_ratio_max': 1.05, 'throughput_ratio_min': .95,
                          'latency_metrics': ['p90_ttft_ms', 'p90_tpot_ms'],
                          'all_repetitions_must_pass': True, 'measurement_gpus': 'entire_four_gpu_allocation',
                          'target': 'matched_independent_rate_blocks', 'max_output_tokens': 128},
                execution_layout=os.environ.get('ECODEP_EXECUTION_LAYOUT', 'sequential_four_gpu_allocation'),
                ports=ports or {k: os.environ.get(k, d) for k, d in (
                    ('ECODEP_API_PORT', '18000'), ('ECODEP_EXPERT_API_PORT', '18001'),
                    ('ECODEP_AFD_PORT', '16239'), ('ECODEP_DP_RPC_BASE_PORT', '29550'),
                    ('ECODEP_CLOCK_URL', 'http://127.0.0.1:9096'))},
                prediction=prediction, heldout_results_used=False,
                files_sha256={str(p.relative_to(ROOT)): sha(p) for p in code_files()})
    if prediction:
        plan['files_sha256'][prediction['path']] = prediction['sha256']
    for p in (ROOT / 'inputs/traces').iterdir():
        plan['files_sha256'][str(p.relative_to(ROOT))] = sha(p)
    plan['files_sha256'][str((ROOT / 'inputs/protocols' / model / 'calibration-max-deployment.json').relative_to(ROOT))] = sha(ROOT / 'inputs/protocols' / model / 'calibration-max-deployment.json')
    campaign.mkdir(parents=True)
    warmup.update(source=str(trace), output=str(ROOT / 'inputs/traces/warmup-8.jsonl'))
    write(campaign / 'warmup-manifest.json', warmup)
    write(campaign / 'trace-isolation-audit.json', audit)
    write(campaign / 'schedule.json', {'selection_split': 'calibration', 'entries': schedule})
    for c in chosen:
        deploy = copy.deepcopy(template)
        deploy.update(method='measured_topology_static_dse', arm='ours', candidate_id=c['id'],
                      calibration_trace=str(trace), calibration_trace_sha256=sha(trace),
                      formal_evaluation_eligible=False)
        ne = 1 if c['topology'] == '2a1e' else 2
        deploy['topology']['ffn_ep'] = ne
        deploy['math_dse_parameter_probe'] = phase == 'parameter_probe'
        deploy['runtime_contract']['stage_trace'] = phase == 'parameter_probe'
        deploy['operating_points'] = {
            'selection_split': 'calibration', 'candidate_id': c['id'],
            'calibration_trace_sha256': sha(trace),
            'by_rps': {str(rate): dict(candidate_id=c['id'], attention_mhz=[c['attention_mhz']] * 2,
                                     expert_mhz=[c['expert_mhz']] * ne,
                                     attention_power_w=[c['attention_power_w']] * 2,
                                     expert_power_w=[c['expert_power_w']] * ne,
                                     request_path_clock_transitions=0) for rate in rates}}
        write(campaign / 'deployments' / (c['id'] + '.json'), deploy)
    plan['campaign_files_sha256'] = {str(p.relative_to(campaign)): sha(p)
                                    for p in campaign.rglob('*') if p.is_file()}
    write(campaign / 'PLAN.json', plan)
    return plan


def verify_plan(path):
    path = Path(path).resolve()
    p = read(path)
    check(p['selection_split'] == 'calibration' and p['heldout_results_used'] is False, 'Not a calibration plan')
    check(path.parent == Path(p['campaign']), 'Campaign moved; create a new plan at its destination')
    for name, expected in p['files_sha256'].items():
        check(sha(ROOT / name) == expected, f'Frozen code/input changed: {name}')
    for name, expected in p['campaign_files_sha256'].items():
        check(sha(path.parent / name) == expected, f'Frozen campaign artifact changed: {name}')
    marker = path.parent / 'STARTED.json'
    if marker.exists():
        check(read(marker)['plan_sha256'] == sha(path), 'Plan changed after first launch')
    return p


def command_for(p, entry):
    c = next(c for c in p['candidates'] if c['id'] == entry.get('candidate_id', entry['arm']))
    a, e = p['allocation'][:2], p['allocation'][2:3 if c['topology'] == '2a1e' else 4]
    env = {k: v for k, v in os.environ.items() if not k.startswith('ECODEP_')}
    env.update(p['ports'])
    env.update(ECODEP_CONTAINER_RUNTIME='native', ECODEP_ATTENTION_GPUS=','.join(map(str, a)),
               ECODEP_EXPERT_GPUS=','.join(map(str, e)), ECODEP_MEASUREMENT_GPUS=','.join(map(str, p['allocation'])),
               ECODEP_ATTENTION_CLOCKS_MHZ='1410,1410', ECODEP_EXPERT_CLOCKS_MHZ=','.join(['1410'] * len(e)),
               ECODEP_ATTENTION_POWER_W='400,400', ECODEP_EXPERT_POWER_W=','.join(['400'] * len(e)),
               ECODEP_SCHEDULE_PATH=str(Path(p['campaign']) / 'schedule.json'),
               ECODEP_SCHEDULE_POSITION=str(entry['position']), ECODEP_REPETITION_ID=str(entry['repetition']),
               ECODEP_WARMUP_TRACE=str(ROOT / 'inputs/traces/warmup-8.jsonl'),
               ECODEP_WARMUP_MANIFEST=str(Path(p['campaign']) / 'warmup-manifest.json'),
               ECODEP_MAX_OUTPUT_TOKENS=str(p['contract']['max_output_tokens']),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for k in ('ECODEP_GPU_WAIT_TIMEOUT_S', 'ECODEP_GPU_STABLE_FOR_S', 'ECODEP_MINIMUM_FREE_GIB'):
        if k in os.environ:
            env[k] = os.environ[k]
    cmd = ['bash', str(ROOT / 'scripts/afd/run_v026_ours_rate_suite.sh'),
           str(Path(p['campaign']) / 'deployments' / (c['id'] + '.json')),
           str(p.get('trace_path', ROOT / 'inputs/traces/calibration-200.jsonl')), entry['suite_id'],
           ','.join(map(str, entry['rates_rps']))]
    return cmd, env


def paths(p, entry):
    return (ROOT / 'results/afd_suites' / p['model_tag'] / entry['suite_id'],
            ROOT / 'results/afd_serving' / p['model_tag'] / entry['suite_id'])


def collect_entry(p, entry):
    suite, run = paths(p, entry)
    measurement_gpus = p.get('measurement_gpus', p['allocation'])
    check((suite / 'COMPLETE').is_file(), f'Missing complete calibration suite: {suite}')
    manifest = read(suite / 'manifest.json')
    deployment = Path(p['campaign']) / 'deployments' / (entry.get('candidate_id', entry['arm']) + '.json')
    c = next(c for c in p['candidates'] if c['id'] == entry.get('candidate_id', entry['arm']))
    # Shared collector also validates BO plans with six/eight-card mappings.
    # Keep the legacy planner/search space unchanged.
    from bo_topology import parse_topology, validate_groups
    na, ne = parse_topology(c['topology'])
    attention, expert = p['allocation'][:na], p['allocation'][na:na + ne]
    validate_groups(attention, expert, p['allocation'])
    check(len(attention) == na and len(expert) == ne, 'Topology exceeds reserved allocation')
    split = p.get('evaluation_split', 'calibration')
    trace_path = Path(p.get('trace_path', ROOT / 'inputs/traces/calibration-200.jsonl'))
    check(manifest['evaluation_split'] == split and manifest['formal_evaluation_eligible'] == (split == 'heldout'),
          'Heldout or incorrectly labelled measurements cannot enter static selection')
    check(manifest['trace_sha256'] == sha(trace_path)
          and manifest['deployment_sha256'] == sha(deployment)
          and manifest['model'] == p['model_tag'] and manifest['repetition'] == entry['repetition']
          and manifest['rates_rps'] == entry['rates_rps'], 'Calibration provenance mismatch')
    check(manifest['plugin'] == p['template']['plugin'], 'Plugin mismatch')
    check(manifest['schedule']['sha256'] == sha(Path(p['campaign']) / 'schedule.json')
          and manifest['schedule']['position'] == entry['position'], 'Schedule mismatch')
    comparison = manifest['comparison_contract']
    check(comparison['measurement']['gpu_ids'] == measurement_gpus, 'Idle GPU omitted from energy accounting')
    if 'configuration' in p:
        from bo_layout import deployment as layout_deployment
        check(comparison['topology'] == layout_deployment(p['configuration']), 'Incorrect joint topology')
    else:
        check(comparison['topology'] == {'attention_dp': na, 'ffn_ep': ne, 'attention_tp': 1, 'expert_tp': 1}, 'Incorrect topology')
    check(comparison['generation']['max_output_tokens'] == p['contract']['max_output_tokens'], 'Generation contract changed')
    launch = read(run / 'launch_config.json')
    if 'configuration' in p:
        from verify_joint_launch import verify
        verify(p['configuration'], run, p['replicas'], p['microbatch_thresholds'])
        check(all(comparison['server'].get(k) == v for k, v in p['microbatch_thresholds'].items()),
              'Manifest microbatch thresholds differ from frozen protocol')
        from bo_layout import verify_launch
        verify_launch(p['configuration'], launch)
    check(launch['attention_ranks'] == na and launch['expert_ranks'] == ne
          and launch['attention_gpus'] == ','.join(map(str, attention))
          and launch['expert_gpus'] == ','.join(map(str, expert)), 'Actual launch topology mismatch')
    for name in ('allocation-reset-before.json', 'clock-reset.json'):
        reset = read(suite / name)
        check(reset['verified'] and reset['requested_gpus'] == measurement_gpus, 'Full allocation reset missing')
    expected_rows = [json.loads(s) for s in trace_path.read_text().splitlines() if s.strip()]
    expected_by_id = {r['source_index']: r for r in expected_rows}
    output = []
    for rate in entry['rates_rps']:
        case = run / f'rps-{rate}'
        summary, telemetry = read(case / 'summary.json'), read(case / 'telemetry.json')
        check(summary['requests'] == p['request_count'], 'Request count mismatch')
        check(telemetry['returncode'] == 0 and telemetry['request_count'] == p['request_count']
              and telemetry['gpu_ids'] == measurement_gpus and telemetry['sample_time_coverage'] >= .99
              and telemetry['sample_error_count'] == 0
              and telemetry['measurement_window_source'] == 'request_first_submit_to_last_finish', 'Invalid telemetry')
        check(math.isclose(summary['energy_j'], telemetry['energy_j'], rel_tol=1e-9)
              and math.isclose(summary['duration_s'], telemetry['duration_s'], rel_tol=1e-9), 'Energy/window mismatch')
        check(read(case / 'gpu-contamination-validation.json')['verified'], 'GPU contamination')
        ack = read(case / 'operating-point-ack.json')
        point = read(deployment)['operating_points']['by_rps'][str(rate)]
        check(ack['verified'] and not ack['verification_errors']
              and ack['attention_gpus'] == attention
              and ack['expert_gpus'] == expert
              and all(ack['requested_' + key] == point[key] for key in (
                  'attention_mhz', 'expert_mhz', 'attention_power_w', 'expert_power_w')),
              'Operating point was not acknowledged')
        if read(deployment).get('math_dynamic_controller'):
            controller_summary = read(case / 'controller-summary.json')
            check(controller_summary['status'] == 'complete' and controller_summary['replay_ended']
                  and controller_summary['outstanding'] == 0 and controller_summary['restored_to_guard'], 'Dynamic controller failed')
        cell = read(case / 'case.json')
        check(cell['evaluation_split'] == split and cell['offered_rps'] == rate
              and cell['repetition'] == entry['repetition']
              and cell['operating_point'] == read(deployment)['operating_points']['by_rps'][str(rate)], 'Rate/operating point mismatch')
        replay = run / f'replay-ecodep-v026-rps-{rate}.jsonl'
        replay_rows = [json.loads(s) for s in replay.read_text().splitlines() if s.strip()]
        check(len(replay_rows) == p['request_count'] and len({r['source_index'] for r in replay_rows}) == p['request_count'], 'Duplicate/missing replay identities')
        for r in replay_rows:
            original = expected_by_id.get(r['source_index'])
            check(original is not None and all(r.get(k) == v for k, v in original.items()), 'Replayed request identity/content mismatch')
            check(r['error'] is None and r['actual_output_tokens'] == min(original['output_tokens'], p['contract']['max_output_tokens']), 'Failed or truncated generation')
        correctness = None
        if 'configuration' in p:
            from output_correctness import compare
            gate = p['output_correctness']
            check(gate['contract']['trace']['sha256'] == sha(trace_path)
                  and gate['contract']['max_output_tokens'] == p['contract']['max_output_tokens']
                  and comparison['model']['config_sha256'] == gate['contract']['model_config']['sha256']
                  and all(comparison['generation'].get(k) == v for k, v in gate['contract']['generation'].items()),
                  'Correctness gate uses a different trace or output limit')
            correctness = compare(gate['contract'], gate['reference'], replay_rows)
            report = case / 'correctness-report.json'
            if report.exists():
                check(read(report) == correctness, 'Correctness report changed')
            else:
                write(report, correctness)
            check(correctness['verified'], f'Output correctness failed: {report}')
        replay_duration = (max(r['finish_wall_ns'] for r in replay_rows)
                           - min(r['submit_wall_ns'] for r in replay_rows)) / 1e9
        check(math.isclose(replay_duration, summary['duration_s'], rel_tol=1e-9), 'Replay measurement window differs')
        for metric in ('ttft_ms', 'tpot_ms'):
            check(math.isclose(percentile([r[metric] for r in replay_rows], .9), summary[metric]['p90'], rel_tol=1e-9), 'Summary tail differs from request records')
        check(math.isclose(sum(r['actual_output_tokens'] for r in replay_rows) / replay_duration,
                           summary['output_token_throughput_tps'], rel_tol=1e-9), 'Summary throughput differs from request records')
        check(len(telemetry['per_gpu_energy_j']) == len(measurement_gpus)
              and math.isclose(sum(telemetry['per_gpu_energy_j']), telemetry['energy_j'], rel_tol=1e-9),
              'Total energy must include the declared measurement GPUs')
        # Topology is intentionally variable; every other experimental setting must match MAX.
        comparable = copy.deepcopy(comparison)
        comparable.pop('topology')
        evidence = [suite / 'manifest.json', suite / 'COMPLETE', suite / 'allocation-reset-before.json',
                    suite / 'clock-reset.json', run / 'launch_config.json', replay,
                    *[case / name for name in ('summary.json', 'telemetry.json', 'case.json',
                                               'gpu-contamination-validation.json', 'operating-point-ack.json')]]
        if 'configuration' in p:
            evidence.extend([run / 'joint-launch.json', *sorted(run.glob('worker-layout-*.json'))])
            evidence.extend([case / 'correctness-report.json', Path(p['output_correctness']['reference']['path'])])
        for name in ('controller-summary.json', 'controller-actions.jsonl', 'events.jsonl'):
            if (case / name).is_file():
                evidence.append(case / name)
        output.append({'selection_split': split, 'candidate_id': entry.get('candidate_id', entry['arm']), 'rate': rate,
                       'repetition': entry['repetition'], 'metrics': metrics(summary),
                       **({'output_correctness': correctness} if correctness is not None else {}),
                       'comparison': comparable, 'evidence_sha256': {str(f): sha(f) for f in evidence}})
    return output


def collect(p):
    rows = [r for entry in p['schedule'] for r in collect_entry(p, entry)]
    baseline = next(r['comparison'] for r in rows if r['candidate_id'] == p['baseline_id'])
    check(all(r['comparison'] == baseline for r in rows), 'Unmatched model/runtime/server/generation/measurement contracts')
    return rows


def run(path, dry_run=False):
    p = verify_plan(path)
    if p.get('evaluation_split') == 'heldout':
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from migration.dynamic_dse import verify_heldout_gate
        verify_heldout_gate(p)
    if dry_run:
        print(json.dumps([{'command': command_for(p, e)[0],
                           'environment': {k: v for k, v in command_for(p, e)[1].items() if k.startswith('ECODEP_')}}
                          for e in p['schedule']], indent=2))
        return
    check(os.geteuid() == 0, 'Execute on the destination root environment')
    (ROOT / 'results').mkdir(exist_ok=True)
    lock = (ROOT / 'results/native-campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, **p['ports'], ECODEP_MODELS=p['model'],
               ECODEP_ATTENTION_GPUS=','.join(map(str, p['allocation'][:2])),
               ECODEP_EXPERT_GPUS=','.join(map(str, p['allocation'][2:])))
    subprocess.run([sys.executable, str(ROOT / 'migration/preflight.py')], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / 'migration/runtime_fingerprint.py'), '--output',
                    str(ROOT / 'environment/native-runtime.json')], check=True)
    campaign = Path(p['campaign'])
    snapshot = {'plan_sha256': sha(path), 'runtime_sha256': sha(ROOT / 'environment/native-runtime.json'),
                'platform_sha256': sha(ROOT / 'environment/native-platform.json')}
    if (campaign / 'STARTED.json').exists():
        check(read(campaign / 'STARTED.json') == snapshot, 'Environment or GPU identity changed; use new campaign')
    else:
        write(campaign / 'STARTED.json', snapshot)
    for entry in p['schedule']:
        verify_plan(path)
        suite, serving = paths(p, entry)
        if (suite / 'COMPLETE').exists():
            collect_entry(p, entry)
            print('SKIP verified:', entry['suite_id'], flush=True)
            continue
        check(not suite.exists() and not serving.exists(), 'Partial run retained; investigate and create a new campaign, never overwrite it')
        cmd, env = command_for(p, entry)
        print('RUN:', entry['suite_id'], flush=True)
        subprocess.run(cmd, env=env, cwd=ROOT, check=True)
        collect_entry(p, entry)
    print('Campaign complete. Phase:', p['phase'], 'Evaluation split:', p.get('evaluation_split', 'calibration'))


def analyze(path, output):
    p = verify_plan(path)
    records = collect(p)
    check(p['phase'] == 'candidate_validation', 'Instrumented probes do not authorize deployment')
    result = select(p, records)
    result.update(plan_sha256=sha(path), plan=str(Path(path).resolve()), workload=p['workload'], records=records)
    write(output, result)
    return result


def mathematical_search(plan_path, parameters, output, rates, weights=None):
    p = verify_plan(plan_path)
    check(p['phase'] == 'parameter_probe', 'A frozen sparse probe plan is required')
    # Only two MAX topology probes; no fitted workload/performance curves.
    evidence = collect(p)
    params = validate_parameters(read(parameters))
    check(params['model'] == p['model_tag'], 'Parameter model mismatch')
    check(params['scheduler'] == dict(max_num_seqs=32, max_num_batched_tokens=3072, microbatches=2, max_output_tokens=128), 'Scheduler must match the frozen probe runtime')
    check(params['calibration_trace_sha256'] == p['workload']['trace_sha256'], 'Parameter trace mismatch')
    sources = params.get('measurement_sources', [])
    check(sources, 'Supply source hashes for measured hardware parameters')
    probe_files = {name: value for row in evidence for name, value in row['evidence_sha256'].items()}
    for entry in p['schedule']:
        _, serving = paths(p, entry)
        sidecars = list(serving.glob('stage-*.jsonl'))
        check(sidecars, 'Missing four-stage parameter probe traces')
        probe_files.update({str(f.resolve()): sha(f) for f in sidecars})
    for source in sources:
        check(source['selection_split'] == 'calibration' and sha(source['path']) == source['sha256'], 'Invalid parameter evidence')
    check(any(source['path'] in probe_files for source in sources), 'Parameters must cite this campaign probe evidence')
    points = []
    for topology in ('2a1e', '2a2e'):
        for af in params['hardware']['attention']['voltage_ratio_by_mhz']:
            for ef in params['hardware']['expert']['voltage_ratio_by_mhz']:
                a, e = int(af), int(ef)
                cid = f'{topology}-max' if a == e == 1410 else f'{topology}-a{a}-e{e}'
                points.append(dict(id=cid, topology=topology, attention_mhz=a, expert_mhz=e,
                                   attention_power_w=400, expert_power_w=400))
    # Historical low-cap points are hypotheses. Unknown throttling is never free saving.
    points.extend([dict(id='2a1e-a810-e1050-p200', topology='2a1e', attention_mhz=810,
                        expert_mhz=1050, attention_power_w=200, expert_power_w=200),
                   dict(id='2a1e-a1050-e1290-p200', topology='2a1e', attention_mhz=1050,
                        expert_mhz=1290, attention_power_w=200, expert_power_w=200)])
    trace = [json.loads(s) for s in (ROOT / 'inputs/traces/calibration-200.jsonl').read_text().splitlines() if s.strip()]
    result = search(params, points, trace, rates, weights)
    result.update(model=p['model_tag'], probe_plan=str(Path(plan_path).resolve()), probe_plan_sha256=sha(plan_path),
                  points=points, rates=rates, request_weights=weights or [1 / len(rates)] * len(rates),
                  parameters=str(parameters.resolve()), parameters_sha256=sha(parameters),
                  probe_evidence_sha256=probe_files, workload=p['workload'])
    write(output, result)


def freeze(path, output):
    p = verify_plan(path)
    rows = collect(p)
    check(p['phase'] == 'candidate_validation', 'Parameter probes cannot freeze a deployment')
    decision = select(p, rows)
    by_id = {c['id']: c for c in p['candidates']}
    chosen = copy.deepcopy(by_id[decision['fixed_configuration']['candidate_id']])
    chosen.update(attention_gpus=p['allocation'][:2], expert_gpus=p['allocation'][2:3 if chosen['topology'] == '2a1e' else 4])
    write(output, {'schema_version': 1, 'selection_split': 'calibration', 'formal_evaluation_eligible': False,
                   'plan': str(Path(path).resolve()), 'plan_sha256': sha(path), 'model': p['model_tag'],
                   'decision': decision, 'workload': p['workload'], 'selected_configuration': chosen,
                   'selected_by_rate': {rate: by_id[cid] for rate, cid in decision['by_rate'].items()},
                   'allocation': p['allocation'], 'request_weights': p['request_weights'],
                   'evidence_sha256': {f: h for r in rows for f, h in r['evidence_sha256'].items()},
                   'dynamic_handoff': {'topology_source': 'selected_configuration',
                                       'required_validation': 'dynamic_calibration_on_selected_topology_and_target_schedule',
                                       'legacy_v9_v10_controller_reusable_without_validation': False,
                                       'mixed_rate_transition_behavior_validated': False}})


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('probe-plan', help='Two topology MAX traces at one rate; no DSE matrix')
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--rate', type=int, choices=(1, 2, 4), default=2)
    p = sub.add_parser('run'); p.add_argument('plan', type=Path); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('search', help='Evaluate mathematical recurrences on CPU')
    p.add_argument('plan', type=Path); p.add_argument('--parameters', type=Path, required=True)
    p.add_argument('--rates', default='1,2,4'); p.add_argument('--request-weights')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('validation-plan', help='Validate at most the selected per-rate and fixed candidates plus MAX')
    p.add_argument('--prediction', type=Path, required=True); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--scope', choices=('per-rate', 'fixed', 'both'), default='both')
    for name in ('analyze', 'freeze'):
        p = sub.add_parser(name); p.add_argument('plan', type=Path); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'probe-plan':
        result = make_plan(args.model, args.campaign, list(map(int, args.gpus.split(','))), [args.rate])
        print(json.dumps({'plan': str(args.campaign / 'PLAN.json'), 'GPU_replays': len(result['schedule']),
                          'requests_per_replay': result['request_count'], 'started': False}, indent=2))
    elif args.action == 'validation-plan':
        pred = read(args.prediction)
        check(pred['selection_split'] == 'calibration' and pred['deployment_authorized'] is False, 'Invalid analytical prediction')
        check(sha(pred['parameters']) == pred['parameters_sha256'], 'Mathematical parameters changed')
        parent = verify_plan(pred['probe_plan'])
        check(sha(pred['probe_plan']) == pred['probe_plan_sha256'], 'Probe plan changed')
        for path, expected in pred['probe_evidence_sha256'].items():
            check(sha(path) == expected, 'Probe evidence changed')
        ids = validation_ids(pred, args.scope)
        chosen = sorted([point for point in pred['points'] if point['id'] in ids], key=lambda c: (c['id'] != BASELINE, c['id']))
        result = make_plan(parent['model'], args.campaign, parent['allocation'], pred['rates'],
                           dict(zip(map(str, pred['rates']), pred['request_weights'])), chosen=chosen,
                           prediction={'path': str(args.prediction.resolve()), 'sha256': sha(args.prediction)}, ports=parent['ports'])
        print(json.dumps({'plan': str(args.campaign / 'PLAN.json'), 'selected_candidates_including_MAX': len(chosen),
                          'validation_cells': len(result['schedule']) * len(result['rates'])}, indent=2))
    elif args.action == 'run':
        run(args.plan, args.dry_run)
    elif args.action == 'search':
        mathematical_search(args.plan, args.parameters, args.output, list(map(int, args.rates.split(','))),
                            json.loads(args.request_weights) if args.request_weights else None)
        print(args.output)
    else:
        globals()[args.action](args.plan, args.output)
        print(args.output)


if __name__ == '__main__':
    main()

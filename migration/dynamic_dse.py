#!/usr/bin/env python3
"""Bind static math DSE to causal DVFS, gate calibration, and freeze heldout runs."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration import static_dse as static
from static_dse.analytical import instantaneous_power
from static_dse.selection import gate, select


def verify_static(path):
    freeze = static.read(path)
    p = static.verify_plan(freeze['plan'])
    static.check(static.sha(freeze['plan']) == freeze['plan_sha256'], 'Static plan changed')
    for name, expected in freeze['evidence_sha256'].items():
        static.check(static.sha(name) == expected, 'Static measurement evidence changed')
    observed = select(p, static.collect(p))
    static.check(freeze['decision'] == observed, 'Static selection differs from measured gates')
    by_id = {c['id']: c for c in p['candidates']}
    chosen = copy.deepcopy(by_id[observed['fixed_configuration']['candidate_id']])
    chosen.update(attention_gpus=p['allocation'][:2], expert_gpus=p['allocation'][2:3 if chosen['topology'] == '2a1e' else 4])
    static.check(freeze['selected_configuration'] == chosen and freeze['allocation'] == p['allocation']
                 and freeze['selected_by_rate'] == {r: by_id[c] for r, c in observed['by_rate'].items()},
                 'Static handoff differs from measured selection')
    prediction = static.read(p['prediction']['path'])
    static.check(static.sha(prediction['parameters']) == prediction['parameters_sha256'], 'Mathematical parameters changed')
    for name, expected in prediction['probe_evidence_sha256'].items():
        static.check(static.sha(name) == expected, 'Parameter probe evidence changed')
    for source in static.read(prediction['parameters'])['measurement_sources']:
        static.check(source['selection_split'] == 'calibration' and static.sha(source['path']) == source['sha256'],
                     'Physical parameter evidence changed')
    return freeze, p, prediction


def controller_config(freeze_path, rate=None):
    freeze, p, prediction = verify_static(freeze_path)
    chosen = copy.deepcopy(freeze['selected_configuration'] if rate is None else freeze['selected_by_rate'][str(rate)])
    ne = 1 if chosen['topology'] == '2a1e' else 2
    chosen['attention_gpus'] = freeze['allocation'][:2]
    chosen['expert_gpus'] = freeze['allocation'][2:2 + ne]
    params = static.read(prediction['parameters'])
    generic = params.get('model_kind') == 'general_operator_physics'
    hw = params['general_hardware'] if generic else params.get('hardware_by_topology', {}).get(chosen['topology'], params['hardware'])
    points = {}
    for point in prediction['points']:
        if point['topology'] != chosen['topology']:
            continue
        try:
            eligible = all((hw['device']['by_mhz'][str(point[role + '_mhz'])]['active_w'] if generic
                            else instantaneous_power(hw[role], point[role + '_mhz'])) <= point[role + '_power_w']
                           for role in ('attention', 'expert'))
        except (ValueError, KeyError):
            eligible = False
        if eligible:
            points[point['id']] = point
    points[chosen['id']] = chosen
    guard = dict(id=chosen['topology'] + '-max', topology=chosen['topology'], attention_mhz=1410,
                 expert_mhz=1410, attention_power_w=400, expert_power_w=400)
    points[guard['id']] = guard
    source = static.ROOT / 'inputs/calibration-max' / p['model'] / 'combined-controller-v10.json'
    pinned = static.read(static.ROOT / 'environment/input-hashes.json')
    static.check(static.sha(source) == pinned[str(source.relative_to(static.ROOT))], 'Inherited MAX guards changed')
    inherited = static.read(source)
    guard_curve = copy.deepcopy(inherited['predictor']['calibration_guard_by_rps'])
    config = dict(schema_version=1, method='static_bound_mathematical_causal_dvfs', selection_split='calibration',
        model=p['model_tag'], static_configuration=chosen, points=list(points.values()), guard_id=guard['id'],
        attention_gpus=chosen['attention_gpus'], expert_gpus=chosen['expert_gpus'], allocation=freeze['allocation'],
        parameters=params, guard_by_rps=guard_curve, guard_mode='inherited_A100_v10_thresholds_not_rebuilt',
        control_interval_ms=100, downshift_hold_ms=500, minimum_transition_interval_ms=500,
        event_stale_ms=1500, relative_service_budget=1.05,
        future_trace_access=False, topology_changes_allowed=False,
        target='fixed_across_1_2_4_rps' if rate is None else f'fixed_for_{rate}_rps',
        files_sha256={str(Path(freeze_path).resolve()): static.sha(freeze_path),
                      prediction['parameters']: static.sha(prediction['parameters']),
                      p['prediction']['path']: static.sha(p['prediction']['path']), str(source): static.sha(source)})
    return config, p


def make_plan(freeze_path, destination, rate=None, calibration_plan=None, calibration_report=None):
    destination = Path(destination).resolve()
    static.check(not destination.exists(), 'Dynamic campaign exists; resume it or use a new name')
    config, original = controller_config(freeze_path, rate)
    split = 'heldout' if calibration_plan else 'calibration'
    rates = [rate] if rate else original['rates']
    trace = static.ROOT / 'inputs/traces' / ('heldout-400.jsonl' if split == 'heldout' else 'calibration-200.jsonl')
    if split == 'heldout':
        static.check(calibration_report is not None, 'A calibration gate report is required')
        report = summarize(calibration_plan)
        static.check(report['all_calibration_gates_pass'], 'Calibration gate failed; heldout remains unused')
        static.check(static.read(calibration_report) == report, 'Calibration gate report differs from current evidence')
        calibrated = static.verify_plan(calibration_plan)
        static.check(static.read(calibrated['controller_path']) == config, 'Controller changed after dynamic calibration')
    # Use a generated skeleton for all runtime/provenance fields, then freeze the final plan.
    skeleton = copy.deepcopy(config['static_configuration'])
    skeleton['id'] = 'max'
    skeleton.update(attention_mhz=1410, expert_mhz=1410, attention_power_w=400, expert_power_w=400)
    selected = dict(config['static_configuration'], id='static')
    dynamic = dict(config['static_configuration'], id='dynamic')
    p = static.make_plan(original['model'], destination, original['allocation'], rates,
        chosen=[skeleton, selected, dynamic], prediction={'path': str(Path(freeze_path).resolve()), 'sha256': static.sha(freeze_path)},
        ports=original['ports'])
    # No subprocess has run. Replace only this just-created draft before freezing it.
    p.update(phase='formal_heldout' if split == 'heldout' else 'dynamic_calibration', baseline_id='max',
             evaluation_split=split, trace_path=str(trace), request_count=len(trace.read_text().splitlines()),
             static_freeze=str(Path(freeze_path).resolve()), static_freeze_sha256=static.sha(freeze_path),
             controller_path=str(destination / 'controller.json'))
    static.write(destination / 'controller.json', config)
    p['repetitions'] = [1, 2] if split == 'heldout' else [1]
    p['schedule'] = []
    for rep in p['repetitions']:
        for c in (p['candidates'] if rep == 1 else list(reversed(p['candidates']))):
            p['schedule'].append(dict(position=len(p['schedule']) + 1, candidate_id=c['id'],
                arm='ours' if split == 'heldout' else c['id'], repetition=rep,
                rates_rps=rates if rep == 1 else list(reversed(rates)),
                suite_id=f'math-{destination.name}-{hashlib.sha256(str(destination).encode()).hexdigest()[:10]}-{p["model"]}-{c["id"]}-r{rep}'))
    (destination / 'schedule.json').write_text(json.dumps({'selection_split': 'calibration', 'entries': p['schedule']}, indent=2) + '\n')
    if split == 'heldout':
        p['calibration_gate'] = dict(plan=str(Path(calibration_plan).resolve()), plan_sha256=static.sha(calibration_plan),
            report=str(Path(calibration_report).resolve()), report_sha256=static.sha(calibration_report),
            controller_sha256=static.sha(p['controller_path']))
    for c in p['candidates']:
        path = destination / 'deployments' / (c['id'] + '.json')
        deployment = static.read(path)
        deployment.update(evaluation_split=split, formal_evaluation_eligible=split == 'heldout',
                          math_method=c['id'], math_parameter_probe=False)
        deployment['runtime_contract']['stage_trace'] = False
        deployment['math_dse_parameter_probe'] = False
        deployment['operating_points']['calibration_trace_sha256'] = p['workload']['trace_sha256']
        if c['id'] == 'dynamic':
            deployment['math_dynamic_controller'] = {'path': p['controller_path'], 'sha256': static.sha(p['controller_path'])}
        if split == 'heldout':
            entry = next(e for e in calibrated['schedule'] if e['candidate_id'] == c['id'])
            suite, _ = static.paths(calibrated, entry)
            deployment['operating_points']['comparison_contract_sha256'] = static.read(suite / 'manifest.json')['comparison_contract_sha256']
            for point in deployment['operating_points']['by_rps'].values():
                point['calibration_feasible'] = True
        path.write_text(json.dumps(deployment, indent=2) + '\n')
    p['campaign_files_sha256'] = {str(f.relative_to(destination)): static.sha(f) for f in destination.rglob('*') if f.is_file() and f.name != 'PLAN.json'}
    for name, expected in config['files_sha256'].items():
        p['files_sha256'][name] = expected
    (destination / 'PLAN.json').write_text(json.dumps(p, indent=2, sort_keys=True) + '\n')
    return p


def summarize(plan_path):
    p = static.verify_plan(plan_path)
    rows = static.collect(p)
    by_key = {(r['candidate_id'], r['rate'], r['repetition']): r for r in rows}
    cells, all_pass = [], True
    for row in rows:
        reference = by_key['max', row['rate'], row['repetition']]
        judged = gate(row['metrics'], reference['metrics'], p['contract'])
        all_pass = all_pass and judged['pass']
        static_reference = by_key['static', row['rate'], row['repetition']]
        cells.append(dict(candidate_id=row['candidate_id'], rate=row['rate'], repetition=row['repetition'],
                          metrics=row['metrics'], relative_MAX_gate=judged,
                          saving_vs_MAX=1 - row['metrics']['energy_j_per_request'] / reference['metrics']['energy_j_per_request'],
                          saving_vs_STATIC=1 - row['metrics']['energy_j_per_request'] / static_reference['metrics']['energy_j_per_request'],
                          energy_claim_eligible=judged['pass']))
    return dict(schema_version=1, evaluation_split=p['evaluation_split'], model=p['model_tag'],
                plan=str(Path(plan_path).resolve()), plan_sha256=static.sha(plan_path),
                static_freeze=p['static_freeze'], static_freeze_sha256=p['static_freeze_sha256'],
                controller_sha256=static.sha(p['controller_path']),
                all_calibration_gates_pass=all_pass if p['evaluation_split'] == 'calibration' else None,
                heldout_evaluation_complete=p['evaluation_split'] == 'heldout', cells=cells,
                request_count_total=p['request_count'] * len(rows),
                topology=static.read(p['controller_path'])['static_configuration']['topology'],
                evidence_sha256={f: h for row in rows for f, h in row['evidence_sha256'].items()},
                comparison_arms=['MAX_on_selected_topology', 'STATIC_DSE', 'MATHEMATICAL_DVFS'],
                inherited_v10_guard_thresholds_rebuilt=False)


def verify_heldout_gate(p):
    g = p['calibration_gate']
    static.check(static.sha(g['plan']) == g['plan_sha256'] and static.sha(g['report']) == g['report_sha256'], 'Calibration gate artifacts changed')
    static.check(static.sha(p['controller_path']) == g['controller_sha256'], 'Formal controller differs from calibrated controller')
    current = summarize(g['plan'])
    static.check(current == static.read(g['report']) and current['all_calibration_gates_pass'], 'Dynamic calibration no longer passes')
    # Re-audit actual source identities immediately before a formal launch.
    import subprocess
    audit = json.loads(subprocess.check_output([sys.executable, str(static.ROOT / 'scripts/audit_trace_isolation.py'),
        '--calibration', str(static.ROOT / 'inputs/traces/calibration-200.jsonl'), '--evaluation', p['trace_path'],
        '--identity-field', 'source_index', '--identity-field', 'source_timestamp'], text=True))
    static.check(audit['status'] == 'PASS', 'Formal trace identity audit failed')


def main():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='action', required=True)
    c = sub.add_parser('plan'); c.add_argument('--static-freeze', type=Path, required=True); c.add_argument('--output-dir', type=Path, required=True)
    c.add_argument('--rate', type=int, choices=(1, 2, 4)); c.add_argument('--calibration-plan', type=Path); c.add_argument('--calibration-report', type=Path)
    c = sub.add_parser('report'); c.add_argument('plan', type=Path); c.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'plan':
        make_plan(args.static_freeze, args.output_dir, args.rate, args.calibration_plan, args.calibration_report)
        print(args.output_dir / 'PLAN.json')
    else:
        result = summarize(args.plan); static.write(args.output, result); print(args.output)
        if result['evaluation_split'] == 'calibration' and not result['all_calibration_gates_pass']:
            raise SystemExit('Calibration gate failed; no heldout launch authorized')

if __name__ == '__main__':
    main()

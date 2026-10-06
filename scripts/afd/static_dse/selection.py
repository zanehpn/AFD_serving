"""Use measurements for final admission after analytical candidate selection.

No synthetic pipeline duration is treated as measured TTFT, TPOT or throughput.
Measured admission is checked separately for each workload and repetition.
"""
from __future__ import annotations

import math
import statistics

METRICS = ('p90_ttft_ms', 'p90_tpot_ms', 'output_tps', 'energy_j_per_request')
KNOBS = ('attention_mhz', 'expert_mhz', 'attention_power_w', 'expert_power_w')


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Metrics must be finite and positive')
    return value


def metrics(summary):
    n = int(summary['requests'])
    if n <= 0 or summary['completed_requests'] != n or summary['failed_requests'] != 0:
        raise ValueError('Incomplete requests cannot enter static DSE')
    return dict(zip(METRICS, map(positive, (
        summary['ttft_ms']['p90'], summary['tpot_ms']['p90'],
        summary['output_token_throughput_tps'], summary['energy_j'] / n,
    ))))


def gate(observed, baseline, contract):
    ratios = {m: positive(observed[m]) / positive(baseline[m]) for m in METRICS}
    failed = [m for m in METRICS[:2] if ratios[m] > contract['latency_ratio_max']]
    if ratios['output_tps'] < contract['throughput_ratio_min']:
        failed.append('output_tps')
    return {'pass': not failed, 'failed_metrics': failed, 'ratios': ratios}


def select(plan, records):
    """Require all frozen repetitions/rates; never silently choose from partial data."""
    expected = {(c['id'], rate, rep) for c in plan['candidates']
                for rate in plan['rates'] for rep in plan['repetitions']}
    by_key = {}
    for row in records:
        if row['selection_split'] != 'calibration':
            raise ValueError('Only calibration measurements may select a policy')
        key = (row['candidate_id'], row['rate'], row['repetition'])
        if key not in expected or key in by_key:
            raise ValueError('Unexpected or duplicate calibration cell')
        for name in METRICS:
            positive(row['metrics'][name])
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError('Static matrix is incomplete; missing cells are not infeasible candidates')
    decisions, by_rate = [], {}
    for rate in plan['rates']:
        scored = []
        for c in plan['candidates']:
            gates, energies = [], []
            for rep in plan['repetitions']:
                obs = by_key[c['id'], rate, rep]['metrics']
                baseline = by_key[plan['baseline_id'], rate, rep]['metrics']
                gates.append({'repetition': rep, **gate(obs, baseline, plan['contract'])})
                energies.append(obs['energy_j_per_request'])
            scored.append({'candidate_id': c['id'], 'rate': rate, 'topology': c['topology'],
                           'status': 'measured_pass' if all(g['pass'] for g in gates) else 'measured_fail',
                           'mean_energy_j_per_request': statistics.mean(energies), 'gates': gates})
        passing = [s for s in scored if s['status'] == 'measured_pass']
        chosen = min(passing, key=lambda s: (s['mean_energy_j_per_request'], s['candidate_id']))
        by_rate[str(rate)] = chosen['candidate_id']
        decisions.extend(scored)
    # One complete fixed configuration must pass every rate before DVFS starts.
    fixed = []
    for c in plan['candidates']:
        rows = [r for r in decisions if r['candidate_id'] == c['id']]
        if all(r['status'] == 'measured_pass' for r in rows):
            energy = sum(r['mean_energy_j_per_request'] * plan['request_weights'][str(r['rate'])]
                         for r in rows)
            fixed.append({'candidate_id': c['id'], 'weighted_energy_j_per_request': energy})
    best = min(fixed, key=lambda r: (r['weighted_energy_j_per_request'], r['candidate_id']))
    return {'selection_split': 'calibration', 'formal_evaluation_eligible': False,
            'by_rate': by_rate, 'fixed_configuration': best, 'cells': decisions,
            'scope': 'best_measured_feasible_candidate_in_frozen_matrix',
            'transition_behavior_validated': False}

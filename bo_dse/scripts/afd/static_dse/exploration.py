"""Budgeted broad exploration for the model-guided arm; no topology favorites."""
import math
from collections import Counter

from .space import configuration, digest, structure


def enabled(options):
    return (options.get('exploration_policy') == 'broad_v2'
            and options['use_model_prior'] and options['method'] == 'bo')


def boundary_repeat_ids(observations, limits, margin):
    ids = set()
    for o in observations:
        if o['status'] != 'ok':
            continue
        m = o['metrics']
        ratios = [m[k]/limits[k] for k in ('ttft_ms', 'tpot_ms', 'tbt_ms') if k in limits]
        ratios.append(limits['min_output_tps']/m['output_tps'])
        if any(abs(r-1) <= margin for r in ratios):
            ids.add(o['candidate_id'])
    return ids


def novelty_pool(candidates, pool, observations, settings):
    """Defer repeats while new points remain, except near a measured SLO boundary."""
    o = settings['bo']
    if not enabled(o):
        return pool
    n = len(observations)
    search_budget = max(1, settings['budget']['evaluations']-settings['setup_cost']['evaluations'])
    if n >= math.ceil(search_budget * o['repeat_after_fraction']):
        return pool
    measured = {r['candidate_id'] for r in observations}
    if not any(candidates[i]['id'] not in measured for i in pool):
        return pool
    boundary = boundary_repeat_ids(observations, settings['limits'], o['repeat_boundary_margin'])
    return [i for i in pool if candidates[i]['id'] not in measured or candidates[i]['id'] in boundary]


def choose(candidates, pool, observations, settings, costs, emean, estd):
    """Spend roughly every third early measurement on an unseen structure.

    Structure probes compare max operating points to avoid confounding topology
    changes with DVFS. Evidence-backed analytical priors contribute to the
    optimistic energy score; uncovered priors remain uncertainty, not physics.
    """
    options = settings['bo']
    if not enabled(options) or not pool:
        return None
    by_id = {c['id']: c for c in candidates}
    seen = {digest(structure(by_id[r['candidate_id']])) for r in observations if r['candidate_id'] in by_id}
    counts = Counter(r['candidate_id'] for r in observations)
    groups = {}
    for i in pool:
        key = digest(structure(candidates[i]))
        if key not in seen:
            groups.setdefault(key, []).append(i)
    n = len(observations)
    budget = max(1, settings['budget']['evaluations']-settings['setup_cost']['evaluations'])
    fraction = options['structure_exploration_fraction']
    quota = math.ceil(budget*fraction)
    used = sum(r.get('proposal_reason') == 'structure_exploration' for r in observations)
    due = used < min(quota, math.floor((n+1)*fraction+1e-9))
    if due and groups:
        representatives = []
        for indices in groups.values():
            # Canonical max operating point for each legal structure, not a
            # hard-coded GPU count. Only affordable/unmeasured points enter.
            representatives.append(max(indices, key=lambda i: (
                tuple(configuration(candidates[i])[k] for k in
                      ('attention_mhz','expert_mhz','attention_power_w','expert_power_w')),
                candidates[i]['id'])))
        def score(i):
            optimistic = emean[i]-options['screen_beta']*estd[i]
            return (-optimistic, estd[i], -costs[i], candidates[i]['id'])
        i = max(representatives, key=score)
        prior = candidates[i].get('prior', {})
        return i, dict(reason='structure_exploration', exploration_quota=quota,
                       exploration_used=used, prior_source=prior.get('source', 'unspecified'),
                       prediction_basis=('unmeasured_uncertainty' if prior.get('source') == 'uncovered_neutral'
                                         else 'candidate_model_prior'),
                       predicted_log_energy=float(emean[i]), log_energy_sd=float(estd[i]))
    reference = by_id[settings['reference_candidate_id']]
    ref = configuration(reference)
    actions = [('expert_mhz','low'), ('attention_mhz','low'),
               ('expert_mhz','middle'), ('attention_mhz','middle'),
               ('expert_power_w','binding'), ('attention_power_w','binding')]
    for control, level in actions:
        choices = [i for i in pool if not counts[candidates[i]['id']]
                   and all(v == ref[k] for k,v in configuration(candidates[i]).items() if k != control)
                   and configuration(candidates[i])[control] < ref[control]]
        all_values = sorted({configuration(c)[control] for c in candidates
                             if all(v == ref[k] for k,v in configuration(c).items() if k != control)})
        if not choices or len(all_values) < 2:
            continue
        if level == 'binding':
            role = control.split('_')[0]
            # p95 of the busiest rank in the replay window. Mean role power
            # alone cannot establish whether a cap will bind.
            evidence = [r['external_observables']['role_peak_rank_p95_power_w'][role]
                        for r in observations if r['status'] == 'ok'
                        and r['candidate_id'] == reference['id']
                        and role in r.get('external_observables', {}).get('role_peak_rank_p95_power_w', {})]
            if not evidence:
                continue
            target = .95 * max(evidence)
            choices = [i for i in choices if configuration(candidates[i])[control] <= target]
            if not choices:
                continue
            desired = max(configuration(candidates[i])[control] for i in choices)
        elif level == 'low':
            desired = all_values[0]
        else:
            interior = all_values[1:-1]
            if not interior:
                continue
            desired = interior[len(interior)//2]
        choices = [i for i in choices if configuration(candidates[i])[control] == desired]
        if choices:
            i = min(choices, key=lambda i: candidates[i]['id'])
            return i, dict(reason='broad_parameter_probe', control=control, level=level,
                           reference_value=ref[control], target_value=desired,
                           collect_four_stage_observation=settings.get('mechanism_model') != 'external_power_duration_v1')
    return None

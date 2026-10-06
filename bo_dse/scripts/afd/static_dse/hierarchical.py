"""Structure selection with calibrated priors, followed by structure-local BO."""
import copy
import math
from collections import Counter

import numpy as np

from .space import digest, structure


def enabled(options):
    return (options.get('exploration_policy') == 'hierarchical_v3'
            and options['method'] == 'bo' and options['use_model_prior'])


def propose(candidates, observations, settings, eligible_ids, remaining_gpu_hours):
    from .optimizer import expected_cost, propose as local_propose
    from .exploration import boundary_repeat_ids

    options = settings['bo']
    by_id = {c['id']: c for c in candidates}
    counts = Counter(o['candidate_id'] for o in observations)
    failures = {o['candidate_id'] for o in observations if o['status'] == 'runtime_incompatible'}
    current = by_id.get(observations[-1]['candidate_id']) if observations else None
    pool = [c for c in candidates if c['id'] in eligible_ids and c['id'] not in failures
            and counts[c['id']] < options['max_repeats']
            and expected_cost(c, current, options) <= remaining_gpu_hours]
    if not pool:
        return None, {'reason': 'no_affordable_hierarchical_candidate'}
    reference = settings['reference_candidate_id']
    if not counts[reference] and reference in eligible_ids:
        candidate = next((c for c in pool if c['id'] == reference), None)
        if candidate is None:
            return None, {'reason': 'reference_cost_exceeds_remaining_budget'}
        cost = expected_cost(candidate, current, options)
        return candidate, dict(reason='measure_reference', expected_gpu_hours=cost, base_gpu_hours=cost)
    budget = max(1, settings['budget']['evaluations'] - settings['setup_cost']['evaluations'])
    if len(observations) < math.ceil(budget * options['repeat_after_fraction']):
        boundary = boundary_repeat_ids(observations, settings['limits'], options['repeat_boundary_margin'])
        novel = [c for c in pool if not counts[c['id']] or c['id'] in boundary]
        if novel:
            pool = novel
    groups = {}
    for c in pool:
        groups.setdefault(digest(structure(c)), []).append(c)
    evidence = {}
    for o in observations:
        if o['candidate_id'] in by_id:
            evidence.setdefault(digest(structure(by_id[o['candidate_id']])), []).append(o)

    # Calibrate each structure independently. No residual or SLO evidence from
    # another topology enters either this correction or its local GP.
    ranking = {}
    for key, points in groups.items():
        valid = [o for o in evidence.get(key, []) if o['status'] == 'ok']
        residuals = [math.log(o['metrics']['energy_j']) - by_id[o['candidate_id']]['prior']['log_energy'] for o in valid]
        correction = float(np.mean(residuals)) if residuals else 0.
        representative = min(points, key=lambda c: (c['prior']['log_energy'], c['id']))
        sd = max(float(representative['prior']['energy_sd']) / math.sqrt(1 + len(valid)),
                 float(np.std(residuals)) if residuals else 0., options['noise_log'])
        violations = []
        for o in valid:
            m, limits = o['metrics'], settings['limits']
            ratios = [m[k] / limits[k] for k in ('ttft_ms', 'tpot_ms', 'tbt_ms') if k in limits]
            ratios.append(limits['min_output_tps'] / m['output_tps'])
            violations.append(max(0., math.log(max(ratios))))
        penalty = min(violations) if violations else 0.
        source = representative['prior'].get('source', 'unspecified')
        ranking[key] = dict(score=float(representative['prior']['log_energy'] + correction
                                        - options['screen_beta'] * sd + penalty),
                            uncertainty=sd, observations=len(valid), prior_source=source,
                            prediction_basis=('uncovered_neutral' if source == 'uncovered_neutral'
                                              else 'calibrated_candidate_prior'))
    fraction = options['structure_exploration_fraction']
    used = sum(o.get('proposal_reason') == 'hierarchical_random' for o in observations)
    due = used < min(math.ceil(budget * fraction), math.floor((len(observations) + 1) * fraction + 1e-9))
    rng = np.random.default_rng(options['seed'] + len(observations))
    if due:
        # Sample structures uniformly, avoiding a bias towards structures with
        # more legal knob combinations. Unseen structures have first priority.
        choices = sorted(k for k in groups if k not in evidence)
        if not choices:
            threshold = float(np.median([r['uncertainty'] for r in ranking.values()]))
            choices = sorted(k for k in groups if ranking[k]['uncertainty'] >= threshold)
        key = str(rng.choice(choices))
        points = sorted(groups[key], key=lambda c: c['id'])
        unseen = [c for c in points if not counts[c['id']]]
        candidate = (unseen or points)[int(rng.integers(len(unseen or points)))]
        cost = expected_cost(candidate, current, options)
        decision = dict(reason='hierarchical_random', expected_gpu_hours=cost, base_gpu_hours=cost)
    else:
        # Only the structure selector compares topologies; BO receives exactly
        # one structure and its own measurement history.
        for key in sorted(groups, key=lambda k: (ranking[k]['score'], k)):
            local = copy.deepcopy(settings)
            local['bo'].update(exploration_policy='legacy', initial_parameter_probes=0,
                               initial_joint_probes=0, analytical_probe_every=0)
            local['reference_candidate_id'] = settings['reference_candidate_id']
            # Preserve all candidates of this structure for repeated observations
            # and current-state lookup, while restricting the selectable pool.
            members = [c for c in candidates if digest(structure(c)) == key]
            local_obs = evidence.get(key, [])
            candidate, decision = local_propose(members, local_obs, local,
                                               {c['id'] for c in groups[key]}, remaining_gpu_hours,
                                               _local=True, _current=current)
            if candidate is not None:
                decision = {**decision, 'local_reason': decision['reason'], 'reason': 'hierarchical_local_bo'}
                break
        else:
            return None, {'reason': 'no_affordable_local_bo_candidate'}
    return candidate, {**decision, 'selected_structure': key,
                       'structure_prediction': ranking[key], 'structure_ranking': ranking,
                       'random_quota': math.ceil(budget * fraction), 'random_used': used,
                       'local_observations': len(evidence.get(key, []))}

"""Capacity-guided v2 with budgeted structure coverage and joint knob exploration."""
import copy
import math
from collections import Counter

from .capacity import assessment
from .space import KNOBS, configuration, digest, structure


def enabled(options):
    return options.get('exploration_policy') == 'capacity_v2' and options['use_model_prior'] and options['method'] == 'bo'


def allocation(c):
    t = c['topology']
    return len(t['attention_gpus']), len(t['expert_gpus'])


TUNING_REASONS = {'capacity_local_bo', 'capacity_joint_knob_probe', 'capacity_threshold_probe'}


def joint_probe(points, selectable, anchor, observations):
    """Maximin design in normalized knob ranks; never a feasibility label.

    Probe simultaneous reductions, including frequency. All observed attempts,
    including failures, count as explored locations. Targets depend only on the
    legal levels and this arm's observations, not on another method's winners.
    """
    levels = {k: sorted({c['knobs'][k] for c in points}) for k in KNOBS}
    def vector(c):
        return tuple(levels[k].index(c['knobs'][k]) / max(1, len(levels[k])-1) for k in KNOBS)
    measured = {o['candidate_id'] for o in observations}
    seen = [vector(c) for c in points if c['id'] in measured] or [vector(anchor)]
    pool = [c for c in points if c['id'] in selectable and c['id'] not in measured
            and all(c['knobs'][k] <= anchor['knobs'][k] for k in KNOBS)
            and sum(c['knobs'][k] != anchor['knobs'][k] for k in KNOBS) >= 2
            and any(c['knobs'][r+'_mhz'] < anchor['knobs'][r+'_mhz'] for r in ('attention', 'expert'))]
    def rank(c):
        v = vector(c)
        distance = min(sum((a-b)**2 for a,b in zip(v,other)) for other in seen)
        return (-distance, sum(v[:2]), sum(v[2:]), tuple(c['knobs'][k] for k in KNOBS), c['id'])
    return min(pool, key=rank) if pool else None


def violation(o, limits):
    if o['status'] != 'ok':
        return math.inf
    m = o['metrics']
    return max([m[k] / limits[k] for k in ('ttft_ms', 'tpot_ms', 'tbt_ms') if k in limits]
               + [limits['min_output_tps'] / m['output_tps']])


def _nonnegative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def feedback(observation, limits):
    """Joint role pressure, not a causal decomposition of end-to-end latency.

    Optional timings must be measured over the same window for both roles.
    Public replay/NVML do not provide these timings; absent values stay absent.
    """
    if observation is None or observation['status'] != 'ok':
        return dict(attention=0., expert=0., basis='no_successful_feedback')
    m = observation['metrics']
    ext = observation.get('external_observables', {})
    raw_util = ext.get('role_mean_gpu_utilization_pct') or {}
    util = {r: raw_util[r] for r in ('attention', 'expert')
            if _nonnegative(raw_util.get(r)) and raw_util[r] <= 100}
    ttft = max(0., m['ttft_ms'] / limits['ttft_ms'] - 1)
    lag = ext.get('client_queue_lag_ms_p90')
    lag = lag if _nonnegative(lag) else None
    client_lag_dominant = lag is not None and lag > .1 * max(m['ttft_ms'], 1.)
    if client_lag_dominant:
        ttft = 0.
    # All latency metrics include both roles and communication. None identifies
    # a role by itself, even when output throughput already meets its SLO.
    deficit = (ttft + max(0., limits['min_output_tps'] / m['output_tps'] - 1)
                   + max([0.] + [max(0., m[k] / limits[k] - 1)
                                 for k in ('tpot_ms', 'tbt_ms') if k in limits]))
    weights = {'attention': .5, 'expert': .5}
    evidence = []
    if len(util) == 2:
        # Bounded saturation pressure emphasizes the busier role. This is a
        # ranking heuristic, not an estimate of queue duration or GPU occupancy.
        pressure = {r: u / max(5., 100. - u) for r, u in util.items()}
        total = sum(pressure.values())
        if total:
            weights = {r: pressure[r] / total for r in weights}
            evidence.append('nvml_role_utilization')
    timings = ext.get('role_timing_ms')
    timing_fields = ('compute_ms', 'queue_wait_ms', 'peer_wait_ms', 'communication_ms')
    timing_valid = (ext.get('role_timing_provenance') == 'measured_same_window'
                    and isinstance(timings, dict)
                    and all(isinstance(timings.get(r), dict)
                            and all(_nonnegative(timings[r].get(k)) for k in timing_fields)
                            for r in weights))
    communication_fraction = None
    if timing_valid:
        # Waiting on the peer supports pressure on that peer, not on the waiter.
        work = {r: timings[r]['compute_ms'] + timings[r]['queue_wait_ms']
                   + timings[other]['peer_wait_ms']
                for r, other in (('attention', 'expert'), ('expert', 'attention'))}
        communication = sum(timings[r]['communication_ms'] for r in weights)
        total = sum(work.values()) + communication
        communication_fraction = communication / total if total else 0.
        if sum(work.values()):
            measured = {r: work[r] / sum(work.values()) for r in weights}
            weights = {r: (weights[r] + measured[r]) / 2 if evidence else measured[r]
                       for r in weights}
        # Communication cannot be attributed to either role's compute capacity.
        # When it dominates, do not infer that adding one particular role helps.
        if communication_fraction >= .5:
            weights = dict(attention=.5, expert=.5)
        evidence.append('measured_role_timing')
    return dict(attention=deficit * weights['attention'], expert=deficit * weights['expert'],
                basis='joint_end_to_end_deficit_and_role_pressure', slo_deficit=deficit,
                role_pressure_weights=weights, evidence=evidence,
                client_queue_lag_ms_p90=lag, client_lag_dominant=client_lag_dominant,
                role_mean_gpu_utilization_pct=util,
                role_timing_ms=timings if timing_valid else None,
                communication_fraction=communication_fraction,
                directional_evidence=bool(evidence) and (communication_fraction is None or communication_fraction < .5),
                interpretation='Resource ranking heuristic; utilization is not queue time, peer wait is not local compute, and communication is not assigned to a role')


def propose(candidates, observations, settings, eligible_ids, remaining_gpu_hours):
    from .optimizer import expected_cost, propose as local_propose, best_measured
    options = settings['bo']
    profile = settings.get('capacity_profile', {})
    by_id = {c['id']: c for c in candidates}
    counts = Counter(o['candidate_id'] for o in observations)
    current = by_id[observations[-1]['candidate_id']] if observations else None
    incompatible = {o['candidate_id'] for o in observations if o['status'] == 'runtime_incompatible'}
    groups = {}
    for c in candidates:
        if (settings.get('reference_evidence_origin') == 'historical_max_reuse'
                and c['id'] == settings['reference_candidate_id']):
            continue  # User requested reuse; do not spend a search trial repeating Max.
        if c['id'] not in eligible_ids or c['id'] in incompatible or assessment(c, profile)['status'] == 'proven_impossible':
            continue
        groups.setdefault(digest(structure(c)), []).append(c)
    if not groups:
        return None, {'reason': 'no_capacity_eligible_structure'}
    evidence = {}
    for o in observations:
        if o['candidate_id'] in by_id:
            evidence.setdefault(digest(structure(by_id[o['candidate_id']])), []).append(o)
    representatives = {k: max(cs, key=lambda c: tuple(c['knobs'][q] for q in KNOBS)) for k, cs in groups.items()}
    seen_allocations = {allocation(by_id[o['candidate_id']]) for o in observations if o['candidate_id'] in by_id}
    reference = by_id[settings['reference_candidate_id']]
    preferred_microbatches = reference['microbatches']
    latest = next((o for o in reversed(observations) if o['status'] == 'ok'), None)
    hint = feedback(latest, settings['limits'])

    def risk(c):
        return {'estimated_fit': 0, 'unknown': 1, 'uncertain': 2, 'proven_impossible': 3}[assessment(c, profile)['status']]

    def rank(c, first=False):
        a, e = allocation(c)
        prev_a, prev_e = allocation(by_id[latest['candidate_id']]) if latest else (a, e)
        growth = (hint['attention'] * (a - prev_a) / prev_a
                  + hint['expert'] * (e - prev_e) / prev_e)
        # Explicit semantic ties prefer lower TP and match baseline DBO. No
        # opaque candidate hash controls the structure exploration order.
        t = c['topology']
        semantic = (t['attention_tp'] + t['expert_tp'], t['attention_tp'], t['expert_tp'], a, e,
                    tuple(t['attention_gpus']), tuple(t['expert_gpus']))
        threshold = (c.get('dbo_decode_token_threshold', 2), c.get('dbo_prefill_token_threshold', 12))
        return (risk(c), a + e if first else 0, c['microbatches'] != preferred_microbatches,
                threshold != (reference.get('dbo_decode_token_threshold', 2),
                              reference.get('dbo_prefill_token_threshold', 12)),
                0 if first else -growth, a + e, semantic)

    unseen = [c for k, c in representatives.items() if k not in evidence
              and counts[c['id']] == 0 and expected_cost(c, current, options) <= remaining_gpu_hours]
    fresh_allocations = [c for c in unseen if allocation(c) not in seen_allocations]
    budget = max(1, settings['budget']['evaluations'] - settings['setup_cost']['evaluations'])
    # Reserve six of sixteen attempts for structures by default. DBO pairs
    # share this allowance; every probe consumes the same total budget.
    structure_budget = max(1, min(max(1, budget-2), int(budget * options.get('capacity_structure_fraction', .375))))
    structure_attempts = sum(o.get('proposal_reason', '').startswith('capacity_')
                             and o.get('proposal_reason') not in TUNING_REASONS for o in observations)
    # Threshold observations spend budget but do not replace resource-knob exploration.
    tuning_attempts = sum(o.get('proposal_reason') in TUNING_REASONS - {'capacity_threshold_probe'}
                          for o in observations)
    feasible = best_measured(observations, settings['limits']) is not None
    explore_structure = not feasible or structure_attempts < structure_budget
    initial = min(options.get('capacity_initial_structures', 2), structure_budget, max(1, budget - 2),
                  len({allocation(c) for c in representatives.values()}))
    covered = len(seen_allocations)
    interval = options.get('capacity_explore_every', 3)
    periodic = covered >= initial and len(observations) >= initial and (len(observations) - initial + 1) % interval == 0

    def high_clock_pick(pool, reason):
        c = min(pool, key=lambda c: rank(c, first=not observations))
        cost = expected_cost(c, current, options)
        return c, dict(reason=reason, capacity=assessment(c, profile), feedback=hint,
                       allocations_covered=covered, initial_allocation_target=initial,
                       structure_attempts=structure_attempts, structure_budget=structure_budget,
                       expected_gpu_hours=cost, base_gpu_hours=cost,
                       uniform_high_operating_point=all(c['knobs'][k] == max(p['knobs'][k] for p in groups[digest(structure(c))]) for k in KNOBS))

    # Two budgeted probes compare threshold profiles at an observed operating
    # point. The child receives only its own new measurement, never parent labels.
    threshold_profiles = {(c.get('dbo_decode_token_threshold', 2),
                           c.get('dbo_prefill_token_threshold', 12)) for c in candidates}
    threshold_trials = sum(o.get('proposal_reason') == 'capacity_threshold_probe' for o in observations)
    if (len(threshold_profiles) > 1 and len(observations) >= 2
            and (len(observations)-2) % 4 == 0
            and threshold_trials < len(threshold_profiles)-1):
        measured_best = best_measured(observations, settings['limits'])
        anchor_observation = (next(o for o in observations if o['candidate_id'] == measured_best['candidate_id'])
                              if measured_best else latest)
        if anchor_observation is not None:
            anchor = by_id[anchor_observation['candidate_id']]
            def threshold(c):
                return c.get('dbo_decode_token_threshold', 2), c.get('dbo_prefill_token_threshold', 12)
            profile_counts = Counter(threshold(by_id[o['candidate_id']]) for o in observations)
            pairs = [c for points in groups.values() for c in points
                     if c['topology'] == anchor['topology'] and c['knobs'] == anchor['knobs']
                     and c['microbatches'] == anchor['microbatches']
                     and threshold(c) != threshold(anchor) and not counts[c['id']]
                     and expected_cost(c, current, options) <= remaining_gpu_hours]
            if pairs:
                chosen = min(pairs, key=lambda c: (profile_counts[threshold(c)], threshold(c), c['id']))
                cost = expected_cost(chosen, current, options)
                return chosen, dict(reason='capacity_threshold_probe', parent_candidate_id=anchor['id'],
                    topology_and_operating_controls_preserved=True, feasibility_assumed=False,
                    threshold_profile=list(threshold(chosen)), expected_gpu_hours=cost, base_gpu_hours=cost,
                    capacity=assessment(chosen, profile), feedback=hint)

    # A DBO mode is a distinct measured structure, not an inherited performance
    # label. Pair each newly visited topology at the same operating controls.
    if explore_structure and options.get('capacity_pair_dbo') and current is not None:
        pairs = [c for key, points in groups.items() if key not in evidence
                 for c in points if c['topology'] == current['topology']
                 and c.get('execution_mode') == current.get('execution_mode')
                 and c['knobs'] == current['knobs'] and c['microbatches'] != current['microbatches']
                 and not counts[c['id']] and expected_cost(c, current, options) <= remaining_gpu_hours]
        if pairs:
            paired = min(pairs, key=lambda c: c['microbatches'])
            cost = expected_cost(paired, current, options)
            return paired, dict(reason='capacity_dbo_pair_probe', parent_candidate_id=current['id'],
                topology_and_operating_controls_preserved=True, expected_gpu_hours=cost,
                base_gpu_hours=cost, capacity=assessment(paired, profile), feedback=hint)

    if explore_structure and covered < initial and fresh_allocations:
        return high_clock_pick(fresh_allocations, 'capacity_structure_probe')
    # Before tuning an infeasible neighborhood, cover the reference allocation.
    # A historical Max sets SLOs but is not a new search observation. Selecting
    # another point in its allocation remains an ordinary charged search trial.
    if covered >= initial and not best_measured(observations, settings['limits']):
        reference_neighbors = []
        if allocation(reference) not in seen_allocations:
            for c in groups.get(digest(structure(reference)), []):
                changes = [k for k in KNOBS if c['knobs'][k] != reference['knobs'][k]]
                if len(changes) != 1 or counts[c['id']] or expected_cost(c, current, options) > remaining_gpu_hours:
                    continue
                k = changes[0]
                levels = sorted({point['knobs'][k] for point in groups[digest(structure(reference))]})
                if abs(levels.index(c['knobs'][k]) - levels.index(reference['knobs'][k])) == 1:
                    reference_neighbors.append(c)
        if reference_neighbors:
            return high_clock_pick(reference_neighbors, 'capacity_reference_neighborhood_probe')
    # Spend the first post-coverage and periodic exploration slots on a legal
    # nearest legal contraction of either role before unrelated allocations. Preserve
    # operating controls so this measurement isolates the allocation change.
    # Parent results rank probes only; they never become child observations.
    if explore_structure and covered >= initial and (tuning_attempts == 0 or periodic):
        parents = sorted(
            (o for o in observations if o['candidate_id'] in by_id
             and violation(o, settings['limits']) <= 1.1),
            key=lambda o: (o['metrics']['energy_j'], violation(o, settings['limits']), o['candidate_id']))
        reductions = []
        for parent_index, parent_obs in enumerate(parents):
            parent = by_id[parent_obs['candidate_id']]
            pa, pe = allocation(parent)
            pt = parent['topology']
            # Coverage belongs to the topology and threshold profile together.
            # A trial at another profile does not cover this child structure.
            for key, points in groups.items():
                if key in evidence:
                    continue
                for child in points:
                    ca, ce = allocation(child)
                    ct = child['topology']
                    if (not ((ca < pa and ce == pe) or (ca == pa and ce < pe))
                            or child['microbatches'] != parent['microbatches']
                            or child['knobs'] != parent['knobs']
                            or any(child.get(k, default) != parent.get(k, default)
                                   for k, default in (('dbo_decode_token_threshold', 2),
                                                      ('dbo_prefill_token_threshold', 12)))
                            or any(ct.get(k) != pt.get(k) for k in
                                   ('attention_tp', 'expert_tp', 'parallelism_semantics'))
                            or counts[child['id']]
                            or expected_cost(child, current, options) > remaining_gpu_hours):
                        continue
                    reductions.append(((risk(child), parent_index, (pa-ca)+(pe-ce), ca, ce, child['id']),
                                       child, parent))
        if reductions:
            _, child, parent = min(reductions, key=lambda item: item[0])
            cost = expected_cost(child, current, options)
            return child, dict(reason='capacity_neighbor_reduction_probe',
                               parent_candidate_id=parent['id'],
                               removed_role='attention' if allocation(child)[0] < allocation(parent)[0] else 'expert',
                               removed_gpu_count=sum(allocation(parent))-sum(allocation(child)),
                               operating_controls_preserved=True,
                               capacity=assessment(child, profile), feedback=hint,
                               allocations_covered=covered, initial_allocation_target=initial,
                               expected_gpu_hours=cost, base_gpu_hours=cost)
    if explore_structure and periodic and unseen:
        return high_clock_pick(fresh_allocations or unseen, 'capacity_structure_audit')

    promising = []
    for key, cs in groups.items():
        obs = evidence.get(key, [])
        if not obs:
            continue
        best = best_measured(obs, settings['limits'])
        successful = [o for o in obs if o['status'] == 'ok']
        if best:
            promising.append((0, best['metrics']['energy_j'], key, best['candidate_id']))
        elif successful:
            closest = min(successful, key=lambda o: violation(o, settings['limits']))
            if violation(closest, settings['limits']) <= 1.1:
                promising.append((1, violation(closest, settings['limits']), key, closest['candidate_id']))
    if not promising and unseen:
        return high_clock_pick(fresh_allocations or unseen, 'capacity_feasibility_expansion')
    promising.sort()
    # Preserve trials for a second promising allocation instead of spending all
    # remaining budget on the first feasible result.
    if len(promising) > 1 and (len(observations) - initial) % 3 == 2:
        promising[0], promising[1] = promising[1], promising[0]
    for _, _, key, anchor_id in promising:
        cs = groups[key]
        local = copy.deepcopy(settings)
        local['bo'].update(exploration_policy='legacy', initial_parameter_probes=0,
                           initial_joint_probes=0, analytical_probe_every=0)
        local['reference_candidate_id'] = anchor_id
        anchor = by_id[anchor_id]
        local_ids = {c['id'] for c in cs if counts[c['id']] < options['max_repeats']
                     and expected_cost(c, current, options) <= remaining_gpu_hours}
        # Prefer fresh points; repeats are allowed only after local exhaustion.
        fresh = {cid for cid in local_ids if not counts[cid]}
        local_ids = fresh or local_ids
        # The first tuning slot and every third thereafter explicitly explore
        # joint reductions. A failed probe consumes the slot and budget normally.
        if (best_measured(evidence[key], settings['limits']) is not None
                and tuning_attempts % options.get('capacity_joint_probe_every', 3) == 0):
            probe = joint_probe(cs, local_ids, anchor, evidence[key])
            if probe is not None:
                cost = expected_cost(probe, current, options)
                return probe, dict(reason='capacity_joint_knob_probe', design='normalized_rank_maximin',
                    anchor_candidate_id=anchor_id, structure_key=key,
                    changed_controls=[k for k in KNOBS if probe['knobs'][k] != anchor['knobs'][k]],
                    feasibility_assumed=False, capacity=assessment(probe, profile),
                    expected_gpu_hours=cost, base_gpu_hours=cost,
                    structure_attempts=structure_attempts, structure_budget=structure_budget)
        # BO can propose any legal joint or nonadjacent intervention within this
        # measured structure. Observations from other structures stay excluded.
        c, decision = local_propose(cs, evidence[key], local, local_ids,
                                   remaining_gpu_hours, _local=True, _current=current)
        if c is not None:
            return c, dict(**{k: v for k, v in decision.items() if k != 'reason'},
                           reason='capacity_local_bo', local_reason=decision['reason'],
                           capacity=assessment(c, profile), feedback=hint, structure_key=key,
                           anchor_candidate_id=anchor_id, local_neighbors=False,
                           structure_attempts=structure_attempts, structure_budget=structure_budget)
    if unseen and explore_structure:
        return high_clock_pick(fresh_allocations or unseen, 'capacity_structure_audit')
    return None, {'reason': 'no_affordable_promising_candidate'}

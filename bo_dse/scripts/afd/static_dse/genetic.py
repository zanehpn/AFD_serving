"""Resumable steady-state GA over the same finite legal configuration space.

Only this arm's measured observations define fitness. Topology and paired DBO
thresholds are atomic genes; repair selects an untried, affordable legal point.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .space import KNOBS, digest

DEFAULT_GA = dict(ga_population_size=4, ga_tournament_size=2,
                  ga_crossover_rate=.9, ga_mutation_rate=.2)
GENES = ('topology', 'execution', 'dbo_threshold_profile', *KNOBS)


def validate_options(options):
    for key in ('ga_population_size', 'ga_tournament_size'):
        value = options[key]
        if type(value) is not int or value < 2:
            raise ValueError(f'{key} must be an integer >= 2')
    if options['ga_tournament_size'] > options['ga_population_size']:
        raise ValueError('GA tournament size exceeds population size')
    for key in ('ga_crossover_rate', 'ga_mutation_rate'):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key} must be a finite probability')
    if options.get('use_model_prior') or options.get('model_screening'):
        raise ValueError('GA baseline requires use_model_prior=false and model_screening=false')


def chromosome(candidate):
    return (digest(candidate['topology']),
            (candidate.get('execution_mode', 'eager'), candidate['microbatches']),
            (candidate.get('dbo_decode_token_threshold'), candidate.get('dbo_prefill_token_threshold')),
            *(candidate['knobs'][key] for key in KNOBS))


def fitness(rows, limits):
    """Constraint domination: feasible energy, then violation, then failures."""
    keys = ('energy_j', 'ttft_ms', 'tpot_ms', 'output_tps') + (('tbt_ms',) if 'tbt_ms' in limits else ())
    if not rows or any(row.get('status') != 'ok' or row.get('telemetry_valid') is False for row in rows):
        return (2, math.inf, math.inf)
    if any(any(isinstance(row.get('metrics', {}).get(k), bool)
                   or not isinstance(row.get('metrics', {}).get(k), (int, float))
                   or not math.isfinite(row['metrics'][k]) or row['metrics'][k] <= 0 for k in keys) for row in rows):
        return (2, math.inf, math.inf)
    metrics = {k: sum(row['metrics'][k] for row in rows)/len(rows) for k in keys}
    violation = sum(max(0., (limit/metrics['output_tps'] if k == 'min_output_tps' else metrics[k]/limit)-1)
                    for k, limit in limits.items())
    if violation == 0:
        return (0, metrics['energy_j'], 0.)
    return (1, violation, metrics['energy_j'])


def propose(candidates, observations, settings, eligible_ids, remaining_gpu_hours):
    # Import lazily to avoid the optimizer dispatch cycle.
    from .optimizer import expected_cost
    options = {**DEFAULT_GA, **settings['bo']}
    validate_options(options)
    all_points = {c['id']: c for c in candidates}
    history = defaultdict(list)
    for row in observations:
        if row['candidate_id'] not in all_points:
            raise ValueError('GA observation is outside this arm candidate space')
        history[row['candidate_id']].append(row)
    current = all_points[observations[-1]['candidate_id']] if observations else None
    costs = {cid: expected_cost(c, current, options) for cid,c in all_points.items()}
    legal = [c for cid,c in sorted(all_points.items()) if cid in eligible_ids]
    pool = [c for c in legal if c['id'] not in history and costs[c['id']] <= remaining_gpu_hours]
    if not pool:
        return None, dict(reason='ga_no_untried_affordable_candidate')
    rng = random.Random(f"ga-v1:{options['seed']}:{len(observations)}")

    def decision(candidate, reason, **extra):
        return candidate, dict(reason=reason, algorithm='steady_state_ga_v1',
                               expected_gpu_hours=costs[candidate['id']], base_gpu_hours=costs[candidate['id']],
                               evaluated_candidates=len(history), **extra)

    reference = settings['reference_candidate_id']
    if not observations and reference in eligible_ids:
        if costs[reference] > remaining_gpu_hours:
            return None, dict(reason='reference_cost_exceeds_remaining_budget')
        return decision(all_points[reference], 'measure_reference', ga_phase='initial_population')
    if len(history) < options['ga_population_size']:
        return decision(rng.choice(pool), 'ga_initial_population')

    ranking = {cid: fitness(rows, settings['limits']) for cid, rows in history.items()}
    population = sorted(history, key=lambda cid: (ranking[cid], cid))[:options['ga_population_size']]
    def tournament(exclude=None):
        ids = [cid for cid in population if cid != exclude]
        sample = rng.sample(ids, min(options['ga_tournament_size'], len(ids)))
        return min(sample, key=lambda cid: (ranking[cid], cid))
    parent_a = tournament()
    parent_b = tournament(parent_a)
    genes_a, genes_b = chromosome(all_points[parent_a]), chromosome(all_points[parent_b])
    crossed = rng.random() < options['ga_crossover_rate']
    child = [rng.choice((a,b)) if crossed else a for a,b in zip(genes_a, genes_b)]
    encodings = {c['id']: chromosome(c) for c in legal}
    levels = [sorted({genes[i] for genes in encodings.values()}, key=repr) for i in range(len(GENES))]
    mutated = []
    for i, values in enumerate(levels):
        alternatives = [v for v in values if v != child[i]]
        if alternatives and rng.random() < options['ga_mutation_rate']:
            child[i] = rng.choice(alternatives)
            mutated.append(GENES[i])
    # Repair sparse or previously measured offspring against the public legal
    # space. Hamming distance avoids treating GPU IDs or threshold pairs as
    # ordinal values; fitness is never consulted for unmeasured candidates.
    distances = [(sum(a != b for a,b in zip(encodings[c['id']], child)), c) for c in pool]
    nearest = min(distance for distance,c in distances)
    selected = rng.choice([c for distance,c in distances if distance == nearest])
    return decision(selected, 'ga_offspring', population=population, parent_candidate_ids=[parent_a,parent_b],
                    crossover_applied=crossed, mutated_genes=mutated, repair_gene_distance=nearest,
                    repaired=bool(nearest), fitness_rule='feasible_energy_then_normalized_violation_then_failure',
                    population_reconstructed_from_own_observations=True)

import copy
import math

import pytest
from test_broad_exploration import fixture, obs
from static_dse import hierarchical, optimizer
from static_dse.space import digest, structure


def case():
    cs, ref, settings = fixture()
    settings['bo']['exploration_policy'] = 'hierarchical_v3'
    settings['default_energy_j'] = 1000
    for c in cs:
        c['prior'].update(log_energy=math.log(1000), energy_sd=.5)
    def measured(c, **extra):
        row = obs(c, cost={'gpu_hours': .1}, proposal_expected_gpu_hours=.1, **extra)
        row['metrics']['energy_j'] = 1000
        return row
    return cs, ref, settings, measured


def test_reference_first_and_hard_budget():
    cs, ref, s, _ = case()
    c, d = optimizer.propose(cs, [], s, {c['id'] for c in cs}, 8)
    assert c['id'] == ref['id'] and d['reason'] == 'measure_reference'
    c, _ = optimizer.propose(cs, [], s, {c['id'] for c in cs}, 0)
    assert c is None


def test_random_covers_unseen_structure_and_is_reproducible():
    cs, ref, s, measured = case()
    observations = [measured(ref), measured(ref)]
    a = optimizer.propose(cs, observations, s, {c['id'] for c in cs}, 8)
    b = optimizer.propose(cs, observations, s, {c['id'] for c in cs}, 8)
    assert a == b
    assert a[1]['reason'] == 'hierarchical_random'
    assert structure(a[0]) != structure(ref)
    assert a[0]['id'] not in {o['candidate_id'] for o in observations}


def test_local_bo_receives_only_selected_structure_and_its_history(monkeypatch):
    cs, ref, s, measured = case()
    other = next(c for c in cs if structure(c) != structure(ref))
    observations = [measured(ref), measured(other, proposal_reason='hierarchical_random')]
    for c in cs:
        if structure(c) == structure(ref):
            c['prior']['log_energy'] = math.log(10)
    # Calibration residual cancels the low prior at measured points; create a
    # real knob-dependent predicted improvement in the selected structure.
    ref['prior']['log_energy'] = math.log(1000)
    calls = []
    def local(members, rows, settings, eligible, budget, **kw):
        assert len({digest(structure(c)) for c in members}) == 1
        ids = {c['id'] for c in members}
        assert all(o['candidate_id'] in ids for o in rows)
        assert kw['_current'] == other
        calls.append((members, rows))
        return next(c for c in members if c['id'] in eligible), {'reason':'find_feasible','expected_gpu_hours':.1,'base_gpu_hours':.1}
    monkeypatch.setattr(optimizer, 'propose', local)
    candidate, decision = hierarchical.propose(cs, observations, s, {c['id'] for c in cs}, 8)
    assert calls and structure(candidate) == structure(ref)
    assert decision['reason'] == 'hierarchical_local_bo'


def test_unmeasured_structure_local_bo_and_exhausted_observation_retained():
    cs, ref, s, measured = case()
    other = [c for c in cs if structure(c) != structure(ref)]
    for c in other:
        c['prior']['log_energy'] = math.log(1)
    candidate, decision = optimizer.propose(cs, [measured(ref)], s, {c['id'] for c in cs}, 8)
    assert structure(candidate) != structure(ref)
    assert decision['reason'] == 'hierarchical_local_bo'
    assert decision['local_observations'] == 0
    point = other[0]
    rows = [measured(point), measured(point)]
    local_s = copy.deepcopy(s)
    local_s['bo'].update(exploration_policy='legacy', initial_parameter_probes=0, initial_joint_probes=0, analytical_probe_every=0)
    candidate, decision = optimizer.propose(other, rows, local_s, {c['id'] for c in other if c != point}, 8, _local=True, _current=ref)
    assert candidate != point
    assert decision['reason'] != 'find_feasible'  # retained data supplies feasible incumbent


def test_failures_not_sampled_and_baselines_do_not_enable_hierarchy():
    cs, ref, s, measured = case()
    observations = [measured(ref), measured(ref)]
    allowed = next(c for c in cs if structure(c) != structure(ref))
    failed = next(c for c in cs if c != allowed and structure(c) != structure(ref))
    observations.append(dict(candidate_id=failed['id'], status='runtime_incompatible'))
    candidate, _ = optimizer.propose(cs, observations, s, {allowed['id'], failed['id']}, 8)
    assert candidate == allowed
    for method, prior in [('random', True), ('bo', False)]:
        options = {**s['bo'], 'method':method, 'use_model_prior':prior}
        assert not hierarchical.enabled(options)

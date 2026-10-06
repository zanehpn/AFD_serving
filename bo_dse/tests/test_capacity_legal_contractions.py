"""Regressions for sparse legal topology grids, without other arms' results."""
import copy

from test_capacity_v2 import fixture, observe, pick
from static_dse import capacity_search as policy
from static_dse.space import enumerate_candidates, digest, structure


def scenario():
    _, _, settings = fixture()
    layouts = [(1, 1), (4, 4), (4, 2), (4, 1)]
    cs = enumerate_candidates(dict(selection_split='calibration',
        topologies=[dict(attention_gpus=list(range(a)), expert_gpus=list(range(a, a+e)),
                        attention_dp=a, attention_tp=1, expert_dp=e, expert_tp=1,
                        expert_ep=e) for a, e in layouts],
        attention_frequencies_mhz=[1560, 2100], expert_frequencies_mhz=[1560, 2100],
        attention_power_caps_w=[200, 300], expert_power_caps_w=[200, 300],
        microbatches=[2], dbo_threshold_profiles=[[2, 12], [32, 512], [64, 1024]]))
    for c in cs:
        c['prior'] = dict(log_energy=9., energy_sd=1., source='uncovered_neutral')
    def high(layout):
        return next(c for c in cs if policy.allocation(c) == layout
                    and c['dbo_decode_token_threshold'] == 32
                    and list(c['knobs'].values()) == [2100, 2100, 300, 300])
    ref = high((4, 4))
    settings['reference_candidate_id'] = ref['id']
    d = dict(reason='capacity_structure_probe', expected_gpu_hours=.1, base_gpu_hours=.1)
    obs = [observe(high((1, 1)), d, good=False), observe(ref, d)]
    # A threshold measurement must not count as resource-knob exploration.
    threshold = next(c for c in cs if c['topology'] == ref['topology']
                     and c['knobs'] == ref['knobs'] and c['dbo_decode_token_threshold'] == 64)
    obs.append(observe(threshold, {**d, 'reason': 'capacity_threshold_probe'}, energy=6000))
    return cs, settings, obs


def test_sparse_grid_reaches_smaller_expert_world_without_illegal_intermediate():
    cs, settings, obs = scenario()
    original = copy.deepcopy(obs)
    for layout, removed in [((4, 2), 2), ((4, 1), 1)]:
        child, decision = pick(cs, obs, settings)
        assert decision['reason'] == 'capacity_neighbor_reduction_probe'
        assert policy.allocation(child) == layout
        assert decision['removed_gpu_count'] == removed
        parent = next(c for c in cs if c['id'] == decision['parent_candidate_id'])
        assert child['knobs'] == parent['knobs']
        assert child['dbo_decode_token_threshold'] == parent['dbo_decode_token_threshold'] == 32
        assert pick(list(reversed(cs)), obs, settings) == (child, decision)
        obs.append(observe(child, decision, energy=4000 if removed == 2 else 3000))
    assert obs[:3] == original


def test_contraction_obeys_capacity_eligibility_and_cost():
    cs, settings, obs = scenario()
    ids = {c['id'] for c in cs if policy.allocation(c) != (4, 2)}
    assert policy.allocation(pick(cs, obs, settings, ids=ids)[0]) == (4, 1)
    assert pick(cs, obs, settings, hours=1e-10)[0] is None
    settings['capacity_profile'] = {'structures': {
        digest(structure(c)): {'status': 'proven_impossible'} for c in cs
        if policy.allocation(c) == (4, 2)}}
    assert policy.allocation(pick(cs, obs, settings)[0]) == (4, 1)


def test_initial_probe_matches_actual_reference_threshold_not_legacy_constants():
    cs, settings, _ = scenario()
    c, _ = pick(cs, [], settings)
    assert (c['dbo_decode_token_threshold'], c['dbo_prefill_token_threshold']) == (32, 512)

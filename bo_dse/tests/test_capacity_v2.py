import copy
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/afd'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from static_dse import capacity, capacity_search
from static_dse.campaign import DEFAULT_BO
from static_dse.optimizer import propose
from static_dse.space import enumerate_candidates, digest, structure


def fixture():
    topologies = []
    for na, ne in [(1, 1), (2, 1), (3, 1), (2, 2)]:
        for at in ([1, 2] if na == 2 else [1]):
            topologies.append(dict(attention_gpus=list(range(na)), expert_gpus=list(range(na, na+ne)),
                attention_dp=na//at, attention_tp=at, expert_dp=ne, expert_tp=1, expert_ep=ne))
    cs = enumerate_candidates(dict(selection_split='calibration', topologies=topologies,
        attention_frequencies_mhz=[1050, 1290, 1410], expert_frequencies_mhz=[1050, 1290, 1410],
        attention_power_caps_w=[200, 300, 400], expert_power_caps_w=[200, 300, 400], microbatches=[2]))
    for c in cs:
        c['prior'] = dict(log_energy=9., energy_sd=1., source='uncovered_neutral')
    ref = next(c for c in cs if capacity_search.allocation(c) == (2, 2) and c['topology']['attention_tp'] == 1
               and list(c['knobs'].values()) == [1410, 1410, 400, 400])
    settings = dict(bo={**DEFAULT_BO, 'exploration_policy': 'capacity_v2'},
        reference_candidate_id=ref['id'], default_energy_j=8000., mechanism_model='external_power_duration_v1',
        limits=dict(ttft_ms=100, tpot_ms=10, tbt_ms=10, min_output_tps=100),
        budget=dict(evaluations=17, gpu_hours=32), setup_cost=dict(evaluations=1))
    return cs, ref, settings


def observe(c, decision, *, good=True, energy=5000, role='attention'):
    return dict(candidate_id=c['id'], status='ok', proposal_reason=decision['reason'],
        proposal_expected_gpu_hours=decision['expected_gpu_hours'], proposal_base_gpu_hours=decision['base_gpu_hours'],
        cost={'gpu_hours': .1}, metrics=dict(energy_j=energy, ttft_ms=50 if good else 200,
                                           tpot_ms=5, tbt_ms=5, output_tps=200 if good else 50),
        external_observables=dict(role_mean_gpu_utilization_pct={role: 95, 'expert' if role == 'attention' else 'attention': 10},
                                 client_queue_lag_ms_p90=1))


def pick(cs, obs, settings, hours=32, ids=None):
    return propose(cs, obs, settings, {c['id'] for c in cs} if ids is None else ids, hours)


def test_short_budget_starts_with_two_allocations_then_joint_tuning():
    cs, ref, settings = fixture()
    obs, layouts = [], []
    original_limits = copy.deepcopy(settings['limits'])
    for n in range(2):
        c, d = pick(cs, obs, settings)
        layouts.append(capacity_search.allocation(c))
        assert d['reason'] == 'capacity_structure_probe'
        assert c['knobs'] == ref['knobs']
        assert c['topology']['attention_tp'] == 1
        obs.append(observe(c, d, good=n > 0))
    assert layouts[0] == (1, 1)
    assert len(set(layouts)) == 2
    assert settings['limits'] == original_limits
    c, d = pick(cs, obs, settings)
    if d['reason'] == 'capacity_neighbor_reduction_probe':
        obs.append(observe(c,d,energy=4000))
        c,d=pick(cs,obs,settings)
    assert d['reason'] == 'capacity_joint_knob_probe'
    assert sum(c['knobs'][k] != ref['knobs'][k] for k in ref['knobs']) >= 2


def test_client_lag_is_not_mislabelled_as_server_queue_or_stage_time():
    cs, _, settings = fixture()
    c, d = pick(cs, [], settings)
    o = observe(c, d, good=False)
    hint = capacity_search.feedback(o, settings['limits'])
    assert hint['attention'] > hint['expert']
    o['external_observables']['client_queue_lag_ms_p90'] = 1000
    assert capacity_search.feedback(o, settings['limits'])['attention'] < hint['attention']


@pytest.mark.parametrize('parent_layout,child_layout', [((7, 1), (6, 1)), ((1, 7), (1, 6)), ((3, 1), (2, 1))])
def test_neighbor_reduction_preserves_controls_then_tunes_own_measurement(parent_layout, child_layout):
    _, _, settings = fixture()
    settings['bo']['capacity_initial_structures'] = 1
    topologies = [dict(attention_gpus=list(range(a)), expert_gpus=list(range(a, a+e)),
                      attention_dp=a, attention_tp=1, expert_dp=e, expert_tp=1, expert_ep=e)
                  for a, e in (parent_layout, child_layout)]
    cs = enumerate_candidates(dict(selection_split='calibration', topologies=topologies,
        enumerate_parallelism=False, attention_frequencies_mhz=[1290, 1410],
        expert_frequencies_mhz=[1290, 1410], attention_power_caps_w=[350, 400],
        expert_power_caps_w=[350, 400], microbatches=[2]))
    for c in cs:
        c['prior'] = dict(log_energy=9., energy_sd=1., source='uncovered_neutral')
    parent = next(c for c in cs if capacity_search.allocation(c) == parent_layout
                  and c['knobs'] == dict(attention_mhz=1410, expert_mhz=1410,
                                         attention_power_w=400, expert_power_w=350))
    settings['reference_candidate_id'] = parent['id']
    obs = [observe(parent, dict(reason='capacity_structure_probe', expected_gpu_hours=.1, base_gpu_hours=.1))]
    # A near-SLO parent is sufficient; no assumption that fewer cards pass SLO.
    obs[0]['metrics']['ttft_ms'] = 105
    before = copy.deepcopy(obs)
    child, decision = pick(cs, obs, settings)
    assert capacity_search.allocation(child) == child_layout
    assert decision['reason'] == 'capacity_neighbor_reduction_probe'
    assert decision['parent_candidate_id'] == parent['id']
    assert child['knobs'] == parent['knobs'] and child['microbatches'] == parent['microbatches']
    assert obs == before
    parent_ids = {c['id'] for c in cs if capacity_search.allocation(c) == parent_layout}
    assert pick(cs, obs, settings, ids=parent_ids)[0]['id'] in parent_ids
    assert pick(cs, obs, settings, hours=1e-10)[0] is None
    blocked = copy.deepcopy(settings)
    blocked['capacity_profile'] = {'structures': {
        digest(structure(c)): {'status': 'proven_impossible'} for c in cs
        if capacity_search.allocation(c) == child_layout}}
    assert pick(cs, obs, blocked)[0]['id'] in parent_ids
    # Once measured, the child is tuned around its own measured anchor.
    obs.append(observe(child, decision, energy=4000))
    tuned, detail = pick(cs, obs, settings)
    assert detail['reason'] == 'capacity_joint_knob_probe'
    assert detail['anchor_candidate_id'] == child['id']
    assert capacity_search.allocation(tuned) == child_layout
    assert sum(tuned['knobs'][k] != child['knobs'][k] for k in child['knobs']) >= 2


def test_far_from_slo_parent_does_not_trigger_reduction():
    cs, ref, settings = fixture()
    settings['bo']['capacity_initial_structures'] = 1
    obs = [observe(ref, dict(reason='capacity_structure_probe', expected_gpu_hours=.1, base_gpu_hours=.1), good=False)]
    assert pick(cs, obs, settings)[1]['reason'] != 'capacity_neighbor_reduction_probe'


def test_missing_optional_telemetry_keeps_policy_usable():
    cs, _, settings = fixture()
    c, d = pick(cs, [], settings)
    o = observe(c, d, good=False)
    o['external_observables'] = {'client_queue_lag_ms_p90': None}
    assert pick(cs, [o], settings)[0] is not None


def test_ttft_only_violation_tracks_expert_pressure_even_with_enough_throughput():
    cs, _, settings = fixture()
    c, d = pick(cs, [], settings)
    o = observe(c, d, good=True)
    o['metrics']['ttft_ms'] = 105
    o['external_observables']['role_mean_gpu_utilization_pct'] = {'attention': 58, 'expert': 99}
    hint = capacity_search.feedback(o, settings['limits'])
    assert hint['expert'] > 10 * hint['attention'] > 0
    selected, _ = pick(cs, [o], settings)
    assert capacity_search.allocation(selected) == (2, 2)
    o['external_observables']['role_mean_gpu_utilization_pct'] = {'attention': 99, 'expert': 58}
    selected, _ = pick(cs, [o], settings)
    assert capacity_search.allocation(selected) == (3, 1)


@pytest.mark.parametrize('metric', ['ttft_ms', 'tpot_ms', 'tbt_ms'])
def test_latency_metric_alone_does_not_identify_a_role(metric):
    cs, _, settings = fixture()
    o = observe(*pick(cs, [], settings))
    o['metrics'][metric] = settings['limits'][metric] * 1.1
    o['external_observables'] = {}
    h = capacity_search.feedback(o, settings['limits'])
    assert h['attention'] == h['expert'] > 0
    assert h['role_timing_ms'] is None and h['communication_fraction'] is None
    assert h['directional_evidence'] is False


@pytest.mark.parametrize('invalid', [None, float('nan'), float('inf'), -1, 101, '99', True])
def test_partial_or_invalid_utilization_remains_neutral(invalid):
    cs, _, settings = fixture()
    o = observe(*pick(cs, [], settings), good=False)
    o['external_observables']['role_mean_gpu_utilization_pct'] = {'attention': invalid, 'expert': 99}
    h = capacity_search.feedback(o, settings['limits'])
    assert h['attention'] == h['expert'] > 0


def test_measured_peer_wait_points_to_peer_and_communication_stays_unattributed():
    cs, _, settings = fixture()
    o = observe(*pick(cs, [], settings))
    o['metrics']['ttft_ms'] = 105
    timing = {r: dict(compute_ms=1, queue_wait_ms=0, peer_wait_ms=0, communication_ms=0)
              for r in ('attention', 'expert')}
    timing['attention']['peer_wait_ms'] = 20
    o['external_observables'] = dict(role_timing_ms=timing)
    h = capacity_search.feedback(o, settings['limits'])
    assert h['attention'] == h['expert']  # Unproven timing is ignored.
    o['external_observables']['role_timing_provenance'] = 'measured_same_window'
    h = capacity_search.feedback(o, settings['limits'])
    assert h['expert'] > h['attention'] and h['directional_evidence']
    timing['expert']['communication_ms'] = 100
    h = capacity_search.feedback(o, settings['limits'])
    assert h['communication_fraction'] > .5
    assert h['attention'] == h['expert'] and not h['directional_evidence']


def test_infeasible_local_search_first_covers_reference_neighbor_without_remeasuring_max():
    from official_space import structures
    cs = enumerate_candidates(dict(selection_split='calibration', topologies=structures(list(range(8)), 'qwen36'),
        enumerate_parallelism=False, attention_frequencies_mhz=[1350, 1410], expert_frequencies_mhz=[1350, 1410],
        attention_power_caps_w=[350, 400], expert_power_caps_w=[350, 400], microbatches=[2]))
    for c in cs:
        c['prior'] = dict(log_energy=9., energy_sd=1., source='uncovered_neutral')
    def point(alloc):
        return next(c for c in cs if capacity_search.allocation(c) == alloc
                    and c['topology']['attention_tp'] == c['topology']['expert_tp'] == 1
                    and list(c['knobs'].values()) == [1410, 1410, 400, 400])
    ref = point((4, 4))
    _, _, settings = fixture()
    settings['reference_candidate_id'] = ref['id']
    settings['reference_evidence_origin'] = 'historical_max_reuse'
    obs = []
    for alloc in [(1, 1), (7, 1), (6, 2), (6, 1)]:
        o = observe(point(alloc), dict(reason='capacity_structure_probe', expected_gpu_hours=.1, base_gpu_hours=.1))
        o['metrics']['ttft_ms'] = 105
        obs.append(o)
    before = copy.deepcopy(obs)
    c, d = pick(cs, obs, settings)
    assert capacity_search.allocation(c) == (4, 4)
    assert c['id'] != ref['id']
    assert sum(c['knobs'][k] != ref['knobs'][k] for k in ref['knobs']) == 1
    assert d['reason'] == 'capacity_reference_neighborhood_probe'
    assert not d['uniform_high_operating_point']
    assert obs == before
    # The extra exploration must still respect physical eligibility and budget.
    eligible = {c['id'] for c in cs if capacity_search.allocation(c) != (4, 4)}
    assert capacity_search.allocation(pick(cs, obs, settings, ids=eligible)[0]) != (4, 4)
    assert pick(cs, obs, settings, hours=1e-10)[0] is None

    # Also avoid remeasuring inherited Max during the initial exploration.
    expert_bound = copy.deepcopy(obs[0])
    expert_bound['external_observables']['role_mean_gpu_utilization_pct'] = {'attention': 10, 'expert': 99}
    c, d = pick(cs, [expert_bound], settings)
    assert capacity_search.allocation(c) == (4, 4)
    assert c['id'] != ref['id']
    assert not d['uniform_high_operating_point']


def test_unavailable_or_unaffordable_structure_never_selected():
    cs, _, settings = fixture()
    ids = {c['id'] for c in cs if capacity_search.allocation(c) != (1, 1)}
    c, _ = pick(cs, [], settings, ids=ids)
    assert capacity_search.allocation(c) == (2, 1)
    c, d = pick(cs, [], settings, hours=1e-10)
    assert c is None


def test_periodic_structure_exploration_survives_local_tuning():
    cs, _, settings = fixture()
    obs = []
    for n in range(8):
        c, d = pick(cs, obs, settings)
        assert c is not None
        if n == 7:
            assert d['reason'] == 'capacity_structure_audit'
        obs.append(observe(c, d, energy=5000 + n*10))


@pytest.mark.parametrize('method', ['bo', 'random'])
def test_baselines_unchanged_by_capacity_policy(method):
    cs, _, settings = fixture()
    settings['bo'].update(method=method, use_model_prior=False, model_screening=False)
    first = pick(cs, [], settings)
    obs = [observe(*first)]
    with_policy = pick(cs, obs, settings)
    settings['bo']['exploration_policy'] = 'legacy'
    assert pick(cs, obs, settings) == with_policy


def checkpoint(tmp_path, expert_bytes=20*capacity.MIB):
    cfg = dict(model_type='deepseek_v2', num_hidden_layers=1, num_attention_heads=2,
               num_key_value_heads=2, hidden_size=8, kv_lora_rank=4, qk_rope_head_dim=2)
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    names = {'model.layers.0.self_attn.q_proj.weight': 2*capacity.MIB,
             'model.layers.1.mlp.experts.0.up_proj.weight': expert_bytes}
    header, offset = {}, 0
    for name, size in names.items():
        header[name] = dict(dtype='BF16', shape=[size//2], data_offsets=[offset, offset+size])
        offset += size
    data = json.dumps(header).encode()
    with (tmp_path / 'model.safetensors').open('wb') as stream:
        stream.write(struct.pack('<Q', len(data)) + data)
        stream.truncate(8 + len(data) + offset)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {n: 'model.safetensors' for n in names}}))


def test_capacity_uses_all_expert_weights_and_ep_world_once(tmp_path):
    checkpoint(tmp_path)
    cs, _, settings = fixture()
    hw = {'devices': {str(i): {'memory_mib': 15} for i in range(4)}}
    profile = capacity.build_profile(tmp_path, cs, hw, max_model_len=1, max_num_seqs=1, workspace_mib=0)
    settings['capacity_profile'] = profile
    c, _ = pick(cs, [], settings)
    assert capacity_search.allocation(c) == (2, 2)
    estimate = capacity.assessment(c, profile)
    assert estimate['status'] == 'estimated_fit'
    assert estimate['per_gpu']['2']['estimated_mib'] == pytest.approx(10)
    assert capacity.assessment(next(c for c in cs if capacity_search.allocation(c) == (1, 1)), profile)['status'] == 'proven_impossible'


def test_capacity_estimate_does_not_hard_reject_workspace_uncertainty(tmp_path):
    checkpoint(tmp_path, expert_bytes=2*capacity.MIB)
    cs, _, _ = fixture()
    hw = {'devices': {str(i): {'memory_mib': 15} for i in range(4)}}
    profile = capacity.build_profile(tmp_path, cs, hw, workspace_mib=100)
    assert {x['status'] for x in profile['structures'].values()} == {'uncertain'}


def test_unknown_model_retains_smallest_structure(tmp_path):
    (tmp_path / 'config.json').write_text('{"model_type":"future_architecture"}')
    cs, _, settings = fixture()
    settings['capacity_profile'] = capacity.build_profile(tmp_path, cs, {'devices': {}})
    assert capacity_search.allocation(pick(cs, [], settings)[0]) == (1, 1)


def test_corrupt_checkpoint_headers_fail_explicitly(tmp_path):
    checkpoint(tmp_path)
    (tmp_path / 'model.safetensors').write_bytes(struct.pack('<Q', 10**12))
    with pytest.raises(ValueError, match='header length'):
        capacity.inventory(tmp_path)

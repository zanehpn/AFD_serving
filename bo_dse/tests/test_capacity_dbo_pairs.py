import copy
from test_capacity_v2 import fixture, observe, pick
from static_dse.space import digest, structure


def setup():
    cs, ref, settings = fixture()
    extra = []
    for c in cs:
        other = copy.deepcopy(c)
        other['microbatches'] = 1
        other['id'] = c['id']+'-dbo-off'
        extra.append(other)
    settings['bo']['capacity_pair_dbo'] = True
    return cs+extra, settings


def test_toggle_is_measured_at_identical_controls_without_mutating_parent():
    cs, settings = setup()
    parent, decision = pick(cs, [], settings)
    obs = [observe(parent, decision)]
    original = copy.deepcopy(obs)
    child, detail = pick(cs, obs, settings)
    assert detail['reason'] == 'capacity_dbo_pair_probe'
    assert child['microbatches'] != parent['microbatches']
    assert child['topology'] == parent['topology'] and child['knobs'] == parent['knobs']
    assert obs == original
    obs.append(observe(child, detail, energy=4500))
    following, _ = pick(cs, obs, settings)
    assert following['topology'] != parent['topology']


def test_pair_respects_eligibility_capacity_and_budget():
    cs, settings = setup()
    parent, d = pick(cs, [], settings)
    obs = [observe(parent, d)]
    child, _ = pick(cs, obs, settings)
    allowed = {c['id'] for c in cs if c['microbatches'] == parent['microbatches']}
    assert pick(cs, obs, settings, ids=allowed)[1]['reason'] != 'capacity_dbo_pair_probe'
    assert pick(cs, obs, settings, hours=1e-10)[0] is None
    settings['capacity_profile'] = {'structures': {digest(structure(child)): {'status': 'proven_impossible'}}}
    assert pick(cs, obs, settings)[1]['reason'] != 'capacity_dbo_pair_probe'


def test_single_mode_search_behavior_remains_unchanged():
    cs, _, settings = fixture()
    settings['bo']['capacity_pair_dbo'] = True
    first, d = pick(cs, [], settings)
    _, following = pick(cs, [observe(first, d)], settings)
    assert following['reason'] != 'capacity_dbo_pair_probe'

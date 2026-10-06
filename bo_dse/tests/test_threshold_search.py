import copy
import pytest
from test_official import space
from test_capacity_v2 import fixture, observe, pick
from static_dse.space import enumerate_candidates, configuration, digest


def test_profiles_expand_real_configuration_identity():
    spec = space()
    spec['microbatches'] = [2]
    original = enumerate_candidates(spec)
    spec['dbo_threshold_profiles'] = [[2, 12], [8, 128], [32, 512]]
    expanded = enumerate_candidates(spec)
    assert len(expanded) == 3*len(original)
    assert len({c['id'] for c in expanded}) == len(expanded)
    assert {tuple(configuration(c)[k] for k in ('dbo_decode_token_threshold', 'dbo_prefill_token_threshold'))
            for c in expanded} == {(2, 12), (8, 128), (32, 512)}
    assert all(c['microbatches'] == 2 for c in expanded)


@pytest.mark.parametrize('profiles', [[], [[2,12],[2,12]], [[0,12]], [[2]], [[True,12]]])
def test_invalid_profile_grid_rejected(profiles):
    spec = space(); spec['microbatches'] = [2]; spec['dbo_threshold_profiles'] = profiles
    with pytest.raises(ValueError):
        enumerate_candidates(spec)


def test_threshold_probes_use_new_measurements_and_preserve_controls():
    original, reference, settings = fixture()
    candidates = []
    for pair in [(2,12),(8,128),(32,512)]:
        for c in original:
            c = copy.deepcopy(c)
            c.update(dbo_decode_token_threshold=pair[0], dbo_prefill_token_threshold=pair[1])
            c['id'] = 'dse-'+digest(configuration(c))[:20]
            candidates.append(c)
    ref = next(c for c in candidates if c['topology']==reference['topology']
               and c['knobs']==reference['knobs'] and c['dbo_decode_token_threshold']==2)
    settings['reference_candidate_id'] = ref['id']
    observations = []
    probes = []
    for step in range(10):
        before = copy.deepcopy(observations)
        c, decision = pick(candidates, observations, settings)
        assert observations == before and c is not None
        if decision['reason'] == 'capacity_threshold_probe':
            parent = next(x for x in candidates if x['id']==decision['parent_candidate_id'])
            assert c['topology']==parent['topology'] and c['knobs']==parent['knobs']
            assert not decision['feasibility_assumed']
            assert c['id'] not in {o['candidate_id'] for o in observations}
            probes.append(c)
        observations.append(observe(c, decision, good=step>0, energy=5000+step))
    assert len(probes)==2
    assert {c['dbo_decode_token_threshold'] for c in probes} == {8,32}

import copy
import pytest
from test_capacity_v2 import fixture, observe, pick
from static_dse import capacity_search, optimizer
from static_dse.space import digest, structure
from test_capacity_dbo_pairs import setup as paired_setup


def single_structure():
    cs,ref,settings=fixture()
    cs=[c for c in cs if structure(c)==structure(ref)]
    observation=observe(ref,dict(reason='capacity_structure_probe',expected_gpu_hours=.1,base_gpu_hours=.1))
    return cs,ref,settings,[observation]


def test_first_joint_probe_reaches_low_frequencies_without_winner_hints():
    cs,ref,settings,obs=single_structure()
    before=copy.deepcopy((settings,obs))
    c,d=pick(cs,obs,settings)
    assert d['reason']=='capacity_joint_knob_probe'
    assert c['knobs']['attention_mhz']==c['knobs']['expert_mhz']==1050
    assert len(d['changed_controls'])==4
    assert not d['feasibility_assumed']
    assert (settings,obs)==before
    assert pick(list(reversed(cs)),obs,settings)==(c,d)


def test_failed_joint_probe_is_charged_exploration_and_not_repeated():
    cs,_,settings,obs=single_structure()
    c,d=pick(cs,obs,settings)
    bad=observe(c,d);bad.update(status='failed',metrics=None)
    obs.append(bad)
    following,detail=pick(cs,obs,settings)
    assert following['id']!=c['id']
    assert detail['reason']=='capacity_local_bo'
    assert not detail['local_neighbors']


def test_regular_bo_can_choose_joint_nonadjacent_points(monkeypatch):
    cs,ref,settings,obs=single_structure()
    c,d=pick(cs,obs,settings)
    obs.append(observe(c,d,energy=6000))
    original=optimizer.propose
    target=next(c for c in cs if c['knobs']==dict(attention_mhz=1050,expert_mhz=1290,
        attention_power_w=300,expert_power_w=200))
    def inspect(candidates,observations,local,eligible,hours,**kwargs):
        if kwargs.get('_local'):
            assert target['id'] in eligible
            assert {o['candidate_id'] for o in observations}=={o['candidate_id'] for o in obs}
            return target,dict(reason='test_acquisition',expected_gpu_hours=.1,base_gpu_hours=.1)
        return original(candidates,observations,local,eligible,hours,**kwargs)
    monkeypatch.setattr(optimizer,'propose',inspect)
    assert pick(cs,obs,settings)[0]==target


def test_joint_design_respects_eligibility_capacity_cost_and_failed_locations():
    cs,ref,settings,obs=single_structure()
    c,d=pick(cs,obs,settings)
    allowed={p['id'] for p in cs}-{c['id']}
    assert pick(cs,obs,settings,ids=allowed)[0]['id'] in allowed
    assert pick(cs,obs,settings,hours=1e-10)[0] is None
    blocked=copy.deepcopy(settings)
    blocked['capacity_profile']={'structures':{digest(structure(ref)):{'status':'proven_impossible'}}}
    assert pick(cs,obs,blocked)[0] is None
    following=capacity_search.joint_probe(cs,{p['id'] for p in cs},ref,
        obs+[dict(candidate_id=c['id'],status='failed')])
    assert following is not None and following['id']!=c['id']


def test_sixteen_trials_reserve_at_least_ten_for_knobs_after_early_feasibility():
    cs,ref,settings=fixture();obs=[]
    for n in range(16):
        c,d=pick(cs,obs,settings)
        assert c is not None
        obs.append(observe(c,d,good=n>0,energy=5000+n))
    tuning=sum(o['proposal_reason'] in capacity_search.TUNING_REASONS for o in obs)
    assert tuning>=10
    assert any(o['proposal_reason']=='capacity_joint_knob_probe' for o in obs)
    assert len({o['candidate_id'] for o in obs})==16


def test_no_feasible_result_can_expand_beyond_structure_allowance():
    cs,_,settings=fixture();settings['bo']['capacity_structure_fraction']=.0625
    obs=[]
    for _ in range(5):
        c,d=pick(cs,obs,settings)
        assert c is not None
        obs.append(observe(c,d,good=False))
    assert len({digest(structure(next(c for c in cs if c['id']==o['candidate_id']))) for o in obs})>1


def test_paired_dbo_reserves_ten_knob_trials_in_sixteen_attempts():
    cs,settings=paired_setup();obs=[]
    for n in range(16):
        c,d=pick(cs,obs,settings)
        assert c is not None
        obs.append(observe(c,d,good=n>1,energy=5000+n))
    assert sum(o['proposal_reason'] in capacity_search.TUNING_REASONS for o in obs)>=10
    assert any(o['proposal_reason']=='capacity_dbo_pair_probe' for o in obs)
    assert {next(c['microbatches'] for c in cs if c['id']==o['candidate_id']) for o in obs}=={1,2}

import copy
from test_capacity_v2 import fixture, observe
from static_dse.space import configuration, digest
from static_dse import capacity_search as policy


def scenario():
    points, reference, settings = fixture()
    extra = [copy.deepcopy(c) for c in points if policy.allocation(c)==(2,2) and c['topology']['attention_tp']==1]
    for c in extra:
        c['topology'].update(attention_gpus=[0], expert_gpus=[1,2], attention_dp=1)
    points += extra
    candidates = []
    for pair in ((2,12),(8,128),(32,512)):
        for source in points:
            c = copy.deepcopy(source)
            c.update(dbo_decode_token_threshold=pair[0],dbo_prefill_token_threshold=pair[1])
            c['id']='dse-'+digest(configuration(c))[:20]
            candidates.append(c)
    ref=next(c for c in candidates if c['topology']==reference['topology'] and c['knobs']==reference['knobs'] and c['dbo_decode_token_threshold']==2)
    settings['reference_candidate_id']=ref['id']
    small=next(c for c in candidates if policy.allocation(c)==(1,1) and c['dbo_decode_token_threshold']==2 and c['knobs']==ref['knobs'])
    threshold=next(c for c in candidates if c['topology']==ref['topology'] and c['knobs']==ref['knobs'] and c['dbo_decode_token_threshold']==8)
    obs=[observe(small,dict(reason='capacity_structure_probe',expected_gpu_hours=.1,base_gpu_hours=.1),good=False),
         observe(ref,dict(reason='capacity_structure_probe',expected_gpu_hours=.1,base_gpu_hours=.1),energy=5000),
         observe(threshold,dict(reason='capacity_threshold_probe',expected_gpu_hours=.1,base_gpu_hours=.1),energy=4500)]
    return candidates,settings,obs,threshold


def test_threshold_probe_does_not_cancel_first_resource_contraction():
    candidates,settings,obs,parent=scenario()
    before=copy.deepcopy(obs)
    child,decision=policy.propose(candidates,obs,settings,{c['id'] for c in candidates},60)
    assert decision['reason']=='capacity_neighbor_reduction_probe'
    assert sum(policy.allocation(child))==sum(policy.allocation(parent))-1
    assert child['knobs']==parent['knobs']
    assert (child['dbo_decode_token_threshold'],child['dbo_prefill_token_threshold'])==(8,128)
    assert obs==before


def test_contraction_preserves_threshold_even_with_adversarial_id_order():
    candidates,settings,obs,parent=scenario()
    # Make a different profile sort first; no opaque ID should change the threshold.
    for c in candidates:
        if sum(policy.allocation(c))==3 and c['dbo_decode_token_threshold']!=8:
            c['id']='000-'+c['id']
    child,decision=policy.propose(candidates,obs,settings,{c['id'] for c in candidates},60)
    assert decision['reason']=='capacity_neighbor_reduction_probe'
    assert child['dbo_decode_token_threshold']==parent['dbo_decode_token_threshold']
    assert child['dbo_prefill_token_threshold']==parent['dbo_prefill_token_threshold']


def test_both_one_card_reductions_fit_before_knob_exploration():
    candidates,settings,obs,_=scenario()
    allocations=[]
    for _ in range(2):
        child,decision=policy.propose(candidates,obs,settings,{c['id'] for c in candidates},60)
        assert decision['reason']=='capacity_neighbor_reduction_probe'
        allocations.append(policy.allocation(child))
        failed=observe(child,decision);failed.update(status='failed',metrics=None)
        obs.append(failed)
    assert set(allocations)=={(1,2),(2,1)}

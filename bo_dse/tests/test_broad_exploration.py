import copy
import sys
from pathlib import Path

import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/afd'))
from static_dse import exploration
from static_dse.campaign import DEFAULT_BO, validate_settings
from static_dse.space import enumerate_candidates, configuration


def fixture():
    topologies = [dict(attention_gpus=[0,1], expert_gpus=[2,3], attention_dp=2, attention_tp=1,
                       expert_dp=2, expert_ep=2, expert_tp=1),
                  dict(attention_gpus=[0,1], expert_gpus=[2], attention_dp=2, attention_tp=1,
                       expert_dp=1, expert_ep=1, expert_tp=1)]
    cs = enumerate_candidates(dict(selection_split='calibration', topologies=topologies, enumerate_parallelism=False,
        attention_frequencies_mhz=[1050,1290,1410], expert_frequencies_mhz=[1050,1290,1410],
        attention_power_caps_w=[200,300,400], expert_power_caps_w=[200,300,400],microbatches=[1], execution_mode='eager'))
    for c in cs:c['prior']={'source':'uncovered_neutral'}
    ref = next(c for c in cs if len(c['topology']['expert_gpus'])==2 and all(v==1410 for k,v in c['knobs'].items() if k.endswith('mhz')) and all(v==400 for k,v in c['knobs'].items() if k.endswith('power_w')))
    settings = dict(bo=copy.deepcopy(DEFAULT_BO), reference_candidate_id=ref['id'],budget={'evaluations':33},setup_cost={'evaluations':1},
                    limits={'ttft_ms':100,'tpot_ms':10,'min_output_tps':10,'tbt_ms':10})
    settings['bo']['exploration_policy'] = 'broad_v2'
    return cs,ref,settings


def obs(c, **extra):
    return dict(candidate_id=c['id'],status='ok',metrics={'ttft_ms':50,'tpot_ms':5,'output_tps':20,'tbt_ms':5},**extra)


def choose(cs, observations, settings, pool=None):
    n=len(cs)
    return exploration.choose(cs, list(range(n)) if pool is None else pool, observations,settings,np.ones(n),np.ones(n)*9,np.ones(n))


def test_frequency_probe_reaches_low_endpoint_then_midpoint():
    cs,ref,s=fixture(); s['bo']['structure_exploration_fraction']=.01
    observations=[obs(ref)]
    targets=[]
    for _ in range(4):
        i,d=choose(cs,observations,s);targets.append((d['control'],d['target_value']))
        observations.append(obs(cs[i],proposal_reason=d['reason']))
    assert targets==[('expert_mhz',1050),('attention_mhz',1050),('expert_mhz',1290),('attention_mhz',1290)]


def test_third_measurement_covers_unseen_structure_at_max_without_relaxing_slo():
    cs,ref,s=fixture();before=copy.deepcopy(s['limits'])
    i,d=choose(cs,[obs(ref),obs(ref)],s)
    assert d['reason']=='structure_exploration'
    assert len(cs[i]['topology']['expert_gpus'])==1
    assert cs[i]['knobs']==ref['knobs']
    assert s['limits']==before
    assert d['prediction_basis']=='unmeasured_uncertainty'


def test_quota_and_affordability_are_respected():
    cs,ref,s=fixture();s['budget']['evaluations']=4
    observations=[obs(ref,proposal_reason='structure_exploration') for _ in range(10)]
    pick=choose(cs,observations,s)
    assert pick is None or pick[1]['reason']!='structure_exploration'
    pool=[i for i,c in enumerate(cs) if len(c['topology']['expert_gpus'])==2]
    i,d=choose(cs,[obs(ref),obs(ref)],s,pool)
    assert i in pool and d['reason']!='structure_exploration'


def test_early_repeat_deferred_but_boundary_repeat_allowed():
    cs,ref,s=fixture();idx=next(i for i,c in enumerate(cs) if c['id']==ref['id']);pool=list(range(len(cs)))
    assert idx not in exploration.novelty_pool(cs,pool,[obs(ref)],s)
    near=obs(ref);near['metrics']['ttft_ms']=99.5
    assert idx in exploration.novelty_pool(cs,pool,[near],s)
    assert idx in exploration.novelty_pool(cs,pool,[obs(ref)]*24,s)


def test_cap_probe_requires_binding_evidence_and_skips_inactive_caps():
    cs,ref,s=fixture();s['bo']['structure_exploration_fraction']=.001
    observations=[obs(ref)]
    for _ in range(4):
        i,d=choose(cs,observations,s);observations.append(obs(cs[i]))
    assert choose(cs,observations,s) is None
    observations[0]['external_observables']={'role_peak_rank_p95_power_w':{'expert':180,'attention':180}}
    assert choose(cs,observations,s) is None
    observations[0]['external_observables']['role_peak_rank_p95_power_w']['expert']=330
    i,d=choose(cs,observations,s)
    assert d['control']=='expert_power_w' and d['target_value']==300


@pytest.mark.parametrize('method,use_prior',[('bo',False),('random',False)])
def test_baselines_keep_existing_selection_pool(method,use_prior):
    cs,ref,s=fixture();s['bo'].update(method=method,use_model_prior=use_prior)
    pool=list(range(len(cs)))
    assert choose(cs,[obs(ref)],s) is None
    assert exploration.novelty_pool(cs,pool,[obs(ref)],s)==pool

@pytest.mark.parametrize('method', ['bo', 'random'])
def test_baseline_proposal_is_identical_with_v2_enabled(method):
    from static_dse.optimizer import propose
    cs,ref,s=fixture()
    for c in cs:c['prior'].update(log_energy=9.,energy_sd=1.)
    s.update(default_energy_j=8500.,mechanism_model='external_power_duration_v1')
    s['bo'].update(method=method,use_model_prior=False,model_screening=False)
    measured=obs(ref);measured['metrics']['energy_j']=8500.
    measured.update(cost={'gpu_hours':.1},proposal_expected_gpu_hours=.1,proposal_base_gpu_hours=.1)
    ids={c['id'] for c in cs}
    new=propose(cs,[measured],s,ids,8.)
    s['bo']['exploration_policy']='legacy'
    old=propose(cs,[measured],s,ids,8.)
    assert new==old

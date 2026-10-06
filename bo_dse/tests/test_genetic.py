import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/afd'))
from static_dse import genetic
from static_dse.campaign import DEFAULT_BO, ask, tell, status, read_json, write_json
from static_dse.comparison import create_comparison, comparison_report
from static_dse.demo import make_demo, synthetic_result
from static_dse.optimizer import propose
from static_dse.space import enumerate_candidates, configuration


def fixture():
    candidates = enumerate_candidates(dict(selection_split='calibration',
        topologies=[dict(attention_gpus=[0,1],expert_gpus=[2],attention_dp=2),
                    dict(attention_gpus=[0,1],expert_gpus=[2,4],attention_dp=2,expert_dp=2,expert_ep=2)],
        attention_frequencies_mhz=[1050,1410],expert_frequencies_mhz=[1050,1410],
        attention_power_caps_w=[200,400],expert_power_caps_w=[200,400],
        microbatches=[2],dbo_threshold_profiles=[[2,12],[8,128],[32,512]]))
    settings=dict(bo={**DEFAULT_BO,**genetic.DEFAULT_GA,'method':'ga', 'use_model_prior':False,'model_screening':False},
                  limits=dict(ttft_ms=100,tpot_ms=10,tbt_ms=12,min_output_tps=100),
                  reference_candidate_id=candidates[0]['id'])
    return candidates, settings


def observation(candidate, energy=100, good=True):
    return dict(candidate_id=candidate['id'],status='ok',telemetry_valid=True,
                metrics=dict(energy_j=energy,ttft_ms=50 if good else 200,tpot_ms=5,tbt_ms=6,output_tps=200))


def test_constraint_domination_includes_tbt_and_failed_trials():
    cs,settings=fixture();limits=settings['limits']
    feasible=observation(cs[0],energy=1000)
    violating=observation(cs[1],energy=1,good=False)
    only_tbt=observation(cs[2],energy=.1);only_tbt['metrics']['tbt_ms']=13
    failed=dict(candidate_id=cs[3]['id'],status='failed',metrics=feasible['metrics'])
    assert genetic.fitness([feasible],limits)<genetic.fitness([only_tbt],limits)<genetic.fitness([violating],limits)<genetic.fitness([failed],limits)
    assert genetic.fitness([feasible,failed],limits)[0]==2


def test_resume_determinism_no_repeats_and_offspring_use_own_measurements():
    cs,s=fixture();ids={c['id'] for c in cs};obs=[];reasons=[]
    for i in range(16):
        c,d=propose(cs,obs,s,ids,100)
        restored=json.loads(json.dumps(obs))
        c2,d2=propose(list(reversed(cs)),restored,copy.deepcopy(s),ids,100)
        assert (c['id'],d)==(c2['id'],d2)
        assert c['id'] not in {o['candidate_id'] for o in obs}
        assert configuration(c)['microbatches']==2
        assert (c['dbo_decode_token_threshold'],c['dbo_prefill_token_threshold']) in {(2,12),(8,128),(32,512)}
        if d['reason']=='ga_offspring':
            assert set(d['parent_candidate_ids']) <= {o['candidate_id'] for o in obs}
            assert len(d['population'])==4
        reasons.append(d['reason']);o=observation(c,100+i,good=i%2==0)
        if i==2:o['status']='failed'
        obs.append(o)
    assert reasons[0]=='measure_reference'
    assert reasons[1:4]==['ga_initial_population']*3
    assert reasons[4:]==['ga_offspring']*12


def test_ga_does_not_read_priors_and_obeys_eligibility_budget_and_exhaustion():
    cs,s=fixture();history=[observation(c,i+10) for i,c in enumerate(cs[:4])]
    allowed={c['id'] for c in cs[4:12]}
    baseline=propose(cs,history,s,allowed,100)
    changed=copy.deepcopy(cs)
    for c in changed:c.update(prior={'log_energy':-99999},mechanism={'impossible':'not available to GA'})
    changed_candidate, changed_decision = propose(changed,history,s,allowed,100)
    assert changed_candidate['id'] == baseline[0]['id']
    assert changed_decision == baseline[1]
    assert baseline[0]['id'] in allowed
    assert propose(cs,history,s,allowed,0)[0] is None
    assert propose(cs,[observation(c) for c in cs],s,{c['id'] for c in cs},100)[0] is None


def test_crossover_and_mutation_repair_to_sparse_legal_space():
    cs,s=fixture();cs=cs[::9];s['bo'].update(ga_crossover_rate=1.,ga_mutation_rate=1.)
    history=[observation(c,i+1) for i,c in enumerate(cs[:4])]
    c,d=propose(cs,history,s,{p['id'] for p in cs},100)
    assert c in cs and c['id'] not in {o['candidate_id'] for o in history}
    assert d['crossover_applied'] and d['mutated_genes']
    assert len(genetic.chromosome(c))==len(genetic.GENES)


@pytest.mark.parametrize('override', [dict(ga_population_size=1),dict(ga_tournament_size=5),dict(ga_mutation_rate=float('nan')),dict(ga_crossover_rate=True),dict(use_model_prior=True)])
def test_invalid_ga_settings_rejected(override):
    _,s=fixture()
    with pytest.raises(ValueError):genetic.validate_options({**s['bo'],**override})


def test_full_ask_tell_budget_resume_and_separate_ga_arm(tmp_path):
    config=make_demo(tmp_path/'inputs',evaluations=16)
    original=read_json(config);original['require_four_stage']=False
    write_json(config,original)
    comparison=tmp_path/'comparison'
    manifest=create_comparison(config,comparison,seeds=[0],methods=['ga','random'])
    ga=Path(next(a['directory'] for a in manifest['campaigns'] if a['method']=='ga'))
    random_arm=Path(next(a['directory'] for a in manifest['campaigns'] if a['method']=='random'))
    untouched=(random_arm/'state.json').read_bytes()
    for i in range(16):
        request=ask(ga)
        assert ask(ga)==request  # Pending proposal is durable across retries.
        result=synthetic_result(request)
        if i==2:
            result['status']='failed';result.pop('metrics');result.pop('four_stage')
            result['failure_reason']='Injected synthetic failure for accounting test'
        tell(ga,result)
    assert ask(ga)['stopped']
    assert status(ga)['cost']['evaluations']==16
    assert len({o['candidate_id'] for o in read_json(ga/'state.json')['observations']})==16
    assert (random_arm/'state.json').read_bytes()==untouched
    assert create_comparison(config,comparison,seeds=[0],resume=True)==manifest
    report=comparison_report(comparison)
    assert 'ga' in json.dumps(report)


def test_default_comparison_includes_four_methods_and_explicit_subset(tmp_path):
    config=make_demo(tmp_path/'inputs',evaluations=4)
    manifest=create_comparison(config,tmp_path/'all',seeds=[0])
    assert {a['method'] for a in manifest['campaigns']}=={'v2','generic_bo','random','ga'}
    manifest=create_comparison(config,tmp_path/'ga_only',seeds=[0],methods=['ga'])
    assert len(manifest['campaigns'])==1 and manifest['campaigns'][0]['method']=='ga'


def test_resuming_three_arms_preserves_observations_and_does_not_add_ga(tmp_path):
    config = make_demo(tmp_path/'inputs', evaluations=4)
    directory = tmp_path/'old-comparison'
    manifest = create_comparison(config, directory, seeds=[0], methods=['v2','generic_bo','random'])
    arm = Path(manifest['campaigns'][0]['directory'])
    tell(arm, synthetic_result(ask(arm)))
    original = (arm/'state.json').read_bytes()
    assert create_comparison(config, directory, seeds=[0], resume=True) == manifest
    assert (arm/'state.json').read_bytes() == original
    assert not (directory/'ga-seed0').exists()
    with pytest.raises(ValueError, match='inputs/seeds changed'):
        create_comparison(config, directory, seeds=[0], resume=True, methods=['v2','generic_bo','random','ga'])


def test_seed_changes_initial_population():
    cs, settings = fixture()
    history = [observation(cs[0])]
    choices = set()
    for seed in range(5):
        settings['bo']['seed'] = seed
        choices.add(propose(cs, history, settings, {c['id'] for c in cs}, 100)[0]['id'])
    assert len(choices) > 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0, -1, True, None])
def test_invalid_measurements_cannot_become_elites(bad):
    cs, settings = fixture()
    row = observation(cs[0])
    row['metrics']['energy_j'] = bad
    assert genetic.fitness([row], settings['limits'])[0] == 2

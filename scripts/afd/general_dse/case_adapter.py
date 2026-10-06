"""Run the existing four-GPU AFD case protocol with portable mathematical physics.

The deployment backend stays bounded to two tested model families and 2A1E/2A2E.
The underlying operator/GPU model is the same one used for arbitrary predictions.
"""
import copy
import json
from pathlib import Path

from . import engine,model_ir,primitives


def point(candidate, allocation):
    ne=1 if candidate['topology']=='2a1e' else 2
    return dict(id=candidate['id'],placement='disaggregated',allocation=allocation,
        attention_groups=[[g] for g in allocation[:2]],expert_groups=[[g] for g in allocation[2:2+ne]],
        **{k:candidate[k] for k in ('attention_mhz','expert_mhz','attention_power_w','expert_power_w')})


def search_case(root, model_key, model_tag, allocation, profile_path, parameters_path, output):
    root=Path(root);hp=Path(profile_path).resolve()
    h=json.loads(hp.read_text());engine.validate_hardware(h);primitives.verify_sources(h)
    if not h.get('sources'):raise ValueError('Portable case prediction needs measured primitive sources')
    m=model_ir.from_config(root/'artifacts/models'/model_tag/'config.json')
    params=dict(model_kind='general_operator_physics',selection_split='calibration',model=model_tag,
        general_model=m,general_hardware=h,
        scheduler=dict(max_num_seqs=32,max_num_batched_tokens=3072,microbatches=2,max_output_tokens=128,
                       workspace_bytes_per_gpu=2*1024**3,memory_fraction=.9),
        measurement_sources=[dict(path=name,sha256=digest,selection_split='calibration')
            for name,digest in {str(hp):primitives.sha(hp),**h['sources'],**m['sources']}.items()])
    if Path(parameters_path).exists():
        if json.loads(Path(parameters_path).read_text())!=params:raise ValueError('Portable parameters changed')
    else:primitives.write(parameters_path,params)
    frequencies=sorted(map(int,h['device']['by_mhz']))
    if 1410 not in frequencies:raise ValueError('Existing A100 case protocol needs measured 1410 MHz')
    candidates=[]
    for topo in ('2a1e','2a2e'):
        for af in frequencies:
            for ef in frequencies:
                if not (810<=af<=1410 and 1050<=ef<=1410):continue
                cid=f'{topo}-max' if af==ef==1410 else f'{topo}-a{af}-e{ef}'
                candidates.append(dict(id=cid,topology=topo,attention_mhz=af,expert_mhz=ef,attention_power_w=400,expert_power_w=400))
    # Historical hypotheses remain eligible for measurement even if their frequency/cap is unknown.
    for af,ef in ((810,1050),(1050,1290)):
        candidates.append(dict(id=f'2a1e-a{af}-e{ef}-p200',topology='2a1e',attention_mhz=af,expert_mhz=ef,attention_power_w=200,expert_power_w=200))
    requests=[json.loads(s) for s in (root/'inputs/traces/calibration-200.jsonl').read_text().splitlines()]
    result=engine.search(m,h,[point(c,allocation) for c in candidates],requests,params['scheduler'],[1,2,4],'2a2e-max')
    if result['predicted_fixed_configuration'] is None:
        raise ValueError('Portable AFD reference could not be predicted feasible; inspect primitives/memory instead of falling back silently')
    result.update(points=candidates,model=model_tag,model_key=model_key,parameters=str(Path(parameters_path).resolve()),
        parameters_sha256=primitives.sha(parameters_path),physics_source='portable_primitives',
        probe_evidence_sha256={r['path']:r['sha256'] for r in params['measurement_sources']},
        deployment_authorized=False,case_scope='AFD case backend restricts deployment candidates; general prediction CLI has no such topology restriction')
    primitives.write(output,result)
    return result

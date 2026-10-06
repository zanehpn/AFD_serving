#!/usr/bin/env python3
"""General LLM/GPU/workload analytical DSE; explicit CPU and GPU boundaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/afd'))
from general_dse import engine, model_ir, primitives


def read(path):
    return json.loads(Path(path).read_text())


def integers(value):
    return list(map(int,value.split(',')))


def code_hashes():
    files = [* (ROOT/'scripts/afd/general_dse').glob('*.py'), ROOT/'migration/general_dse.py',
             ROOT/'scripts/afd/static_dse/analytical.py']
    return {str(p.resolve()):primitives.sha(p) for p in files}


def make_search_plan(model_path, hardware_path, trace_path, evaluation_trace, output, allocation,
                     rates, adp, atp, ep, etp, frequencies, placements, baseline_id, candidates_path=None):
    paths = [Path(p).resolve() for p in (model_path,hardware_path,trace_path,evaluation_trace)]
    model, hardware = read(paths[0]),read(paths[1])
    model_ir.validate(model);engine.validate_hardware(hardware)
    primitives.verify_sources(model);primitives.verify_sources(hardware)
    if not hardware.get('sources'):
        raise ValueError('Real prediction plans require source hashes for measured hardware primitives')
    audit=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/audit_trace_isolation.py'),
        '--calibration',str(paths[2]),'--evaluation',str(paths[3]),
        '--identity-field','source_index','--identity-field','source_timestamp'],text=True))
    if audit['status']!='PASS':raise ValueError('Trace isolation failed')
    trace=[json.loads(s) for s in paths[2].read_text().splitlines()]
    if any(r.get('evaluation_split')!='calibration' for r in trace):
        raise ValueError('Only calibration requests may select a configuration')
    frequencies=frequencies or [max(map(int,hardware['device']['by_mhz']))]
    points=(read(candidates_path) if candidates_path else engine.enumerate_points(hardware,allocation,adp,atp,ep,etp,frequencies,placements))
    if any(p['allocation']!=allocation for p in points):raise ValueError('Candidate allocation differs from plan')
    if baseline_id not in {p['id'] for p in points}:raise ValueError('Baseline must be an explicit member of the candidate set')
    for p in points:engine.topology(p,hardware)
    sources={str(p):primitives.sha(p) for p in paths}
    if candidates_path:sources[str(Path(candidates_path).resolve())]=primitives.sha(candidates_path)
    sources.update(model.get('sources',{}));sources.update(hardware.get('sources',{}));sources.update(code_hashes())
    plan=dict(schema='general_search_plan_v1',selection_split='calibration',model=str(paths[0]),hardware=str(paths[1]),
        trace=str(paths[2]),evaluation_identity_audit=audit,candidates=points,baseline_id=baseline_id,rates=rates,
        scheduler=dict(max_num_seqs=32,max_num_batched_tokens=3072,max_output_tokens=128,microbatches=2,
            workspace_bytes_per_gpu=2*1024**3,memory_fraction=.9),
        contract=dict(latency_ratio_max=1.05,throughput_ratio_min=.95),sources=sources,
        allocation=allocation,formal_evaluation_eligible=False, GPU_execution_requested=False)
    primitives.write(output,plan)
    return plan


def execute_search(plan_path, output):
    plan=read(plan_path);primitives.verify_sources(plan)
    if plan['schema']!='general_search_plan_v1' or plan['selection_split']!='calibration':raise ValueError('Invalid search plan')
    result=engine.search(read(plan['model']),read(plan['hardware']),plan['candidates'],
        [json.loads(s) for s in Path(plan['trace']).read_text().splitlines()],plan['scheduler'],plan['rates'],plan['baseline_id'],
        latency_ratio=plan['contract']['latency_ratio_max'],throughput_ratio=plan['contract']['throughput_ratio_min'])
    result.update(plan=str(Path(plan_path).resolve()),plan_sha256=primitives.sha(plan_path),
        sources=plan['sources'],model_source=plan['model'],hardware_source=plan['hardware'],
        validation_required_for=['new_model_operator_mapping','unmeasured_topologies','shape_extrapolation','native_runtime_dispatch'],
        evidence_scope='hardware primitive calibration; no candidate-specific end-to-end measurements used')
    primitives.write(output,result)
    return result


def validate_predictions(prediction_path, measurements_path, output):
    """Audit prediction errors without selecting/tuning anything from validation outcomes."""
    pred=read(prediction_path);primitives.verify_sources(pred)
    if primitives.sha(pred['plan'])!=pred['plan_sha256']:raise ValueError('Prediction plan changed')
    obs=read(measurements_path)
    if obs['prediction_sha256']!=primitives.sha(prediction_path):raise ValueError('Measurements reference another prediction')
    if obs['purpose']!='prediction_validation_only' or obs['used_for_parameter_selection'] is not False:
        raise ValueError('Validation outcomes cannot be used for parameter selection')
    primitives.verify_sources(obs)
    if not obs.get('sources'):raise ValueError('Validation requires raw measurement source hashes')
    rows={(r['candidate_id'],r['rate']):r['prediction'] for r in pred['rows']}
    checked=[];seen=set()
    for r in obs['rows']:
        key=(r['candidate_id'],r['rate']);identity=(*key,r['repetition'])
        if identity in seen or key not in rows:raise ValueError('Duplicate or unknown validation cell')
        seen.add(identity);p=rows[key]
        if p['status']!='predicted_requires_validation':raise ValueError('Cannot score a missing prediction')
        errors={}
        for m in ('p90_ttft_ms','p90_tpot_ms','output_tps','energy_j_per_request'):
            engine.finite(r['metrics'][m],m)
            errors[m]=dict(predicted=p[m],measured=r['metrics'][m],relative_error=(p[m]-r['metrics'][m])/r['metrics'][m])
        checked.append(dict(candidate_id=key[0],rate=key[1],repetition=r['repetition'],errors=errors))
    if not checked:raise ValueError('Empty validation observations')
    result=dict(schema='general_prediction_validation_v1',purpose='prediction_validation_only',cells=checked,
        deployment_authorized=False,selection_performed=False,formal_generalization_claim_authorized=False,
        prediction_sha256=primitives.sha(prediction_path),measurement_sha256=primitives.sha(measurements_path),
        scope='reported cells only; does not establish unseen-model/topology independence without an external frozen split protocol')
    primitives.write(output,result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('model');c.add_argument('--config',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('probe-plan');c.add_argument('--gpus',type=integers,required=True);c.add_argument('--frequencies',type=integers,required=True);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('probe-run');c.add_argument('plan',type=Path);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('build-hardware');c.add_argument('measurements',type=Path);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('candidates');c.add_argument('--hardware',type=Path,required=True);c.add_argument('--allocation',type=integers,required=True)
    def dimensions(c):
        c.add_argument('--attention-dp',type=integers,default=[1,2,4]);c.add_argument('--attention-tp',type=integers,default=[1])
        c.add_argument('--expert-ep',type=integers,default=[1,2,4]);c.add_argument('--expert-tp',type=integers,default=[1])
        c.add_argument('--frequencies',type=integers);c.add_argument('--placements',default='disaggregated,colocated')
    dimensions(c);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('plan');c.add_argument('--model',type=Path,required=True);c.add_argument('--hardware',type=Path,required=True)
    c.add_argument('--trace',type=Path,default=ROOT/'inputs/traces/calibration-200.jsonl')
    c.add_argument('--evaluation-trace',type=Path,default=ROOT/'inputs/traces/heldout-400.jsonl')
    c.add_argument('--allocation',type=integers,required=True);c.add_argument('--rates',type=integers,default=[1,2,4])
    c.add_argument('--baseline-id',required=True);c.add_argument('--candidates',type=Path);dimensions(c);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('search');c.add_argument('plan',type=Path);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('validate');c.add_argument('--prediction',type=Path,required=True);c.add_argument('--measurements',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='model':primitives.write(a.output,model_ir.from_config(a.config))
    elif a.action=='probe-plan':primitives.write(a.output,primitives.probe_plan(ROOT,a.gpus,a.frequencies))
    elif a.action=='probe-run':primitives.run(a.plan,a.output)
    elif a.action=='build-hardware':
        raw=read(a.measurements)
        if raw.get('restored') is not True:raise ValueError('Primitive run did not verify allocation restoration')
        source_plan=read(raw['provenance']['plan_path']);primitives.verify_sources(source_plan)
        if primitives.sha(raw['provenance']['plan_path'])!=raw['provenance']['plan_sha256']:raise ValueError('Primitive plan changed')
        primitives.write(a.output,primitives.build(raw,a.measurements))
    elif a.action=='candidates':
        h=read(a.hardware);engine.validate_hardware(h)
        points=engine.enumerate_points(h,a.allocation,a.attention_dp,a.attention_tp,a.expert_ep,a.expert_tp,
            a.frequencies or [max(map(int,h['device']['by_mhz']))],a.placements.split(','))
        primitives.write(a.output,points)
    elif a.action=='plan':make_search_plan(a.model,a.hardware,a.trace,a.evaluation_trace,a.output,a.allocation,a.rates,
        a.attention_dp,a.attention_tp,a.expert_ep,a.expert_tp,a.frequencies,a.placements.split(','),a.baseline_id,a.candidates)
    elif a.action=='search':execute_search(a.plan,a.output)
    else:validate_predictions(a.prediction,a.measurements,a.output)
    print(a.output)

if __name__=='__main__':main()

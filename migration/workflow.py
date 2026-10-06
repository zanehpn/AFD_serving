#!/usr/bin/env python3
"""Complete native mathematical static + dynamic workflow, with immutable phases."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration import static_dse as static
from migration import build_math_parameters as parameters
from migration import dynamic_dse as dynamic


def status(base, phase, **extra):
    path = base / 'STATUS.json'
    path.write_text(json.dumps(dict(phase=phase, **extra), indent=2) + '\n')
    print(phase, flush=True)


def expected_spec(model, gpus, target, hardware_profile=None):
    return dict(schema_version=1, model=model, allocation=gpus, target=target,
                hardware_profile=str(Path(hardware_profile).resolve()) if hardware_profile else None,
                hardware_profile_sha256=static.sha(hardware_profile) if hardware_profile and Path(hardware_profile).exists() else None,
                files_sha256={str(f): static.sha(f) for f in static.code_files()},
                root=str(static.ROOT), selection_split='calibration',
                stages=([ 'portable_primitive_mathematical_search' ] if hardware_profile else
                        ['two_topology_probes', 'generate_parameters', 'CPU_mathematical_search']) + [
                        'validate_static_candidates', 'freeze_static', 'calibrate_dynamic_all_targets',
                        'freeze_all_formal_plans', 'run_heldout_all_targets', 'summarize'],
                rates=[1, 2, 4], inherited_MAX_thresholds='reuse_without_rebuilding',
                dynamic_calibration_repetitions=1, formal_repetitions=2)


def run(model, base, gpus, target='fixed', dry_run=False, hardware_profile=None):
    base = Path(base).resolve()
    spec = expected_spec(model, gpus, target, hardware_profile)
    if dry_run:
        print(json.dumps({k: v for k, v in spec.items() if k != 'files_sha256'}, indent=2))
        return
    base.mkdir(parents=True, exist_ok=True)
    lock = (static.ROOT / 'results/math-workflow.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec_path = base / 'WORKFLOW.json'
        if spec_path.exists():
            static.check(static.read(spec_path) == spec, 'Workflow code, target, or allocation changed; use a new checkout/campaign')
        else:
            static.write(spec_path, spec)
        heldout_started = (base / 'HELDOUT_STARTED.json').exists()
        def need(path):
            if heldout_started and not Path(path).exists():
                raise ValueError('Missing prerequisite after heldout started: refuse to regenerate/tune')
            return not Path(path).exists()
        def calibration_run(path):
            if heldout_started:
                # Verification only: never recollect calibration after opening heldout.
                static.collect(static.verify_plan(path))
            else:
                static.run(path)
        try:
            param = base / 'physical-parameters.json'
            prediction = base / 'PREDICTION.json'
            if hardware_profile:
                from general_dse.case_adapter import search_case
                status(base, 'portable_primitive_mathematical_search')
                if need(prediction):
                    search_case(static.ROOT, model, static.MODELS[model], gpus, hardware_profile, param, prediction)
                pred = static.read(prediction)
                static.check(pred['parameters_sha256'] == static.sha(param), 'Portable parameter lineage changed')
                for source, expected in pred['probe_evidence_sha256'].items():
                    static.check(static.sha(source) == expected, 'Portable primitive evidence changed')
                ports = None
            else:
                status(base, 'parameter_probes')
                probe = base / 'probes/PLAN.json'
                if need(probe):
                    static.make_plan(model, probe.parent, gpus)
                calibration_run(probe)
                param = base / 'physical-parameters.json'
                status(base, 'parameter_generation')
                if need(param):
                    parameters.build(probe, param)
                prediction = base / 'PREDICTION.json'
                status(base, 'CPU_mathematical_search')
                if need(prediction):
                    static.mathematical_search(probe, param, prediction, [1, 2, 4])
                pred = static.read(prediction)
                static.check(pred['parameters_sha256'] == static.sha(param) and pred['probe_plan_sha256'] == static.sha(probe), 'Prediction lineage changed')
                ports = static.read(probe)['ports']
            status(base, 'static_candidate_validation')
            validation = base / 'static-validation/PLAN.json'
            if need(validation):
                ids = static.validation_ids(pred)
                chosen = sorted([p for p in pred['points'] if p['id'] in ids], key=lambda c: (c['id'] != '2a2e-max', c['id']))
                static.make_plan(model, validation.parent, gpus, [1, 2, 4], chosen=chosen,
                    prediction={'path': str(prediction), 'sha256': static.sha(prediction)},
                    ports=ports)
            calibration_run(validation)
            report = base / 'STATIC_REPORT.json'
            if need(report):
                static.analyze(validation, report)
            freeze = base / 'STATIC_FREEZE.json'
            if need(freeze):
                static.freeze(validation, freeze)
            dynamic.verify_static(freeze)
            conditions = [None] if target == 'fixed' else [1, 2, 4]
            calibrated = []
            # Complete/freeze every target before opening any heldout results.
            for rate in conditions:
                label = 'fixed' if rate is None else f'rps-{rate}'
                status(base, 'dynamic_calibration', target=label)
                cal = base / label / 'dynamic-calibration/PLAN.json'
                cal_report = base / label / 'CALIBRATION_REPORT.json'
                if need(cal):
                    dynamic.make_plan(freeze, cal.parent, rate)
                calibration_run(cal)
                fresh = dynamic.summarize(cal)
                if need(cal_report):
                    static.write(cal_report, fresh)
                else:
                    static.check(static.read(cal_report) == fresh, 'Calibration evidence changed')
                if not fresh['all_calibration_gates_pass']:
                    status(base, 'calibration_gate_failed', target=label, heldout_started=heldout_started)
                    raise ValueError('Dynamic/static calibration failed relative MAX gates; heldout remains unstarted')
                calibrated.append((rate, label, cal, cal_report))
            formal_plans = []
            status(base, 'freeze_all_formal_plans')
            for rate, label, cal, cal_report in calibrated:
                formal = base / label / 'heldout/PLAN.json'
                if need(formal):
                    dynamic.make_plan(freeze, formal.parent, rate, cal, cal_report)
                dynamic.verify_heldout_gate(static.verify_plan(formal))
                formal_plans.append(formal)
            manifest = {str(p): static.sha(p) for p in formal_plans}
            if heldout_started:
                static.check(static.read(base / 'HELDOUT_STARTED.json')['formal_plans_sha256'] == manifest, 'Formal plan changed after start')
            else:
                static.write(base / 'HELDOUT_STARTED.json', dict(formal_plans_sha256=manifest,
                    static_freeze_sha256=static.sha(freeze), code_sha256=spec['files_sha256']))
            for formal in formal_plans:
                status(base, 'heldout_running', plan=str(formal))
                static.run(formal)
            reports = [dynamic.summarize(p) for p in formal_plans]
            result = dict(status='complete', model=model, target=target,
                          static_freeze=str(freeze), static_freeze_sha256=static.sha(freeze),
                          heldout_results=reports, formal_request_count=sum(r['request_count_total'] for r in reports),
                          calibration_used_for_selection_only=True)
            if (base / 'RESULTS.json').exists():
                static.check(static.read(base / 'RESULTS.json') == result, 'Final evidence changed')
            else:
                static.write(base / 'RESULTS.json', result)
            status(base, 'complete', results=str(base / 'RESULTS.json'))
        except BaseException as exc:
            previous = static.read(base / 'STATUS.json') if (base / 'STATUS.json').exists() else {}
            status(base, 'failed', failed_phase=previous.get('phase'), error=str(exc),
                   heldout_started=(base / 'HELDOUT_STARTED.json').exists())
            raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', choices=static.MODELS, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--target', choices=('fixed', 'per-rate'), default='fixed')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--hardware-profile', type=Path, help='Use a premeasured portable primitive profile')
    mode.add_argument('--legacy-topology-probes', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    profile = None if args.legacy_topology_probes else (args.hardware_profile or static.ROOT / 'results/primitives/HARDWARE.json')
    if profile and not args.dry_run:
        if args.hardware_profile and not profile.exists():
            raise FileNotFoundError('Explicit --hardware-profile must already exist')
        import subprocess
        subprocess.run([sys.executable, str(static.ROOT / 'migration/ensure_hardware.py'), '--profile', str(profile), '--gpus', args.gpus], check=True)
    run(args.model, args.campaign, list(map(int, args.gpus.split(','))), args.target, args.dry_run, profile)

if __name__ == '__main__':
    main()

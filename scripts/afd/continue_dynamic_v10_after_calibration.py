#!/usr/bin/env python3
"""Continue queued calibration to frozen evaluation when development gates pass."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / 'results/afd_suites/dynamic-fbss-v10-inputs-20260906'
STATUS = TRACES / 'CONTINUATION_STATUS.json'
MODELS = [('deepseek-v2-lite', 'DeepSeek-V2-Lite-Chat', 'deepseek-v2-lite-v026-dynamic-fbss-v9-20260906'),
          ('qwen36', 'Qwen3.6-35B-A3B', 'qwen36-v026-dynamic-fbss-v9b-20260906')]
if os.environ.get('ECODEP_MODELS'):
    selected = os.environ['ECODEP_MODELS'].split()
    if not selected or len(selected) != len(set(selected)) or set(selected) - {r[0] for r in MODELS}:
        raise ValueError('invalid ECODEP_MODELS selection')
    MODELS = [r for r in MODELS if r[0] in selected]
CALIBRATION_SUFFIX = os.environ.get('ECODEP_V10_CALIBRATION_SUFFIX', '')


def status(state, **kwargs):
    data = dict(status=state, updated_at=datetime.now(timezone.utc).isoformat(), **kwargs)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(data), flush=True)


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def main():
    deadline = time.monotonic() + 7200
    status('waiting_for_dynamic_calibration', heldout_started=False)
    while True:
        completed = []
        for model, tag, _ in MODELS:
            completed.append((ROOT / 'results/afd_suites' / tag / f'{model}-v026-dynamic-fbss-v10-calibration-dynamic{CALIBRATION_SUFFIX}-20260906' / 'COMPLETE').exists())
        if all(completed):
            break
        if time.monotonic() >= deadline:
            status('calibration_wait_timeout', heldout_started=False, reason='Calibration did not complete; inspect GPU availability and calibration-dynamic.log. No heldout outcome was opened.')
            return
        time.sleep(10)
    reports = []
    for model, tag, source in MODELS:
        p = ROOT / 'results/afd_protocols' / f'{model}-v026-dynamic-fbss-v10-20260906'
        suites = ROOT / 'results/afd_suites' / tag
        run('python3', 'scripts/afd/summarize_dynamic_v10_calibration.py', p,
            suites / f'{model}-v026-dynamic-fbss-v10-calibration-max-20260906',
            suites / f'{model}-v026-dynamic-fbss-v10-calibration-dynamic{CALIBRATION_SUFFIX}-20260906')
        reports.append(json.loads((p / 'CALIBRATION_RESULTS_V10.json').read_text()))
    def gate_values(report):
        if report.get('calibration_comparison_scope') == 'cross_server_inherited_max_reference':
            return report['all_calibration_reference_gates_pass'], report['pooled_reference_raw_saving']
        return report['all_calibration_slo_pass'], report['pooled_raw_saving']

    if not all(gate_values(r)[0] and gate_values(r)[1] > 0 for r in reports):
        status('calibration_requires_further_optimization', heldout_started=False,
               reports=[{'model': r['model'], 'gate_pass': gate_values(r)[0], 'reference_raw_saving': gate_values(r)[1], 'scope': r.get('calibration_comparison_scope', 'same_server')} for r in reports],
               reason='Retain all calibration cells; fresh heldout remains unused.')
        return
    status('freezing_evaluation', heldout_started=False)
    for model, tag, source in MODELS:
        p = ROOT / 'results/afd_protocols' / f'{model}-v026-dynamic-fbss-v10-20260906'
        if not (p / 'EVALUATION_FREEZE.json').exists():
            run('python3', 'scripts/afd/freeze_dynamic_v10.py', p, ROOT / 'results/afd_protocols' / source, TRACES)
        run('python3', 'scripts/afd/audit_dynamic_v10_freeze.py', p, TRACES)
    status('running_frozen_three_arm_heldout', heldout_started=True)
    run('bash', 'scripts/afd/run_dynamic_v10_heldout.sh')
    results = []
    for model, tag, _ in MODELS:
        p = ROOT / 'results/afd_protocols' / f'{model}-v026-dynamic-fbss-v10-20260906'
        pair = {str(revision): json.loads((p / f'DYNAMIC_RESULTS_V{revision}.json').read_text()) for revision in (9, 10)}
        results.append(dict(model=tag, comparisons=pair))
    filename = ('CROSS_MODEL_DYNAMIC_FBSS_V10_RESULTS_20260906.json' if len(MODELS) == 2
                else f'DYNAMIC_FBSS_V10_RESULTS_{MODELS[0][0]}_20260906.json')
    output = ROOT / 'results/afd_protocols' / filename
    output.write_text(json.dumps({'status': 'complete', 'models': results, 'repetitions': 1, 'heldout_source_indices': [6000, 6399], 'total_completed_requests': 3600 * len(MODELS)}, indent=2) + '\n')
    status('complete', heldout_started=True, result=str(output))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
        status('failed', heldout_started=previous.get('heldout_started', False), error=f'{type(exc).__name__}: {exc}')
        raise

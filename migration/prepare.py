#!/usr/bin/env python3
"""Prepare native dynamic calibration with inherited A100 MAX guards."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/afd'))
from inherited_max import prepare_controller

TRACES = ROOT / 'results/afd_suites/dynamic-fbss-v10-inputs-20260906'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocate(value):
    if isinstance(value, str):
        return value.replace('dynamic-fbss-v9-inputs-20260906/calibration-200.jsonl', 'dynamic-fbss-v10-inputs-20260906/calibration-200.jsonl')
    if isinstance(value, list):
        return [relocate(x) for x in value]
    if isinstance(value, dict):
        return {relocate(k): relocate(v) for k, v in value.items()}
    return value


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    os.chdir(ROOT)
    for relative, expected in json.loads((ROOT / 'environment/input-hashes.json').read_text()).items():
        if digest(ROOT / relative) != expected:
            raise SystemExit(f'Migration input changed: {relative}')
    marker = ROOT / 'environment/PREPARED.json'
    if marker.exists():
        raise SystemExit('Already prepared. Resume with run.sh; use a fresh clone for a new campaign.')
    for model in ['deepseek-v2-lite', 'qwen36']:
        target = ROOT / f'results/afd_protocols/{model}-v026-dynamic-fbss-v10-20260906'
        if target.exists():
            raise SystemExit(f'Existing protocol: {target}. Refusing to overwrite run state.')
    for path in (ROOT / 'inputs/traces').iterdir():
        out = TRACES / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            write(out, relocate(json.loads(path.read_text())))
        else:
            shutil.copyfile(path, out)
    for model, old in [('deepseek-v2-lite', 'deepseek-v2-lite-v026-dynamic-fbss-v9-20260906'), ('qwen36', 'qwen36-v026-dynamic-fbss-v9b-20260906')]:
        src = ROOT / 'inputs/protocols' / model
        target = ROOT / f'results/afd_protocols/{model}-v026-dynamic-fbss-v10-20260906'
        source = ROOT / 'results/afd_protocols' / old
        for name in ['FREEZE.json', 'expanded-profile-v6.json', 'fbss-decision-v6.json', 'routing-profile-v9.json']:
            # Portable policy templates use repository-relative paths.
            source.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / name, source / name)
        write(source / 'combined-controller-v9.json', relocate(json.loads((src / 'combined-controller-v9.json').read_text())))
        for name in ['calibration-max-deployment.json', 'calibration-fbss-deployment.json', 'calibration-schedule.json']:
            write(target / name, relocate(json.loads((src / name).read_text())))
        freeze = relocate(json.loads((src / 'PRE_DYNAMIC_CALIBRATION_CODE_FREEZE.json').read_text()))
        inherited = ROOT / 'inputs/calibration-max' / model
        write(target / 'combined-controller-v10.json', prepare_controller(ROOT, model, relocate))
        freeze['migration'] = {'runtime': 'native', 'old_max_results_reused': True,
                               'max_calibration_mode': 'reuse_source_a100_max',
                               'inherited_max_reference': str(inherited),
                               'inherited_max_provenance_sha256': digest(inherited / 'PROVENANCE.json'),
                               'source_policy': 'inherited_A100_v9', 'source_freeze_sha256': digest(src / 'PRE_DYNAMIC_CALIBRATION_CODE_FREEZE.json')}
        code_paths = [*ROOT.glob('scripts/**/*.py'), *ROOT.glob('scripts/**/*.sh'), *ROOT.glob('migration/*.py'), *ROOT.glob('migration/*.sh'), ROOT / 'services/nvcontrold.py']
        freeze['files_sha256'] = {str(path.resolve()): digest(path) for path in code_paths if not path.name.startswith('test_')}
        write(target / 'PRE_DYNAMIC_CALIBRATION_CODE_FREEZE.json', freeze)
    output = subprocess.check_output(['python3', str(ROOT / 'scripts/audit_trace_isolation.py'), '--calibration', str(TRACES / 'calibration-200.jsonl'), '--evaluation', str(TRACES / 'heldout-400.jsonl'), '--identity-field', 'source_index', '--identity-field', 'source_timestamp'], text=True)
    assert json.loads(output)['status'] == 'PASS'
    (TRACES / 'trace-isolation-audit.json').write_text(output)
    subprocess.run(['python3', str(ROOT / 'migration/runtime_fingerprint.py'), '--output', str(ROOT / 'environment/native-runtime.json')], check=True)
    write(marker, {'root': str(ROOT), 'gpu_groups': [os.environ.get('ECODEP_ATTENTION_GPUS', '0,1'), os.environ.get('ECODEP_EXPERT_GPUS', '2,3')], 'runtime': 'native', 'heldout_started': False,
                   'selected_models': os.environ.get('ECODEP_MODELS', 'deepseek-v2-lite qwen36').split(),
                   'service_ports': {key: int(os.environ.get(key, default)) for key, default in [('ECODEP_API_PORT', '18000'), ('ECODEP_EXPERT_API_PORT', '18001'), ('ECODEP_AFD_PORT', '16239'), ('ECODEP_DP_RPC_BASE_PORT', '29550')]},
                   'execution_layout': os.environ.get('ECODEP_EXECUTION_LAYOUT', 'sequential_4gpu'),
                   'calibration_results_reused': True, 'max_calibration_mode': 'reuse_source_a100_max', 'source_policy': 'inherited_A100_v9',
                   'source_manifest': {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'inputs').rglob('*') if p.is_file()}})
    print('Prepared native dynamic calibration with inherited A100 MAX guards. No destination MAX run or fabricated COMPLETE marker.')


if __name__ == '__main__':
    main()

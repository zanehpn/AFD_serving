#!/usr/bin/env python3
"""Upgrade an unstarted prepared checkout to inherited A100 MAX calibration."""
import json
from pathlib import Path
import shutil
from prepare import ROOT, digest, relocate, write, prepare_controller


def main():
    marker = ROOT / 'environment/PREPARED.json'
    prepared = json.loads(marker.read_text())
    if prepared.get('max_calibration_mode') == 'reuse_source_a100_max':
        return
    for tag in ['DeepSeek-V2-Lite-Chat', 'Qwen3.6-35B-A3B']:
        if list((ROOT / 'results/afd_suites' / tag).glob('*dynamic-fbss-v10-*')):
            raise SystemExit('A v10 suite already started. Keep its evidence and use a fresh clone for the inherited-MAX campaign.')
    for model in ['deepseek-v2-lite', 'qwen36']:
        protocol = ROOT / f'results/afd_protocols/{model}-v026-dynamic-fbss-v10-20260906'
        if any((protocol / name).exists() for name in ['EVALUATION_FREEZE.json', 'CALIBRATION_RESULTS_V10.json', 'FREEZE.json']):
            raise SystemExit('Calibration/evaluation results already exist; refusing to revise this experiment.')
    for relative, expected in json.loads((ROOT / 'environment/input-hashes.json').read_text()).items():
        if digest(ROOT / relative) != expected:
            raise SystemExit(f'Migration input changed: {relative}')
    for model in ['deepseek-v2-lite', 'qwen36']:
        protocol = ROOT / f'results/afd_protocols/{model}-v026-dynamic-fbss-v10-20260906'
        original = protocol / 'PRE_DYNAMIC_CALIBRATION_CODE_FREEZE.json'
        archive = ROOT / 'inputs/calibration-max' / model
        config = prepare_controller(ROOT, model, relocate)
        shutil.copyfile(original, protocol / 'PRE_NATIVE_MAX_REUSE_CODE_FREEZE.json')
        freeze = json.loads(original.read_text())
        freeze['migration'].update(old_max_results_reused=True, max_calibration_mode='reuse_source_a100_max',
                                   inherited_max_reference=str(archive), inherited_max_provenance_sha256=digest(archive / 'PROVENANCE.json'))
        paths = [*ROOT.glob('scripts/**/*.py'), *ROOT.glob('scripts/**/*.sh'), *ROOT.glob('migration/*.py'), *ROOT.glob('migration/*.sh'), ROOT / 'services/nvcontrold.py']
        freeze['files_sha256'] = {str(path.resolve()): digest(path) for path in paths if not path.name.startswith('test_')}
        write(protocol / 'combined-controller-v10.json', config)
        write(original, freeze)
    shutil.copyfile(marker, ROOT / 'environment/PREPARED-before-max-reuse.json')
    prepared.update(calibration_results_reused=True, max_calibration_mode='reuse_source_a100_max',
                    source_manifest={str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'inputs').rglob('*') if p.is_file()})
    write(marker, prepared)
    print('Upgraded unstarted campaign to inherited A100 MAX guards; prior preparation metadata retained.')


if __name__ == '__main__':
    main()

"""Carry preparation costs across an explicitly requested new-host protocol change.

No observations, priors, or feasible labels are imported.
"""
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def carry(config, prior, inputs):
    prior, inputs = Path(prior).resolve(), Path(inputs)
    old = read(prior/'official-config.json')
    for key in ('model', 'model_path', 'gpus', 'rps', 'max_model_len',
                'max_output_tokens', 'warmup_requests', 'evaluation_requests', 'time_scale'):
        if config[key] != old[key]:
            raise ValueError('Preparation cost transition changes workload/hardware: '+key)
    for gpu in config['gpus']:
        if config['hardware']['devices'][str(gpu)]['uuid'] != old['hardware']['devices'][str(gpu)]['uuid']:
            raise ValueError('Preparation cost transition changes physical GPU')
    for name in ('calibration.jsonl', 'heldout.jsonl', 'warmup.jsonl'):
        if artifact(inputs/name)['sha256'] != artifact(prior/'inputs'/name)['sha256']:
            raise ValueError('Preparation cost transition changes trace: '+name)
    for path in (prior/'comparison').glob('*/state.json'):
        state = read(path)
        if state['observations'] or state.get('pending'):
            raise ValueError('This transition only supports preparation before search')
    context = read(old['context_manifest'])
    for path, expected in context['files_sha256'].items():
        if artifact(path)['sha256'] != expected:
            raise ValueError('Old frozen context changed: '+path)
    for path, expected in context['model_files'].items():
        stat = Path(path).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError('Old frozen model changed: '+path)
    for marker in prior.rglob('STARTED.json'):
        if not (marker.parent/'worker-result.json').exists():
            raise ValueError('Old preparation remains interrupted; recover it first')
    paths = sorted((prior/'validation').glob('*/worker-result.json'))
    if not paths:
        raise ValueError('No completed preparation cost receipts')
    rows = read(prior/'inputs/prior-preparation.json')['receipts'] if (prior/'inputs/prior-preparation.json').exists() else []
    rows += [{'artifact': artifact(path)} for path in paths]
    if len({row['artifact']['path'] for row in rows}) != len(rows):
        raise ValueError('Duplicate preparation cost receipt')
    for row in rows:
        ref = row['artifact']
        if artifact(ref['path']) != ref:
            raise ValueError('Preparation receipt changed')
        path = Path(ref['path'])
        receipt = read(path)
        if receipt.get('cleanup_required'):
            resolved = path.parent/'CLEANUP_RESOLVED.json'
            if not resolved.exists() or read(resolved).get('original_receipt') != ref:
                raise ValueError('Unresolved preparation cleanup')
        for key in ('wall_seconds', 'gpu_hours', 'tuning_energy_j'):
            value = receipt['cost'][key]
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid preparation cost')
        for evidence in receipt.get('artifacts', []):
            if artifact(evidence['path']) != evidence:
                raise ValueError('Preparation evidence changed')
    record = dict(previous_directory=str(prior), receipts=rows,
                  reason='User requested local MAX SLO and new DBO threshold candidates; retain every earlier preparation cost',
                  old_config=artifact(prior/'official-config.json'), old_limits=old['limits'],
                  old_reference=old['reference_configuration'], new_reference=config['reference_configuration'],
                  observations_imported=False)
    (inputs/'prior-preparation.json').write_text(json.dumps(record, indent=2)+'\n')

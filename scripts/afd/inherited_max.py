"""Validate and relocate inherited A100 calibration evidence, never heldout results."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reference(directory):
    directory = Path(directory)
    provenance = json.loads((directory / 'PROVENANCE.json').read_text())
    if (provenance['kind'] != 'inherited_calibration_max'
            or provenance['selection_split'] != 'calibration'
            or provenance['destination_measurement'] is not False
            or provenance['source_hardware'] != 'NVIDIA A100-SXM4-80GB'):
        raise ValueError('not an inherited A100 MAX calibration reference')
    for name, expected in provenance['files_sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError(f'inherited calibration evidence changed: {name}')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['evaluation_split'] != 'calibration':
        raise ValueError('heldout evidence cannot supply calibration thresholds')
    config = json.loads((directory / 'combined-controller-v10.json').read_text())
    if config['compatibility']['model'] != manifest['model']:
        raise ValueError('inherited controller/model mismatch')
    for row in config['predictor']['calibration_guard_by_rps']:
        case = directory / f"rps-{row['rps']}"
        summary = json.loads((case / 'summary.json').read_text())
        if summary['completed_requests'] != 200 or summary['failed_requests']:
            raise ValueError('inherited MAX calibration is incomplete')
        expected = [summary['ttft_ms']['p90'] * 1.05,
                    summary['tpot_ms']['p99'] * 1.05 + config['events']['progress_event_interval_ms'] + config['control_interval_ms']]
        if [row['prefill_age_guard_ms'], row['progress_gap_guard_ms']] != expected:
            raise ValueError('inherited thresholds do not match source MAX calibration')
    return provenance


def prepare_controller(root, model, relocate):
    root = Path(root)
    archive = root / 'inputs/calibration-max' / model
    provenance = validate_reference(archive)
    original = json.loads((archive / 'combined-controller-v10.json').read_text())
    source_suite = provenance['source_suite']
    source = Path(source_suite)
    source_serving = str(source.parents[2] / 'afd_serving' / source.parent.name / source.name)
    old_v9 = 'deepseek-v2-lite-v026-dynamic-fbss-v9-20260906' if model == 'deepseek-v2-lite' else 'qwen36-v026-dynamic-fbss-v9b-20260906'
    old_controller = str(source.parents[2] / 'afd_protocols' / old_v9 / 'combined-controller-v9.json')

    def translate(value):
        if isinstance(value, str):
            for before, after in [(source_suite, str(archive)), (source_serving, str(archive)),
                                  (old_controller, str(root / 'inputs/protocols' / model / 'combined-controller-v9.json'))]:
                value = value.replace(before, after)
            return relocate(value)
        if isinstance(value, list):
            return [translate(x) for x in value]
        if isinstance(value, dict):
            return {translate(k): translate(v) for k, v in value.items()}
        return value
    config = translate(original)
    for path, expected in config['predictor']['guard_calibration']['files_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'inherited guard dependency changed: {path}')
    config['predictor']['guard_calibration']['migration'] = {
        'mode': 'reuse_source_a100_max', 'destination_max_calibration_run': False,
        'source_controller_sha256': digest(archive / 'combined-controller-v10.json'),
        'source_provenance': str(archive / 'PROVENANCE.json'),
        'source_provenance_sha256': digest(archive / 'PROVENANCE.json')}
    return config

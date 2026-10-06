"""Import an archived Max as historical evidence, without serving a request."""
import json
import hashlib
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inherit(config, prior, inputs):
    prior = Path(prior).resolve()
    old = read(prior/'official-config.json')
    reference = read(prior/'inputs/slo-reference.json')
    receipt_path = prior/'validation'/old['reference_candidate_id']/'worker-result.json'
    receipt = read(receipt_path)
    if not config.get('direct_search') or config.get('slo_mode') != 'relative_max':
        raise ValueError('Reference reuse requires direct search with relative Max SLOs')
    for key in ('model', 'rps', 'evaluation_requests', 'warmup_requests', 'max_model_len',
                'max_output_tokens', 'time_scale', 'memory_clock_mhz', 'output_validation',
                'tbt_slo', 'slo_ratios', 'reference_configuration', 'reference_candidate_id'):
        if config.get(key) != old.get(key):
            raise ValueError('Inherited Max protocol mismatch: '+key)
    for name in ('calibration.jsonl', 'warmup.jsonl', 'heldout.jsonl'):
        if sha(inputs/name) != sha(prior/'inputs'/name):
            raise ValueError('Inherited Max trace mismatch: '+name)
    if sha(Path(config['model_path'])/'config.json') != old['correctness']['model_config']['sha256']:
        raise ValueError('Inherited Max model configuration mismatch')
    if sha(receipt_path) != reference['receipt']['sha256']:
        raise ValueError('Inherited Max receipt checksum mismatch')
    if (receipt['status'] != 'ok' or not receipt.get('execution_verified')
            or not receipt.get('telemetry_valid') or receipt.get('failed_requests') != 0
            or receipt.get('completed_requests') != config['evaluation_requests']):
        raise ValueError('Inherited Max must have completed all requests with valid telemetry')
    if reference['metrics'] != receipt['metrics'] or reference['configuration'] != config['reference_configuration']:
        raise ValueError('Inherited Max summary differs from measured receipt')
    limits = {k: receipt['metrics']['output_tps' if k == 'min_output_tps' else k]*ratio
              for k, ratio in config['slo_ratios'].items()}
    if limits != reference['limits'] or limits != old['limits']:
        raise ValueError('Inherited Max frozen SLO thresholds differ')
    # Device identity is recorded, not equated across machines. No local execution
    # verification is inferred from historical measurements.
    for source, name in ((prior/'official-config.json', 'inherited-max-config.json'),
                         (prior/'inputs/slo-reference.json', 'inherited-max-reference.json'),
                         (receipt_path, 'inherited-max-receipt.json')):
        (inputs/name).write_bytes(source.read_bytes())
    record = dict(origin='historical_max_reuse', source_directory=str(prior),
                  receipt=dict(path=str(inputs/'inherited-max-receipt.json'), sha256=sha(receipt_path)),
                  reference=dict(path=str(inputs/'inherited-max-reference.json'),
                                 sha256=sha(prior/'inputs/slo-reference.json')),
                  source_hardware=old['hardware'], current_hardware=config['hardware'],
                  local_measurement_performed=False, local_execution_verified=False,
                  limits=limits, cost_scope='Historical Max cost charged once; no new Max attempt')
    (inputs/'inherited-max.json').write_text(json.dumps(record, indent=2)+'\n')
    config['inherited_max'] = record

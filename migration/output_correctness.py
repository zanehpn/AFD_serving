"""Exact output regression against a frozen, colocated stock-vLLM reference.

This checks the calibration cohort only. Exact tokens are conservative across
parallel floating-point reductions; a mismatch is diagnostic, not proof of a
particular communication bug. Never relax the gate using heldout outputs.
"""
import hashlib
import json
from pathlib import Path

PROTOCOL = 'stock_vllm_exact_tokens_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_rows(contract):
    if contract['protocol'] != PROTOCOL or contract['selection_split'] != 'calibration':
        raise ValueError('Correctness reference must use the calibration protocol')
    if contract['generation'] != {'temperature': 0, 'seed': 0, 'ignore_eos': True, 'generation_config': 'vllm'}:
        raise ValueError('Correctness generation settings differ from the supported protocol')
    if sha(contract['trace']['path']) != contract['trace']['sha256']:
        raise ValueError('Correctness calibration trace changed')
    if sha(contract['model_config']['path']) != contract['model_config']['sha256']:
        raise ValueError('Correctness reference model changed')
    rows = [json.loads(s) for s in Path(contract['trace']['path']).read_text().splitlines() if s.strip()]
    if (not rows or len(rows) != contract['trace']['requests']
            or len({r['source_index'] for r in rows}) != len(rows)
            or any(r.get('evaluation_split') != 'calibration' for r in rows)):
        raise ValueError('Invalid correctness calibration cohort')
    return rows


def token_ids(value, length):
    return (isinstance(value, list) and len(value) == length
            and all(type(t) is int and t >= 0 for t in value))


def validate_reference(reference, contract):
    rows = trace_rows(contract)
    if reference.get('contract') != contract:
        raise ValueError('Correctness reference contract differs from frozen protocol')
    runtime = reference.get('runtime', {})
    if (runtime.get('backend') != 'stock_vllm_colocated' or runtime.get('plugins') != []
            or runtime.get('tensor_parallel_size') != contract['reference_tp']
            or runtime.get('microbatches') != 1 or runtime.get('afd_enabled') is not False
            or runtime.get('vllm_version') != '0.26.0'):
        raise ValueError('Reference must be stock vLLM 0.26.0 without AFD or microbatching')
    expected = {r['source_index']: r for r in rows}
    actual = reference.get('outputs', [])
    if len(actual) != len(rows) or {r['source_index'] for r in actual} != set(expected):
        raise ValueError('Reference has duplicate/missing request identities')
    for item in actual:
        row = expected[item['source_index']]
        if (any(item.get(k) != v for k, v in row.items())
                or not token_ids(item.get('output_token_ids'), min(row['output_tokens'], contract['max_output_tokens']))):
            raise ValueError('Reference request identity/content or output token IDs are invalid')
    return {r['source_index']: r['output_token_ids'] for r in actual}


def compare(contract, reference_artifact, replay_rows):
    if sha(reference_artifact['path']) != reference_artifact['sha256']:
        raise ValueError('Frozen correctness reference changed')
    reference = json.loads(Path(reference_artifact['path']).read_text())
    expected = validate_reference(reference, contract)
    rows = {r['source_index']: r for r in trace_rows(contract)}
    if len(replay_rows) != len(rows) or {r['source_index'] for r in replay_rows} != set(rows):
        raise ValueError('Correctness replay has duplicate/missing request identities')
    mismatches = []
    for row in replay_rows:
        source = rows[row['source_index']]
        if any(row.get(k) != v for k, v in source.items()):
            raise ValueError('Correctness replay request identity/content changed')
        wanted = expected[row['source_index']]
        actual = row.get('actual_output_token_ids')
        valid = token_ids(actual, len(wanted))
        if not valid or actual != wanted:
            first = next((i for i, (a, b) in enumerate(zip(actual, wanted)) if a != b), None) if valid else None
            mismatches.append({'source_index': row['source_index'], 'first_mismatch_position': first,
                               'reason': 'token_mismatch' if valid else 'missing_or_invalid_token_ids'})
    return {'verified': not mismatches, 'protocol': PROTOCOL,
            'reference_sha256': reference_artifact['sha256'], 'selection_split': 'calibration',
            'requests': len(rows), 'output_tokens': sum(map(len, expected.values())),
            'mismatched_requests': len(mismatches), 'mismatches': mismatches}

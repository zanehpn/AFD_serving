"""Calibration output gate rejects corrupt content even at the right length."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'migration'))
from output_correctness import PROTOCOL, compare, sha, validate_reference


@pytest.fixture
def cohort(tmp_path):
    rows = [{'source_index': i, 'source_timestamp': str(i), 'request_id': i,
             'evaluation_split': 'calibration', 'input_tokens': 4, 'output_tokens': 3} for i in range(2)]
    trace = tmp_path / 'calibration.jsonl'
    trace.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    model = tmp_path / 'model.json'; model.write_text('{}')
    contract = {'protocol': PROTOCOL, 'selection_split': 'calibration',
                'trace': {'path': str(trace), 'sha256': sha(trace), 'requests': 2},
                'model_config': {'path': str(model), 'sha256': sha(model)},
                'reference_tp': 2, 'max_output_tokens': 3,
                'generation': {'temperature': 0, 'seed': 0, 'ignore_eos': True, 'generation_config': 'vllm'}}
    reference = {'contract': contract, 'runtime': {'backend': 'stock_vllm_colocated', 'plugins': [],
                  'tensor_parallel_size': 2, 'microbatches': 1, 'afd_enabled': False, 'vllm_version': '0.26.0'},
                 'outputs': [{**r, 'output_token_ids': [2, 3, 4]} for r in rows]}
    path = tmp_path / 'reference.json'; path.write_text(json.dumps(reference))
    replay = [{**r, 'actual_output_token_ids': [2, 3, 4]} for r in rows]
    return contract, reference, {'path': str(path), 'sha256': sha(path)}, replay


def test_correct_length_wrong_content_is_rejected(cohort):
    contract, _, source, rows = cohort
    assert compare(contract, source, rows)['verified']
    rows[0]['actual_output_token_ids'][1] = 99
    report = compare(contract, source, rows)
    assert not report['verified'] and report['mismatched_requests'] == 1
    assert report['mismatches'][0]['first_mismatch_position'] == 1


@pytest.mark.parametrize('tokens', [None, [], [2, 3], [2, 3, 4, 5], [2, True, 4], [2, -1, 4]])
def test_missing_or_invalid_ids_fail_closed(cohort, tokens):
    contract, _, source, rows = cohort
    rows[0]['actual_output_token_ids'] = tokens
    assert compare(contract, source, rows)['verified'] is False


@pytest.mark.parametrize('damage', ['reference_hash', 'model_hash', 'trace_hash', 'identity', 'duplicate', 'heldout', 'afd', 'generation'])
def test_mismatched_reference_and_cohort_are_rejected(cohort, damage):
    contract, reference, source, rows = cohort
    if damage == 'reference_hash':
        Path(source['path']).write_text('{}')
    elif damage == 'model_hash':
        Path(contract['model_config']['path']).write_text('{"changed":true}')
    elif damage == 'trace_hash':
        Path(contract['trace']['path']).write_text('')
    elif damage == 'identity':
        rows[0]['source_timestamp'] = 'different'
    elif damage == 'duplicate':
        rows[0] = copy.deepcopy(rows[1])
    else:
        if damage == 'heldout':
            reference['outputs'][0]['evaluation_split'] = 'heldout'
        elif damage == 'afd':
            reference['runtime']['plugins'] = ['afd']
        else:
            contract['generation']['seed'] = 1
        with pytest.raises(ValueError):
            validate_reference(reference, contract)
        return
    with pytest.raises(ValueError):
        compare(contract, source, rows)

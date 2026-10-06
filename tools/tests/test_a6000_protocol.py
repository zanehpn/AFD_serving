"""New-host protocol tests, copied into the isolated algorithm snapshot."""
import copy
import json
import sys

import pytest

from test_official import prepared, official, official_worker, read_json


def test_four_arms_keep_absolute_tbt_and_hardware_protocol(prepared):
    args, _, _, _ = prepared
    args = copy.copy(args)
    args.directory = args.directory.parent/'a6000-four-arms'
    args.comparison_methods = ['v2', 'generic_bo', 'random', 'ga']
    args.expected_gpu_model = 'NVIDIA RTX A6000'
    args.tbt_ms = 164.88368295003397
    args.slo_mode = 'absolute'
    args.direct_search = True
    args.output_validation = 'request_completion'
    args.microbatches = [2]
    args.dbo_threshold_profiles = ['16:256', '32:512', '64:1024']
    args.reference_dbo_thresholds = '32:512'
    config = official.initialize(args)
    assert config['expected_gpu_model'] == args.expected_gpu_model
    assert config['cuda_architecture'] == 'sm_86'
    assert config['reference_configuration']['dbo_decode_token_threshold'] == 32
    assert config['reference_configuration']['dbo_prefill_token_threshold'] == 512
    original = dict(config['limits'])
    official.prepare(config)
    config = read_json(args.directory/'official-config.json')
    assert config['limits'] == original
    assert config['limits']['tbt_ms'] == args.tbt_ms
    manifest = read_json(args.directory/'comparison/comparison.json')
    assert {arm['method'] for arm in manifest['campaigns']} == set(args.comparison_methods)
    for arm in manifest['campaigns']:
        bundle = read_json(args.directory/'comparison'/f"{arm['method']}-seed0"/'bundle.json')
        assert bundle['settings']['limits'] == original


@pytest.mark.parametrize('detected, accepted', [('NVIDIA RTX A6000', True), ('NVIDIA A100-SXM4-80GB', False)])
def test_declared_a6000_model_is_enforced(tmp_path, monkeypatch, detected, accepted):
    import capture_hardware
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(gpus=[0,1,2,3], expected_gpu_model='NVIDIA RTX A6000',
                                     cuda_architecture='sm_86', clock_url='http://fixture')))
    monkeypatch.setattr(sys, 'argv', ['worker', 'preflight', '--config', str(config), '--directory', str(tmp_path)])
    monkeypatch.setattr(official_worker, 'check_ports', lambda c: None)
    monkeypatch.setattr(official_worker, 'verify_runtime', lambda: {})
    seen = []
    monkeypatch.setattr(official_worker, 'compiler_preflight', lambda p, arch: seen.append(arch))
    monkeypatch.setattr(capture_hardware, 'capture', lambda ids, nv: {'devices': {str(i): {'name': detected} for i in ids}})
    monkeypatch.setattr(official_worker, 'http', lambda url: dict(protocol='nvcontrold.applied_ack.v2', allowed=[0,1,2,3]))
    if accepted:
        official_worker.main()
        assert (tmp_path/'preflight.json').exists()
    else:
        with pytest.raises(ValueError, match='GPU model differs'):
            official_worker.main()
        assert not (tmp_path/'preflight.json').exists()
    assert seen == ['sm_86']


def test_preparation_revision_carries_costs_without_measurement_labels(prepared):
    from preparation_revision import carry
    args, config, _, _ = prepared
    current = copy.deepcopy(config)
    current['limits'] = None
    current['slo_mode'] = 'relative_max'
    current['reference_configuration']['dbo_decode_token_threshold'] = 32
    current['reference_configuration']['dbo_prefill_token_threshold'] = 512
    inputs = args.directory.parent/'new-inputs'
    inputs.mkdir()
    for name in ('calibration.jsonl', 'heldout.jsonl', 'warmup.jsonl'):
        (inputs/name).write_bytes((args.directory/'inputs'/name).read_bytes())
    before = (args.directory/'official-config.json').read_bytes()
    carry(current, args.directory, inputs)
    value = read_json(inputs/'prior-preparation.json')
    assert value['observations_imported'] is False
    assert len(value['receipts']) == len(list((args.directory/'validation').glob('*/worker-result.json')))
    assert (args.directory/'official-config.json').read_bytes() == before
    current['hardware']['devices']['0']['uuid'] = 'another-device'
    with pytest.raises(ValueError, match='physical GPU'):
        carry(current, args.directory, inputs)


def test_preparation_revision_rejects_search_observations(prepared):
    from preparation_revision import carry
    args, config, _, _ = prepared
    state_path = args.directory/'comparison/v2-seed0/state.json'
    state = read_json(state_path)
    state['observations'] = [{'candidate_id': 'already-measured'}]
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='before search'):
        carry(config, args.directory, args.directory/'inputs')

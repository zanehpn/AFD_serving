import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_resolves_dependencies_without_old_results(tmp_path):
    checkout = tmp_path / 'checkout'
    for directory in ['inputs', 'migration', 'scripts', 'environment', 'services']:
        shutil.copytree(ROOT / directory, checkout / directory, ignore=shutil.ignore_patterns('__pycache__', 'PREPARED.json', 'native-runtime.json'))
    subprocess.run([sys.executable, str(checkout / 'migration/prepare.py')], check=True)
    assert not list(checkout.glob('results/**/COMPLETE'))
    for model, old in [('deepseek-v2-lite', 'deepseek-v2-lite-v026-dynamic-fbss-v9-20260906'), ('qwen36', 'qwen36-v026-dynamic-fbss-v9b-20260906')]:
        protocol = checkout / f'results/afd_protocols/{model}-v026-dynamic-fbss-v10-20260906'
        freeze = json.loads((protocol / 'PRE_DYNAMIC_CALIBRATION_CODE_FREEZE.json').read_text())
        controller = json.loads(Path(freeze['source_v9_controller']).read_text())
        profile = Path(controller['predictor']['calibration_profile']['path'])
        assert profile.is_file()
        for name in ['calibration-max-deployment.json', 'calibration-fbss-deployment.json']:
            assert Path(json.loads((protocol / name).read_text())['calibration_trace']).is_file()
        inherited = json.loads((checkout / 'inputs/calibration-max' / model / 'combined-controller-v10.json').read_text())
        current = json.loads((protocol / 'combined-controller-v10.json').read_text())
        assert freeze['migration']['old_max_results_reused'] is True
        for original_row, current_row in zip(inherited['predictor']['calibration_guard_by_rps'], current['predictor']['calibration_guard_by_rps']):
            for key in ['rps', 'prefill_age_guard_ms', 'progress_gap_guard_ms']:
                assert original_row[key] == current_row[key]
            assert Path(current_row['source']).is_file()
    result = subprocess.run([sys.executable, str(checkout / 'migration/prepare.py')], capture_output=True)
    assert result.returncode != 0
    audit = json.loads((checkout / 'results/afd_suites/dynamic-fbss-v10-inputs-20260906/trace-isolation-audit.json').read_text())
    assert audit['status'] == 'PASS' and all(x == 0 for x in audit['overlap_counts'].values())


@pytest.mark.parametrize('na,ne', [(2, 2), (4, 2), (5, 1), (4, 4), (6, 2), (7, 1)])
def test_native_launch_has_host_paths_and_truthful_metadata(tmp_path, monkeypatch, na, ne):
    checkout = tmp_path / 'checkout'
    shutil.copytree(ROOT / 'scripts/afd', checkout / 'scripts/afd')
    plugin = checkout / 'plugin'
    (plugin / 'afd_plugin').mkdir(parents=True)
    (plugin / 'afd_plugin/__init__.py').write_text('')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    wrapper = bin_dir / 'python3'
    wrapper.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
if sys.argv[1].endswith('native_service.py'):
    d = pathlib.Path(sys.argv[3])
    (d / 'native-service.pid').write_text(str(os.getpid()))
    (d / 'captured.json').write_text(json.dumps(sys.argv[5:]))
else:
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
''')
    wrapper.chmod(0o755)
    ids = [1, 3, 5, 7, 0, 2, 4, 6]
    attention, expert = ','.join(map(str, ids[:na])), ','.join(map(str, ids[na:na + ne]))
    env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'], ECODEP_MODEL_PATH=str(checkout / 'model'),
               ECODEP_MODEL_TAG='test-model', ECODEP_RUN_ID='native-smoke', ECODEP_AFD_PLUGIN_ROOT=str(plugin),
               ECODEP_ATTENTION_GPUS=attention, ECODEP_EXPERT_GPUS=expert,
               ECODEP_ATTENTION_RANKS=str(na), ECODEP_EXPERT_RANKS=str(ne), ECODEP_ENABLE_ROUTING_SIDECAR='1',
               ECODEP_API_PORT='18100', ECODEP_EXPERT_API_PORT='18101', ECODEP_AFD_PORT='16339', ECODEP_DP_RPC_BASE_PORT='29650')
    subprocess.run(['bash', str(checkout / 'scripts/afd/start_server_native.sh')], env=env, check=True)
    run = checkout / 'results/afd_serving/test-model/native-smoke'
    config = json.loads((run / 'launch_config.json').read_text())
    assert config['runtime'] == 'native' and config['memory_limit_bytes'] is None
    assert config['attention_gpus'] == attention and config['expert_gpus'] == expert
    assert config['attention_ranks'] == na and config['expert_ranks'] == ne
    args = json.loads((run / 'captured.json').read_text())
    assert '/model' not in args and '/results' not in args
    assert args[args.index('--results') + 1] == str(run)
    for flag, value in [('--api-port', '18100'), ('--expert-api-port', '18101'), ('--afd-port', '16339'), ('--dp-rpc-base-port', '29650')]:
        assert args[args.index(flag) + 1] == value
    assert config['api_port'] == 18100
    spec = importlib.util.spec_from_file_location('tested_native_launcher', ROOT / 'scripts/afd/launch_pair.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(sys, 'argv', args[1:])
    parsed = launcher.parse_args()
    for role, count in [('attention', na), ('expert', ne)]:
        command = launcher.command(parsed, role)
        assert command[command.index('--data-parallel-size') + 1] == str(count)
        topology = json.loads(command[command.index('--additional-config') + 1])['afd']
        assert topology['num_attention_ranks'] == na and topology['num_ffn_ranks'] == ne
        assert '--enforce-eager' in command and '--enable-dbo' in command


def test_native_stop_owns_only_recorded_process_group(tmp_path):
    service = ROOT / 'migration/native_service.py'
    subprocess.run([sys.executable, str(service), 'start', str(tmp_path), '--', sys.executable, '-c', 'import time; time.sleep(120)'], check=True)
    state = json.loads((tmp_path / 'native-service.json').read_text())
    try:
        duplicate = subprocess.run([sys.executable, str(service), 'start', str(tmp_path), '--', 'true'], capture_output=True)
        assert duplicate.returncode != 0
    finally:
        subprocess.run([sys.executable, str(service), 'stop', str(tmp_path)], check=True)
    # A second stop must be harmless, including when a zombie awaits reaping.
    subprocess.run([sys.executable, str(service), 'stop', str(tmp_path)], check=True)


def test_hardware_rejects_wrong_family_and_duplicate_gpus():
    spec = importlib.util.spec_from_file_location('check_gpus', ROOT / 'migration/check_gpus.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [[str(i), ' NVIDIA A100-SXM4-80GB', '81920'] for i in range(8)]
    assert len(module.validate([1, 3, 5, 7], rows)) == 4
    for count in (6, 8):
        assert len(module.validate(list(range(count)), rows, (count,))) == count
        with pytest.raises(ValueError):
            module.validate(list(range(count)), rows)
    with pytest.raises(ValueError):
        module.validate([0, 1, 1, 3], rows)
    with pytest.raises(ValueError):
        module.validate([0, 1, 2, 3], [[str(i), 'H100', '81920'] for i in range(4)])


def test_native_freeze_and_audit_end_to_end(tmp_path):
    """Synthetic calibration report exercises provenance plumbing, never GPU outcomes."""
    import hashlib
    checkout = tmp_path / 'checkout'
    plugin = ROOT / 'third_party/afd-plugin-ecodep-v026-dvfs-v6-four-stage'
    if not (plugin / '.git').exists():
        pytest.skip('restore_plugins.py is needed for exact-commit freeze integration')
    # Restore the inherited freeze's actual commit; the installed native BO
    # plugin may now be newer. Never relabel the historical fixture.
    historical = json.loads((ROOT / 'inputs/protocols/qwen36/FREEZE.json').read_text())['plugin_commit']
    historical_plugin = tmp_path / 'historical-plugin'
    subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(plugin), str(historical_plugin)], check=True)
    subprocess.run(['git', '-C', str(historical_plugin), 'checkout', '--detach', historical], check=True)
    plugin = historical_plugin
    for directory in ['inputs', 'migration', 'scripts', 'environment', 'services']:
        shutil.copytree(ROOT / directory, checkout / directory, ignore=shutil.ignore_patterns('__pycache__', 'PREPARED.json', 'native-runtime.json'))
    subprocess.run([sys.executable, 'migration/prepare.py'], cwd=checkout, check=True)
    target = checkout / 'results/afd_protocols/qwen36-v026-dynamic-fbss-v10-20260906'
    source = checkout / 'results/afd_protocols/qwen36-v026-dynamic-fbss-v9b-20260906'
    traces = checkout / 'results/afd_suites/dynamic-fbss-v10-inputs-20260906'
    cfg = json.loads((source / 'combined-controller-v9.json').read_text())
    cfg.update(controller_revision=10, method='fbss_constrained_causal_stage_routing_joint_frequency_power_v10')
    controller_path = target / 'combined-controller-v10.json'
    controller_path.write_text(json.dumps(cfg))
    (target / 'CALIBRATION_RESULTS_V10.json').write_text(json.dumps(dict(
        status='complete_calibration_only', heldout_outcomes_read=0,
        controller_sha256=hashlib.sha256(controller_path.read_bytes()).hexdigest())))
    path = target / 'calibration-max-deployment.json'
    dep = json.loads(path.read_text())
    dep['plugin']['root'] = str(plugin)
    path.write_text(json.dumps(dep))
    model = checkout / 'artifacts/models/Qwen3.6-35B-A3B'
    model.mkdir(parents=True)
    (model / 'config.json').write_text('{"fixture": true}')
    subprocess.run([sys.executable, 'scripts/afd/freeze_dynamic_v10.py', str(target), str(source), str(traces)], cwd=checkout, check=True, capture_output=True)
    subprocess.run([sys.executable, 'scripts/afd/audit_dynamic_v10_freeze.py', str(target), str(traces)], cwd=checkout, check=True)
    # A code change after freeze must make the launch audit reject the run.
    with (checkout / 'migration/native_service.py').open('a') as out:
        out.write('\n# changed after freeze\n')
    rejected = subprocess.run([sys.executable, 'scripts/afd/audit_dynamic_v10_freeze.py', str(target), str(traces)], cwd=checkout, capture_output=True)
    assert rejected.returncode != 0


def test_inherited_max_summary_is_not_labeled_same_server(tmp_path):
    checkout = tmp_path / 'checkout'
    for directory in ['inputs', 'migration', 'scripts', 'environment', 'services']:
        shutil.copytree(ROOT / directory, checkout / directory, ignore=shutil.ignore_patterns('__pycache__', 'PREPARED.json', 'native-runtime.json'))
    subprocess.run([sys.executable, 'migration/prepare.py'], cwd=checkout, check=True)
    source = checkout / 'inputs/calibration-max/qwen36'
    protocol = checkout / 'results/afd_protocols/qwen36-v026-dynamic-fbss-v10-20260906'
    suite = checkout / 'results/afd_suites/Qwen3.6-35B-A3B/fixture-dynamic'
    serving = checkout / 'results/afd_serving/Qwen3.6-35B-A3B/fixture-dynamic'
    suite.mkdir(parents=True)
    (suite / 'COMPLETE').touch()
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest['comparison_contract']['container'] = {'image': 'native-fixture', 'image_id': 'fixture'}
    manifest['controller'] = {'source_config': str(protocol / 'combined-controller-v10.json')}
    (suite / 'manifest.json').write_text(json.dumps(manifest))
    for rate in [1, 2, 4]:
        case = serving / f'rps-{rate}'
        shutil.copytree(source / f'rps-{rate}', case)
        telemetry = json.loads((case / 'telemetry.json').read_text())
        telemetry['request_log'] = str(case / 'requests.jsonl')
        (case / 'telemetry.json').write_text(json.dumps(telemetry))
        (case / 'controller-summary.json').write_text(json.dumps({'status': 'complete', 'restored_to_guard': True, 'routing_updates': 1}))
        (case / 'controller-actions.jsonl').write_text('')
    subprocess.run([sys.executable, 'scripts/afd/summarize_dynamic_v10_calibration.py', str(protocol), str(tmp_path / 'missing-destination-max'), str(suite)], cwd=checkout, check=True, capture_output=True)
    report = json.loads((protocol / 'CALIBRATION_RESULTS_V10.json').read_text())
    assert report['destination_max_calibration_run'] is False
    assert report['same_server_saving_measured'] is False
    assert report['calibration_comparison_scope'] == 'cross_server_inherited_max_reference'
    assert 'pooled_raw_saving' not in report
    assert report['pooled_reference_raw_saving'] == 0
    # Fail closed if source calibration evidence is modified after preparation.
    (source / 'rps-1/summary.json').write_text('{}')
    failed = subprocess.run([sys.executable, 'scripts/afd/summarize_dynamic_v10_calibration.py', str(protocol), str(tmp_path / 'missing-destination-max'), str(suite)], cwd=checkout, capture_output=True)
    assert failed.returncode != 0


def test_reuse_max_upgrade_only_before_measurement(tmp_path):
    checkout = tmp_path / 'checkout'
    for directory in ['inputs', 'migration', 'scripts', 'environment', 'services']:
        shutil.copytree(ROOT / directory, checkout / directory, ignore=shutil.ignore_patterns('__pycache__', 'PREPARED*.json', 'native-runtime.json'))
    subprocess.run([sys.executable, 'migration/prepare.py'], cwd=checkout, check=True)
    marker = checkout / 'environment/PREPARED.json'
    old = json.loads(marker.read_text())
    old.pop('max_calibration_mode')
    old['calibration_results_reused'] = False
    marker.write_text(json.dumps(old))
    subprocess.run([sys.executable, 'migration/reuse_max.py'], cwd=checkout, check=True)
    assert json.loads(marker.read_text())['max_calibration_mode'] == 'reuse_source_a100_max'
    assert (checkout / 'environment/PREPARED-before-max-reuse.json').is_file()
    assert not list(checkout.glob('results/**/COMPLETE'))
    marker.write_text(json.dumps(old))
    started = checkout / 'results/afd_suites/Qwen3.6-35B-A3B/qwen36-v026-dynamic-fbss-v10-calibration-dynamic-20260906'
    started.mkdir(parents=True)
    rejected = subprocess.run([sys.executable, 'migration/reuse_max.py'], cwd=checkout, capture_output=True)
    assert rejected.returncode != 0
    assert not json.loads(marker.read_text())['calibration_results_reused']


def test_parallel_plan_explores_both_topologies_and_isolates_resources():
    spec = importlib.util.spec_from_file_location('parallel', ROOT / 'migration/parallel_8gpu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.plan(ROOT, 'test-plan')
    module.validate_dse(ROOT, rows)
    all_ports = [p for row in rows for p in row['ports']]
    assert len(all_ports) == len(set(all_ports))
    groups = [{int(v) for key in ['ECODEP_ATTENTION_GPUS', 'ECODEP_EXPERT_GPUS'] for v in row['environment'][key].split(',')} for row in rows]
    assert groups == [{0, 1, 2, 3}, {4, 5, 6, 7}]
    assert rows[0]['workspace'] != rows[1]['workspace']
    assert all(row['static_dse']['eight_gpu_global_optimality_claimed'] is False for row in rows)
    for row in rows:
        assert row['static_dse']['candidate_topologies'] == ['2A1E', '2A2E']
        assert row['static_dse']['allocation_includes_inactive_gpu'] is True
    with pytest.raises(ValueError):
        module.plan(ROOT, '../escape')


def test_selected_model_continuation_excludes_other_model():
    for selected, expected in [('deepseek-v2-lite', 'DeepSeek-V2-Lite-Chat'), ('qwen36', 'Qwen3.6-35B-A3B')]:
        code = "import runpy, json; d=runpy.run_path('scripts/afd/continue_dynamic_v10_after_calibration.py'); print(json.dumps(d['MODELS']))"
        out = subprocess.check_output([sys.executable, '-c', code], cwd=ROOT, env=dict(os.environ, ECODEP_MODELS=selected), text=True)
        models = json.loads(out)
        assert len(models) == 1 and models[0][1] == expected

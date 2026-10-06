"""Check package isolation and the read-only hardware export contract."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bo_capture_hardware", ROOT / "capture_hardware.py")
hardware = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hardware)


def test_cli_from_unrelated_working_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "cli.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    assert "compare-round" in result.stdout
    assert "plan-calibration" in result.stdout


@pytest.mark.parametrize("overlap", [False, True])
def test_provisioning_cli_checks_request_isolation_before_reading_workload_lengths(tmp_path, overlap):
    profile = {"selection_split": "calibration", "ratio_scope": "attention_workers_per_ffn_service_group",
               "batch_size_per_attention": 16, "ratios": [1, 2, 3],
               "power_model": {"type": "constant_role_power_v1", "selection_split": "calibration",
                               "assumption": "ratio_independent_cycle_average_role_power",
                               "attention_worker_w": 200., "ffn_service_group_w": 200.},
               "latency_coefficients_ms": {stage: {"slope": .01, "intercept": 1.}
                                            for stage in ("attention", "ffn", "communication_roundtrip")}}
    (tmp_path / 'profile.json').write_text(json.dumps(profile))
    (tmp_path / 'calibration.jsonl').write_text(json.dumps({"source_index": 1, "prompt_tokens": 100, "decode_steps": 10}) + '\n')
    # Heldout needs identity only; its lengths must not enter the estimator.
    (tmp_path / 'heldout.jsonl').write_text(json.dumps({"source_index": 1 if overlap else 2}) + '\n')
    command = [sys.executable, str(ROOT / 'cli.py'), 'provision-ratios']
    for name in ('profile', 'calibration', 'heldout', 'output'):
        suffix = '.jsonl' if name in ('calibration', 'heldout') else '.json'
        command += ['--' + name, str(tmp_path / (name + suffix))]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    if overlap:
        assert result.returncode != 0 and 'identity overlap' in result.stderr
        assert not (tmp_path / 'output.json').exists()
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads((tmp_path / 'output.json').read_text())
        assert report['workload_moments']['theta'] == 104.5
        assert report['split_audit']['overlap'] == {'source_index': 0}
        assert report['execution_eligibility_changed'] is False
        assert report['energy_advice']['recommendation'] == report['recommendation']


def test_capabilities_are_normalized_and_shutdown_on_failure():
    calls = []
    nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetPowerManagementLimitConstraints=lambda h: (100000, 400000),
        nvmlDeviceGetSupportedMemoryClocks=lambda h: [1593],
        nvmlDeviceGetSupportedGraphicsClocks=lambda h, m: [1050, 1410],
        nvmlDeviceGetUUID=lambda h: b"GPU-fixture",
        nvmlDeviceGetName=lambda h: b"fixture",
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=80 * 2**30),
    )
    dev = hardware.capture([2], nvml)["devices"]["2"]
    assert dev["min_power_w"] == 100 and dev["max_power_w"] == 400
    assert dev["memory_mib"] == 81920 and dev["uuid"] == "GPU-fixture"
    assert dev["clock_pairs"] == [{"memory_mhz": 1593, "graphics_mhz": f} for f in (1050, 1410)]
    assert calls == ["init", "shutdown"]
    def unsupported(handle):
        raise RuntimeError("clock query unsupported")
    nvml.nvmlDeviceGetSupportedMemoryClocks = unsupported
    with pytest.raises(RuntimeError, match="unsupported"):
        hardware.capture([2], nvml)
    assert calls == ["init", "shutdown", "init", "shutdown"]

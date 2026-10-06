"""Guard fitting must use valid calibration evidence, including transport delay."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("make_v10", Path(__file__).with_name("make_dynamic_v10_controller.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source.json"
    config = json.loads((root / "inputs/protocols/qwen36/combined-controller-v9.json").read_text())
    config["predictor"]["calibration_profile"]["path"] = str(root / "inputs/protocols/qwen36/routing-profile-v9.json")
    write(source, config)
    profile = json.loads(Path(config["predictor"]["calibration_profile"]["path"]).read_text())
    suite = tmp_path / "results/afd_suites/Qwen3.6-35B-A3B/test-calibration"
    manifest = {"evaluation_split": "calibration", "model": "Qwen3.6-35B-A3B", "trace_sha256": profile["trace"]["sha256"]}
    write(suite / "manifest.json", manifest)
    (suite / "COMPLETE").touch()
    serving = tmp_path / "results/afd_serving/Qwen3.6-35B-A3B/test-calibration"
    for rate in [1, 2, 4]:
        case = serving / f"rps-{rate}"
        write(case / "summary.json", {"completed_requests": 200, "failed_requests": 0,
              "ttft_ms": {"p90": 1000 * rate}, "tpot_ms": {"p99": 200}})
        write(case / "telemetry.json", {"returncode": 0, "sample_error_count": 0, "sample_time_coverage": 1})
        write(case / "gpu-contamination-validation.json", {"verified": True, "foreign_process_samples": 0})
    return source, suite, serving


def test_uses_calibration_p90_and_progress_delivery_allowance(evidence):
    source, suite, _ = evidence
    config = module.build(source, suite)
    curve = config["predictor"]["calibration_guard_by_rps"]
    assert [r["prefill_age_guard_ms"] for r in curve] == [1050, 2100, 4200]
    assert all(r["progress_gap_guard_ms"] == 510 for r in curve)
    assert config["predictor"]["relative_service_budget_ratio"] == 1.05
    assert config["predictor"]["guard_calibration"]["development_requests"] == 600


def test_refuses_heldout_latency_feedback(evidence):
    source, suite, _ = evidence
    path = suite / "manifest.json"
    value = json.loads(path.read_text())
    value["evaluation_split"] = "heldout"
    write(path, value)
    with pytest.raises(ValueError, match="held-out"):
        module.build(source, suite)


def test_refuses_incomplete_or_contaminated_calibration(evidence):
    source, suite, serving = evidence
    path = serving / "rps-2/gpu-contamination-validation.json"
    write(path, {"verified": True, "foreign_process_samples": 1})
    with pytest.raises(ValueError, match="ownership"):
        module.build(source, suite)
    (suite / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        module.build(source, suite)

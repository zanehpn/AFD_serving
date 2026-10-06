"""Behavior checks for causal batching, relative service gates, and guards."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).with_name("controller.py")
SPEC = importlib.util.spec_from_file_location("dvfs_v10_test_target", MODULE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
Controller = module.CausalControllerV10
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def config():
    path = ROOT / "inputs/protocols/qwen36/combined-controller-v9.json"
    cfg = json.loads(path.read_text())
    cfg["controller_revision"] = 10
    cfg["signals"]["rate_window_ms"] = 1000
    cfg["predictor"]["relative_service_budget_ratio"] = 1.05
    cfg["predictor"]["role_gpu_counts"] = {"attention": 2, "expert": 2}
    cfg["predictor"]["calibration_guard_by_rps"] = [
        {"rps": 1, "prefill_age_guard_ms": 1000, "progress_gap_guard_ms": 600},
        {"rps": 2, "prefill_age_guard_ms": 2000, "progress_gap_guard_ms": 700},
        {"rps": 4, "prefill_age_guard_ms": 4000, "progress_gap_guard_ms": 900},
    ]
    return cfg


def submit(controller, request, stamp):
    controller.consume({"event": "submit", "wall_ns": stamp, "request_id": request,
                        "input_tokens": 100, "requested_output_tokens": 100})


def test_arrival_rate_counts_requests_not_reciprocal_gaps(config):
    controller = Controller(config, "dynamic-ae")
    submit(controller, "a", 1_000_000_000)
    submit(controller, "b", 1_000_000_001)
    assert controller.signals(1_100_000_000)["arrival_rate_ewma_rps"] == 2
    assert controller.signals(2_100_000_000)["arrival_rate_ewma_rps"] == 0


def test_decode_rate_is_invariant_to_batch_packet_spacing(config):
    first = Controller(config, "dynamic-ae")
    second = Controller(config, "dynamic-ae")
    for controller in [first, second]:
        submit(controller, "a", 1_000_000_000)
        submit(controller, "b", 1_000_000_000)
    first.consume({"event": "progress_batch", "wall_ns": 1_250_000_000,
                   "requests": [{"request_id": "a", "output_chunks": 10},
                                {"request_id": "b", "output_chunks": 20}]})
    for request, count, stamp in [("a", 10, 1_250_000_000), ("b", 20, 1_250_000_001)]:
        second.consume({"event": "progress", "request_id": request,
                        "output_chunks": count, "wall_ns": stamp})
    for controller in [first, second]:
        assert controller.signals(1_300_000_000)["decode_output_token_rate_ewma"] == 30
        assert controller.signals(2_300_000_000)["decode_output_token_rate_ewma"] == 0


def test_duplicate_and_finished_progress_cannot_inflate_work(config):
    controller = Controller(config, "dynamic-ae")
    submit(controller, "a", 1_000_000_000)
    for event, count in [("progress", 10), ("progress", 10), ("finish", 12), ("progress", 100)]:
        controller.consume({"event": event, "wall_ns": 1_200_000_000,
                            "request_id": "a", "output_chunks": count})
    assert controller.signals(1_300_000_000)["decode_output_token_rate_ewma"] == 12
    assert not controller.requests


def test_window_never_counts_future_timestamp(config):
    controller = Controller(config, "dynamic-ae")
    submit(controller, "a", 2_000_000_000)
    assert controller._arrival_ewma_at(1_000_000_000) == 0


def test_guard_curve_interpolates_and_clamps(config):
    controller = Controller(config, "dynamic-ae")
    assert controller._guard_limits(0) == (1000, 600)
    assert controller._guard_limits(3) == (3000, 800)
    assert controller._guard_limits(8) == (4000, 900)


def test_startup_and_stale_events_still_force_max(config):
    controller = Controller(config, "dynamic-ae")
    submit(controller, "a", 1_000_000_000)
    states, _ = controller.desired(1_100_000_000)
    assert set(states.values()) == {"f1410-p400"}
    _, signals = controller.desired(2_100_000_000)
    assert signals["safety_fallback"] == 1


def selection_case(config, capacity):
    cfg = copy.deepcopy(config)
    cfg["predictor"]["work_weights"] = {role: {"prefill": 1, "decode": 0}
                                         for role in ["attention", "expert"]}
    for role, model in cfg["predictor"]["role_models"].items():
        for point in model["operating_points"]:
            point["capacity_reference_work_per_s"] = capacity if point["state"] == "f1050-p400" else 4
        for state, fitted in model["power_model"]["states"].items():
            fitted["idle_intercept_w"] = 50 if state == "f1050-p400" else 100
            fitted["dynamic_slope_w"] = 0
    controller = Controller(cfg, "dynamic-ae")
    signals = controller.signals(2_000_000_000)
    signals.update(startup_window_complete=True, arrival_rate_ewma_rps=8,
                   submitted_input_tokens_ewma=cfg["predictor"]["reference_workload"]["input_tokens_mean"])
    return controller, signals


def test_relative_gate_allows_near_max_capacity_during_overload(config):
    controller, signals = selection_case(config, 3.9)
    states, reason = controller._select_joint_v5(signals)
    assert states["expert"] == "f1050-p400"
    assert reason == "v10_min_power_relative_service_feasible_joint"


def test_relative_gate_rejects_material_capacity_loss(config):
    controller, signals = selection_case(config, 3)
    states, _ = controller._select_joint_v5(signals)
    assert states["expert"] == "f1410-p400"


def test_calibration_slo_failure_excludes_cheap_action(config):
    config["predictor"]["joint_operating_points"][1]["calibration_slo_eligible"] = False
    controller, signals = selection_case(config, 3.9)
    states, _ = controller._select_joint_v5(signals)
    assert states["expert"] == "f1410-p400"


@pytest.mark.parametrize("field,value,expected", [
    ("oldest_prefill_ms", 4001, "v10_calibrated_prefill_age_guard"),
    ("oldest_progress_gap_ms", 901, "v10_calibrated_progress_stall_guard"),
])
def test_calibrated_latency_guard_has_priority(config, field, value, expected):
    controller, signals = selection_case(config, 3.9)
    signals.update(decode=1)
    signals[field] = value
    states, reason = controller._select_joint_v5(signals)
    assert reason == expected
    assert set(states.values()) == {"f1410-p400"}


def test_configuration_revision_cannot_silently_run_old_policy(config):
    config["controller_revision"] = 9
    with pytest.raises(ValueError, match="revision-10"):
        Controller(config, "dynamic-ae")

from __future__ import annotations

import copy

from .runtime import select_runtime_operating_point
from .test_model import _candidate


def _runtime_candidates() -> list[dict]:
    guard = _candidate("guard")
    guard["knobs"].update(
        {"attention_power_w": 400, "expert_power_w": 400}
    )
    low = copy.deepcopy(guard)
    low["id"] = "low"
    low["knobs"].update(
        {"attention_mhz": 1290, "expert_mhz": 1290, "attention_power_w": 250, "expert_power_w": 250}
    )
    low["power_model"]["dynamic_slope_w"] = 300.0
    other_topology = copy.deepcopy(low)
    other_topology["id"] = "other"
    other_topology["topology"]["expert_gpus"] = 1
    other_topology["topology"]["expert_ep"] = 1
    return [guard, low, other_topology]


def test_runtime_never_changes_frozen_topology() -> None:
    candidates = _runtime_candidates()
    decision = select_runtime_operating_point(
        candidates,
        frozen_topology=candidates[0]["topology"],
        reference_workload={},
        signals={"arrival_rate_ewma_rps": 1.0, "offered_utilization": 0.2},
        current_candidate_id="guard",
        routing_calibration_p95=1.2,
    )
    assert decision.candidate_id == "low"


def test_urgent_request_age_immediately_selects_guard() -> None:
    candidates = _runtime_candidates()
    decision = select_runtime_operating_point(
        candidates,
        frozen_topology=candidates[0]["topology"],
        reference_workload={},
        signals={
            "arrival_rate_ewma_rps": 1.0,
            "offered_utilization": 0.2,
            "oldest_prefill_ms": 351.0,
        },
        current_candidate_id="low",
        routing_calibration_p95=1.2,
    )
    assert decision.candidate_id == "guard"
    assert decision.reason == "urgent_tail_guard"
    assert decision.immediate_upshift


def test_stale_routing_uses_calibration_tail() -> None:
    candidates = _runtime_candidates()
    decision = select_runtime_operating_point(
        candidates,
        frozen_topology=candidates[0]["topology"],
        reference_workload={},
        signals={
            "arrival_rate_ewma_rps": 1.0,
            "offered_utilization": 0.2,
            "routing_imbalance": 1.0,
            "routing_age_ms": 1000.0,
        },
        current_candidate_id="guard",
        routing_calibration_p95=1.4,
    )
    assert decision.routing_source == "calibration_p95_stale_fallback"

from __future__ import annotations

import copy

import pytest

from .model import STAGES, evaluate_candidate, pipeline_time_ms, select_candidate


def _candidate(candidate_id: str = "safe") -> dict:
    return {
        "id": candidate_id,
        "selection_split": "calibration",
        "parallelism_legal": True,
        "memory_feasible": True,
        "microbatches": 2,
        "layers": 28,
        "requests_per_pipeline": 1,
        "guard_pipeline_time_ms": 20.3,
        "stage_models": {
            stage: {
                "intercept_ms": duration,
                "prefill_ms_per_token": 0.0,
                "decode_ms_per_token": 0.0,
                "p90_residual_ms": 0.0,
            }
            for stage, duration in zip(STAGES, (10.0, 2.0, 8.0, 1.0), strict=True)
        },
        "calibration_latency_ratios": {"p90_ttft": 1.01, "p90_tpot": 1.02},
        "power_model": {
            "idle_intercept_w": 400.0,
            "dynamic_slope_w": 600.0,
            "total_power_cap_w": 1600.0,
        },
        "topology": {"attention_gpus": 2, "expert_gpus": 2},
        "knobs": {"attention_mhz": 1350, "expert_mhz": 1290},
    }


WORKLOAD = {
    "arrival_rate_rps": 1.0,
    "offered_utilization": 0.5,
    "routing_imbalance": 1.0,
}


def test_pipeline_is_set_by_longest_stage_plus_fill_drain() -> None:
    stage_times = dict(zip(STAGES, (10.0, 2.0, 8.0, 1.0), strict=True))
    assert pipeline_time_ms(stage_times, microbatches=2, layers=28) == pytest.approx(
        20.0 + 11.0 / 28.0
    )


def test_slower_non_bottleneck_only_changes_fill_drain() -> None:
    base = dict(zip(STAGES, (10.0, 2.0, 8.0, 1.0), strict=True))
    slower = {**base, "ffn_compute": 9.0}
    assert pipeline_time_ms(slower, microbatches=4, layers=28) - pipeline_time_ms(
        base, microbatches=4, layers=28
    ) == pytest.approx(1.0 / 28.0)


def test_speeding_bottleneck_reduces_steady_state_time() -> None:
    base = dict(zip(STAGES, (10.0, 2.0, 8.0, 1.0), strict=True))
    faster = {**base, "attention_compute": 9.0}
    assert pipeline_time_ms(faster, microbatches=4, layers=28) < pipeline_time_ms(
        base, microbatches=4, layers=28
    )


def test_selector_minimizes_energy_after_safety_gates() -> None:
    high = _candidate("high")
    low = _candidate("low")
    low["power_model"]["dynamic_slope_w"] = 400.0
    selected, rows = select_candidate([high, low], WORKLOAD)
    assert len(rows) == 2
    assert selected is not None
    assert selected.candidate_id == "low"


def test_latency_budget_fails_closed() -> None:
    candidate = _candidate()
    candidate["calibration_latency_ratios"]["p90_tpot"] = 1.051
    result = evaluate_candidate(candidate, WORKLOAD)
    assert not result.feasible
    assert "p90_tpot_calibration_budget" in result.rejection_reasons


def test_non_calibration_profile_is_rejected() -> None:
    candidate = _candidate()
    candidate["selection_split"] = "heldout"
    result = evaluate_candidate(candidate, WORKLOAD)
    assert not result.feasible
    assert "non_calibration_profile" in result.rejection_reasons


def test_missing_stage_fails_closed() -> None:
    candidate = _candidate()
    del candidate["stage_models"]["f2a_combine"]
    result = evaluate_candidate(candidate, WORKLOAD)
    assert not result.feasible
    assert any(reason.startswith("incomplete_model:") for reason in result.rejection_reasons)


def test_infeasible_low_power_point_cannot_win() -> None:
    unsafe = _candidate("unsafe")
    unsafe["power_model"]["dynamic_slope_w"] = 1.0
    unsafe["memory_feasible"] = False
    safe = copy.deepcopy(_candidate("safe"))
    selected, _ = select_candidate([unsafe, safe], WORKLOAD)
    assert selected is not None
    assert selected.candidate_id == "safe"


def test_primary_relative_contract_does_not_require_absolute_headroom() -> None:
    candidate = _candidate()
    candidate["requests_per_pipeline"] = 0.001
    relative = evaluate_candidate(candidate, WORKLOAD)
    absolute = evaluate_candidate(candidate, WORKLOAD, capacity_headroom=1.15)
    assert relative.feasible
    assert not absolute.feasible
    assert "capacity_headroom" in absolute.rejection_reasons

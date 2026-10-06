from __future__ import annotations

import copy

import pytest

from .model import evaluate_candidate, select_candidate
from .search import bottleneck_guided_search
from .workload_response import LATENCY_METRICS, predict_response


def curve(latencies, powers, durations):
    return {
        "type": "finite_workload_piecewise_response_v1", "selection_split": "calibration",
        "workload_key": "model-hardware-shapes-arrivals", "measurement_gpu_ids": [0, 1, 2, 3],
        "request_count": 100, "output_tokens": 1000,
        "knots": [dict(rate_rps=rate, duration_s=duration, power_w=power,
                       **dict.fromkeys(LATENCY_METRICS, latency))
                  for rate, latency, power, duration in zip((1., 2., 4.), latencies, powers, durations)],
    }


def candidates():
    baseline = curve([100, 100, 300], [400, 500, 600], [110, 60, 40])
    reduced = curve([95, 200, 800], [300, 350, 400], [110, 75, 65])
    return [dict(id=name, performance_model="workload_response_v1", selection_split="calibration",
                 parallelism_legal=True, memory_feasible=True,
                 workload_response=response, reference_workload_response=baseline,
                 topology={"attention_gpus": [0, 1], "expert_gpus": experts}, knobs={})
            for name, response, experts in [("large", baseline, [2, 3]), ("small", reduced, [2])]]


def workload(rate=1.):
    return {"arrival_rate_rps": rate, "workload_key": "model-hardware-shapes-arrivals", "gpu_budget": 4}


def test_configuration_switches_with_load_without_topology_rules():
    points = candidates()
    assert select_candidate(points, workload(1))[0].candidate_id == "small"
    assert select_candidate(points, workload(2))[0].candidate_id == "large"
    assert select_candidate(points, workload(4))[0].candidate_id == "large"
    # IDs and topology names do not prescribe which action to select.
    points[0]["id"], points[1]["id"] = "renamed-one", "renamed-two"
    assert select_candidate(list(reversed(points)), workload(1))[0].candidate_id == "renamed-two"


def test_fbss_search_accepts_response_curves_without_invented_stage_times():
    result = bottleneck_guided_search(
        candidates(), [workload(1), workload(2), workload(4)],
        guard_candidate_id="large", max_search_path_points=8)
    assert result.predicted_best_by_workload == ("small", "large", "large")
    assert result.steps[0].bottleneck is None
    assert result.steps[0].reason == "min_energy_with_workload_latency_guard"


def test_low_load_does_not_make_reduced_configuration_automatically_safe():
    points = candidates()
    points[1]["workload_response"]["knots"][0]["p99_ttft_ms"] = 200
    assert select_candidate(points, workload(1))[0].candidate_id == "large"


def test_low_load_does_not_make_reduced_configuration_automatically_cheaper():
    points = candidates()
    points[1]["workload_response"]["knots"][0]["power_w"] = 600
    assert select_candidate(points, workload(1))[0].candidate_id == "large"


def test_service_ratio_no_longer_substitutes_for_low_load_end_to_end_latency():
    small = candidates()[1]
    small.update(guard_pipeline_time_ms=100, calibration_latency_ratios={"p90_ttft": 2.6})
    result = evaluate_candidate(small, workload())
    assert result.feasible
    assert result.capacity_rps is None
    assert result.pipeline_time_ms is None
    assert result.workload_prediction["completed_requests_per_s"] == pytest.approx(100 / 110)


def test_complete_workload_energy_includes_drain_time():
    prediction = predict_response(candidates()[0]["workload_response"], workload(1))
    assert prediction["energy_j_per_request"] == 440
    assert prediction["energy_j_per_request"] != prediction["power_w"] / 1


def test_continuous_interpolation_explains_switch_between_rates():
    points = candidates()
    assert select_candidate(points, workload(1.05))[0].candidate_id == "small"
    chosen, rows = select_candidate(points, workload(1.2))
    assert chosen.candidate_id == "large"
    small = next(row for row in rows if row.candidate_id == "small")
    assert "p90_ttft_ms_workload_budget" in small.rejection_reasons
    assert small.workload_prediction["support_bracket_rps"] == [1, 2]
    assert small.workload_prediction["evidence"] == "unvalidated_interpolation"


@pytest.mark.parametrize("rate", [0, -1, .5, 4.1, float("nan"), float("inf")])
def test_unsupported_or_invalid_rates_fail_closed(rate):
    assert select_candidate(candidates(), workload(rate))[0] is None


@pytest.mark.parametrize("key,value", [("workload_key", "different-model-or-shape"), ("gpu_budget", 8)])
def test_different_workload_or_gpu_budget_is_not_silently_extrapolated(key, value):
    assert select_candidate(candidates(), {**workload(), key: value})[0] is None


def test_missing_curve_does_not_fall_back_to_legacy_pipeline():
    point = candidates()[1]
    del point["workload_response"]
    assert not evaluate_candidate(point, workload()).feasible


def test_heldout_curve_rejected_even_if_outer_candidate_says_calibration():
    point = candidates()[1]
    point["workload_response"]["selection_split"] = "heldout"
    assert not evaluate_candidate(point, workload()).feasible


def test_absolute_capacity_requires_independent_capacity_evidence():
    result = evaluate_candidate(candidates()[0], workload(), capacity_headroom=1.15)
    assert not result.feasible
    assert "absolute_capacity_not_identified" in result.rejection_reasons


def test_inactive_gpu_energy_scope_must_match_baseline():
    point = candidates()[1]
    point["workload_response"]["measurement_gpu_ids"] = [0, 1, 2]
    assert not evaluate_candidate(point, workload()).feasible


def test_optional_throughput_gate_is_applied_separately():
    point = candidates()[1]
    point["workload_response"]["knots"][0]["duration_s"] = 150
    assert evaluate_candidate(point, workload()).feasible  # original latency-only contract
    point["output_throughput_min_ratio"] = .95
    assert "output_throughput_workload_budget" in evaluate_candidate(point, workload()).rejection_reasons


def test_reference_curve_requires_matched_requests():
    point = copy.deepcopy(candidates()[1])
    point["reference_workload_response"]["request_count"] = 400
    assert not evaluate_candidate(point, workload()).feasible


@pytest.mark.parametrize("mutation", ["duplicate_rate", "nan_power", "missing_latency"])
def test_malformed_curve_fails_closed(mutation):
    point = candidates()[1]
    knots = point["workload_response"]["knots"]
    if mutation == "duplicate_rate":
        knots[1]["rate_rps"] = 1
    elif mutation == "nan_power":
        knots[0]["power_w"] = float("nan")
    else:
        del knots[0]["p99_tpot_ms"]
    assert not evaluate_candidate(point, workload()).feasible

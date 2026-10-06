"""Finite-workload response curves for load-dependent configuration selection.

These empirical curves complement four-stage service diagnostics. They do not
turn service-time ratios into latency percentiles, extrapolate to new request
shapes, or interpret completed requests / makespan as saturation capacity.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


LATENCY_METRICS = ("p90_ttft_ms", "p99_ttft_ms", "p90_tpot_ms", "p99_tpot_ms")


def positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def interpolate(x: float, left: float, right: float, a: float, b: float) -> float:
    if left == right:
        return a
    return a + (b - a) * (x - left) / (right - left)


def predict_response(curve: Mapping[str, Any], workload: Mapping[str, Any]) -> dict:
    if curve["selection_split"] != "calibration":
        raise ValueError("non_calibration_response")
    if curve["type"] != "finite_workload_piecewise_response_v1":
        raise ValueError("unsupported_response_type")
    # The key includes model, hardware, request identities/shapes and arrival
    # pattern. Only the time scaling (rate) may vary within this curve.
    if not workload.get("workload_key") or workload["workload_key"] != curve["workload_key"]:
        raise ValueError("unsupported_workload")
    allocated = list(curve["measurement_gpu_ids"])
    if not allocated or len(allocated) != len(set(allocated)):
        raise ValueError("invalid_measurement_gpu_ids")
    if int(workload["gpu_budget"]) != len(allocated):
        raise ValueError("unsupported_gpu_budget")
    rate = positive(workload["arrival_rate_rps"], "arrival_rate_rps")
    requests = positive(curve["request_count"], "request_count")
    output_tokens = positive(curve["output_tokens"], "output_tokens")
    knots = curve["knots"]
    rates = [positive(k["rate_rps"], "knot rate") for k in knots]
    if not rates or rates != sorted(set(rates)):
        raise ValueError("response rates must be unique and increasing")
    for knot in knots:
        for field in (*LATENCY_METRICS, "duration_s", "power_w"):
            positive(knot[field], field)
    if not rates[0] <= rate <= rates[-1]:
        raise ValueError("rate_outside_calibration_support")
    left = right = knots[0]
    for knot in knots:
        if knot["rate_rps"] == rate:
            left = right = knot
            break
        if knot["rate_rps"] < rate:
            left = knot
        else:
            right = knot
            break
    lo, hi = float(left["rate_rps"]), float(right["rate_rps"])
    # Arrival span is proportional to inverse rate for a fixed request cohort.
    # Measured drain time is retained at the anchors, not mistaken for service
    # capacity or an idle/sleep state on the fourth GPU.
    duration = interpolate(1 / rate, 1 / lo, 1 / hi,
                           float(left["duration_s"]), float(right["duration_s"]))
    throughput = requests / duration
    power = interpolate(throughput, requests / float(left["duration_s"]),
                        requests / float(right["duration_s"]),
                        float(left["power_w"]), float(right["power_w"]))
    # Equal-throughput anchors can still have different powers.
    if left["duration_s"] == right["duration_s"]:
        power = interpolate(rate, lo, hi, float(left["power_w"]), float(right["power_w"]))
    return {
        "rate_rps": rate,
        "support_bracket_rps": [lo, hi],
        "evidence": "calibration_anchor" if lo == hi else "unvalidated_interpolation",
        "latencies_ms": {
            metric: interpolate(rate, lo, hi, float(left[metric]), float(right[metric]))
            for metric in LATENCY_METRICS
        },
        "duration_s": duration,
        "completed_requests_per_s": throughput,
        "output_tokens_per_s": output_tokens / duration,
        "power_w": power,
        "energy_j_per_request": power * duration / requests,
        "energy_scope": "all_allocated_gpus_including_inactive",
        "capacity_rps": None,
    }


def assess_response(candidate: Mapping[str, Any], workload: Mapping[str, Any],
                    latency_budget_ratio: float) -> tuple[dict, list[str]]:
    curve = candidate["workload_response"]
    reference = candidate["reference_workload_response"]
    if curve["measurement_gpu_ids"] != reference["measurement_gpu_ids"]:
        raise ValueError("unmatched_energy_scope")
    for key in ("request_count", "output_tokens", "workload_key"):
        if curve[key] != reference[key]:
            raise ValueError(f"unmatched_{key}")
    predicted = predict_response(curve, workload)
    baseline = predict_response(reference, workload)
    ratios = {metric: predicted["latencies_ms"][metric] / baseline["latencies_ms"][metric]
              for metric in LATENCY_METRICS}
    reasons = [f"{metric}_workload_budget" for metric, ratio in ratios.items()
               if ratio > latency_budget_ratio]
    # Match the registered calibration contract; a throughput gate is optional
    # and must be explicit, never silently imported from another protocol.
    tps_ratio = predicted["output_tokens_per_s"] / baseline["output_tokens_per_s"]
    minimum_tps = candidate.get("output_throughput_min_ratio")
    if minimum_tps is not None and tps_ratio < positive(minimum_tps, "minimum throughput ratio"):
        reasons.append("output_throughput_workload_budget")
    predicted.update({
        "latency_ratios_to_matched_max": ratios,
        "output_throughput_ratio_to_matched_max": tps_ratio,
        "energy_saving_ratio_to_matched_max":
            1 - predicted["energy_j_per_request"] / baseline["energy_j_per_request"],
        "limiting_latency_metric": max(ratios, key=ratios.get),
        "baseline": baseline,
    })
    return predicted, reasons

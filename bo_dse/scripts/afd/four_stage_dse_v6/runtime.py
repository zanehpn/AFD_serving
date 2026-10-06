#!/usr/bin/env python3
"""Fast-layer runtime decisions within a one-shot FBSS topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .model import CandidateEvaluation, evaluate_candidate


def topology_signature(topology: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Hardware identity is irrelevant; parallel shape is the static choice."""
    fields = (
        "attention_dp",
        "expert_dp",
        "attention_tp",
        "expert_tp",
        "expert_ep",
    )
    counts = {
        "attention_gpu_count": len(topology["attention_gpus"])
        if isinstance(topology["attention_gpus"], list)
        else int(topology["attention_gpus"]),
        "expert_gpu_count": len(topology["expert_gpus"])
        if isinstance(topology["expert_gpus"], list)
        else int(topology["expert_gpus"]),
    }
    return tuple((field, topology.get(field, 1)) for field in fields) + tuple(
        sorted(counts.items())
    )


def runtime_workload(
    reference: Mapping[str, float],
    signals: Mapping[str, float],
    *,
    routing_calibration_p95: float,
    routing_stale_ms: float,
) -> tuple[dict[str, float], str]:
    routing_age = float(signals.get("routing_age_ms", float("inf")))
    if routing_age > routing_stale_ms:
        routing = routing_calibration_p95
        routing_source = "calibration_p95_stale_fallback"
    else:
        routing = max(float(signals.get("routing_imbalance", 1.0)), 1.0)
        routing_source = "online_causal"
    workload = {
        "prefill_tokens_per_microbatch": max(
            float(signals.get("prefill_tokens_per_microbatch", 0.0)),
            float(reference.get("prefill_tokens_per_microbatch", 0.0)) * 0.1,
        ),
        "decode_tokens_per_microbatch": max(
            float(signals.get("decode_tokens_per_microbatch", 0.0)),
            float(reference.get("decode_tokens_per_microbatch", 0.0)) * 0.1,
        ),
        "communication_bytes_scale": max(
            float(signals.get("communication_bytes_scale", 1.0)), 0.0
        ),
        "routing_imbalance": routing,
        "arrival_rate_rps": max(float(signals.get("arrival_rate_ewma_rps", 0.0)), 0.0),
        "offered_utilization": min(
            max(float(signals.get("offered_utilization", 0.0)), 0.0), 1.0
        ),
    }
    return workload, routing_source


def _resource_level(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    knobs = candidate["knobs"]
    return (
        float(knobs["attention_mhz"]),
        float(knobs["attention_power_w"]),
        float(knobs["expert_mhz"]),
        float(knobs["expert_power_w"]),
    )


@dataclass(frozen=True)
class RuntimeDecision:
    candidate_id: str
    reason: str
    routing_source: str
    evaluation: CandidateEvaluation
    immediate_upshift: bool


def select_runtime_operating_point(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frozen_topology: Mapping[str, Any],
    reference_workload: Mapping[str, float],
    signals: Mapping[str, float],
    current_candidate_id: str | None,
    routing_calibration_p95: float,
    routing_stale_ms: float = 750.0,
    urgent_prefill_age_ms: float = 350.0,
    urgent_progress_gap_ms: float = 250.0,
    latency_budget_ratio: float = 1.05,
    capacity_headroom: float = 1.15,
) -> RuntimeDecision:
    """Choose A/E clocks and caps without changing the frozen topology."""
    signature = topology_signature(frozen_topology)
    eligible = [
        candidate
        for candidate in candidates
        if topology_signature(candidate["topology"]) == signature
        and candidate.get("selection_split") == "calibration"
    ]
    if not eligible:
        raise ValueError("no calibration candidate matches the frozen topology")
    workload, routing_source = runtime_workload(
        reference_workload,
        signals,
        routing_calibration_p95=routing_calibration_p95,
        routing_stale_ms=routing_stale_ms,
    )
    evaluations = {
        str(candidate["id"]): evaluate_candidate(
            candidate,
            workload,
            latency_budget_ratio=latency_budget_ratio,
            capacity_headroom=capacity_headroom,
        )
        for candidate in eligible
    }
    guard = max(eligible, key=_resource_level)
    urgent = (
        float(signals.get("oldest_prefill_ms", 0.0)) >= urgent_prefill_age_ms
        or float(signals.get("oldest_progress_gap_ms", 0.0)) >= urgent_progress_gap_ms
        or float(signals.get("queue_growth_rps", 0.0)) > 0.0
        and not any(row.feasible for row in evaluations.values())
    )
    if urgent:
        selected = evaluations[str(guard["id"])]
        reason = "urgent_tail_guard"
    else:
        feasible = [row for row in evaluations.values() if row.feasible]
        if feasible:
            selected = min(
                feasible,
                key=lambda row: (
                    float(row.predicted_energy_j_per_request),
                    float(row.pipeline_time_ms),
                    row.candidate_id,
                ),
            )
            reason = "four_stage_min_energy_slo_feasible"
        else:
            selected = evaluations[str(guard["id"])]
            reason = "no_safe_point_guard"
    by_id = {str(candidate["id"]): candidate for candidate in eligible}
    current = by_id.get(str(current_candidate_id))
    target = by_id[selected.candidate_id]
    immediate_upshift = current is None or any(
        target_value > current_value
        for target_value, current_value in zip(
            _resource_level(target), _resource_level(current), strict=True
        )
    )
    return RuntimeDecision(
        candidate_id=selected.candidate_id,
        reason=reason,
        routing_source=routing_source,
        evaluation=selected,
        immediate_upshift=immediate_upshift,
    )

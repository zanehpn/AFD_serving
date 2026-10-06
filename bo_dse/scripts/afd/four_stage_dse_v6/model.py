#!/usr/bin/env python3
"""Four-stage performance/energy model for Attention--FFN disaggregation.

The model deliberately separates the pipeline stages that share only two
hardware actuator domains:

    Attention compute -> A2F dispatch -> FFN compute -> F2A combine

Static topology and runtime operating points use the same feasibility test,
but topology changes are restricted to a slow control window.  The selector
only consumes a profile fitted on the calibration split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .attention_workload import predict_rank_model
from .schedule import schedule_time


STAGES = (
    "attention_compute",
    "a2f_dispatch",
    "ffn_compute",
    "f2a_combine",
)


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def validate_stage_times(stage_times_ms: Mapping[str, float]) -> dict[str, float]:
    """Validate and normalize a complete four-stage timing vector."""
    missing = [stage for stage in STAGES if stage not in stage_times_ms]
    extra = [stage for stage in stage_times_ms if stage not in STAGES]
    if missing or extra:
        raise ValueError(f"invalid stage vector: missing={missing}, extra={extra}")
    return {
        stage: _positive(stage_times_ms[stage], f"stage_times_ms[{stage}]")
        for stage in STAGES
    }


def pipeline_time_ms(
    stage_times_ms: Mapping[str, float],
    *,
    microbatches: int,
    layers: int,
    schedule_model: Mapping[str, Any] | None = None,
) -> float:
    """Schedule finite microbatches through A/dispatch/F/combine and layers.

    Missing layer records use uniform weights. Dispatch and combine share a
    resource by default; a new layer starts only after the preceding combine.
    The bottleneck approximation is available only by its explicit function.
    """
    stages = validate_stage_times(stage_times_ms)
    if schedule_model is None:
        schedule_model = {'type': 'finite_microbatch_fifo_v1', 'communication': 'shared_roundtrip'}
    return schedule_time(tuple(stages[s] for s in STAGES), microbatches, layers, schedule_model)


def bottleneck_pipeline_time_ms(stage_times_ms: Mapping[str, float], *, microbatches: int, layers: int) -> float:
    """Historical simplified-pipeline approximation, never the default prior."""
    stages = validate_stage_times(stage_times_ms)
    if microbatches < 1:
        raise ValueError("microbatches must be >= 1")
    if layers < 1:
        raise ValueError("layers must be >= 1")
    bottleneck = max(STAGES, key=lambda stage: stages[stage])
    fill_drain_ms = sum(
        value for stage, value in stages.items() if stage != bottleneck
    ) / float(layers)
    return float(microbatches) * stages[bottleneck] + fill_drain_ms


def bottleneck_stage(stage_times_ms: Mapping[str, float]) -> str:
    stages = validate_stage_times(stage_times_ms)
    return max(STAGES, key=lambda stage: stages[stage])


def imbalance_ratio(stage_times_ms: Mapping[str, float]) -> float:
    stages = validate_stage_times(stage_times_ms)
    return max(stages.values()) / min(stages.values())


def predict_stage_times(
    stage_models: Mapping[str, Mapping[str, float]],
    workload: Mapping[str, float],
    *,
    risk_quantile: str = "p90_residual_ms",
) -> dict[str, float]:
    """Predict a risk-adjusted duration for every physical pipeline stage."""
    prefill = max(float(workload.get("prefill_tokens_per_microbatch", 0.0)), 0.0)
    decode = max(float(workload.get("decode_tokens_per_microbatch", 0.0)), 0.0)
    routing = max(float(workload.get("routing_imbalance", 1.0)), 1.0)
    bytes_scale = max(float(workload.get("communication_bytes_scale", 1.0)), 0.0)
    prediction: dict[str, float] = {}
    for stage in STAGES:
        model = stage_models.get(stage)
        if model is None:
            raise ValueError(f"missing stage model {stage}")
        duration = (
            float(model.get("intercept_ms", 0.0))
            + float(model.get("prefill_ms_per_token", 0.0)) * prefill
            + float(model.get("decode_ms_per_token", 0.0)) * decode
        )
        if stage == "attention_compute":
            calibrated = predict_rank_model(model.get("attention_workload_model"), workload,
                                            risk_quantile=risk_quantile)
            if calibrated["available"]:
                # Replace the max-rank fit, never add another barrier penalty
                # on top of a stage time which already includes that maximum.
                prediction[stage] = calibrated["duration_ms"]
                continue
        if stage == "ffn_compute":
            reference = max(float(model.get("routing_imbalance_reference", 1.0)), 1.0)
            sensitivity = max(float(model.get("routing_sensitivity", 1.0)), 0.0)
            duration *= 1.0 + sensitivity * max(routing / reference - 1.0, 0.0)
        if stage in {"a2f_dispatch", "f2a_combine"}:
            duration *= bytes_scale
        duration += max(float(model.get(risk_quantile, 0.0)), 0.0)
        prediction[stage] = max(duration, 1e-9)
    return prediction


def predict_total_power_w(
    power_model: Mapping[str, Any], workload: Mapping[str, float]
) -> float:
    """Predict total GPU power, respecting the candidate's measured cap."""
    utilization = min(max(float(workload.get("offered_utilization", 0.0)), 0.0), 1.0)
    idle = max(float(power_model.get("idle_intercept_w", 0.0)), 0.0)
    slope = max(float(power_model.get("dynamic_slope_w", 0.0)), 0.0)
    cap = _positive(power_model["total_power_cap_w"], "total_power_cap_w")
    return min(idle + slope * utilization, cap)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    feasible: bool
    rejection_reasons: tuple[str, ...]
    bottleneck: str | None
    stage_times_ms: dict[str, float]
    pipeline_time_ms: float | None
    capacity_rps: float | None
    predicted_power_w: float | None
    predicted_energy_j_per_request: float | None
    imbalance_ratio: float | None
    topology: dict[str, Any]
    knobs: dict[str, Any]
    workload_prediction: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_candidate(
    candidate: Mapping[str, Any],
    workload: Mapping[str, Any],
    *,
    latency_budget_ratio: float = 1.05,
    capacity_headroom: float | None = None,
) -> CandidateEvaluation:
    """Apply legality, memory, relative-tail, then energy tests.

    The primary contract is relative to the matched MAX configuration at the
    same offered load.  Absolute arrival-rate headroom is therefore disabled
    by default; callers may opt into it for a separate absolute-capacity
    experiment by passing ``capacity_headroom``.
    """
    candidate_id = str(candidate.get("id", "unnamed"))
    topology = dict(candidate.get("topology", {}))
    knobs = dict(candidate.get("knobs", {}))
    reasons: list[str] = []
    if not bool(candidate.get("parallelism_legal", False)):
        reasons.append("parallelism_illegal")
    if not bool(candidate.get("memory_feasible", False)):
        reasons.append("memory_infeasible")
    if candidate.get("selection_split") != "calibration":
        reasons.append("non_calibration_profile")

    if candidate.get("performance_model") == "workload_response_v1":
        from .workload_response import assess_response

        prediction = None
        try:
            prediction, failures = assess_response(candidate, workload, latency_budget_ratio)
            reasons.extend(failures)
            if capacity_headroom is not None:
                # Finite-cohort completion throughput is not service capacity.
                reasons.append("absolute_capacity_not_identified")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            reasons.append(f"incomplete_workload_model:{error}")
        return CandidateEvaluation(
            candidate_id=candidate_id, feasible=not reasons,
            rejection_reasons=tuple(reasons), bottleneck=None,
            stage_times_ms={}, pipeline_time_ms=None, capacity_rps=None,
            predicted_power_w=prediction["power_w"] if prediction else None,
            predicted_energy_j_per_request=prediction["energy_j_per_request"] if prediction else None,
            imbalance_ratio=None, topology=topology, knobs=knobs,
            workload_prediction=prediction,
        )

    try:
        stages = predict_stage_times(candidate["stage_models"], workload)
        microbatches = int(candidate.get("microbatches", 1))
        layers = int(candidate["layers"])
        pipeline = pipeline_time_ms(
            stages, microbatches=microbatches, layers=layers,
            schedule_model=(candidate.get("analytical_provisioning") or {}).get("schedule_model")
        )
        requests_per_pipeline = _positive(
            candidate.get("requests_per_pipeline", 1.0),
            "requests_per_pipeline",
        )
        capacity = requests_per_pipeline * 1000.0 / pipeline
        arrival = max(float(workload.get("arrival_rate_rps", 0.0)), 0.0)
        if capacity_headroom is not None and capacity < arrival * capacity_headroom:
            reasons.append("capacity_headroom")
        guard_pipeline = _positive(
            candidate["guard_pipeline_time_ms"], "guard_pipeline_time_ms"
        )
        if pipeline / guard_pipeline > latency_budget_ratio:
            reasons.append("pipeline_tail_budget")
        calibration_ratios = candidate.get("calibration_latency_ratios", {})
        for metric in ("p90_ttft", "p90_tpot"):
            ratio = float(calibration_ratios.get(metric, float("inf")))
            if ratio > latency_budget_ratio:
                reasons.append(f"{metric}_calibration_budget")
        power = predict_total_power_w(candidate["power_model"], workload)
        energy_per_request = power / max(min(arrival, capacity), 1e-9)
        imbalance = imbalance_ratio(stages)
        bottleneck = bottleneck_stage(stages)
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"incomplete_model:{error}")
        stages = {}
        pipeline = None
        capacity = None
        power = None
        energy_per_request = None
        imbalance = None
        bottleneck = None

    return CandidateEvaluation(
        candidate_id=candidate_id,
        feasible=not reasons,
        rejection_reasons=tuple(reasons),
        bottleneck=bottleneck,
        stage_times_ms=stages,
        pipeline_time_ms=pipeline,
        capacity_rps=capacity,
        predicted_power_w=power,
        predicted_energy_j_per_request=energy_per_request,
        imbalance_ratio=imbalance,
        topology=topology,
        knobs=knobs,
    )


def select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    workload: Mapping[str, Any],
    *,
    latency_budget_ratio: float = 1.05,
    capacity_headroom: float | None = None,
) -> tuple[CandidateEvaluation | None, list[CandidateEvaluation]]:
    """Select the lowest-energy feasible point with balance as tie-breaker."""
    evaluations = [
        evaluate_candidate(
            candidate,
            workload,
            latency_budget_ratio=latency_budget_ratio,
            capacity_headroom=capacity_headroom,
        )
        for candidate in candidates
    ]
    feasible = [row for row in evaluations if row.feasible]
    if not feasible:
        return None, evaluations
    selected = min(
        feasible,
        key=lambda row: (
            float(row.predicted_energy_j_per_request),
            float(row.pipeline_time_ms) if row.pipeline_time_ms is not None else float("inf"),
            float(row.imbalance_ratio) if row.imbalance_ratio is not None else float("inf"),
            row.candidate_id,
        ),
    )
    return selected, evaluations

#!/usr/bin/env python3
"""Four-Stage Bottleneck-Guided Safe Search (FBSS).

FBSS uses cheap four-stage predictions to make a one-shot deployment choice.
It does not inspect the held-out trace and does not run a candidate-validation
round. Exhaustive evaluation is retained only as an explicit oracle-ablation
helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .model import STAGES, CandidateEvaluation, evaluate_candidate


COMMUNICATION_STAGES = frozenset(("a2f_dispatch", "f2a_combine"))


def criticality_weights(
    stage_times_ms: Mapping[str, float], *, microbatches: int, layers: int
) -> dict[str, float]:
    """Derivative of pipeline time with respect to each stage duration."""
    if microbatches < 1 or layers < 1:
        raise ValueError("microbatches and layers must be positive")
    if set(stage_times_ms) != set(STAGES):
        raise ValueError("criticality requires all four stages")
    bottleneck = max(STAGES, key=lambda stage: float(stage_times_ms[stage]))
    return {
        stage: float(microbatches) if stage == bottleneck else 1.0 / float(layers)
        for stage in STAGES
    }


def recommended_controls(bottleneck: str) -> tuple[str, ...]:
    """Map a physical bottleneck to the smallest useful control subspace."""
    if bottleneck == "attention_compute":
        return (
            "attention_gpus",
            "attention_dp",
            "attention_tp",
            "attention_mhz",
            "attention_power_w",
            "max_num_batched_tokens",
        )
    if bottleneck == "ffn_compute":
        return (
            "expert_gpus",
            "expert_dp",
            "expert_ep",
            "expert_mhz",
            "expert_power_w",
            "routing_placement",
        )
    if bottleneck in COMMUNICATION_STAGES:
        return (
            "attention_tp",
            "expert_ep",
            "microbatches",
            "placement",
            "connector",
            "max_num_batched_tokens",
        )
    raise ValueError(f"unknown stage {bottleneck!r}")


def _flat_configuration(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(candidate.get("topology", {})),
        **dict(candidate.get("knobs", {})),
        "microbatches": candidate.get("microbatches"),
    }


def changed_controls(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> frozenset[str]:
    left_flat = _flat_configuration(left)
    right_flat = _flat_configuration(right)
    keys = set(left_flat) | set(right_flat)
    return frozenset(
        key for key in keys if left_flat.get(key) != right_flat.get(key)
    )


def _directed_neighbor_score(
    current: CandidateEvaluation,
    neighbor: CandidateEvaluation,
    controls: frozenset[str],
) -> tuple[float, float, float, str]:
    """Rank a one-coordinate move without mixing feasibility and energy."""
    relevant = set(recommended_controls(current.bottleneck)) if current.bottleneck else set()
    direction_match = 1.0 if controls & relevant else 0.0
    current_time = _latency_score(current)
    next_time = _latency_score(neighbor)
    current_energy = float(current.predicted_energy_j_per_request or float("inf"))
    next_energy = float(neighbor.predicted_energy_j_per_request or float("inf"))
    if not current.feasible:
        # Restore safety first; capacity/tail improvement dominates power.
        return (
            float(neighbor.feasible) * 1e6 + direction_match * 1e3,
            current_time - next_time,
            current_energy - next_energy,
            neighbor.candidate_id,
        )
    if not neighbor.feasible:
        return (-1e9, -next_time, -next_energy, neighbor.candidate_id)
    # Once safe, recover power from slack stages.  The tail term prevents a
    # low-power point immediately adjacent to the SLO boundary from winning a
    # materially safer point for negligible energy gain.
    relative_energy_gain = (current_energy - next_energy) / max(current_energy, 1e-9)
    relative_time_cost = max(next_time - current_time, 0.0) / max(current_time, 1e-9)
    return (
        direction_match * 1e-3 + relative_energy_gain - 0.1 * relative_time_cost,
        -next_time,
        -next_energy,
        neighbor.candidate_id,
    )


def _latency_score(row: CandidateEvaluation) -> float:
    if row.workload_prediction is not None:
        return max(row.workload_prediction["latencies_ms"].values())
    return float(row.pipeline_time_ms) if row.pipeline_time_ms is not None else float("inf")


@dataclass(frozen=True)
class SearchStep:
    workload_index: int
    from_candidate: str | None
    candidate_id: str
    bottleneck: str | None
    changed_controls: tuple[str, ...]
    reason: str
    evaluation: CandidateEvaluation


@dataclass(frozen=True)
class SearchResult:
    search_path: tuple[str, ...]
    steps: tuple[SearchStep, ...]
    predicted_best_by_workload: tuple[str | None, ...]
    evaluated_predictions: int
    max_search_path_points: int


def bottleneck_guided_search(
    candidates: Sequence[Mapping[str, Any]],
    workloads: Sequence[Mapping[str, float]],
    *,
    guard_candidate_id: str,
    max_search_path_points: int,
    latency_budget_ratio: float = 1.05,
    capacity_headroom: float | None = None,
) -> SearchResult:
    """Make one-shot choices using bounded stage-directed coordinate paths.

    All candidates are scored by the cheap model. ``max_search_path_points``
    only bounds the recorded explanation path; it does not authorize physical
    candidate validation. Each move changes one parameter where possible.
    """
    if max_search_path_points < 1:
        raise ValueError("max_search_path_points must be >= 1")
    by_id = {str(candidate["id"]): candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate ids must be unique")
    if guard_candidate_id not in by_id:
        raise ValueError("guard candidate is absent")
    if not workloads:
        raise ValueError("at least one calibration workload is required")

    search_path: list[str] = [guard_candidate_id]
    steps: list[SearchStep] = []
    best_ids: list[str | None] = []
    prediction_count = 0

    for workload_index, workload in enumerate(workloads):
        evaluations = {
            candidate_id: evaluate_candidate(
                candidate,
                workload,
                latency_budget_ratio=latency_budget_ratio,
                capacity_headroom=capacity_headroom,
            )
            for candidate_id, candidate in by_id.items()
        }
        prediction_count += len(evaluations)
        feasible = [row for row in evaluations.values() if row.feasible]
        predicted_best = (
            min(
                feasible,
                key=lambda row: (
                    float(row.predicted_energy_j_per_request),
                    _latency_score(row),
                    row.candidate_id,
                ),
            )
            if feasible
            else None
        )
        best_ids.append(None if predicted_best is None else predicted_best.candidate_id)

        current_id = guard_candidate_id
        visited = {current_id}
        while len(search_path) < max_search_path_points:
            if predicted_best is not None and current_id == predicted_best.candidate_id:
                break
            current = evaluations[current_id]
            unvisited = [key for key in by_id if key not in visited]
            if not unvisited:
                break
            distances = {
                key: len(changed_controls(by_id[current_id], by_id[key]))
                for key in unvisited
            }
            nearest_distance = min(distances.values())
            pool = [key for key in unvisited if distances[key] == nearest_distance]
            next_id = max(
                pool,
                key=lambda key: _directed_neighbor_score(
                    current,
                    evaluations[key],
                    changed_controls(by_id[current_id], by_id[key]),
                ),
            )
            controls = tuple(sorted(changed_controls(by_id[current_id], by_id[next_id])))
            reason = (
                "restore_slo_along_bottleneck"
                if not current.feasible
                else "reclaim_noncritical_energy_with_slo_guard"
            )
            if current.workload_prediction is not None:
                reason = "min_energy_with_workload_latency_guard"
            steps.append(
                SearchStep(
                    workload_index=workload_index,
                    from_candidate=current_id,
                    candidate_id=next_id,
                    bottleneck=current.bottleneck,
                    changed_controls=controls,
                    reason=reason,
                    evaluation=evaluations[next_id],
                )
            )
            visited.add(next_id)
            if next_id not in search_path:
                search_path.append(next_id)
            current_id = next_id

    return SearchResult(
        search_path=tuple(search_path[:max_search_path_points]),
        steps=tuple(steps),
        predicted_best_by_workload=tuple(best_ids),
        evaluated_predictions=prediction_count,
        max_search_path_points=max_search_path_points,
    )


def exhaustive_oracle_ids(
    candidates: Sequence[Mapping[str, Any]],
    workloads: Sequence[Mapping[str, float]],
) -> tuple[str | None, ...]:
    """Explicit calibration-only oracle used to quantify search regret."""
    output: list[str | None] = []
    for workload in workloads:
        rows = [evaluate_candidate(candidate, workload) for candidate in candidates]
        feasible = [row for row in rows if row.feasible]
        output.append(
            None
            if not feasible
            else min(
                feasible,
                key=lambda row: float(row.predicted_energy_j_per_request),
            ).candidate_id
        )
    return tuple(output)

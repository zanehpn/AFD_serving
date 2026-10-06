#!/usr/bin/env python3
"""Sparse-anchor elasticity surrogate for FBSS candidate generation."""

from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .model import STAGES, pipeline_time_ms, predict_stage_times


FEATURES = (
    "attention_mhz",
    "expert_mhz",
    "attention_power_w",
    "expert_power_w",
    "attention_gpus",
    "expert_gpus",
    "attention_tp",
    "expert_ep",
    "microbatches",
)

STAGE_PRIORS = {
    "attention_compute": (0.85, 0.00, 0.15, 0.00, 0.85, 0.00, 0.05, 0.00, 0.10),
    "a2f_dispatch": (0.05, 0.05, 0.00, 0.00, 0.35, 0.35, 0.20, 0.20, 0.20),
    "ffn_compute": (0.00, 0.85, 0.00, 0.15, 0.00, 0.85, 0.00, 0.10, 0.10),
    "f2a_combine": (0.05, 0.05, 0.00, 0.00, 0.35, 0.35, 0.20, 0.20, 0.20),
}


def _gpu_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else int(value)


def configuration(candidate: Mapping[str, Any]) -> dict[str, float]:
    topology = candidate["topology"]
    knobs = candidate["knobs"]
    return {
        "attention_mhz": float(knobs["attention_mhz"]),
        "expert_mhz": float(knobs["expert_mhz"]),
        "attention_power_w": float(knobs["attention_power_w"]),
        "expert_power_w": float(knobs["expert_power_w"]),
        "attention_gpus": float(_gpu_count(topology["attention_gpus"])),
        "expert_gpus": float(_gpu_count(topology["expert_gpus"])),
        "attention_tp": float(topology.get("attention_tp", 1)),
        "expert_ep": float(topology.get("expert_ep", 1)),
        "microbatches": float(candidate["microbatches"]),
    }


def log_resource_features(
    guard_config: Mapping[str, float], candidate_config: Mapping[str, float]
) -> np.ndarray:
    """Positive values mean the candidate has less of a resource than guard."""
    return np.asarray(
        [
            math.log(float(guard_config[name]) / float(candidate_config[name]))
            for name in FEATURES
        ],
        dtype=float,
    )


def fit_elasticity_surrogate(
    measured_candidates: Sequence[Mapping[str, Any]],
    reference_workload: Mapping[str, float],
    *,
    guard_candidate_id: str,
    prior_strength: float = 2.0,
) -> dict[str, Any]:
    """Fit stage elasticities with structural priors from sparse anchors."""
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    by_id = {str(row["id"]): row for row in measured_candidates}
    if guard_candidate_id not in by_id:
        raise ValueError("guard candidate is absent from measured anchors")
    guard = by_id[guard_candidate_id]
    guard_config = configuration(guard)
    guard_times = predict_stage_times(guard["stage_models"], reference_workload)
    design = np.asarray(
        [log_resource_features(guard_config, configuration(row)) for row in measured_candidates],
        dtype=float,
    )
    fits: dict[str, Any] = {}
    for stage in STAGES:
        target = np.asarray(
            [
                math.log(
                    predict_stage_times(row["stage_models"], reference_workload)[stage]
                    / guard_times[stage]
                )
                for row in measured_candidates
            ],
            dtype=float,
        )
        prior = np.asarray(STAGE_PRIORS[stage], dtype=float)
        regularized = design.T @ design + prior_strength * np.eye(len(FEATURES))
        rhs = design.T @ target + prior_strength * prior
        elasticity = np.linalg.solve(regularized, rhs)
        # More hardware resource should not increase a stage time in the
        # sparse-data surrogate. Real non-monotonic effects are handled by
        # measured-candidate replacement and validation.
        elasticity = np.clip(elasticity, 0.0, 2.0)
        residual = target - design @ elasticity
        fits[stage] = {
            "elasticity": {
                name: float(value) for name, value in zip(FEATURES, elasticity, strict=True)
            },
            "log_residual_p90": float(
                np.quantile(np.abs(residual), 0.90, method="higher")
            ),
            "anchors": len(measured_candidates),
        }
    return {
        "schema_version": 1,
        "type": "regularized_log_elasticity_with_four_stage_structural_priors",
        "guard_candidate_id": guard_candidate_id,
        "guard_configuration": guard_config,
        "features": list(FEATURES),
        "prior_strength": prior_strength,
        "stage_fits": fits,
    }


def _topology_legal(topology: Mapping[str, Any]) -> bool:
    attention_gpus = _gpu_count(topology["attention_gpus"])
    expert_gpus = _gpu_count(topology["expert_gpus"])
    return (
        attention_gpus
        == int(topology.get("attention_dp", 1)) * int(topology.get("attention_tp", 1))
        and expert_gpus
        == int(topology.get("expert_dp", 1))
        * int(topology.get("expert_ep", 1))
        * int(topology.get("expert_tp", 1))
    )


def synthesize_candidate(
    guard: Mapping[str, Any],
    surrogate: Mapping[str, Any],
    *,
    candidate_id: str,
    topology: Mapping[str, Any],
    knobs: Mapping[str, Any],
    microbatches: int,
    reference_workload: Mapping[str, float],
    memory_feasible: bool,
) -> dict[str, Any]:
    candidate_stub = {
        "topology": dict(topology),
        "knobs": dict(knobs),
        "microbatches": microbatches,
    }
    features = log_resource_features(
        surrogate["guard_configuration"], configuration(candidate_stub)
    )
    stage_models: dict[str, dict[str, float]] = {}
    uncertainty_terms: list[float] = []
    for stage in STAGES:
        fit = surrogate["stage_fits"][stage]
        beta = np.asarray([fit["elasticity"][name] for name in FEATURES], dtype=float)
        scale = math.exp(float(features @ beta))
        uncertainty = float(fit["log_residual_p90"]) + 0.03 * float(
            np.count_nonzero(np.abs(features) > 1e-12)
        )
        uncertainty_terms.append(uncertainty)
        base = guard["stage_models"][stage]
        stage_models[stage] = {
            key: float(base.get(key, 0.0)) * scale
            for key in (
                "intercept_ms",
                "prefill_ms_per_token",
                "decode_ms_per_token",
                "p90_residual_ms",
                "p95_residual_ms",
            )
        }
        nominal = (
            stage_models[stage]["intercept_ms"]
            + stage_models[stage]["prefill_ms_per_token"]
            * float(reference_workload.get("prefill_tokens_per_microbatch", 0.0))
            + stage_models[stage]["decode_ms_per_token"]
            * float(reference_workload.get("decode_tokens_per_microbatch", 0.0))
        )
        stage_models[stage]["p90_residual_ms"] += nominal * (math.exp(uncertainty) - 1.0)
        stage_models[stage]["routing_imbalance_reference"] = float(
            base.get("routing_imbalance_reference", 1.0)
        )
        stage_models[stage]["routing_sensitivity"] = float(
            base.get("routing_sensitivity", 0.0)
        )

    predicted_times = predict_stage_times(stage_models, reference_workload)
    candidate_pipeline = pipeline_time_ms(
        predicted_times,
        microbatches=microbatches,
        layers=int(guard["layers"]),
    )
    guard_pipeline = float(guard["guard_pipeline_time_ms"])
    worst_uncertainty = max(uncertainty_terms)
    latency_ratio = candidate_pipeline / guard_pipeline
    guard_config = configuration(guard)
    candidate_config = configuration(candidate_stub)
    guard_gpu_count = guard_config["attention_gpus"] + guard_config["expert_gpus"]
    candidate_gpu_count = (
        candidate_config["attention_gpus"] + candidate_config["expert_gpus"]
    )
    guard_total_cap = (
        guard_config["attention_gpus"] * guard_config["attention_power_w"]
        + guard_config["expert_gpus"] * guard_config["expert_power_w"]
    )
    candidate_total_cap = (
        candidate_config["attention_gpus"] * candidate_config["attention_power_w"]
        + candidate_config["expert_gpus"] * candidate_config["expert_power_w"]
    )
    gpu_scale = candidate_gpu_count / guard_gpu_count
    cap_scale = candidate_total_cap / guard_total_cap
    return {
        "id": candidate_id,
        "selection_split": "calibration",
        "validation_status": "surrogate_unvalidated",
        "parallelism_legal": _topology_legal(topology),
        "memory_feasible": bool(memory_feasible),
        "topology": dict(topology),
        "knobs": dict(knobs),
        "microbatches": int(microbatches),
        "layers": int(guard["layers"]),
        "requests_per_pipeline": float(guard["requests_per_pipeline"]),
        "guard_pipeline_time_ms": guard_pipeline,
        "stage_models": stage_models,
        "power_model": {
            "idle_intercept_w": float(guard["power_model"]["idle_intercept_w"])
            * gpu_scale,
            "dynamic_slope_w": float(guard["power_model"]["dynamic_slope_w"])
            * cap_scale,
            "total_power_cap_w": candidate_total_cap,
            "fit_status": "surrogate_unvalidated",
        },
        "calibration_latency_ratios": {
            "p90_ttft": latency_ratio,
            "p90_tpot": latency_ratio,
        },
        "surrogate_uncertainty_log_p90": worst_uncertainty,
    }


def expand_candidate_space(
    measured_candidates: Sequence[Mapping[str, Any]],
    surrogate: Mapping[str, Any],
    specification: Mapping[str, Any],
    reference_workload: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Enumerate cheap predictions, replacing exact anchors with measurements."""
    guard_id = str(surrogate["guard_candidate_id"])
    measured_by_config = {
        tuple(configuration(row)[name] for name in FEATURES): copy.deepcopy(dict(row))
        for row in measured_candidates
    }
    guard = next(row for row in measured_candidates if str(row["id"]) == guard_id)
    output: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    counter = 0
    for topology, attention_mhz, expert_mhz, attention_cap, expert_cap, microbatches in itertools.product(
        specification["topologies"],
        specification["attention_frequencies_mhz"],
        specification["expert_frequencies_mhz"],
        specification["attention_power_caps_w"],
        specification["expert_power_caps_w"],
        specification["microbatches"],
    ):
        knobs = {
            "attention_mhz": int(attention_mhz),
            "expert_mhz": int(expert_mhz),
            "attention_power_w": int(attention_cap),
            "expert_power_w": int(expert_cap),
        }
        stub = {"topology": topology, "knobs": knobs, "microbatches": microbatches}
        key = tuple(configuration(stub)[name] for name in FEATURES)
        if key in seen:
            continue
        seen.add(key)
        if key in measured_by_config:
            measured = measured_by_config[key]
            measured["validation_status"] = "physically_measured_anchor"
            output.append(measured)
            continue
        memory_feasible = bool(topology.get("memory_feasible", False))
        output.append(
            synthesize_candidate(
                guard,
                surrogate,
                candidate_id=f"s{counter:05d}",
                topology=topology,
                knobs=knobs,
                microbatches=int(microbatches),
                reference_workload=reference_workload,
                memory_feasible=memory_feasible,
            )
        )
        counter += 1
    return output

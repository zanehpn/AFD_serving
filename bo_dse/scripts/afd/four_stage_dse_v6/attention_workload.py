"""Calibrated context-work and rank-service models for Attention.

Context tokens are logical causal query/key pairs, NOT measured HBM bytes.
Rank skew is a service-barrier proxy, NOT an observed synchronization wait.
Neither quantity establishes a tail-latency guarantee or execution eligibility.
"""
from __future__ import annotations

import math

import numpy as np


CONTEXT_FIELDS = ("prefill_context_tokens", "decode_context_tokens")
CONTEXT_TARGETS = tuple(name + "_per_microbatch" for name in CONTEXT_FIELDS)
TOKEN_TARGETS = ("prefill_tokens_per_microbatch", "decode_tokens_per_microbatch")
BASE_TERMS = ("intercept", *TOKEN_TARGETS)
MODEL_TYPE = "attention_context_rank_service_v1"


def validate_context(value, prefill, decode):
    """Validate an optional, exact summary from causal request-token spans."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid attention workload schema")
    if value.get("source") != "request_token_spans":
        raise ValueError("attention workload requires causal request-token spans")
    names = (*CONTEXT_FIELDS, "kv_sequence_tokens", "active_requests", "prefill_tokens", "decode_tokens")
    for name in names:
        number = value.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"invalid attention workload {name}")
    if value["prefill_tokens"] != prefill or value["decode_tokens"] != decode:
        raise ValueError("attention context/token counts disagree")
    if value["active_requests"] > prefill + decode:
        raise ValueError("attention active requests exceed real tokens")
    if bool(value["active_requests"]) != bool(prefill + decode):
        raise ValueError("attention context lacks active-request evidence")
    if not value["active_requests"] <= value["kv_sequence_tokens"] <= sum(value[k] for k in CONTEXT_FIELDS):
        raise ValueError("invalid logical KV sequence load")
    for field, tokens in zip(CONTEXT_FIELDS, (prefill, decode)):
        if value[field] < tokens or (tokens == 0 and value[field] != 0):
            raise ValueError("attention context work inconsistent with phase tokens")
    return dict(value)


def rank_summary(ranks, *, comparable_events):
    """Retain rank evidence, including idle participants; never invent ranks."""
    values = [float(row["duration_ms"]) for row in ranks.values()]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("rank service durations must be finite and positive")
    mean, maximum = float(np.mean(values)), max(values)
    return {
        "rank_count": len(values), "rank_mean_ms": mean,
        "rank_max_ms": maximum, "rank_std_ms": float(np.std(values)),
        "service_barrier_uplift_ms": maximum - mean,
        "potential_wait_gpu_ms": sum(maximum - value for value in values),
        "comparable_events": bool(comparable_events),
        "scope": "same_causal_transaction_and_microbatch",
        "interpretation": "max_minus_mean_service_not_measured_wall_clock_wait",
    }


def context_totals(group):
    """Sum real Attention-DP context work once, never over layers or E peers."""
    ranks = group.get("attention_rank_observations", {})
    if not ranks or any(row.get("context") is None for row in ranks.values()):
        return None
    return {key: sum(row["context"][key] for row in ranks.values())
            for key in (*CONTEXT_FIELDS, "kv_sequence_tokens", "active_requests")}


def reference_context(groups):
    rows = [context_totals(group) for group in groups]
    if not rows or any(row is None for row in rows):
        return {}
    return {target: float(np.mean([row[field] for row in rows]))
            for field, target in zip(CONTEXT_FIELDS, CONTEXT_TARGETS)}


def _design(groups, terms):
    rows = []
    for group in groups:
        context = context_totals(group)
        row = [1., float(group["prefill_tokens"]), float(group["decode_tokens"])]
        if len(terms) > len(BASE_TERMS):
            if context is None:
                return None
            row.extend(float(context[key]) for key in CONTEXT_FIELDS)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _target(workload, terms):
    values = [1.]
    for name in terms[1:]:
        if name not in workload:
            return None
        value = workload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid attention workload target {name}")
        values.append(float(value))
    return np.asarray(values)


def support(design, target):
    if target is None:
        return False, "missing_context_target"
    scale = np.maximum(np.max(np.abs(design), axis=0), 1.)
    x, point = design / scale, target / scale
    if np.linalg.norm(point - point @ np.linalg.pinv(x) @ x) > 1e-7:
        return False, "unidentifiable_workload"
    if np.any(point < x.min(axis=0) - 1e-7) or np.any(point > x.max(axis=0) + 1e-7):
        return False, "outside_calibration_range"
    return True, "supported_interpolation"


def common_context_target(group_sets, token_workload):
    """Freeze one actual pooled context target supported by every topology.

    Restrict to the already selected common token shape. An empty intersection
    leaves the new predictor unavailable; it must not break a valid legacy run.
    """
    terms = (*BASE_TERMS, *CONTEXT_TARGETS)
    designs, pool = [], []
    for groups in group_sets:
        selected = [g for g in groups if all(
            float(g[key]) == float(token_workload[target])
            for key, target in zip(("prefill_tokens", "decode_tokens"), TOKEN_TARGETS))]
        x = _design(selected, terms) if selected else None
        if x is None or not len(x):
            return {}, "missing_context_evidence_at_common_token_shape"
        designs.append(x)
        pool.extend(x.tolist())
    center = np.median(np.asarray(pool), axis=0)
    scale = np.maximum(np.max(np.abs(pool), axis=0), 1.)
    candidates = sorted({tuple(row) for row in pool},
                        key=lambda row: (float(np.linalg.norm((np.asarray(row) - center) / scale)), row))
    for row in candidates:
        if all(support(x, np.asarray(row))[0] for x in designs):
            return dict(zip(CONTEXT_TARGETS, row[len(BASE_TERMS):])), "shared_observed_context_supported"
    # Means are valid only when supported by every calibrated design.
    point = np.mean(np.asarray(pool), axis=0)
    if all(support(x, point)[0] for x in designs):
        return dict(zip(CONTEXT_TARGETS, point[len(BASE_TERMS):].tolist())), "shared_context_interpolation"
    return {}, "no_common_context_support"


def fit_rank_model(groups, nnls):
    eligible = [g for g in groups if g.get("attention_rank_summary", {}).get("comparable_events")]
    if len(eligible) != len(groups) or not eligible:
        return {"type": MODEL_TYPE, "enabled": False, "reason": "incomparable_or_missing_rank_events"}
    scopes = {g["attention_rank_summary"]["scope"] for g in eligible}
    if len(scopes) != 1:
        return {"type": MODEL_TYPE, "enabled": False, "reason": "mixed_layer_barrier_coverage"}
    with_context = all(context_totals(group) is not None for group in eligible)
    terms = (*BASE_TERMS, *CONTEXT_TARGETS) if with_context else BASE_TERMS
    x = _design(eligible, terms)
    scale = np.maximum(np.max(np.abs(x), axis=0), 1.)
    scaled = x / scale
    rank = int(np.linalg.matrix_rank(scaled))
    responses = {}
    for label, key in (("mean_service", "rank_mean_ms"), ("barrier_uplift", "service_barrier_uplift_ms")):
        y = np.asarray([g["attention_rank_summary"][key] for g in eligible])
        coefficients = nnls(scaled, y) / scale
        responses[label] = {"coefficients": coefficients.tolist(),
                            "rmse_ms": float(np.sqrt(np.mean((x @ coefficients - y) ** 2)))}
    predicted = x @ (np.asarray(responses["mean_service"]["coefficients"]) + np.asarray(responses["barrier_uplift"]["coefficients"]))
    observed = np.asarray([g["attention_rank_summary"]["rank_max_ms"] for g in eligible])
    return {
        "type": MODEL_TYPE, "enabled": True, "selection_split": "calibration",
        "terms": list(terms), "context_enabled": with_context,
        "design": np.unique(x, axis=0).tolist(), "design_rank": rank,
        "coefficients_individually_identified": rank == x.shape[1],
        "responses": responses, "samples": len(eligible),
        "rank_counts": sorted({g["attention_rank_summary"]["rank_count"] for g in eligible}),
        "barrier_basis": next(iter(scopes)),
        "rmse_ms": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "p90_residual_ms": float(np.quantile(np.maximum(observed - predicted, 0), .9, method="higher")),
        "p95_residual_ms": float(np.quantile(np.maximum(observed - predicted, 0), .95, method="higher")),
        "scope": "within_calibrated_structure_and_operating_point",
        "interpretation": "mean_rank_service_plus_empirical_barrier_uplift_replaces_existing_max",
    }


def predict_rank_model(model, workload, *, risk_quantile="nominal_no_margin"):
    if not model or not model.get("enabled"):
        return {"available": False, "reason": (model or {}).get("reason", "missing_rank_model")}
    if model.get("type") != MODEL_TYPE or model.get("selection_split") != "calibration":
        raise ValueError("invalid or non-calibration attention rank model")
    terms = model["terms"]
    if tuple(terms) not in (BASE_TERMS, (*BASE_TERMS, *CONTEXT_TARGETS)):
        raise ValueError("invalid attention model terms")
    target = _target(workload, terms)
    valid, reason = support(np.asarray(model["design"]), target)
    if not valid:
        return {"available": False, "reason": reason}
    values = {}
    for name, response in model["responses"].items():
        coefficients = np.asarray(response["coefficients"])
        if coefficients.shape != target.shape or not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0):
            raise ValueError("invalid attention model coefficients")
        values[name] = float(target @ coefficients)
    margin = max(float(model.get(risk_quantile, 0.)), 0.)
    return {"available": True, "reason": reason, **values,
            "duration_ms": max(values["mean_service"] + values["barrier_uplift"] + margin, 1e-9),
            "context_enabled": model["context_enabled"],
            "interpretation": "service_barrier_proxy_not_measured_synchronization_wait"}

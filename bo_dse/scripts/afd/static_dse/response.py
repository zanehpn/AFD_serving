"""Identifiable frequency/cap interactions and conservative structural priors."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import nnls

from .space import KNOBS, configuration, digest, structure
from four_stage_dse_v6.model import STAGES


JOINT_TERMS = ["intercept", *KNOBS, "attention_frequency_x_cap", "expert_frequency_x_cap"]


def joint_features(knobs, reference):
    x = [math.log(knobs[k] / reference[k]) for k in KNOBS]
    return [1., *x, x[0] * x[2], x[1] * x[3]]


def fit_joint(anchors):
    """A seven-term response needs seven independent interventions, not repeats."""
    reference = anchors[0]["knobs"]
    x = np.asarray([joint_features(a["knobs"], reference) for a in anchors])
    rank = int(np.linalg.matrix_rank(x))
    report = {"terms": JOINT_TERMS, "rank": rank, "required_rank": len(JOINT_TERMS),
              "identified": rank == len(JOINT_TERMS), "samples": len(anchors),
              "reference": reference,
              "ranges": {k: [min(a["knobs"][k] for a in anchors), max(a["knobs"][k] for a in anchors)] for k in KNOBS},
              "assumption": "local log-bilinear response; cross-role interactions remain unmodelled"}
    if not report["identified"]:
        return report
    targets = {s: [a["stage_ms"][s] for a in anchors] for s in STAGES}
    targets["power_w"] = [a["power_w"] for a in anchors]
    # Actual clocks have their own response; never replace requested clocks by
    # hypothetical cap-derived clocks in the measurement labels.
    report["responses"] = {}
    for name, values in targets.items():
        y = np.log(values)
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        report["responses"][name] = {"coefficients": coefficients.tolist(),
                                     "rmse_log": float(np.sqrt(np.mean((x @ coefficients - y) ** 2)))}
    measured = [a for a in anchors if a.get("operating_state")]
    clock_x = np.asarray([joint_features(a["knobs"], reference) for a in measured])
    report["actual_clock_rank"] = int(np.linalg.matrix_rank(clock_x)) if len(measured) else 0
    if report["actual_clock_rank"] == len(JOINT_TERMS):
        for role in ("attention", "expert"):
            y = np.log([a["operating_state"][role]["effective_mhz"] for a in measured])
            coef = np.linalg.lstsq(clock_x, y, rcond=None)[0]
            report["responses"][role + "_effective_mhz"] = {
                "coefficients": coef.tolist(), "rmse_log": float(np.sqrt(np.mean((clock_x @ coef - y) ** 2)))}
    report["needs_repetition"] = len(anchors) <= len(JOINT_TERMS) or any(a.get("repetitions", 1) < 2 for a in anchors)
    return report


def joint_predict(model, knobs):
    x = np.asarray(joint_features(knobs, model["reference"]))
    return {name: math.exp(float(np.clip(x @ row["coefficients"], -30, 30)))
            for name, row in model["responses"].items()}


def normalized_workload(workload, candidate):
    """Explicit fixed-total-token protocol; legacy profiles keep their old unit.

    The division by Attention DP is a balanced-batch hypothesis. Actual token
    shapes must support the target when raw feedback is normalized.
    """
    result = dict(workload)
    c = configuration(candidate)
    for phase in ("prefill", "decode"):
        key = phase + "_tokens_per_step"
        if key in workload:
            result[phase + "_tokens_per_microbatch"] = float(workload[key]) / (c["microbatches"] * c["attention_dp"])
    return result


STRUCTURAL_TERMS = ["startup", "attention_work", "expert_work", "attention_collective",
                    "expert_collective", "dispatch_fanout", "microbatch_launch"]


def structural_features(c):
    """Dimensionless mechanism bases under fixed global token work.

    Coefficients are empirical nonnegative costs, not asymptotic guarantees.
    Per-stage fits decide which bases contribute; collinear coefficients are
    never published as identified parameter effects.
    """
    a = c["attention_dp"] * c["attention_tp"]
    e = c["expert_dp"] * c["expert_ep"] * c["expert_tp"]
    m = c["microbatches"]
    return [1., 1 / (m * a), 1 / (m * e),
            math.log2(c["attention_tp"]) / m, math.log2(c["expert_tp"]) / m,
            (1 - 1 / c["expert_ep"]) / (m * c["expert_dp"]), float(m)]


def fit_structural(groups, workload):
    # Legacy fixed-microbatch calibration cannot identify fixed-total-work
    # scaling. Require both workload phases, including an explicit zero phase.
    enabled = all(k in workload for k in ("prefill_tokens_per_step", "decode_tokens_per_step"))
    buckets = {}
    if enabled:
        for group in groups.values():
            for anchor in group["anchors"]:
                # Control frequency/cap and execution mode; GPU fabric is part
                # of the frozen campaign hardware/context, not a learned claim.
                key = digest([anchor["knobs"], group["structure"]["execution_mode"]])
                buckets.setdefault(key, []).append({**anchor, "structure": group["structure"]})
    models = {}
    for key, rows in buckets.items():
        x = np.asarray([structural_features(r["structure"]) for r in rows])
        rank = int(np.linalg.matrix_rank(x))
        models[key] = {"design": x.tolist(), "rank": rank, "terms": STRUCTURAL_TERMS,
                       "structures": len({digest(r["structure"]) for r in rows}),
                       "knobs": rows[0]["knobs"], "execution_mode": rows[0]["structure"]["execution_mode"],
                       "responses": {}, "anchors": [r["id"] for r in rows]}
        for name in [*STAGES, "power_w", "requests_per_pipeline"]:
            values = [r["stage_ms"][name] if name in STAGES else r[name] for r in rows]
            coef, _ = nnls(x, np.asarray(values))
            prediction = x @ coef
            models[key]["responses"][name] = {"predictor_weights": coef.tolist(),
                                               "identified_coefficients": coef.tolist() if rank == x.shape[1] else None,
                                               "relative_rmse": float(np.sqrt(np.mean(((prediction - values) / values) ** 2)))}
        models[key]["layers"] = rows[0]["layers"]
        if any(r["layers"] != rows[0]["layers"] for r in rows):
            raise ValueError("structural calibration layer counts differ")
    return {"enabled": enabled, "models": models,
            "required_workload": ["prefill_tokens_per_step", "decode_tokens_per_step"],
            "assumptions": "balanced fixed-total-token work; same operating points and execution mode; no independent claim for collinear DP/EP/TP effects"}


def structural_predict(report, candidate):
    c = configuration(candidate)
    key = digest([candidate["knobs"], c["execution_mode"]])
    model = report.get("models", {}).get(key)
    if not model or model["structures"] < 2:
        return None
    x = np.asarray(model["design"])
    target = np.asarray(structural_features(c))
    if np.linalg.norm(target - target @ np.linalg.pinv(x) @ x) > 1e-7:
        return None
    if np.any(target < x.min(axis=0) - 1e-7) or np.any(target > x.max(axis=0) + 1e-7):
        return None
    predicted = {name: float(target @ row["predictor_weights"]) for name, row in model["responses"].items()}
    if any(value <= 0 or not math.isfinite(value) for value in predicted.values()):
        return None
    return {"values": predicted, "layers": model["layers"], "rank": model["rank"],
            "anchors": model["anchors"], "uncertainty_log": 1.0 + max(r["relative_rmse"] for r in model["responses"].values())}

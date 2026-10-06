"""Analytical A/F ratio advice under the paper's saturated-decode assumptions.

Equations 3, 8--12 of Song et al., arXiv:2601.21351v4, plus an optional
constant-role-power energy extension. This interface does not give SLO guarantees,
does not equate an EP rank with an independent FFN service group, and does not
grant execution eligibility. Native BO still uses its charged topology probes.
"""
from __future__ import annotations

import math

from scipy.integrate import quad
from scipy.special import log_ndtr


def workload_moments(requests):
    """Renewal-reward moments from explicit decode-slot lifetimes.

    decode_steps is deliberately distinct from output_tokens/max_tokens: a
    caller must establish the scheduler's convention rather than assume it.
    """
    if not requests:
        raise ValueError("empty calibration requests")
    identities, lifetimes, first, second = set(), [], [], []
    for row in requests:
        identity = row["source_index"]
        if identity in identities:
            raise ValueError("duplicate calibration request identity")
        identities.add(identity)
        p, d = row["prompt_tokens"], row["decode_steps"]
        if type(p) is not int or p < 0 or type(d) is not int or d < 1:
            raise ValueError("explicit nonnegative prompt_tokens and positive decode_steps required")
        lifetimes.append(d)
        first.append(d * p + d * (d - 1) / 2)
        second.append(d * p * p + p * d * (d - 1) + d * (d - 1) * (2 * d - 1) / 6)
    denominator = sum(lifetimes)
    theta = math.fsum(first) / denominator
    variance = max(math.fsum(second) / denominator - theta * theta, 0.)
    return {"theta": theta, "variance": variance, "requests": len(requests),
            "decode_slot_steps": denominator, "sampling": "decode_step_not_wall_clock",
            "assumption": "stationary_iid_request_cycles_with_immediate_slot_replenishment"}


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"invalid provisioning {name}")
    return float(value)


def cycle_time(coefficients, moments, batch_size, ratio, *, barrier):
    a, b = coefficients["attention"], coefficients["ffn"]
    c = coefficients["communication_roundtrip"]
    mean = a["slope"] * batch_size * moments["theta"] + a["intercept"]
    other = max(b["slope"] * ratio * batch_size + b["intercept"],
                c["slope"] * ratio * batch_size + c["intercept"])
    sigma = a["slope"] * math.sqrt(batch_size * moments["variance"])
    if not barrier or sigma == 0:
        return max(mean, other)
    # E[max(G, mu + sigma * max_j Z_j)] using the tail integral. The
    # cross-worker maximum is INSIDE the expectation, not max of means.
    z = (other - mean) / sigma
    tail = lambda x: -math.expm1(ratio * float(log_ndtr(x)))
    # Split at zero so extreme standardized thresholds do not confuse quad's
    # infinite-interval transformation. Below -12 the tail differs negligibly
    # from one for the supported finite ratio list.
    lower = max(z, -12.)
    integral = max(-12. - z, 0.)
    if lower < 0:
        integral += quad(tail, lower, 0., epsabs=1e-9)[0]
    integral += quad(tail, max(lower, 0.), math.inf, epsabs=1e-9)[0]
    return other + sigma * integral


def _mean_field_candidates(coefficients, moments, batch, ratios, *, resource_ratio=1.):
    """Breakpoints and stationary points with FFN/Attention resource weights.

    For constant role weights w_A, w_F, minimizing
    (r*w_A+w_F)*(a*B*r+b)/(r*B) has r=sqrt(b*w_F/(a*B*w_A)).
    Only this stationary point changes; the latency breakpoints do not.
    """
    a, f, c = (coefficients[key] for key in ("attention", "ffn", "communication_roundtrip"))
    mean_a = a["slope"] * batch * moments["theta"] + a["intercept"]
    continuous = [float(min(ratios)), float(max(ratios))]
    thresholds = [(mean_a - stage["intercept"]) / (stage["slope"] * batch)
                  for stage in (c, f) if stage["slope"] > 0]
    if thresholds:
        continuous.append(min(thresholds))
    for stage in (c, f):
        if stage["slope"] > 0:
            continuous.append(math.sqrt(stage["intercept"] / (stage["slope"] * batch) * resource_ratio))
    if f["slope"] != c["slope"]:
        continuous.append((c["intercept"] - f["intercept"]) / (batch * (f["slope"] - c["slope"])))
    return sorted({r for r in continuous if min(ratios) <= r <= max(ratios)})


def _energy_advice(power_model, rows, coefficients, moments, batch, ratios):
    """Optional saturated-decode energy prior; supplied watts are not cap values."""
    if power_model is None:
        return {"available": False, "reason": "missing_calibrated_role_power"}
    if not isinstance(power_model, dict) or power_model.get("type") != "constant_role_power_v1":
        raise ValueError("provisioning requires an explicit constant role power model")
    if power_model.get("assumption") != "ratio_independent_cycle_average_role_power":
        raise ValueError("role power must describe cycle averages independent of the candidate ratio")
    if power_model.get("selection_split") != "calibration":
        raise ValueError("role power requires calibration inputs")
    pa = _number(power_model.get("attention_worker_w"), "attention_worker_w")
    pf = _number(power_model.get("ffn_service_group_w"), "ffn_service_group_w")
    if pa <= 0 or pf <= 0:
        raise ValueError("role power must be strictly positive")
    predictions = []
    for row in rows:
        r = row["ratio"]
        power = r * pa + pf
        energy = {name: power * milliseconds / (1000 * r * batch)
                  for name, milliseconds in row["cycle_ms"].items()}
        if not math.isfinite(power) or any(not math.isfinite(e) or e <= 0 for e in energy.values()):
            raise ValueError("invalid provisioning energy prediction")
        predictions.append({"ratio": r, "active_power_w": power,
                            "predicted_j_per_decode_token": energy})
    return {"available": True, "objective": "saturated_decode_j_per_token",
            "power_model": dict(power_model),
            "mean_field_candidate_ratios": _mean_field_candidates(
                coefficients, moments, batch, ratios, resource_ratio=pf / pa),
            "recommendation": {name: min(predictions, key=lambda row: row["predicted_j_per_decode_token"][name])["ratio"]
                               for name in ("mean_field", "gaussian_barrier")},
            "ranked_ratios": sorted(predictions, key=lambda row: (row["predicted_j_per_decode_token"]["gaussian_barrier"], row["ratio"])),
            "interpretation": "conditional_model_advice_not_measured_energy_or_finite_cohort_objective",
            "assumptions": ["fixed_latency_coefficients_and_full_batches",
                            "role_watts_are_cycle_averages_including_waiting_on_participating_gpus",
                            "ffn_service_group_w_includes_all_of_its_ep_tp_ranks",
                            "unused_gpu_power_is_excluded", "power_caps_are_not_average_power"]}


def recommend_ratios(profile, requests):
    if profile.get("selection_split") != "calibration":
        raise ValueError("provisioning requires calibration inputs")
    if profile.get("ratio_scope") != "attention_workers_per_ffn_service_group":
        raise ValueError("provisioning requires an explicit FFN service-group mapping")
    batch = profile["batch_size_per_attention"]
    ratios = profile["ratios"]
    if type(batch) is not int or batch < 1 or not isinstance(ratios, list) or not ratios:
        raise ValueError("positive batch size and nonempty ratio list required")
    if any(type(r) is not int or r < 1 for r in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("ratio candidates must be unique positive integer fan-ins")
    coefficients = profile["latency_coefficients_ms"]
    for stage in ("attention", "ffn", "communication_roundtrip"):
        for field in ("slope", "intercept"):
            _number(coefficients[stage][field], f"{stage}.{field}")
    moments = workload_moments(requests)
    rows = []
    for r in sorted(ratios):
        times = {name: cycle_time(coefficients, moments, batch, r, barrier=barrier)
                 for name, barrier in (("mean_field", False), ("gaussian_barrier", True))}
        if any(not math.isfinite(value) or value <= 0 for value in times.values()):
            raise ValueError("provisioning predicted nonpositive or invalid cycle time")
        rows.append({"ratio": r, "cycle_ms": times,
                     "tokens_per_second_per_instance": {name: 1000 * r * batch / ((r + 1) * t) for name, t in times.items()}})
    return {"schema_version": 1, "selection_split": "calibration", "workload_moments": moments,
            "source": "https://arxiv.org/html/2601.21351v4", "ratio_scope": profile["ratio_scope"],
            "mean_field_candidate_ratios": _mean_field_candidates(coefficients, moments, batch, ratios),
            "energy_advice": _energy_advice(profile.get("power_model"), rows, coefficients, moments, batch, ratios),
            "recommendation": {name: max(rows, key=lambda row: row["tokens_per_second_per_instance"][name])["ratio"]
                               for name in ("mean_field", "gaussian_barrier")},
            "ranked_ratios": sorted(rows, key=lambda row: (-row["tokens_per_second_per_instance"]["gaussian_barrier"], row["ratio"])),
            "objective": "saturated_decode_throughput_per_instance_not_energy",
            "execution_eligibility_changed": False, "slo_feasibility_established": False,
            "assumptions": ["full_microbatches_immediate_replenishment", "iid_slot_loads_for_gaussian_barrier",
                            "large_batch_gaussian_approximation_requires_validation",
                            "fixed_calibrated_hardware_operating_point_and_ffn_mapping",
                            "roundtrip_communication_is_one_resource_cost_not_two_independent_stages"]}

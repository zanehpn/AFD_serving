"""Learn parameter effects from physical anchors, not synthesized v6 labels."""
from __future__ import annotations

import math
import statistics

from four_stage_dse_v6.model import STAGES, pipeline_time_ms, predict_stage_times, predict_total_power_w
from four_stage_dse_v6.attention_workload import predict_rank_model
from .space import KNOBS, configuration, digest, structure, positive
from .response import (fit_joint, joint_predict, fit_structural, structural_predict,
                       normalized_workload)
from .analytical import predict as analytical_predict


PARAMETER_EFFECTS = {
    "frequency": "Changes compute service time; measured stage elasticity identifies frequency-sensitive stages.",
    "power_cap": "May bind actual clocks and total power; frequency/cap effects are not assumed additive savings.",
    "attention_dp": "Changes requests and token batch per replica; expert aggregation depends on backend scheduling.",
    "attention_tp": "Partitions attention work and introduces collectives; requires shape and communication calibration.",
    "expert_ep": "Changes expert residency, dispatch/combine, and rank imbalance; does not change model expert count.",
    "expert_tp": "Partitions expert GEMMs and adds collectives; joint native deployment uses EP=1 within each replica.",
    "expert_dp": "Replicates complete AFD service groups with request dispatch; independent-replica scheduling priors need separate calibration.",
    "microbatches": "At fixed total tokens B, per-microbatch tokens are approximately B/M; overlap competes with small-GEMM and message startup overhead.",
}


def build_mechanism(profile, workload):
    if profile.get("selection_split") != "calibration":
        raise ValueError("mechanism fitting requires calibration data")
    anchors = [c for c in profile.get("candidates", [])
               if c.get("validation_status") == "physically_measured_anchor"]
    groups = {}
    for anchor in anchors:
        if anchor.get("selection_split") != "calibration":
            raise ValueError("held-out anchor")
        key = digest(structure(anchor))
        group = groups.setdefault(key, {"structure": structure(anchor), "anchors": [], "effects": {}})
        # Estimate response from nominal fitted service, not from differences in
        # independently fitted risk margins. Uncertainty is handled separately.
        stages = predict_stage_times(anchor["stage_models"], normalized_workload(workload, anchor), risk_quantile="nominal_no_margin")
        power = predict_total_power_w(anchor["power_model"], workload)
        for stage, value in stages.items():
            positive(value, f"anchor stage {stage}")
        positive(power, "anchor power")
        positive(anchor["requests_per_pipeline"], "anchor requests per pipeline")
        rank_model = anchor["stage_models"]["attention_compute"].get("attention_workload_model")
        attention = predict_rank_model(rank_model, normalized_workload(workload, anchor))
        basis = ("context_rank" if attention.get("context_enabled") else "rank_service") if attention["available"] else "legacy_tokens"
        group["anchors"].append({"id": anchor["id"], "knobs": anchor["knobs"],
                                  "stage_ms": stages, "power_w": power,
                                  "microbatches": anchor["microbatches"], "layers": anchor["layers"],
                                  "requests_per_pipeline": anchor["requests_per_pipeline"],
                                  "operating_state": anchor.get("operating_state")})
        group["anchors"][-1]["analytical_provisioning"] = anchor.get("analytical_provisioning")
        group["anchors"][-1].update(attention_workload=attention, attention_basis=basis,
                                     context_fallback=bool(rank_model and (not rank_model.get("context_enabled") or not attention["available"])))
    for group in groups.values():
        bases = {(row["attention_basis"], bool(row.get("analytical_provisioning"))) for row in group["anchors"]}
        group["attention_workload_consistent"] = len(bases) == 1
        # Repeated measurements of one point improve its estimate; they do not
        # create extra independent single-factor interventions.
        repeated = {}
        for row in group["anchors"]:
            repeated.setdefault(digest(row["knobs"]), []).append(row)
        pooled = []
        for rows in repeated.values():
            state_rows = [r for r in rows if r.get("operating_state")]
            operating = ({role: {"effective_mhz": statistics.mean(r["operating_state"][role]["effective_mhz"] for r in state_rows),
                                 "power_cap_active_fraction": statistics.mean(r["operating_state"][role]["power_cap_active_fraction"] for r in state_rows)}
                          for role in ("attention", "expert")} if state_rows else None)
            pooled.append({**rows[0], "id": "+".join(r["id"] for r in rows),
                           "stage_ms": {s: statistics.mean(r["stage_ms"][s] for r in rows) for s in STAGES},
                           "power_w": statistics.mean(r["power_w"] for r in rows),
                           "requests_per_pipeline": statistics.mean(r["requests_per_pipeline"] for r in rows),
                           "repetitions": len(rows), "operating_state": operating})
        group["anchors"] = pooled
        rows = group["anchors"]
        for knob in KNOBS:
            evidence = []
            for i, left in enumerate(rows):
                for right in rows[i + 1:]:
                    if left["knobs"][knob] == right["knobs"][knob] or any(
                        left["knobs"][k] != right["knobs"][k] for k in KNOBS if k != knob
                    ):
                        continue
                    high, low = sorted((left, right), key=lambda r: r["knobs"][knob], reverse=True)
                    dx = math.log(high["knobs"][knob] / low["knobs"][knob])
                    evidence.append({"high_anchor": high["id"], "low_anchor": low["id"],
                                     "high": high["knobs"][knob], "low": low["knobs"][knob],
                                     "time_elasticity": {s: math.log(low["stage_ms"][s] / high["stage_ms"][s]) / dx for s in STAGES},
                                     "power_elasticity": math.log(high["power_w"] / low["power_w"]) / dx})
            if evidence:
                group["effects"][knob] = {
                    "time_elasticity": {s: statistics.median(e["time_elasticity"][s] for e in evidence) for s in STAGES},
                    "power_elasticity": statistics.median(e["power_elasticity"] for e in evidence),
                    "range": [min(e["low"] for e in evidence), max(e["high"] for e in evidence)],
                    "evidence": evidence,
                    "confidence": "empirical; paired endpoints are not independent repetitions",
                    "needs_repetition": len(evidence) < 3,
                    "interpretation": "Negative or small elasticity can reflect noise, nonbinding caps, or overlap; it is not a monotonic guarantee.",
                }
        group["joint_response"] = fit_joint(rows)
        if not group["attention_workload_consistent"]:
            # Different workload-normalization bases must not masquerade as
            # paired frequency or power interventions.
            group["effects"] = {}
            group["joint_response"]["identified"] = False
            group["joint_response"]["reason"] = "mixed_attention_workload_support"
    return {"schema_version": 1, "parameter_effects": PARAMETER_EFFECTS, "groups": groups,
            "structural_response": fit_structural(groups, workload),
            "physical_anchor_count": len(anchors), "uses_synthesized_labels_for_fit": False,
            "stage_response_basis": "nominal stage models fitted on physical calibration; empirical response, not raw paired latency or a confidence bound",
            "unidentified_effects": "New parallelism/microbatch structures require additional calibration; no universal scaling exponent is assumed."}


def inform_candidates(candidates, report, workload, request_count):
    """Apply calibrated local elasticity and publish bottleneck/slack explanation.

    No effects transfer across unmeasured structures. Joint changes use a
    separable local approximation with inflated uncertainty, never hard pruning.
    """
    for candidate in candidates:
        if configuration(candidate)['expert_dp'] > 1:
            candidate['prior']['energy_sd'] = max(candidate['prior']['energy_sd'], 1.0)
            candidate['prior']['source'] = 'broad_empirical_bo_fallback'
            candidate['mechanism'] = {'status': 'analytical_model_unavailable', 'identified_knobs': [],
                                      'calibration_priority': 'structure_probe',
                                      'reason': 'independent_replica_arrival_and_schedule_model_unavailable'}
            continue
        group = report["groups"].get(digest(structure(candidate)))
        if not group:
            candidate["mechanism"] = {"status": "needs_structure_anchor", "identified_knobs": [],
                                      "calibration_priority": "structure_probe"}
            # Old resource-count priors cannot identify a newly enumerated layout.
            candidate["prior"]["energy_sd"] = max(candidate["prior"]["energy_sd"], 1.0)
            predicted = structural_predict(report.get("structural_response", {}), candidate)
            if predicted:
                v = predicted["values"]
                stages = {s: v[s] for s in STAGES}
                pipeline = pipeline_time_ms(stages, microbatches=candidate["microbatches"], layers=predicted["layers"])
                capacity = v["requests_per_pipeline"] * 1000 / pipeline
                cfg = configuration(candidate)
                power = min(v["power_w"], sum(len(cfg[r + "_gpus"]) * cfg[r + "_power_w"] for r in ("attention", "expert")))
                candidate["prior"].update(log_energy=math.log(power * request_count / min(workload["arrival_rate_rps"], capacity)),
                                          energy_sd=predicted["uncertainty_log"], source="structural_mechanism_interpolation")
                candidate["mechanism"].update(status="structural_prior_needs_validation", stage_ms=stages,
                                               structural_rank=predicted["rank"], anchors=predicted["anchors"],
                                               bottleneck=max(stages, key=stages.get))
            continue
        cfg = configuration(candidate)
        nearest = min(group["anchors"], key=lambda a: sum(abs(math.log(cfg[k] / a["knobs"][k])) for k in KNOBS))
        changed = [k for k in KNOBS if cfg[k] != nearest["knobs"][k]]
        missing = [k for k in changed if k not in group["effects"]]
        stages = dict(nearest["stage_ms"])
        power = nearest["power_w"]
        outside = []
        for knob in changed:
            if knob in missing:
                continue
            effect = group["effects"][knob]
            step = math.log(cfg[knob] / nearest["knobs"][knob])
            if not effect["range"][0] <= cfg[knob] <= effect["range"][1]:
                outside.append(knob)
            for stage in STAGES:
                # Extreme empirical slopes indicate an unreliable prior, not
                # certainty; bounding the numeric exponent prevents overflow.
                stages[stage] *= math.exp(max(-10, min(10, -effect["time_elasticity"][stage] * step)))
            power *= math.exp(max(-10, min(10, effect["power_elasticity"] * step)))
        bottleneck = max(stages, key=stages.get)
        weak = [k for k in changed if k in group["effects"] and group["effects"][k]["needs_repetition"]]
        uncertainty = 0.25 + 0.25 * bool(weak) + 0.30 * max(len(changed) - 1, 0) + 0.5 * len(outside) + len(missing)
        joint = group["joint_response"]
        joint_values = None
        if joint["identified"]:
            joint_values = joint_predict(joint, candidate["knobs"])
            anchor_values = joint_predict(joint, nearest["knobs"])
            # Preserve the measured anchor exactly; apply the fitted response
            # as a ratio so refitting never changes an anchor into a new label.
            stages = {s: nearest["stage_ms"][s] * joint_values[s] / anchor_values[s] for s in STAGES}
            power = nearest["power_w"] * joint_values["power_w"] / anchor_values["power_w"]
            missing = []
            cross_role = any(k.startswith("attention") for k in changed) and any(k.startswith("expert") for k in changed)
            uncertainty = max(.35, max(v["rmse_log"] for v in joint["responses"].values())) + .5 * len(outside) + .25 * joint["needs_repetition"] + .3 * cross_role
        bottleneck = max(stages, key=stages.get)
        power = min(power, sum(len(cfg[r + "_gpus"]) * cfg[r + "_power_w"] for r in ("attention", "expert")))
        analytical = analytical_predict(nearest.get("analytical_provisioning"), stages, cfg, nearest["layers"], power)
        schedule = analytical["schedule_model"] if analytical["available"] else None
        pipeline = pipeline_time_ms(stages, microbatches=nearest["microbatches"], layers=nearest["layers"], schedule_model=schedule)
        baseline_time = pipeline_time_ms(nearest["stage_ms"], microbatches=nearest["microbatches"], layers=nearest["layers"], schedule_model=schedule)
        # Equal reference requests per pipeline, within the same structure.
        capacity = nearest["requests_per_pipeline"] * 1000 / pipeline
        rate = max(float(workload["arrival_rate_rps"]), 1e-9)
        na, ne = len(cfg["attention_gpus"]), len(cfg["expert_gpus"])
        power = min(power, na * cfg["attention_power_w"] + ne * cfg["expert_power_w"])
        if (nearest.get("context_fallback") or not group["attention_workload_consistent"]
                or analytical.get("available") and analytical["layer_barrier_coverage"] < 1):
            uncertainty = max(uncertainty, 1.)
            candidate["prior"]["energy_sd"] = max(candidate["prior"]["energy_sd"], uncertainty)
        if not missing:
            candidate["prior"].update(log_energy=math.log(power / min(rate, capacity) * request_count),
                                      energy_sd=uncertainty, source="measured_joint_response" if joint_values else "measured_parameter_elasticity")
        candidate["mechanism"] = {
            "status": "local_effects_estimated" if not missing else "needs_parameter_anchor",
            "anchor": nearest["id"], "identified_knobs": sorted(group["effects"]),
            "attention_workload": nearest["attention_workload"],
            "attention_prediction_basis": nearest["attention_basis"],
            "attention_workload_consistent": group["attention_workload_consistent"],
            "analytical_provisioning": analytical,
            "unidentified_knobs": missing, "extrapolated_knobs": outside,
            "uncertain_knobs": [k for k, e in group["effects"].items() if e["needs_repetition"]],
            "joint_effects_unmeasured": len(changed) > 1 and (not joint["identified"] or cross_role),
            "joint_response_identified": joint["identified"],
            "joint_response_rank": joint["rank"],
            "predicted_effective_clocks_mhz": {r: min(cfg[r + "_mhz"], joint_values[r + "_effective_mhz"]) for r in ("attention", "expert")
                                                if joint_values and r + "_effective_mhz" in joint_values},
            "bottleneck": bottleneck, "stage_ms": stages,
            "stage_slack_ms": {s: max(stages.values()) - stages[s] for s in STAGES},
            "pipeline_ratio_to_anchor": pipeline / baseline_time,
            "note": "Stage slack and sensitivities are model explanations, not end-to-end SLO guarantees.",
            "calibration_priority": "parameter_probe" if missing else "model_guided",
        }
    return candidates

"""Automatic calibration-to-provisioning bridge for native four-stage BO.

All coefficients and workload statistics originate from the receipt's clipped
calibration events. Unknown support is reported, never turned into eligibility.
EP ranks constitute a complete FFN service group; connector fan-in is separate.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import nnls

from four_stage_dse_v6.model import STAGES, pipeline_time_ms
from .provisioning import cycle_time, workload_moments


def _fit(x, y, terms):
    if not x:
        return {"available": False, "reason": "missing_decode_calibration"}
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    scale = np.maximum(np.max(np.abs(x), axis=0), 1.)
    coefficient, _ = nnls(x / scale, y)
    coefficient /= scale
    rank = int(np.linalg.matrix_rank(x / scale))
    return {"available": True, "terms": terms, "coefficients": coefficient.tolist(),
            "design_rank": rank, "coefficients_identified": rank == x.shape[1],
            "ranges": np.stack([x.min(axis=0), x.max(axis=0)]).tolist(),
            "rmse_ms": float(np.sqrt(np.mean((x @ coefficient - y) ** 2))), "samples": len(y)}


def calibrated_model(groups, cfg, layers, workload, role_power_w, evidence):
    mapping = {"attention_workers": cfg["attention_dp"], "attention_tp": cfg["attention_tp"],
               "ffn_service_groups": cfg["expert_dp"], "ffn_ep": cfg["expert_ep"], "ffn_tp": cfg["expert_tp"],
               "connector_peer_fan_in": len(cfg["attention_gpus"]) / len(cfg["expert_gpus"]),
               "barrier_scope": "observed_causal_participants_per_layer",
               "global_fan_in": cfg["attention_dp"] / cfg["expert_dp"]}
    complete = [g for g in groups if g.get("layer_barrier_available")
                and [r["layer_idx"] for r in g["layer_observations"]] == list(range(layers))]
    weights = None
    if len(complete) == len(groups):
        totals = np.mean([[list(row["stage_ms"][s] for s in STAGES) for row in g["layer_observations"]]
                          for g in complete], axis=0)
        weights = (totals / totals.sum(axis=0)).tolist()
    schedule = {"type": "finite_microbatch_fifo_v1", "communication": "shared_roundtrip",
                "layer_weights": weights,
                "layer_basis": "calibration_per_layer_service" if weights else "uniform_layers_missing_layer_evidence",
                "resource_assumption": "serialized_roundtrip_conservative_until_independent_resources_calibrated"}
    ax, ay, fx, fy, cy = [], [], [], [], []
    contexts, requests = [], {}
    for g in groups:
        for row in g["attention_rank_observations"].values():
            for span in row.get("decode_request_spans", []):
                key = span["request_id"]
                item = requests.setdefault(key, {"prompt_tokens": span["prompt_tokens"], "positions": set()})
                if item["prompt_tokens"] != span["prompt_tokens"]:
                    raise ValueError("calibration request prompt changed")
                item["positions"].update(range(span["first_token_position"], span["first_token_position"] + span["token_count"]))
        if g["prefill_tokens"] or not g["decode_tokens"]:
            continue
        fx.append([1., float(g["decode_tokens"])])
        fy.append(g["stage_times_ms"]["ffn_compute"] / layers)
        cy.append((g["stage_times_ms"]["a2f_dispatch"] + g["stage_times_ms"]["f2a_combine"]) / layers)
        for row in g["attention_rank_observations"].values():
            context = row.get("context")
            if context is None or not row["decode_tokens"]:
                continue
            ax.append([1., context["decode_context_tokens"]])
            ay.append(row["duration_ms"] / layers)
            # A query-weighted empirical context mean remains useful without
            # claiming independent slots or complete request lifetimes.
            contexts.append((context["decode_context_tokens"], row["decode_tokens"]))
    fits = {"attention": _fit(ax, ay, ["intercept_ms", "ms_per_context_token"]),
            "ffn": _fit(fx, fy, ["intercept_ms", "ms_per_group_query"]),
            "communication_roundtrip": _fit(fx, cy, ["intercept_ms", "ms_per_group_query"])}
    moments = {"available": False, "reason": "missing_complete_decode_request_spans"}
    lifetimes = []
    for identity, item in requests.items():
        positions = sorted(item["positions"])
        if positions and positions == list(range(item["prompt_tokens"], positions[-1] + 1)):
            lifetimes.append({"source_index": identity, "prompt_tokens": positions[0] + 1, "decode_steps": len(positions)})
    if (lifetimes and len(lifetimes) == len(requests) == evidence.get("expected_decode_requests")
            and sum(row["decode_steps"] for row in lifetimes) == evidence.get("expected_decode_queries")):
        moments = {"available": True, **workload_moments(lifetimes),
                   "context_convention": "inclusive_query_context_first_decode_position_plus_one",
                   "origin": "deduplicated_causal_calibration_decode_spans",
                   "complete_request_count": len(lifetimes)}
    moments["observed_requests"] = len(requests)
    moments["expected_completed_requests"] = evidence.get("completed_requests")
    moments["observed_decode_queries"] = sum(len(row["positions"]) for row in requests.values())
    moments["expected_decode_queries"] = evidence.get("expected_decode_queries")
    coefficients = {k: {"intercept": v["coefficients"][0], "slope": v["coefficients"][1]}
                    for k, v in fits.items() if v["available"]}
    return {"type": "calibrated_provisioning_v1", "selection_split": "calibration",
            "mapping": mapping, "schedule_model": schedule, "coefficient_fits": fits,
            "latency_coefficients_ms": coefficients, "workload_moments": moments,
            "empirical_context_mean": sum(v for v, _ in contexts) / sum(n for _, n in contexts) if contexts else None,
            "role_power_w": role_power_w, "evidence": evidence,
            "layer_barrier_coverage": len(complete) / len(groups),
            "observed_participant_counts": sorted({g["attention_rank_summary"]["rank_count"] for g in groups}),
            "balanced_query_shapes": sorted({g["decode_tokens"] for g in groups if not g["prefill_tokens"]
                and len(g["attention_rank_observations"]) == cfg["attention_dp"]
                and len({r["decode_tokens"] for r in g["attention_rank_observations"].values()}) == 1}),
            "reference_workload": dict(workload), "execution_eligibility_changed": False}


def predict(model, stage_ms, cfg, layers, power_w):
    if cfg['expert_dp'] > 1:
        return {'available': False, 'reason': 'independent_replica_arrival_and_schedule_model_unavailable',
                'fallback': 'measured_end_to_end_bo_with_broad_prior'}
    if not model:
        return {"available": False, "reason": "missing_calibrated_provisioning"}
    if model.get("type") != "calibrated_provisioning_v1" or model.get("selection_split") != "calibration":
        raise ValueError("invalid or non-calibration provisioning model")
    mapping = model["mapping"]
    if any(mapping[k] != cfg[c] for k, c in (("attention_workers", "attention_dp"), ("attention_tp", "attention_tp"),
                                            ("ffn_service_groups", "expert_dp"), ("ffn_ep", "expert_ep"), ("ffn_tp", "expert_tp"))):
        return {"available": False, "reason": "service_group_mapping_changed"}
    schedule = model["schedule_model"]
    pipeline = pipeline_time_ms(stage_ms, microbatches=cfg["microbatches"], layers=layers, schedule_model=schedule)
    work = model["reference_workload"]
    queries = float(work.get("decode_tokens_per_microbatch", 0))
    result = {"available": True, "pipeline_ms": pipeline, "mapping": mapping,
              "schedule_model": schedule, "layer_barrier_coverage": model["layer_barrier_coverage"],
              "predicted_active_power_w": power_w,
              "predicted_decode_j_per_token": power_w * pipeline / (1000 * queries * cfg["microbatches"]) if queries > 0 and not work.get("prefill_tokens_per_microbatch", 0) else None,
              "coefficient_identification": {k: v.get("coefficients_identified", False) for k, v in model["coefficient_fits"].items()},
              "gaussian_cycle_ms": None, "mean_field_cycle_ms": None,
              "synchronization_wait_measured": False, "execution_eligibility_changed": False}
    moments = model["workload_moments"]
    ratio = mapping["global_fan_in"]
    # Gaussian fan-in must represent the actual observed synchronized group.
    supported = (moments.get("available") and mapping["ffn_service_groups"] == 1 and mapping["attention_tp"] == 1
                 and model["observed_participant_counts"] == [mapping["attention_workers"]]
                 and queries in model["balanced_query_shapes"] and queries / ratio >= 1 and (queries / ratio).is_integer()
                 and all(result["coefficient_identification"].values()) and queries > 0)
    if supported:
        targets = {"attention": queries / ratio * moments["theta"], "ffn": queries,
                   "communication_roundtrip": queries}
        supported = all(fit["ranges"][0][1] <= targets[key] <= fit["ranges"][1][1]
                        for key, fit in model["coefficient_fits"].items())
    if supported:
        coefficient = {k: dict(v) for k, v in model["latency_coefficients_ms"].items()}
        batch = queries / ratio
        reference = model.get("reference_stage_ms", stage_ms)
        for key, stages in (("attention", ["attention_compute"]), ("ffn", ["ffn_compute"]),
                            ("communication_roundtrip", ["a2f_dispatch", "f2a_combine"])):
            factor = sum(stage_ms[s] for s in stages) / sum(reference[s] for s in stages)
            coefficient[key] = {k: v * factor for k, v in coefficient[key].items()}
        result["gaussian_cycle_ms"] = cycle_time(coefficient, moments, batch, int(ratio), barrier=True)
        result["mean_field_cycle_ms"] = cycle_time(coefficient, moments, batch, int(ratio), barrier=False)
    result["gaussian_status"] = "calibrated_iid_prior_not_a_latency_certificate" if supported else "insufficient_moments_coefficients_or_group_support"
    return result

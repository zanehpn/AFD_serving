"""Enumerate configurations without treating mathematical legality as execution evidence."""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
from collections import Counter

from four_stage_dse_v6.model import evaluate_candidate


PARALLEL = ("attention_dp", "attention_tp", "expert_dp", "expert_ep", "expert_tp")
KNOBS = ("attention_mhz", "expert_mhz", "attention_power_w", "expert_power_w")
DBO_THRESHOLDS = ('dbo_decode_token_threshold', 'dbo_prefill_token_threshold')
CATEGORIES = {
    "allocation": ["attention_gpus", "expert_gpus"],
    "attention_parallelism": ["attention_dp", "attention_tp"],
    "expert_parallelism": ["expert_dp", "expert_ep", "expert_tp"],
    "microbatch": ["microbatches"], "dbo_threshold": list(DBO_THRESHOLDS), "frequency": list(KNOBS[:2]),
    "power_cap": list(KNOBS[2:]),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def configuration(candidate):
    t = candidate["topology"]
    result = {k: list(t[k]) for k in ("attention_gpus", "expert_gpus")}
    result.update({k: t.get(k, 1) for k in PARALLEL})
    if 'parallelism_semantics' in t:
        result['parallelism_semantics'] = t['parallelism_semantics']
    result.update({k: candidate["knobs"][k] for k in KNOBS})
    result["microbatches"] = candidate["microbatches"]
    result["execution_mode"] = candidate.get("execution_mode", "eager")
    for key in DBO_THRESHOLDS:
        if key in candidate:
            result[key] = candidate[key]
    return result


def structure(candidate):
    c = configuration(candidate)
    return {k: v for k, v in c.items() if k not in KNOBS}


def layout(candidate):
    """Capability matching uses role sizes; physical GPU identities remain in candidate IDs."""
    c = structure(candidate)
    c["attention_gpus"] = len(c["attention_gpus"])
    c["expert_gpus"] = len(c["expert_gpus"])
    return c


def enumerate_candidates(spec):
    if spec.get("selection_split") != "calibration":
        raise ValueError("candidate specification must be calibration-only")
    grid_keys = ("attention_frequencies_mhz", "expert_frequencies_mhz",
                 "attention_power_caps_w", "expert_power_caps_w", "microbatches")
    for key in grid_keys:
        if not spec.get(key):
            raise ValueError(f"empty grid: {key}")
        for value in spec[key]:
            integer(value, key)
    rows = {}
    for original in spec["topologies"]:
        for role in ("attention", "expert"):
            gpus = original[f"{role}_gpus"]
            if not isinstance(gpus, list) or not gpus:
                raise ValueError("explicit nonempty GPU index lists are required")
            for gpu in gpus:
                integer(gpu, "GPU index", 0)
        na, ne = len(original["attention_gpus"]), len(original["expert_gpus"])
        if spec.get("enumerate_parallelism", False):
            parallel = [(ad, at, ed, ep, et)
                        for ad in range(1, na + 1) for at in range(1, na + 1)
                        for ed in range(1, ne + 1) for ep in range(1, ne + 1)
                        for et in range(1, ne + 1) if ad * at == na and ed * ep * et == ne]
        else:
            parallel = [tuple(original.get(k, 1) for k in PARALLEL)]
        for degrees, values in itertools.product(parallel, itertools.product(*(spec[k] for k in grid_keys))):
            t = {k: copy.deepcopy(original[k]) for k in ("attention_gpus", "expert_gpus")}
            t.update(dict(zip(PARALLEL, degrees)))
            if 'parallelism_semantics' in original:
                if spec.get('enumerate_parallelism'):
                    raise ValueError('Official launch degrees require explicit topology mappings')
                t['parallelism_semantics'] = original['parallelism_semantics']
            candidate = {"topology": t, "knobs": dict(zip(KNOBS, values[:4])),
                         "microbatches": values[4], "execution_mode": spec.get("execution_mode", "eager"),
                         "selection_split": "calibration"}
            key = digest(configuration(candidate))
            candidate["id"] = "dse-" + key[:20]
            rows[key] = candidate
    profiles = spec.get('dbo_threshold_profiles')
    if profiles is None:
        return list(rows.values())
    if not profiles or len({tuple(p) for p in profiles}) != len(profiles):
        raise ValueError('DBO threshold profiles must be nonempty and distinct')
    expanded = []
    for pair in profiles:
        if len(pair) != 2:
            raise ValueError('DBO profile must contain decode and prefill thresholds')
        for key, value in zip(DBO_THRESHOLDS, pair):
            integer(value, key)
        for original in rows.values():
            if original['microbatches'] != 2:
                raise ValueError('Threshold search requires DBO fixed on')
            candidate = copy.deepcopy(original)
            candidate.update(dict(zip(DBO_THRESHOLDS, pair)))
            candidate['id'] = 'dse-' + digest(configuration(candidate))[:20]
            expanded.append(candidate)
    return expanded


def hardware_from_snapshot(snapshot):
    """Accept the checked-in nvidia-smi snapshot without using the live GPU at search time."""
    devices = {}
    clock_map = {r["gpu_index"]: r["supported_clock_pairs"] for r in snapshot["clock_capabilities"]}
    for row in csv.DictReader(snapshot["gpu_metadata_csv"].splitlines(), skipinitialspace=True):
        idx = int(row["index"])
        if idx not in clock_map:
            continue
        devices[str(idx)] = {"uuid": row["uuid"], "memory_mib": float(row["memory.total [MiB]"]),
                             "min_power_w": float(row["power.min_limit [W]"]),
                             "max_power_w": float(row["power.max_limit [W]"]),
                             "clock_pairs": clock_map[idx]}
    return {"host": snapshot["host"], "devices": devices, "source_date": snapshot["date"]}


def hard_filter(candidate, hardware, runtime):
    c = configuration(candidate)
    reasons, pending = [], []
    ag, eg = c["attention_gpus"], c["expert_gpus"]
    gpus = ag + eg
    if len(set(gpus)) != len(gpus):
        reasons.append("duplicate_or_overlapping_gpus")
    if len(gpus) > runtime["gpu_budget"] or not set(gpus) <= set(runtime["allowed_gpus"]):
        reasons.append("gpu_budget_or_allocation")
    for k in PARALLEL:
        integer(c[k], k)
    if len(ag) != c["attention_dp"] * c["attention_tp"]:
        reasons.append("attention_rank_product")
    official = runtime['adapter'] == 'official_v026'
    if not official and len(eg) != c["expert_dp"] * c["expert_ep"] * c["expert_tp"]:
        reasons.append("expert_rank_product")
    if c.get('parallelism_semantics') and not official:
        reasons.append('parallelism_semantics_adapter_mismatch')
    if runtime.get('capacity_profile'):
        from .capacity import assessment
        entry = assessment(candidate, runtime['capacity_profile'])
        if entry['status'] == 'proven_impossible':
            reasons.append('proven_capacity_weight_lower_bound')
    if runtime.get("connector") == "p2p_nccl" and (len(ag) < len(eg) or len(ag) % len(eg)):
        reasons.append("p2p_attention_expert_rank_ratio")
    # launch_pair.py enables native EP; it cannot deploy independent replicated
    # expert groups or intra-expert TP under this candidate schema.
    if runtime["adapter"] == "v026_launch_pair":
        if c["expert_dp"] != 1 or c["expert_tp"] != 1 or c["expert_ep"] != len(eg):
            reasons.append("launcher_expert_parallelism_unrepresentable")
        if c["microbatches"] not in (1, 2):
            reasons.append("launcher_microbatch_count_unrepresentable")
    elif runtime["adapter"] == "v026_joint_replicas":
        from .native_layout import validate
        try:
            validate(c)
        except ValueError as error:
            reasons.append(str(error))
    elif official:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from official_space import validate
        try:
            validate(c)
        except ValueError as error:
            reasons.append(str(error))
    elif runtime["adapter"] != "external":
        raise ValueError("unknown runtime adapter")
    if c["execution_mode"] not in runtime["execution_modes"]:
        reasons.append("execution_mode_unsupported")
    constraints = runtime.get("divisibility", {})
    for key, dimension in constraints.items():
        if key not in PARALLEL:
            raise ValueError(f"unknown divisibility field {key}")
        integer(dimension, "divisibility dimension")
        if dimension % c[key]:
            reasons.append(f"model_divisibility:{key}")
    for role, indices in (("attention", ag), ("expert", eg)):
        for idx in indices:
            dev = hardware["devices"].get(str(idx))
            if dev is None:
                pending.append(f"unknown_gpu:{idx}")
                continue
            if not dev["min_power_w"] <= c[f"{role}_power_w"] <= dev["max_power_w"]:
                reasons.append(f"power_limit:{role}:{idx}")
            pairs = dev.get("clock_pairs", [])
            if not pairs:
                pending.append(f"unknown_clock_capability:{idx}")
            elif not any(p["memory_mhz"] == runtime["memory_clock_mhz"] and
                         p["graphics_mhz"] == c[f"{role}_mhz"] for p in pairs):
                reasons.append(f"unsupported_clock:{role}:{idx}")
    matched = [r for r in runtime.get("structures", []) if r["layout"] == layout(candidate)]
    if len(matched) > 1:
        raise ValueError("duplicate structure capability")
    if not matched:
        pending.append("structure_execution_unverified")
    else:
        rule = matched[0]
        if rule["status"] == "unsupported":
            reasons.append("backend_structure_unsupported")
        elif rule['status'] == 'validation_failed':
            reasons.append('measured_structure_validation_failed')
        elif rule["status"] != "verified" or not rule.get("evidence"):
            pending.append("structure_execution_unverified")
        # Only caller-supplied, evidenced lower bounds allow a memory rejection.
        for idx, minimum in rule.get("memory_lower_bound_mib", {}).items():
            positive(minimum, "memory lower bound")
            if not rule.get("memory_evidence"):
                raise ValueError("memory lower bound requires evidence")
            if int(idx) in gpus and idx in hardware["devices"] and minimum > hardware["devices"][idx]["memory_mib"]:
                reasons.append(f"proven_memory_lower_bound:{idx}")
    status = "hard_rejected" if reasons else "pending_validation" if pending else "eligible"
    return {"status": status, "reasons": sorted(set(reasons)), "pending_reasons": sorted(set(pending))}


def model_priors(candidates, profile, workload, request_count, default_energy_j):
    """Use matching v6 stage profiles as priors, never as physical observations.

    Uncovered layouts get a broad neutral prior. All topology fields, including
    Expert TP/DP absent from the old surrogate feature key, participate in matching.
    """
    if profile.get("selection_split") != "calibration":
        raise ValueError("only calibration profiles may inform DSE")
    by_config = {}
    for measured in profile.get("candidates", []):
        if measured.get("selection_split") != "calibration":
            raise ValueError("non-calibration candidate profile")
        try:
            key = digest(configuration(measured))
        except (KeyError, TypeError):
            continue
        if key in by_config:
            raise ValueError("ambiguous model profile configuration")
        by_config[key] = measured
    output = []
    for candidate in candidates:
        prior = {"log_energy": math.log(default_energy_j), "energy_sd": 1.0,
                 "log_constraint_ratios": [0.0, 0.0, 0.0], "constraint_sd": 1.0,
                 "source": "uncovered_neutral"}
        matched = by_config.get(digest(configuration(candidate)))
        if matched is not None:
            ev = evaluate_candidate(matched, workload)
            energy = ev.predicted_energy_j_per_request
            if energy is not None and math.isfinite(energy) and energy > 0:
                prior.update(log_energy=math.log(energy * request_count), source="four_stage_v6",
                             energy_sd=max(0.25, float(matched.get("surrogate_uncertainty_log_p90", 0.25))))
            # Pipeline timing is not a percentile-latency or output-throughput
            # label. Expose its optimistic screening ratio separately, softly.
            if ev.pipeline_time_ms is not None:
                ratio = ev.pipeline_time_ms / float(matched["guard_pipeline_time_ms"])
                prior["pipeline_ratio"] = ratio
                prior["pipeline_sd"] = max(0.35, prior["energy_sd"])
        output.append({**candidate, "prior": prior})
    return output


def audit(candidates, hardware, runtime):
    rows = [{"id": c["id"], **hard_filter(c, hardware, runtime)} for c in candidates]
    return {"categories": CATEGORIES, "raw_unique_candidates": len(rows),
            "structural_layouts": len({digest(layout(c)) for c in candidates}),
            "counts": dict(Counter(r["status"] for r in rows)),
            "reasons": dict(Counter(reason for r in rows for reason in r["reasons"])),
            "reason_counts_are_nonexclusive": True, "candidates": rows}


def launch_environment(candidate, runtime):
    if runtime["adapter"] == "v026_joint_replicas":
        from .native_layout import validate
        c = validate(configuration(candidate))
        return {**{f"ECODEP_{role.upper()}_GPUS": ",".join(map(str, c[role + "_gpus"]))
                   for role in ("attention", "expert")},
                **{f"ECODEP_{key.upper()}": str(c[key]) for key in PARALLEL},
                "ECODEP_MICROBATCHES": str(c["microbatches"]),
                "ECODEP_ENABLE_DBO": str(int(c["microbatches"] == 2))}
    if runtime["adapter"] != "v026_launch_pair":
        return None
    c = configuration(candidate)
    if c["expert_dp"] != 1 or c["expert_tp"] != 1 or c["microbatches"] not in (1, 2):
        raise ValueError("configuration has no launch_pair mapping")
    return {"ECODEP_ATTENTION_GPUS": ",".join(map(str, c["attention_gpus"])),
            "ECODEP_EXPERT_GPUS": ",".join(map(str, c["expert_gpus"])),
            "ECODEP_ATTENTION_RANKS": str(len(c["attention_gpus"])),
            "ECODEP_EXPERT_RANKS": str(len(c["expert_gpus"])),
            "ECODEP_ATTENTION_TP": str(c["attention_tp"]), "ECODEP_EXPERT_TP": "1",
            "ECODEP_ENABLE_DBO": str(int(c["microbatches"] == 2)),
            "ECODEP_CUDA_GRAPH_FULL_DECODE_ONLY": str(int(c["execution_mode"] == "full_decode_only"))}

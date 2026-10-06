"""Matched-budget, separate-state method comparisons; no shared search labels."""
from __future__ import annotations

import copy
import random
import statistics
import time
from pathlib import Path

from .campaign import (ask, create_campaign, read_json, write_json, status, tell,
                       freeze, file_hash)
from .demo import synthetic_result
from .executor import run_one
from .optimizer import best_measured
from .space import digest


VARIANTS = {
    "v2": {"use_model_prior": True, "model_screening": True, "method": "bo", "exploration_policy": "broad_v2"},
    "generic_bo": {"exploration_policy": "legacy", "use_model_prior": False, "model_screening": False, "method": "bo"},
    "random": {"exploration_policy": "legacy", "use_model_prior": False, "model_screening": False, "method": "random"},
    "ga": {"exploration_policy": "legacy", "use_model_prior": False, "model_screening": False, "method": "ga", "max_repeats": 1},
}


def create_comparison(config_path, directory, seeds=(0, 1, 2), resume=False, methods=None):
    directory = Path(directory).resolve()
    if directory.exists() and not resume:
        raise FileExistsError("comparison directory exists")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    base_path = Path(config_path).resolve()
    config = read_json(base_path)
    variants = copy.deepcopy(VARIANTS)
    if config.get('bo', {}).get('exploration_policy') == 'capacity_v2':
        variants['v2']['exploration_policy'] = 'capacity_v2'
    if methods is None and resume and (directory/'comparison.json').exists():
        methods = list(dict.fromkeys(a['method'] for a in read_json(directory/'comparison.json')['campaigns']))
    if methods is not None:
        if not methods or len(set(methods)) != len(methods) or any(m not in variants for m in methods):
            raise ValueError('methods must be distinct supported comparison methods')
        variants = {name: variants[name] for name in methods}
    existing = directory / 'comparison.json'
    if resume and existing.exists():
        manifest = read_json(existing)
        if (manifest['input_config']['sha256'] != file_hash(base_path)
                or {(a['method'], a['seed']) for a in manifest['campaigns']} !=
                   {(method, seed) for method in variants for seed in seeds}):
            raise ValueError('comparison inputs/seeds changed during resume')
        for arm in manifest['campaigns']:
            status(arm['directory'])
        return manifest
    for key in ("specification", "hardware", "runtime", "profile", "calibration_trace", "heldout_trace", "calibration_plan"):
        if config.get(key):
            config[key] = str((base_path.parent / config[key]).resolve())
    config["context_files"] = {k: str((base_path.parent / p).resolve()) for k, p in config.get("context_files", {}).items()}
    config["profile_trace_files"] = [str((base_path.parent / p).resolve()) for p in config.get("profile_trace_files", [])]
    # All arms collect the same telemetry and pay the same declared prior setup
    # and optional common calibration plan. Generic arms do not use its model.
    config["require_four_stage"] = config.get('mechanism_model') != 'external_power_duration_v1'
    directory.mkdir(parents=True, exist_ok=resume)
    campaigns = []
    for seed in seeds:
        for name, overrides in variants.items():
            path = directory / f"{name}-seed{seed}.json"
            arm = copy.deepcopy(config)
            arm["bo"] = {**arm.get("bo", {}), **overrides, "seed": seed}
            carried = config.get('prior_arm_costs', {}).get(f'{name}-seed{seed}', {})
            for key in ('evaluations', 'gpu_hours', 'wall_seconds', 'tuning_energy_j'):
                arm['setup_cost'][key] += carried.get(key, 0)
            if path.exists():
                if read_json(path) != arm:
                    raise ValueError('partial comparison configuration changed')
            else:
                write_json(path, arm)
            campaign = directory / f"{name}-seed{seed}"
            if campaign.exists():
                status(campaign)  # verify existing input hashes and preserve observations
            else:
                temporary = directory / f'.{name}-seed{seed}-initializing'
                if temporary.exists():
                    if (temporary / 'state.json').exists():
                        state = read_json(temporary / 'state.json')
                        if state['observations'] or state['pending']:
                            raise ValueError('initialization directory contains measurements')
                    temporary.rename(directory / f'.{name}-seed{seed}-interrupted-{time.time_ns()}')
                create_campaign(path, temporary)
                temporary.rename(campaign)
            campaigns.append({"method": name, "seed": seed, "directory": str(campaign)})
    manifest = {"schema_version": 1, "mode": config["mode"], "selection_split": "calibration",
                "budget_per_arm": config["budget"], "setup_cost_per_arm": config["setup_cost"],
                "additional_prior_costs_by_arm": config.get('prior_arm_costs', {}),
                "limits": config["limits"], "campaigns": campaigns,
                "input_config": {"path": str(base_path), "sha256": file_hash(base_path)},
                "common_calibration_plan": config.get("calibration_plan"),
                "interpretation": "matched total caps, including setup, failures and repeats; no held-out evaluation; GPU-hour overshoot reported",
                "reference": "best observed calibration point across arms, not exhaustive optimum"}
    write_json(directory / "comparison.json", manifest)
    return manifest


def comparison_round(directory, evaluator=None, synthetic=False, timeout_seconds=1800):
    directory = Path(directory)
    manifest = read_json(directory / "comparison.json")
    if synthetic != (manifest["mode"] == "synthetic"):
        raise ValueError("explicit synthetic/physical mode must match comparison")
    if not synthetic and not evaluator:
        raise ValueError("physical comparison requires an explicit evaluator")
    # Deterministic interleaving changes arm order each round to reduce drift
    # confounding. No parallel GPU launches, even across different seeds.
    counts = [status(c["directory"])["cost"]["evaluations"] for c in manifest["campaigns"]]
    order = list(manifest["campaigns"])
    random.Random(sum(counts)).shuffle(order)
    outcomes = []
    for arm in order:
        campaign = arm["directory"]
        if status(campaign)["frozen"]:
            continue
        request = ask(campaign)
        if request.get("stopped"):
            if status(campaign)["best"]:
                freeze(campaign)
            outcomes.append({**arm, "stopped": True})
            continue
        if synthetic:
            outcome = tell(campaign, synthetic_result(request))
        else:
            outcome = run_one(campaign, evaluator, timeout_seconds)
        outcomes.append({**arm, "stopped": False, "outcome": outcome})
    return {"all_stopped": all(o["stopped"] for o in outcomes), "outcomes": outcomes}


def comparison_report(directory, tolerance=.02, reference_campaign=None):
    if not 0 <= tolerance < 1:
        raise ValueError("invalid energy tolerance")
    manifest = read_json(Path(directory) / "comparison.json")
    statuses = [(arm, status(arm["directory"])) for arm in manifest["campaigns"]]
    energies = [s["best"]["metrics"]["energy_j"] for _, s in statuses if s["best"]]
    reference = min(energies) if energies else None
    reference_scope = "observed_union_only_not_exhaustive_oracle"
    reference_evidence = None
    if reference_campaign is not None:
        if not all(s["frozen"] for _, s in statuses):
            raise ValueError("freeze every arm before inspecting a measured reference")
        ref_status = status(reference_campaign)
        if not ref_status["frozen"] or not ref_status["best"]:
            raise ValueError("reference must be frozen with a measured feasible result")
        ref_bundle = read_json(Path(reference_campaign) / "bundle.json")
        first = read_json(Path(manifest["campaigns"][0]["directory"]) / "bundle.json")
        for key in ("runtime", "hardware", "model_workload", "code_sha256"):
            if ref_bundle[key] != first[key]:
                raise ValueError(f"reference context mismatch: {key}")
        for key in ("limits", "workload", "mode", "model_id"):
            if ref_bundle["settings"].get(key) != first["settings"].get(key):
                raise ValueError(f"reference protocol mismatch: {key}")
        for key, source in first["sources"].items():
            if key.startswith("context:") or key in ("specification", "calibration_trace", "profile"):
                if ref_bundle["sources"].get(key, {}).get("sha256") != source["sha256"]:
                    raise ValueError(f"reference source mismatch: {key}")
        ref_state = read_json(Path(reference_campaign) / "state.json")
        eligible = {c["id"] for c in ref_bundle["audit"]["candidates"] if c["status"] == "eligible"}
        measured = {o["candidate_id"] for o in ref_state["observations"]}
        reference = ref_status["best"]["metrics"]["energy_j"]
        reference_scope = "fully_attempted_initial_eligible_space" if eligible <= measured else "partially_measured_reference"
        reference_evidence = {"directory": str(Path(reference_campaign).resolve()),
                              "bundle_sha256": file_hash(Path(reference_campaign) / "bundle.json"),
                              "state_sha256": file_hash(Path(reference_campaign) / "state.json"),
                              "initial_eligible_points": len(eligible), "attempted_initial_eligible_points": len(eligible & measured),
                              "cost": ref_status["cost"],
                              "limitation": "finite noisy measurements; attempted failures are counted; no proof of global optimality"}
    rows = []
    for arm, s in statuses:
        curve = []
        hit = None
        for h in s["curve"]:
            energy = h["best"]["metrics"]["energy_j"] if h["best"] else None
            point = {"evaluations": h["cost"]["evaluations"], "gpu_hours": h["cost"]["gpu_hours"],
                     "energy_j": energy, "ratio_to_best_observed": energy / reference if energy and reference else None}
            curve.append(point)
        # A later failed repetition can invalidate an incumbent. Require the
        # quality threshold to remain satisfied through the final observation.
        for i, point in enumerate(curve):
            if reference and all(p["energy_j"] is not None and p["energy_j"] <= reference * (1 + tolerance) for p in curve[i:]):
                hit = point
                break
        state = read_json(Path(arm["directory"]) / "state.json")
        rows.append({**arm, "best": s["best"], "cost": s["cost"], "curve": curve,
                     "sustained_target_cost": hit, "pending": s["pending"] is not None,
                     "failures": sum(o["status"] != "ok" for o in state["observations"]),
                     "gpu_hour_overshoot": max(0, s["cost"]["gpu_hours"] - s["budget"]["gpu_hours"])})
    summaries = {}
    for method in VARIANTS:
        arms = [r for r in rows if r["method"] == method]
        hits = [r["sustained_target_cost"] for r in arms if r["sustained_target_cost"]]
        finals = [r["best"]["metrics"]["energy_j"] for r in arms if r["best"]]
        summaries[method] = {"seeds": len(arms), "target_hit_seeds": len(hits),
                             "median_final_energy_j": statistics.median(finals) if finals else None,
                             "median_target_evaluations_among_hits": statistics.median(h["evaluations"] for h in hits) if hits else None,
                             "median_target_gpu_hours_among_hits": statistics.median(h["gpu_hours"] for h in hits) if hits else None}
    return {"mode": manifest["mode"], "selection_split": "calibration",
            "budget_per_arm": manifest["budget_per_arm"], "target_tolerance": tolerance,
            "reference_energy_j": reference, "reference_scope": reference_scope, "reference_evidence": reference_evidence,
            "methods": summaries, "arms": rows, "heldout_evaluation_completed": False}

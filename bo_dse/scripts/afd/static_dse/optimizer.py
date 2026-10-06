"""Finite-space GP residual learning and cost-aware constrained acquisition."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from . import exploration
from .space import KNOBS, configuration, digest, structure
from four_stage_dse_v6.model import STAGES


def features(candidates, *, include_analytical=True):
    structures = sorted({digest(structure(c)) for c in candidates})
    lookup = {key: i for i, key in enumerate(structures)}
    numeric = np.asarray([[math.log(configuration(c)[k]) for k in KNOBS] for c in candidates])
    numeric = (numeric - numeric.min(axis=0)) / np.maximum(np.ptp(numeric, axis=0), 1e-9)
    cats = np.zeros((len(candidates), len(structures)))
    for i, c in enumerate(candidates):
        cats[i, lookup[digest(structure(c))]] = 1.0
    # Bottleneck shape comes only from calibration, not future measurements.
    stages = np.asarray([[float(c["mechanism"]["stage_ms"][s]) for s in STAGES]
                         if len(c.get("mechanism", {}).get("stage_ms", {})) == 4 else [0.] * 4
                         for c in candidates])
    stages /= np.maximum(stages.max(axis=1, keepdims=True), 1e-9)
    if not include_analytical:
        return np.concatenate([numeric, cats, stages], axis=1)
    analytical = []
    for c in candidates:
        mechanism = c.get("mechanism", {})
        row = mechanism.get("analytical_provisioning", {})
        if not row.get("available"):
            analytical.append([0., 0., 0., 0.])
            continue
        gaussian, mean = row.get("gaussian_cycle_ms"), row.get("mean_field_cycle_ms")
        analytical.append([1., row["layer_barrier_coverage"],
                           math.log1p(row["pipeline_ms"] / max(sum(mechanism["stage_ms"].values()), 1e-9)),
                           math.log(gaussian / mean) if gaussian and mean else 0.])
    return np.concatenate([numeric, cats, stages, np.asarray(analytical)], axis=1)


def posterior(x, indices, values, prior, prior_sd, noise):
    prior, prior_sd = np.asarray(prior), np.asarray(prior_sd)
    if not indices:
        return prior.copy(), prior_sd.copy()
    idx = np.asarray(indices)
    gp = GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=2.5),
                                  alpha=(noise / prior_sd[idx]) ** 2 + 1e-8,
                                  optimizer=None, normalize_y=False)
    gp.fit(x[idx], (np.asarray(values) - prior[idx]) / prior_sd[idx])
    # Chunk diagonal prediction to keep full 22,500-space scoring inexpensive.
    means, stds = [], []
    for start in range(0, len(x), 1024):
        mean, std = gp.predict(x[start:start + 1024], return_std=True)
        means.extend(mean)
        stds.extend(std)
    return prior + prior_sd * means, np.maximum(prior_sd * stds, 1e-6)


def aggregate(observations, limits):
    grouped = {}
    for row in observations:
        grouped.setdefault(row["candidate_id"], []).append(row)
    result = {}
    for candidate_id, rows in grouped.items():
        valid = [r for r in rows if r["status"] == "ok"]
        if not valid:
            continue
        metrics = {key: float(np.mean([r["metrics"][key] for r in valid]))
                   for key in ("energy_j", "ttft_ms", "tpot_ms", "output_tps") + (("tbt_ms",) if "tbt_ms" in limits else ())}
        feasible = (len(valid) == len(rows) and metrics["ttft_ms"] <= limits["ttft_ms"] and
                    metrics["tpot_ms"] <= limits["tpot_ms"] and metrics["output_tps"] >= limits["min_output_tps"]
                    and ("tbt_ms" not in limits or metrics["tbt_ms"] <= limits["tbt_ms"]))
        result[candidate_id] = {"metrics": metrics, "feasible": feasible, "repetitions": len(valid)}
    return result


def best_measured(observations, limits):
    valid = [(key, row) for key, row in aggregate(observations, limits).items() if row["feasible"]]
    if not valid:
        return None
    key, row = min(valid, key=lambda item: (item[1]["metrics"]["energy_j"], item[0]))
    return {"candidate_id": key, **row}


def expected_cost(candidate, current, options):
    switching = 0.0
    if current is None or structure(candidate) != structure(current):
        switching = options["structure_switch_seconds"]
    elif configuration(candidate) != configuration(current):
        switching = options["knob_switch_seconds"]
    gpus = len(candidate["topology"]["attention_gpus"]) + len(candidate["topology"]["expert_gpus"])
    # Allocated during a topology switch: use the larger old/new allocation.
    old_gpus = gpus if current is None else sum(len(current["topology"][f"{r}_gpus"]) for r in ("attention", "expert"))
    return max((options["evaluation_seconds"] * gpus + switching * max(gpus, old_gpus)) / 3600, 1e-9)


def lognormal_ei(best, mean, std):
    std = np.maximum(std, 1e-9)
    z = (math.log(best) - mean) / std
    # Truncation prevents overflow in far extrapolation; such points retain
    # exploration through constraint feasibility and periodic deferred audits.
    return np.maximum(best * ndtr(z) - np.exp(np.clip(mean + std ** 2 / 2, -700, 700)) * ndtr(z - std), 0.0)


def propose(candidates, observations, settings, eligible_ids, remaining_gpu_hours, *, _local=False, _current=None):
    if settings['bo']['method'] == 'ga':
        from .genetic import propose as genetic_propose
        return genetic_propose(candidates, observations, settings, eligible_ids, remaining_gpu_hours)
    from . import capacity_search
    if capacity_search.enabled(settings['bo']):
        return capacity_search.propose(candidates, observations, settings, eligible_ids, remaining_gpu_hours)
    all_candidates = {c["id"]: c for c in candidates}
    candidates = [c for c in candidates if _local or c["id"] in eligible_ids]
    if not candidates:
        return None, {"reason": "empty_space"}
    options, limits = settings["bo"], settings["limits"]
    collect_stages = settings.get('mechanism_model') != 'external_power_duration_v1'
    by_id = {c["id"]: i for i, c in enumerate(candidates)}
    counts = Counter(o["candidate_id"] for o in observations)
    known_failures = {o["candidate_id"] for o in observations if o["status"] == "runtime_incompatible"}
    current = _current if _local else (all_candidates[observations[-1]["candidate_id"]] if observations else None)
    model_observations = [o for o in observations if o["candidate_id"] in by_id]
    costs = np.asarray([expected_cost(c, current, options) for c in candidates])
    base_costs = costs.copy()
    selectable = [i for i, c in enumerate(candidates) if c["id"] in eligible_ids and
                  c["id"] not in known_failures and counts[c["id"]] < options["max_repeats"]]
    if not selectable:
        return None, {"reason": "space_exhausted"}
    # Reference is measured first; analytical anchors are never free observations.
    reference = settings["reference_candidate_id"]
    if counts[reference] == 0 and reference in eligible_ids:
        i = by_id[reference]
        if costs[i] > remaining_gpu_hours:
            return None, {"reason": "reference_cost_exceeds_remaining_budget"}
        return candidates[i], {"reason": "measure_reference", "expected_gpu_hours": float(costs[i]), "base_gpu_hours": float(base_costs[i])}
    # The generic BO baseline must not receive mechanism features for free.
    feature_candidates = candidates if options["use_model_prior"] else [{k: v for k, v in c.items() if k != "mechanism"} for c in candidates]
    x = features(feature_candidates, include_analytical=settings.get('mechanism_model') != 'four_stage_fifo_v1')
    successful = [o for o in model_observations if o["status"] == "ok"]
    indices = [by_id[o["candidate_id"]] for o in successful]
    use_prior = options["use_model_prior"]
    baseline = math.log(settings["default_energy_j"])
    eprior = np.asarray([c["prior"]["log_energy"] if use_prior else baseline for c in candidates])
    esd = np.asarray([c["prior"]["energy_sd"] if use_prior else 1.0 for c in candidates])
    emean, estd = posterior(x, indices, [math.log(o["metrics"]["energy_j"]) for o in successful],
                           eprior, esd, options["noise_log"])
    constraint_means, constraint_stds = [], []
    probability = np.ones(len(candidates))
    for key, bound, sign in (("ttft_ms", limits["ttft_ms"], 1),
                             ("tpot_ms", limits["tpot_ms"], 1),
                             ("output_tps", limits["min_output_tps"], -1)) + ((("tbt_ms", limits["tbt_ms"], 1),) if "tbt_ms" in limits else ()):
        vals = [sign * math.log(o["metrics"][key] / bound) for o in successful]
        mean, std = posterior(x, indices, vals, np.zeros(len(x)), np.ones(len(x)) * options["constraint_prior_sd"], options["noise_log"])
        constraint_means.append(mean)
        constraint_stds.append(std)
        probability *= ndtr(-mean / std)
    # Separate execution reliability from performance constraints. Kernel-weighted
    # beta smoothing uses failures without inventing energy or latency labels.
    reliability = np.ones(len(candidates))
    if model_observations:
        successes, failures = np.zeros(len(x)), np.zeros(len(x))
        for observation in model_observations:
            weight = np.exp(-np.sum((x - x[by_id[observation["candidate_id"]]]) ** 2, axis=1))
            if observation["status"] == "ok":
                successes += weight
            else:
                failures += weight
        reliability = (1 + successes) / (2 + successes + failures)
    positive_cost_observations = [o for o in model_observations if o["cost"]["gpu_hours"] > 0]
    if positive_cost_observations:
        obs_indices = [by_id[o["candidate_id"]] for o in positive_cost_observations]
        residual = [math.log(o["cost"]["gpu_hours"] / o.get("proposal_base_gpu_hours", o["proposal_expected_gpu_hours"])) for o in positive_cost_observations]
        corr, _ = posterior(x, obs_indices, residual, np.zeros(len(x)), np.ones(len(x)) * .5, .25)
        costs *= np.exp(np.clip(corr, -3, 3))
    affordable = [i for i in selectable if costs[i] <= remaining_gpu_hours]
    if not affordable:
        return None, {"reason": "estimated_cost_exceeds_remaining_budget"}
    affordable = exploration.novelty_pool(candidates, affordable, observations, settings)
    best = best_measured(observations, limits)
    analytical_points = [j for j in affordable
                         if candidates[j].get("mechanism", {}).get("analytical_provisioning", {}).get("available")
                         and not candidates[j]["mechanism"].get("unidentified_knobs")]
    interval = options.get("analytical_probe_every", 4)
    if (use_prior and options["method"] == "bo" and interval > 0 and len(observations) % interval == 1
            and len(observations) % options["deferred_audit_every"] != 0):
        unseen = [j for j in analytical_points if counts[candidates[j]["id"]] == 0]
        if unseen:
            i = min(unseen, key=lambda j: (candidates[j]["prior"]["log_energy"], costs[j], candidates[j]["id"]))
            return candidates[i], {"reason": "analytical_provisioning_probe",
                                   "expected_gpu_hours": float(costs[i]), "base_gpu_hours": float(base_costs[i]),
                                   "predicted_log_energy": candidates[i]["prior"]["log_energy"],
                                   "collect_four_stage_observation": collect_stages,
                                   "execution_eligibility_changed": False}
    broad = exploration.choose(candidates, affordable, observations, settings, costs, emean, estd)
    if broad is not None:
        i, decision = broad
        return candidates[i], {**decision, "expected_gpu_hours": float(costs[i]),
                               "base_gpu_hours": float(base_costs[i])}
    # Mechanistic experimental design precedes general acquisition: change one
    # unidentified control at a time around the fixed reference. These probes
    # are real measurements, budgeted like BO, and may fail the service gate.
    probe_index = len(observations) - 1
    controls = ("expert_mhz", "attention_mhz", "expert_power_w", "attention_power_w")
    if not exploration.enabled(options) and options["use_model_prior"] and 0 <= probe_index < options["initial_parameter_probes"] and probe_index < len(controls):
        control = controls[probe_index]
        design_reference = (min(analytical_points, key=lambda j: candidates[j]["prior"]["log_energy"])
                            if analytical_points else by_id[reference])
        reference_config = configuration(candidates[design_reference])
        probes = []
        for j in affordable:
            c = candidates[j]
            cfg = configuration(c)
            mechanism = c.get("mechanism", {})
            confident = control in mechanism.get("identified_knobs", []) and control not in mechanism.get("uncertain_knobs", [])
            if counts[c["id"]] or confident:
                continue
            if cfg[control] != reference_config[control] and all(cfg[k] == reference_config[k] for k in cfg if k != control):
                probes.append(j)
        if probes:
            # Nearest supported intervention avoids conflating several changes.
            i = min(probes, key=lambda j: abs(math.log(configuration(candidates[j])[control] / reference_config[control])))
            return candidates[i], {"reason": "single_parameter_probe", "control": control,
                                   "expected_gpu_hours": float(costs[i]), "base_gpu_hours": float(base_costs[i]),
                                   "collect_four_stage_observation": collect_stages}
    joint_index = probe_index - options["initial_parameter_probes"]
    if not exploration.enabled(options) and use_prior and 0 <= joint_index < min(options.get("initial_joint_probes", 0), 2):
        role = ("expert", "attention")[joint_index]
        controls = {role + "_mhz", role + "_power_w"}
        refcfg = configuration(candidates[by_id[reference]])
        probes = [j for j in affordable if counts[candidates[j]["id"]] == 0
                  and not candidates[j].get("mechanism", {}).get("joint_response_identified", False)
                  and {k for k, v in configuration(candidates[j]).items() if v != refcfg[k]} == controls]
        if probes:
            i = min(probes, key=lambda j: sum(abs(math.log(configuration(candidates[j])[k] / refcfg[k])) for k in controls))
            return candidates[i], {"reason": "joint_parameter_probe", "controls": sorted(controls),
                                   "expected_gpu_hours": float(costs[i]), "base_gpu_hours": float(base_costs[i]),
                                   "collect_four_stage_observation": collect_stages}
    acquisition = probability * reliability
    if best is not None:
        acquisition *= lognormal_ei(best["metrics"]["energy_j"], emean, estd)
    if options["cost_aware"]:
        acquisition /= costs
    deferred = []
    if options["model_screening"]:
        beta = options["screen_beta"]
        for i in affordable:
            bad_constraint = any(mean[i] - beta * std[i] > 0 for mean, std in zip(constraint_means, constraint_stds))
            bad_energy = best is not None and emean[i] - beta * estd[i] > math.log(best["metrics"]["energy_j"] * (1 + options["energy_defer_margin"]))
            if (bad_constraint or bad_energy) and candidates[i]["id"] != reference:
                deferred.append(i)
    deferred_set = set(deferred)
    active = [i for i in affordable if i not in deferred_set]
    rng = np.random.default_rng(options["seed"] + len(observations))
    periodic_audit = len(observations) % options["deferred_audit_every"] == 0
    if options["method"] == "random":
        i, reason = int(rng.choice(affordable)), "random"
    elif deferred and (periodic_audit or not active):
        i, reason = int(rng.choice(deferred)), "deferred_audit" if active else "deferred_rescue"
    else:
        # Normal acquisition follows separately budgeted structure exploration.
        scores = acquisition[active]
        if not np.any(scores > 1e-300):
            i = max(active, key=lambda j: (estd[j] * probability[j], candidates[j]["id"]))
            reason = "uncertainty_exploration"
        else:
            i = max(active, key=lambda j: (float(acquisition[j]), -counts[candidates[j]["id"]], candidates[j]["id"]))
            reason = "constrained_ei" if best else "find_feasible"
    return candidates[i], {"reason": reason, "expected_gpu_hours": float(costs[i]), "base_gpu_hours": float(base_costs[i]),
                           "feasible_probability": float(probability[i]), "execution_probability": float(reliability[i]),
                           "acquisition": float(acquisition[i]), "predicted_log_energy": float(emean[i]),
                           "log_energy_sd": float(estd[i]), "active_count": len(active),
                           "deferred_count": len(deferred), "affordable_count": len(affordable),
                           "deferred_ids": [candidates[j]["id"] for j in deferred],
                           "independent_objective_and_constraint_posteriors": True}

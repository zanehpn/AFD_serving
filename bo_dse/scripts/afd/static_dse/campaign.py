"""Immutable inputs, identity isolation, resumable ask/tell, and complete trial accounting."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import time
from decimal import Decimal
from contextlib import contextmanager
from pathlib import Path

from .mechanism import build_mechanism, inform_candidates
from .optimizer import best_measured, propose
from .optimizer import expected_cost
from .genetic import DEFAULT_GA, validate_options as validate_ga_options
from .response import normalized_workload
from .calibration import planned_candidate
from .space import (audit, configuration, digest, enumerate_candidates, hardware_from_snapshot,
                    integer, launch_environment, model_priors, positive, structure)
from four_stage_dse_v6.model import STAGES


DEFAULT_BO = {"method": "bo", "use_model_prior": True, "model_screening": True,
              "cost_aware": True, "seed": 0, "noise_log": .05,
              "constraint_prior_sd": .5, "screen_beta": 2.5,
              "energy_defer_margin": .02, "deferred_audit_every": 5,
              "initial_parameter_probes": 4,
              "exploration_policy": "broad_v2", "structure_exploration_fraction": 1/3,
              "repeat_after_fraction": .75, "repeat_boundary_margin": .01,
              "initial_joint_probes": 2,
              "capacity_initial_structures": 2, "capacity_explore_every": 3,
              "capacity_structure_fraction": .375, "capacity_joint_probe_every": 3,
              "capacity_policy_revision": "legal_contractions_v2",
              "analytical_probe_every": 4,
              "max_repeats": 2, "evaluation_seconds": 90.,
              "structure_switch_seconds": 120., "knob_switch_seconds": 1.}


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    root = Path(__file__).resolve().parent
    files = list(root.glob("*.py")) + list((root.parent / "four_stage_dse_v6").glob("*.py"))
    files += [root.parent / "static_dse_cli.py", root.parent / "launch_pair.py"]
    files += [root.parent / "build_deepseek_v6_four_stage_profile.py", root.parent / "measure_command.py",
              root.parent / "static_dse_feedback_executor.py"]
    return digest({str(p.relative_to(root.parent)): file_hash(p) for p in files if p.is_file()})


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def trace_info(path, fields):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty trace")
    ids = {}
    for field in fields:
        values = []
        for row in rows:
            value = row.get(field)
            if value is None or isinstance(value, (dict, list, bool)):
                raise ValueError(f"missing or invalid immutable identity {field}")
            # Compare identities without treating JSON 1 and 1.0 as different.
            if isinstance(value, (int, float)):
                number = Decimal(str(value))
                if not number.is_finite():
                    raise ValueError("nonfinite trace identity")
                values.append(str(number.normalize()))
            else:
                values.append(value)
        if len(set(values)) != len(rows):
            raise ValueError(f"duplicate trace identity: {field}")
        ids[field] = set(values)
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path), "requests": len(rows)}, ids


def isolation(calibration, heldout, fields):
    if Path(calibration).resolve() == Path(heldout).resolve():
        raise ValueError("calibration and heldout are the same file")
    cal, ci = trace_info(calibration, fields)
    test, ti = trace_info(heldout, fields)
    overlap = {field: len(ci[field] & ti[field]) for field in fields}
    if any(overlap.values()):
        raise ValueError(f"trace identity overlap: {overlap}")
    return {"calibration": cal, "heldout": test, "identity_fields": fields, "overlap": overlap}


def validate_settings(settings):
    if settings.get("selection_split") != "calibration":
        raise ValueError("DSE tuning must be labelled calibration")
    if settings.get("mode") not in ("physical", "synthetic"):
        raise ValueError("explicit physical/synthetic mode required")
    for key in ("ttft_ms", "tpot_ms", "min_output_tps"):
        positive(settings["limits"][key], key)
    if "tbt_ms" in settings["limits"]:
        positive(settings["limits"]["tbt_ms"], "tbt_ms")
    positive(settings["default_energy_j"], "default_energy_j")
    positive(settings["workload"]["arrival_rate_rps"], "arrival rate")
    positive(settings["budget"]["gpu_hours"], "GPU hour budget")
    integer(settings["budget"]["evaluations"], "evaluation budget")
    validate_cost(settings["setup_cost"])
    integer(settings["setup_cost"]["evaluations"], "setup evaluations", 0)
    bo = {**DEFAULT_BO, **settings.get("bo", {})}
    if bo["method"] not in ("bo", "random", "ga"):
        raise ValueError("method must be bo, random or ga")
    if bo["method"] == "ga":
        bo = {**DEFAULT_GA, **bo}
        validate_ga_options(bo)
    for k in ("use_model_prior", "model_screening", "cost_aware"):
        if type(bo[k]) is not bool:
            raise ValueError(f"{k} must be boolean")
    for k in ("noise_log", "constraint_prior_sd", "screen_beta", "evaluation_seconds"):
        positive(bo[k], k)
    for k in ("structure_switch_seconds", "knob_switch_seconds", "energy_defer_margin"):
        if not math.isfinite(bo[k]) or bo[k] < 0:
            raise ValueError(f"invalid {k}")
    for k in ("deferred_audit_every", "max_repeats"):
        integer(bo[k], k)
    integer(bo["initial_parameter_probes"], "initial_parameter_probes", 0)
    integer(bo["initial_joint_probes"], "initial_joint_probes", 0)
    integer(bo["analytical_probe_every"], "analytical_probe_every", 0)
    integer(bo["seed"], "seed", 0)
    for key in ('capacity_initial_structures', 'capacity_explore_every', 'capacity_joint_probe_every'):
        integer(bo[key], key)
    if bo['exploration_policy'] not in ('legacy', 'broad_v2', 'capacity_v2'):
        raise ValueError('unknown exploration_policy')
    for key in ('structure_exploration_fraction', 'repeat_after_fraction', 'capacity_structure_fraction'):
        if isinstance(bo[key], bool) or not math.isfinite(bo[key]) or not 0 < bo[key] <= 1:
            raise ValueError(f'invalid {key}')
    if isinstance(bo['repeat_boundary_margin'], bool) or not math.isfinite(bo['repeat_boundary_margin']) or not 0 <= bo['repeat_boundary_margin'] < 1:
        raise ValueError('invalid repeat_boundary_margin')
    settings["bo"] = bo
    for k in ("require_four_stage", "allow_structure_probes", "require_output_correctness"):
        if type(settings.get(k, False)) is not bool:
            raise ValueError(f"{k} must be boolean")
    if (settings.get("allow_structure_probes") and not settings.get("require_four_stage")
            and settings.get('mechanism_model') != 'external_power_duration_v1'):
        raise ValueError("structure validation requires four-stage feedback")
    if settings.get('mechanism_model') == 'external_power_duration_v1':
        if settings.get('require_four_stage'):
            raise ValueError('Official external model requires no stage labels')


def validate_cost(cost):
    for key in ("gpu_hours", "wall_seconds", "tuning_energy_j"):
        if isinstance(cost[key], bool) or not isinstance(cost[key], (int, float)) or not math.isfinite(cost[key]) or cost[key] < 0:
            raise ValueError(f"invalid cost: {key}")


def create_campaign(config_path, directory):
    started = time.perf_counter()
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("campaign directory exists; resume it or use a new directory")
    settings = copy.deepcopy(read_json(config_path))
    validate_settings(settings)
    base = Path(config_path).resolve().parent
    sources = {}
    def input_file(name, value):
        path = (base / value).resolve()
        sources[name] = {"path": str(path), "sha256": file_hash(path)}
        return path
    paths = {key: input_file(key, settings[key]) for key in
             ("specification", "hardware", "runtime", "profile", "calibration_trace", "heldout_trace")}
    input_file("campaign_config", str(Path(config_path).resolve()))
    context_files = settings.get("context_files", {})
    if settings["mode"] == "physical" and not {"model_config", "plugin_source", "launcher"} <= context_files.keys():
        raise ValueError("physical campaigns must bind model_config, plugin_source, and launcher context files")
    for key, path in context_files.items():
        input_file("context:" + key, path)
    fields = settings.get("identity_fields", ["source_index", "source_timestamp"])
    if not fields or "source_index" not in fields:
        raise ValueError("source_index audit is required")
    split = isolation(paths["calibration_trace"], paths["heldout_trace"], fields)
    hardware = read_json(paths["hardware"])
    if "gpu_metadata_csv" in hardware:
        hardware = hardware_from_snapshot(hardware)
    runtime, profile, spec = (read_json(paths[k]) for k in ("runtime", "profile", "specification"))
    if settings["mode"] == "physical":
        if not settings.get("model_id") or profile.get("model") != settings["model_id"] or runtime.get("model_id") != settings["model_id"]:
            raise ValueError("model-specific profile/runtime evidence must match model_id")
        profile_traces = settings.get("profile_trace_files", [])
        if not profile_traces and profile.get("candidates"):
            raise ValueError("physical profiles require their calibration source traces for isolation audit")
        split["profile_calibration"] = []
        for i, name in enumerate(profile_traces):
            p = input_file(f"profile_trace:{i}", name)
            checked = isolation(p, paths["heldout_trace"], fields)
            split["profile_calibration"].append(checked)
    # Evidence files are immutable campaign inputs, not an unchecked boolean.
    for i, rule in enumerate(runtime.get("structures", [])):
        for j, evidence in enumerate(rule.get("evidence", []) + rule.get("memory_evidence", [])):
            p = (paths["runtime"].parent / evidence["path"]).resolve()
            if file_hash(p) != evidence["sha256"]:
                raise ValueError("runtime evidence hash mismatch")
            input_file(f"runtime_evidence:{i}:{j}", str(p))
    candidates = enumerate_candidates(spec)
    count = split["calibration"]["requests"]
    workload = {**profile.get("reference_workload", {}), **settings["workload"]}
    candidates = model_priors(candidates, profile, workload, count, settings["default_energy_j"])
    mechanism = build_mechanism(profile, workload)
    if settings["mode"] == "physical" and mechanism["physical_anchor_count"]:
        if settings["setup_cost"]["evaluations"] < mechanism["physical_anchor_count"] or settings["setup_cost"]["gpu_hours"] <= 0:
            raise ValueError("physical anchor profiling cost must be included in setup_cost")
    candidates = inform_candidates(candidates, mechanism, workload, count)
    external_observations = profile.get('external_observations', [])
    if settings.get('mechanism_model') == 'external_power_duration_v1':
        from .external_model import inform
        if runtime['adapter'] != 'official_v026' or profile.get('candidates'):
            raise ValueError('External model requires the official adapter and observable-only profile')
        if external_observations and (not settings.get('profile_trace_files') or
                settings['setup_cost']['evaluations'] < len(external_observations)):
            raise ValueError('External calibration requires audited traces and charged measurements')
        candidates, mechanism = inform(candidates, external_observations, settings['default_energy_j'])
    report = audit(candidates, hardware, runtime)
    plan = {"trials": []}
    if settings.get("calibration_plan"):
        plan = read_json(input_file("calibration_plan", settings["calibration_plan"]))
        if plan.get("selection_split") != "calibration":
            raise ValueError("calibration plan split mismatch")
        by_id = {c["id"]: c for c in candidates}
        checks = {r["id"]: r for r in report["candidates"]}
        for trial in plan["trials"]:
            c = by_id[trial["candidate_id"]]
            if trial["configuration"] != configuration(c):
                raise ValueError("calibration plan configuration mismatch")
            check = checks[c["id"]]
            if check["status"] == "hard_rejected" or set(check["pending_reasons"]) - {"structure_execution_unverified"}:
                raise ValueError("calibration plan contains an unsupported configuration")
    if "reference_configuration" in settings:
        target = digest(settings["reference_configuration"])
        matches = [c["id"] for c in candidates if digest(configuration(c)) == target]
        if len(matches) != 1:
            raise ValueError("reference configuration is absent or ambiguous")
        settings["reference_candidate_id"] = matches[0]
    ref = settings["reference_candidate_id"]
    if not any(r["id"] == ref and r["status"] == "eligible" for r in report["candidates"]):
        raise ValueError("reference must pass hardware filters and have structure execution evidence")
    # Canonical settings contain resolved traces and all backend context in proposals.
    settings["calibration_trace"] = str(paths["calibration_trace"])
    settings["heldout_trace"] = str(paths["heldout_trace"])
    bundle = {"schema_version": 1, "settings": settings, "runtime": runtime,
              "hardware": hardware, "candidates": candidates, "audit": report,
              "model_workload": workload,
              "external_observations": external_observations,
              "calibration_plan": plan,
              "mechanism_anchors": [c for c in profile.get("candidates", []) if c.get("validation_status") == "physically_measured_anchor"],
              "isolation": split, "sources": sources, "code_sha256": code_hash()}
    directory.mkdir(parents=True)
    write_json(directory / "bundle.json", bundle)
    write_json(directory / "mechanism.json", mechanism)
    write_json(directory / "audit.json", report)
    state = {"schema_version": 1, "bundle_sha256": file_hash(directory / "bundle.json"),
             "observations": [], "pending": None, "frozen": False, "history": [],
             "initialization_wall_seconds": time.perf_counter() - started,
             "optimizer_wall_seconds": 0.0}
    write_json(directory / "state.json", state)
    return report


@contextmanager
def locked_campaign(directory):
    directory = Path(directory)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(directory / "state.json")
        if state["bundle_sha256"] != file_hash(directory / "bundle.json"):
            raise ValueError("frozen campaign bundle changed")
        bundle = read_json(directory / "bundle.json")
        if bundle["code_sha256"] != code_hash():
            raise ValueError("DSE code changed; start a new campaign")
        for source in bundle["sources"].values():
            if source["sha256"] != file_hash(source["path"]):
                raise ValueError(f"campaign input changed: {source['path']}")
        yield bundle, state
        write_json(directory / "state.json", state)


def costs(bundle, state):
    total = copy.deepcopy(bundle["settings"]["setup_cost"])
    total["evaluations"] += len(state["observations"])
    for observation in state["observations"]:
        for k in ("gpu_hours", "wall_seconds", "tuning_energy_j"):
            total[k] += observation["cost"][k]
    total["optimizer_wall_seconds"] = state.get("optimizer_wall_seconds", 0.)
    total["initialization_wall_seconds"] = state.get("initialization_wall_seconds", 0.)
    total["wall_seconds"] += total["optimizer_wall_seconds"] + total["initialization_wall_seconds"]
    return total


def updated_mechanism(bundle, state):
    extra = [o for o in state["observations"] if o["status"] == "ok" and "four_stage" in o
             and o['four_stage'].get('model_feedback_supported', True)]
    if not extra:
        return None
    by_id = {c["id"]: c for c in bundle["candidates"]}
    anchors = copy.deepcopy(bundle["mechanism_anchors"])
    for observation in extra:
        c = by_id[observation["candidate_id"]]
        stage = observation["four_stage"]
        anchors.append({**copy.deepcopy(c), "id": observation["trial_id"],
                        "validation_status": "physically_measured_anchor",
                        "stage_models": stage.get("stage_models", {s: {"intercept_ms": stage["stage_ms"][s]} for s in STAGES}),
                        "operating_state": stage.get("operating_state"),
                        "analytical_provisioning": stage.get("analytical_provisioning"),
                        "power_model": {"idle_intercept_w": stage["power_w"], "dynamic_slope_w": 0,
                                        "total_power_cap_w": sum(len(c["topology"][r + "_gpus"]) * c["knobs"][r + "_power_w"] for r in ("attention", "expert"))},
                        "layers": stage["layers"], "requests_per_pipeline": stage["requests_per_pipeline"]})
    report = build_mechanism({"selection_split": "calibration", "candidates": anchors}, bundle["model_workload"])
    return {**report, "measurement_mode": bundle["settings"]["mode"], "feedback_observations": len(extra)}


def updated_candidates(bundle, state, report_path=None):
    candidates = copy.deepcopy(bundle["candidates"])
    if bundle['settings'].get('mechanism_model') == 'external_power_duration_v1':
        from .external_model import inform
        candidates, report = inform(candidates, bundle.get('external_observations', []) + state['observations'],
                                    bundle['settings']['default_energy_j'])
        if report_path is not None:
            write_json(report_path, report)
        return candidates
    report = updated_mechanism(bundle, state)
    if report is None:
        return candidates
    if report_path is not None:
        write_json(report_path, report)
    # All GP residuals are recomputed against this updated prior on every ask.
    return inform_candidates(candidates, report, bundle["model_workload"], bundle["isolation"]["calibration"]["requests"])


def ask(directory):
    with locked_campaign(directory) as (bundle, state):
        if state["frozen"]:
            raise ValueError("campaign is frozen")
        if state["pending"] is not None:
            return state["pending"]
        spent = costs(bundle, state)
        settings = bundle["settings"]
        budget = settings["budget"]
        if spent["evaluations"] >= budget["evaluations"] or spent["gpu_hours"] >= budget["gpu_hours"]:
            return {"stopped": True, "reason": "budget_exhausted", "cost": spent}
        eligible = {r["id"] for r in bundle["audit"]["candidates"] if r["status"] == "eligible"}
        verified = {digest(structure(next(c for c in bundle["candidates"] if c["id"] == o["candidate_id"])))
                    for o in state["observations"] if o["status"] == "ok" and ("four_stage" in o or
                    (settings.get('mechanism_model') == 'external_power_duration_v1'
                     and o.get('execution_verified') is True and o.get('telemetry_valid') is True))}
        # Promotion is scoped to this immutable model/backend/workload campaign.
        # Runtime capability files are never edited from a success boolean.
        checks = {r["id"]: r for r in bundle["audit"]["candidates"]}
        eligible |= {c["id"] for c in bundle["candidates"] if digest(structure(c)) in verified
                     and checks[c["id"]]["status"] != "hard_rejected"
                     and not set(checks[c["id"]]["pending_reasons"]) - {"structure_execution_unverified"}}
        started = time.perf_counter()
        active_candidates = updated_candidates(bundle, state, Path(directory) / "current-mechanism.json")
        candidate, trial = (planned_candidate(bundle["calibration_plan"], active_candidates, state["observations"],
                                              bundle["audit"]["candidates"], settings.get("allow_structure_probes", False))
                            if state["observations"] else (None, None))
        if candidate is not None:
            current = next(c for c in active_candidates if c["id"] == state["observations"][-1]["candidate_id"])
            estimated = expected_cost(candidate, current, settings["bo"])
            decision = {"reason": "planned_calibration", "purpose": trial["purpose"],
                        "expected_gpu_hours": estimated, "base_gpu_hours": estimated,
                        "collect_four_stage_observation": True}
            if estimated > budget["gpu_hours"] - spent["gpu_hours"]:
                candidate, decision = None, {"reason": "planned_trial_exceeds_remaining_budget"}
        else:
            selectable = set(eligible)
            if settings.get('allow_structure_probes'):
                selectable |= {row['id'] for row in bundle['audit']['candidates']
                               if row['status'] != 'hard_rejected'
                               and not set(row['pending_reasons']) - {'structure_execution_unverified'}}
            candidate, decision = propose(active_candidates, state["observations"], settings,
                                          selectable, budget["gpu_hours"] - spent["gpu_hours"])
        optimizer_seconds = time.perf_counter() - started
        state["optimizer_wall_seconds"] += optimizer_seconds
        decision["optimizer_wall_seconds"] = optimizer_seconds
        if candidate is None:
            return {"stopped": True, **decision, "cost": spent}
        request = {"schema_version": 1, "trial_id": f"trial-{len(state['observations']):05d}",
                   "candidate_id": candidate["id"], "configuration": configuration(candidate),
                   "configuration_sha256": digest(configuration(candidate)),
                   "context_sha256": state["bundle_sha256"], "selection_split": "calibration",
                   "mode": settings["mode"], "trace": bundle["isolation"]["calibration"],
                   "workload": settings["workload"], "limits": settings["limits"],
                   "model_workload": normalized_workload(bundle["model_workload"], candidate),
                   "model_workload_sha256": digest(normalized_workload(bundle["model_workload"], candidate)),
                   "require_four_stage": settings.get("require_four_stage", False),
                   "require_output_correctness": settings.get("require_output_correctness", False),
                   "structure_validation_trial": candidate["id"] not in eligible,
                   "context_files": {k: v for k, v in bundle["sources"].items() if k.startswith("context:")},
                   "launch_environment": launch_environment(candidate, bundle["runtime"]),
                   "operating_points": candidate["knobs"], "decision": decision,
                   "mechanism": candidate.get("mechanism", {}),
                   "cost_includes": ["loading", "switching", "warmup", "replay", "failed_attempts"]}
        request["request_sha256"] = digest(request)
        state["pending"] = request
        state["history"].append({"event": "ask", "trial_id": request["trial_id"], "decision": decision})
        return request


def validate_result(result, pending, bundle):
    for key in ("trial_id", "candidate_id", "configuration_sha256", "context_sha256", "request_sha256", "mode", "selection_split"):
        if result.get(key) != pending[key]:
            raise ValueError(f"measurement receipt mismatch: {key}")
    if result.get("trace_sha256") != pending["trace"]["sha256"]:
        raise ValueError("measurement trace does not match calibration requests")
    if result.get("origin") != ("physical_measurement" if pending["mode"] == "physical" else "synthetic_test"):
        raise ValueError("model predictions cannot be submitted as measurements")
    validate_cost(result["cost"])
    if result["cost"]["wall_seconds"] <= 0:
        raise ValueError("every trial, including failures, must record elapsed cost")
    status = result.get("status")
    if status not in ("ok", "runtime_incompatible", "failed"):
        raise ValueError("unknown trial status")
    if pending["mode"] == "physical":
        if not result.get("artifacts"):
            raise ValueError("physical measurement requires hashed source artifacts")
        for artifact in result["artifacts"]:
            if file_hash(artifact["path"]) != artifact["sha256"]:
                raise ValueError("measurement artifact hash mismatch")
    if status == "ok":
        if result["cost"]["gpu_hours"] <= 0:
            raise ValueError("successful GPU measurements require GPU cost")
        expected = pending["trace"]["requests"]
        for key in ("requests", "completed_requests", "failed_requests"):
            integer(result.get(key), key, 0)
        if result.get("requests") != expected or result.get("completed_requests") != expected or result.get("failed_requests") != 0:
            raise ValueError("incomplete request cohort must be reported as a failed trial")
        if result.get("execution_verified") is not True or result.get("telemetry_valid") is not True:
            raise ValueError("execution/telemetry verification is missing")
        if pending.get('require_output_correctness'):
            gate = result.get('output_correctness', {})
            reference = pending['context_files']['context:correctness_reference']
            if (gate.get('verified') is not True or gate.get('protocol') != 'stock_vllm_exact_tokens_v1'
                    or gate.get('reference_sha256') != reference['sha256']
                    or gate.get('requests') != expected or gate.get('mismatched_requests') != 0
                    or gate.get('selection_split') != 'calibration' or not gate.get('output_tokens', 0) > 0):
                raise ValueError('Successful trial requires frozen-reference output correctness verification')
        for key in ("energy_j", "ttft_ms", "tpot_ms", "output_tps") + (("tbt_ms",) if "tbt_ms" in bundle["settings"]["limits"] else ()):
            positive(result["metrics"][key], key)
        if bundle['settings'].get('mechanism_model') == 'external_power_duration_v1':
            from .external_model import validate_observable
            validate_observable(result)
        if result["cost"]["tuning_energy_j"] < result["metrics"]["energy_j"]:
            raise ValueError("total tuning energy cannot omit the serving measurement energy")
        if pending.get("require_four_stage") and "four_stage" not in result:
            raise ValueError("this campaign requires measured four-stage feedback for every successful trial")
        if "four_stage" in result:
            stage = result["four_stage"]
            if not stage.get('model_feedback_supported', True):
                if (not pending['workload'].get('allow_model_feedback_fallback')
                        or not stage.get('model_fallback_reason') or 'stage_models' in stage
                        or stage.get('label_basis') != 'observed_unnormalized_stage_means'):
                    raise ValueError('Unnormalized stage feedback requires an explicit frozen fallback contract')
            if stage.get("workload_sha256") != pending["model_workload_sha256"]:
                raise ValueError("four-stage labels must be normalized to the frozen model workload")
            if set(stage["stage_ms"]) != set(STAGES):
                raise ValueError("incomplete four-stage observation")
            for value in stage["stage_ms"].values():
                positive(value, "stage duration")
            positive(stage["power_w"], "stage profile power")
            positive(stage["requests_per_pipeline"], "requests per pipeline")
            integer(stage["layers"], "layers")
            if "stage_models" in stage:
                from four_stage_dse_v6.model import predict_stage_times
                fitted = predict_stage_times(stage["stage_models"], pending["model_workload"], risk_quantile="nominal_no_margin")
                if any(not math.isclose(fitted[s], stage["stage_ms"][s], rel_tol=1e-6) for s in STAGES):
                    raise ValueError("stage models do not reproduce normalized feedback labels")
            if "operating_state" in stage:
                for role in ("attention", "expert"):
                    positive(stage["operating_state"][role]["effective_mhz"], "effective clock")
                    fraction = stage["operating_state"][role]["power_cap_active_fraction"]
                    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                        raise ValueError("invalid power-cap active fraction")
            if pending.get("require_four_stage") and pending["mode"] == "physical":
                if stage.get("provenance") != "raw_calibration_stage_traces_and_nvml" or "operating_state" not in stage:
                    raise ValueError("physical stage feedback must be built from raw traces and actual clocks")
    elif not result.get("failure_reason"):
        raise ValueError("failed trials require a reason")


def tell(directory, result):
    with locked_campaign(directory) as (bundle, state):
        if state["frozen"] or state["pending"] is None:
            raise ValueError("no pending trial or campaign frozen")
        pending = state["pending"]
        validate_result(result, pending, bundle)
        observation = copy.deepcopy(result)
        observation["proposal_reason"] = pending["decision"]["reason"]
        observation["structure_validation_trial"] = pending.get("structure_validation_trial", False)
        observation["proposal_expected_gpu_hours"] = pending["decision"]["expected_gpu_hours"]
        observation["proposal_base_gpu_hours"] = pending["decision"].get("base_gpu_hours", observation["proposal_expected_gpu_hours"])
        state["observations"].append(observation)
        started = time.perf_counter()
        report = updated_mechanism(bundle, state)
        if report is not None:
            write_json(Path(directory) / "current-mechanism.json", report)
        state["optimizer_wall_seconds"] += time.perf_counter() - started
        state["pending"] = None
        best = best_measured(state["observations"], bundle["settings"]["limits"])
        state["history"].append({"event": "tell", "trial_id": result["trial_id"],
                                 "cost": costs(bundle, state), "best": best})
        return {"cost": costs(bundle, state), "best": best}


def status(directory):
    with locked_campaign(directory) as (bundle, state):
        return {"mode": bundle["settings"]["mode"], "frozen": state["frozen"],
                "cost": costs(bundle, state), "budget": bundle["settings"]["budget"],
                "pending": state["pending"], "filter_counts": bundle["audit"]["counts"],
                "four_stage_feedback_count": sum(o["status"] == "ok" and "four_stage" in o for o in state["observations"]),
                "best": best_measured(state["observations"], bundle["settings"]["limits"]),
                "curve": [h for h in state["history"] if h["event"] == "tell"]}


def freeze(directory):
    with locked_campaign(directory) as (bundle, state):
        if state["pending"] is not None:
            raise ValueError("finish or report failure for the pending trial before freezing")
        if state["frozen"]:
            return state["deployment"]
        best = best_measured(state["observations"], bundle["settings"]["limits"])
        if best is None:
            raise ValueError("no measured feasible configuration to freeze")
        candidate = next(c for c in bundle["candidates"] if c["id"] == best["candidate_id"])
        deployment = {"schema_version": 1, "mode": bundle["settings"]["mode"],
                      "selection_split": "calibration", "best": best,
                      "configuration": configuration(candidate), "configuration_sha256": digest(configuration(candidate)),
                      "context_sha256": state["bundle_sha256"], "code_sha256": bundle["code_sha256"],
                      "isolation": bundle["isolation"], "cost": costs(bundle, state),
                      "launch_environment": launch_environment(candidate, bundle["runtime"]),
                      "heldout_evaluation_completed": False,
                      "optimality_claim": "best measured feasible configuration in this search"}
        # The authoritative freeze is committed in state with the same atomic write.
        state["frozen"], state["deployment"] = True, deployment
        return deployment


def summary_result(request, receipt, summary_path):
    """Bridge existing summarize_replay.py output; provenance comes from the executor receipt."""
    summary = read_json(summary_path)
    result = copy.deepcopy(receipt)
    result["requests"] = summary["requests"]
    result["completed_requests"] = summary["completed_requests"]
    result["failed_requests"] = summary["failed_requests"]
    if summary["requests"] != request["trace"]["requests"] or summary["failed_requests"]:
        result["status"] = "failed"
        result["failure_reason"] = "incomplete or failed replay cohort"
    elif result.get("status") == "ok":
        result["metrics"] = {"energy_j": summary["energy_j"], "ttft_ms": summary["ttft_ms"]["p90"],
                             "tpot_ms": summary["tpot_ms"]["p90"], "output_tps": summary["output_token_throughput_tps"]}
    result.setdefault("artifacts", []).append({"path": str(Path(summary_path).resolve()), "sha256": file_hash(summary_path)})
    return result

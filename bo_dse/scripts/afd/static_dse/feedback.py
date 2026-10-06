"""Convert causal stage traces and aligned NVML samples into measured feedback."""
from __future__ import annotations

import copy
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np

from build_deepseek_v6_four_stage_profile import load_four_stage_groups, fit_stage_model
from four_stage_dse_v6.model import STAGES, predict_stage_times, pipeline_time_ms
from four_stage_dse_v6.attention_workload import context_totals, predict_rank_model
from measure_command import integrate_window_energy
from .space import digest, positive, integer
from .analytical import calibrated_model


def make_manifest(request, cell_directory, layers):
    """Build the feedback index from the existing AFD measurement directory."""
    from .campaign import read_json, file_hash
    directory = Path(cell_directory).resolve()
    def artifact(path):
        return {"path": str(path), "sha256": file_hash(path)}
    reset_path = directory / "measurement-trace-reset.json"
    reset = read_json(reset_path)
    telemetry = read_json(directory / "telemetry.json")
    if reset.get("all_files_empty_after_reset") is not True or not 0 < reset["reset_wall_ns"] <= telemetry["started_wall_ns"]:
        raise ValueError("trace reset did not precede measurement")
    traces = []
    for path in sorted((directory / "traces").glob("stage-*.jsonl")):
        match = re.fullmatch(r"stage-(attention|ffn)-(\d+)-\d+\.jsonl", path.name)
        if not match:
            raise ValueError("unrecognized stage rank filename")
        traces.append({**artifact(path), "role": "expert" if match[1] == "ffn" else "attention", "rank": int(match[2])})
    result = {"schema_version": 1, "selection_split": "calibration",
            **{k: request[k] for k in ("request_sha256", "configuration_sha256", "context_sha256")},
            "trace_sha256": request["trace"]["sha256"], "warmup_excluded": True,
            "trace_reset": artifact(reset_path), "layers": integer(layers, "layers"),
            "summary": artifact(directory / "summary.json"), "telemetry": artifact(directory / "telemetry.json"),
            "power_samples": artifact(directory / "power-samples.jsonl"), "stage_traces": traces}
    replay = directory.parent / f"replay-ecodep-v026-{directory.name}.jsonl"
    if request.get("workload", {}).get("native_config_sha256") and replay.exists():
        result["completed_replay"] = artifact(replay)
        result["decode_step_convention"] = "native_eager_one_prefill_output_then_decode_queries"
    return result


def attach_feedback(request, receipt, manifest_path):
    """Manifest binds original files to a trial; no manually supplied stage labels.

    Raw traces are clipped to the replay window before joining. Token regression
    is evaluated only at identifiable, in-range workload features. The manifest
    also binds warmup exclusion and the role-to-file mapping, which the producer
    must establish by resetting observability after warmup.
    """
    from .campaign import file_hash, read_json, summary_result
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    for key in ("request_sha256", "configuration_sha256", "context_sha256"):
        if manifest.get(key) != request[key]:
            raise ValueError(f"feedback manifest mismatch: {key}")
    if manifest.get("selection_split") != "calibration" or manifest.get("trace_sha256") != request["trace"]["sha256"]:
        raise ValueError("feedback must originate from this calibration replay")
    if manifest.get("warmup_excluded") is not True:
        raise ValueError("stage feedback requires warmup exclusion evidence")
    artifacts = [{"path": str(manifest_path), "sha256": file_hash(manifest_path)}]
    def source(item):
        path = (manifest_path.parent / item["path"]).resolve()
        if file_hash(path) != item["sha256"]:
            raise ValueError("feedback source hash mismatch")
        artifacts.append({"path": str(path), "sha256": item["sha256"]})
        return path
    summary_path = source(manifest["summary"])
    telemetry = read_json(source(manifest["telemetry"]))
    samples_path = source(manifest["power_samples"])
    result = summary_result(request, receipt, summary_path)
    if result["status"] != "ok":
        result.setdefault("artifacts", []).extend(artifacts)
        return result
    if (telemetry.get("returncode") != 0 or telemetry.get("sample_error_count") != 0
            or telemetry.get("sample_time_coverage", 0) < .99
            or telemetry.get("request_count") != request["trace"]["requests"]):
        raise ValueError("invalid measurement telemetry")
    begin, end = telemetry["started_wall_ns"], telemetry["finished_wall_ns"]
    if end <= begin:
        raise ValueError("invalid replay interval")
    reset = read_json(source(manifest["trace_reset"]))
    if reset.get("all_files_empty_after_reset") is not True or not 0 < reset["reset_wall_ns"] <= begin:
        raise ValueError("stage reset evidence does not precede measurement")
    cfg = request["configuration"]
    gpu_ids = cfg["attention_gpus"] + cfg["expert_gpus"]
    if telemetry["gpu_ids"] != gpu_ids:
        raise ValueError("telemetry GPU order does not match allocation")
    paths = manifest["stage_traces"]
    expected_roles = {"attention": len(cfg["attention_gpus"]), "expert": len(cfg["expert_gpus"])}
    for role, count in expected_roles.items():
        ranks = [p["rank"] for p in paths if p["role"] == role]
        if sorted(ranks) != list(range(count)):
            raise ValueError("stage trace rank coverage mismatch")
    if any(p["role"] not in expected_roles for p in paths):
        raise ValueError("unknown stage trace role")
    with tempfile.TemporaryDirectory(prefix="dse-stage-") as temporary:
        clipped = []
        for item in paths:
            raw = source(item)
            rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
            if any("start_wall_ns" not in r or "end_wall_ns" not in r for r in rows):
                raise ValueError("stage trace lacks wall-clock boundaries")
            rows = [r for r in rows if begin <= r["start_wall_ns"] <= r["end_wall_ns"] <= end]
            if not rows:
                raise ValueError("empty measurement-stage rank")
            path = Path(temporary) / f"{item['role']}-{item['rank']}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            clipped.append(path)
        groups, coverage = load_four_stage_groups(clipped, minimum_complete_coverage=.95,
            token_scope=request["workload"].get("four_stage_token_scope", "rank_max"),
            attention_tp=cfg.get('attention_tp', 1))
    # Idle DP ranks need not participate in every transaction. Global file
    # coverage is required; per-transaction participation is reported separately.
    for group in groups:
        for stage in STAGES:
            count = expected_roles["expert" if stage == "ffn_compute" else "attention"]
            if not 1 <= group["ranks_by_stage"][stage] <= count:
                raise ValueError("invalid per-transaction rank coverage")
    observed_splits = sorted({r["stage_idx"] for r in groups})
    if observed_splits != list(range(cfg["microbatches"])):
        raise ValueError("observed microbatch split differs from proposed configuration")
    workload = request["model_workload"]
    target = np.array([1., workload.get("prefill_tokens_per_microbatch", 0.),
                       workload.get("decode_tokens_per_microbatch", 0.)])
    design = np.array([[1., r["prefill_tokens"], r["decode_tokens"]] for r in groups])
    fallback = None
    if np.linalg.norm(target - target @ np.linalg.pinv(design) @ design) > 1e-7:
        fallback = 'frozen workload is not identifiable from observed batch shapes'
    elif np.any(target < design.min(axis=0) - 1e-7) or np.any(target > design.max(axis=0) + 1e-7):
        fallback = 'stage normalization would extrapolate token batch shapes'
    if cfg['expert_dp'] > 1:
        fallback = 'independent_replica_service_capacity_model_unavailable'
    if fallback and not request['workload'].get('allow_model_feedback_fallback'):
        raise ValueError(fallback)
    models = {s: fit_stage_model(groups, s, 1., routing_enabled=False) for s in STAGES}
    if cfg.get("attention_tp", 1) != 1 and any(g['attention_tp_deduplicated'] != cfg['attention_tp'] for g in groups):
        models["attention_compute"]["attention_workload_model"] = {
            "enabled": False, "reason": "attention_tp_context_deduplication_not_supported"}
    times = ({s: float(np.mean([g['stage_times_ms'][s] for g in groups])) for s in STAGES} if fallback else
             predict_stage_times(models, workload, risk_quantile="nominal_no_margin"))
    rank_model = models["attention_compute"].get("attention_workload_model")
    attention_prediction = predict_rank_model(rank_model, workload)
    context_count = sum(context_totals(group) is not None for group in groups)
    attention_diagnostics = {
        "context_complete_groups": context_count, "groups": len(groups),
        "context_coverage": context_count / len(groups),
        "context_unit": "logical_causal_query_key_pairs_not_hbm_bytes",
        "prediction": attention_prediction,
        "rank_observations": [{"transaction_id": group["transaction_id"], "stage_idx": group["stage_idx"],
                               "summary": group["attention_rank_summary"],
                               "ranks": group["attention_rank_observations"]} for group in groups],
        "synchronization_wait_measured": False,
        "interpretation": "empirical_service_barrier_proxy_no_cross_device_completion_timestamps",
    }
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    if len(samples) < 2:
        raise ValueError("insufficient operating-state samples")
    timestamps = [r["timestamp_ns"] for r in samples]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("nonmonotone sample timestamps")
    if any(b - a > 2_000_000_000 for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("operating-state sampling gap exceeds two seconds")
    for row in samples:
        if row["gpu_ids"] != gpu_ids or len(row["power_w"]) != len(gpu_ids):
            raise ValueError("power sample allocation mismatch")
        for value in row["power_w"]:
            positive(value, "sample power")
        if len(row.get("operating_state", [])) != len(gpu_ids):
            raise ValueError("actual clocks/caps were not collected; enable --record-operating-state")
    energy, time_coverage = integrate_window_energy(
        [(0., r["timestamp_ns"], r["power_w"]) for r in samples], start_ns=begin, end_ns=end, gpu_count=len(gpu_ids))
    if time_coverage < .99 or not math.isclose(sum(energy), result["metrics"]["energy_j"], rel_tol=.01):
        raise ValueError("reintegrated energy does not match summary")
    operating = {}
    for role, ids in (("attention", cfg["attention_gpus"]), ("expert", cfg["expert_gpus"])):
        clocks, weights, throttled, active_weight = [], [], 0., 0.
        for left, right in zip(samples, samples[1:]):
            dt = max(0, min(end, right["timestamp_ns"]) - max(begin, left["timestamp_ns"])) / 1e9
            for gpu in ids:
                state = left["operating_state"][gpu_ids.index(gpu)]
                clock = positive(state["graphics_mhz"], "actual clock")
                cap = positive(state["power_limit_w"], "enforced cap")
                if abs(cap - cfg[role + "_power_w"]) > 1:
                    raise ValueError("enforced power limit differs from proposal")
                utilization = state["gpu_utilization"]
                if not math.isfinite(utilization) or not 0 <= utilization <= 100:
                    raise ValueError("invalid GPU utilization")
                integer(state["throttle_reasons"], "throttle bitmask", 0)
                if dt and utilization > 0:
                    clocks.append(clock)
                    weights.append(dt * utilization / 100)
                    active_weight += dt
                    # NVML SW power cap bit (0x4), not idle/thermal throttling.
                    throttled += dt * bool(state["throttle_reasons"] & 0x4)
        if not weights:
            raise ValueError("no active clock samples for role")
        operating[role] = {"effective_mhz": float(np.average(clocks, weights=weights)),
                           "requested_mhz": cfg[role + "_mhz"],
                           "power_cap_active_fraction": throttled / active_weight,
                           "weighting": "time_times_gpu_utilization"}
    layers = integer(manifest["layers"], "layers")
    duration = (end - begin) / 1e9
    role_power = {role: sum(energy[gpu_ids.index(gpu)] for gpu in cfg[role + "_gpus"]) / duration
                  for role in ("attention", "expert")}
    evidence = {"manifest_sha256": file_hash(manifest_path), "trace_sha256": request["trace"]["sha256"],
                "completed_requests": request["trace"]["requests"]}
    if manifest.get("completed_replay") and manifest.get("decode_step_convention") == "native_eager_one_prefill_output_then_decode_queries":
        replay = [json.loads(line) for line in source(manifest["completed_replay"]).read_text().splitlines() if line.strip()]
        if len(replay) != request["trace"]["requests"] or any(row.get("error") for row in replay):
            raise ValueError("analytical replay cohort mismatch")
        outputs = [integer(row["actual_output_tokens"], "actual output tokens") for row in replay]
        evidence.update(expected_decode_queries=sum(n - 1 for n in outputs),
                        expected_decode_requests=sum(n > 1 for n in outputs),
                        decode_step_convention=manifest["decode_step_convention"],
                        completed_replay_sha256=manifest["completed_replay"]["sha256"])
    analytical = calibrated_model(groups, cfg, layers, workload, role_power, evidence)
    analytical["reference_stage_ms"] = times
    pipeline = pipeline_time_ms(times, microbatches=cfg["microbatches"], layers=layers,
                                schedule_model=analytical["schedule_model"])
    result["four_stage"] = {"workload_sha256": request["model_workload_sha256"],
                            "stage_ms": times, "stage_models": models, "power_w": sum(energy) / duration,
                            "layers": layers, "requests_per_pipeline": request["trace"]["requests"] / duration * pipeline / 1000,
                            "coverage": coverage, "operating_state": operating,
                            "observed_microbatches": len(observed_splits),
                            "normalization_design_rank": int(np.linalg.matrix_rank(design)),
                            "attention_diagnostics": attention_diagnostics,
                            "analytical_provisioning": analytical,
                            "provenance": "raw_calibration_stage_traces_and_nvml"}
    if fallback:
        stage = result['four_stage']
        stage.pop('stage_models')
        stage.update(model_feedback_supported=False, model_fallback_reason=fallback,
                     label_basis='observed_unnormalized_stage_means',
                     analytical_provisioning={'available': False, 'reason': fallback,
                                             'fallback': 'end_to_end_measurements_only'})
    else:
        result['four_stage'].update(model_feedback_supported=True, label_basis='normalized_stage_model')
    result.setdefault("artifacts", []).extend(artifacts)
    return result

#!/usr/bin/env python3
"""Freeze v10 guard curves from valid calibration MAX cells only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def load(path):
    return Path(path).read_text()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source, suite):
    config = json.loads(load(source))
    suite = Path(suite).resolve()
    if not (suite / "COMPLETE").exists():
        raise ValueError("calibration MAX suite is incomplete")
    manifest = json.loads(load(suite / "manifest.json"))
    if manifest["evaluation_split"] != "calibration":
        raise ValueError("v10 calibration cannot consume held-out results")
    if manifest["model"] != config["compatibility"]["model"]:
        raise ValueError("calibration model differs")
    profile = json.loads(load(config["predictor"]["calibration_profile"]["path"]))
    if manifest["trace_sha256"] != profile["trace"]["sha256"]:
        raise ValueError("calibration identities differ from the source profile")
    serving = suite.parents[2] / "afd_serving" / suite.parent.name / suite.name
    curve = []
    hashes = {str(suite / "manifest.json"): digest(suite / "manifest.json"),
              str(Path(source).resolve()): digest(source)}
    for rate in [1, 2, 4]:
        case = serving / f"rps-{rate}"
        summary = json.loads(load(case / "summary.json"))
        telemetry = json.loads(load(case / "telemetry.json"))
        ownership = json.loads(load(case / "gpu-contamination-validation.json"))
        if summary["completed_requests"] != 200 or summary["failed_requests"]:
            raise ValueError("calibration MAX requires 200 successful requests per rate")
        if telemetry["returncode"] or telemetry["sample_error_count"] or telemetry["sample_time_coverage"] < .99:
            raise ValueError("calibration telemetry failed")
        if not ownership["verified"] or ownership["foreign_process_samples"]:
            raise ValueError("calibration GPU ownership failed")
        # TTFT uses the registered +5% budget. Progress notifications can lag
        # token emission by one producer interval and one controller poll.
        curve.append({"rps": rate,
                      "prefill_age_guard_ms": 1.05 * summary["ttft_ms"]["p90"],
                      "progress_gap_guard_ms": 1.05 * summary["tpot_ms"]["p99"]
                          + config["events"]["progress_event_interval_ms"]
                          + config["control_interval_ms"],
                      "source": str(case / "summary.json")})
        for name in ["summary.json", "telemetry.json", "gpu-contamination-validation.json"]:
            hashes[str(case / name)] = digest(case / name)
    config = copy.deepcopy(config)
    config["controller_revision"] = 10
    config["method"] = "fbss_constrained_causal_stage_routing_joint_frequency_power_v10"
    config["signals"]["rate_window_ms"] = 1000
    config["signals"]["rate_estimator"] = "count_over_fixed_time_window"
    config["predictor"]["relative_service_budget_ratio"] = 1.05
    config["predictor"]["role_gpu_counts"] = {"attention": 2, "expert": 2}
    config["predictor"]["calibration_guard_by_rps"] = curve
    config["predictor"]["guard_calibration"] = {
        "selection_split": "calibration", "files_sha256": hashes,
        "development_requests": 600,
        "rule": "TTFT p90 times 1.05; TPOT p99 times 1.05 plus progress interval and poll; interpolate by causal arrival rate",
    }
    config["predictor"]["type"] = "causal_window_rate_relative_max_service_power_minimizer"
    # These legacy EWMA settings no longer govern either arrival or progress
    # rate. Remove them to avoid a misleading frozen configuration.
    for key in ["arrival_ewma_alpha", "arrival_ewma_decay_ms", "max_instantaneous_arrival_rps"]:
        config["signals"].pop(key, None)
    config["predictor"].pop("decode_progress_ewma_alpha", None)
    config["predictor"].pop("urgent_thresholds", None)
    return config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("calibration_max_suite", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("controller artifacts are immutable")
    config = build(args.source, args.calibration_max_suite)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config["predictor"]["calibration_guard_by_rps"], indent=2))


if __name__ == "__main__":
    main()

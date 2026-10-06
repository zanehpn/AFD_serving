#!/usr/bin/env python3
"""Freeze matched-MAX and one-shot FBSS held-out deployments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


RATES = (1, 2, 4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n")


def runtime_contract(*, compute_gate_on_attention: bool) -> dict[str, object]:
    return {
        "base": "official_vllm_afd_v0.26_role_pruned_data_plane",
        "cuda_graph_full_decode_only": False,
        "ffn_cudagraph": False,
        "enable_dbo": True,
        "compute_gate_on_attention": compute_gate_on_attention,
        "prefix_caching": True,
        "routing_sidecar": False,
        "routing_ffn_sidecar": False,
        "routing_source_role": "attention",
        "routing_async_copy": True,
        "stage_trace": False,
        "expert_boundary_action": False,
        "request_path_clock_transitions": 0,
        "operating_point_semantics": "one_shot_frozen_per_rate_cell",
    }


def deployment(
    *,
    arm: str,
    candidate_id: str,
    points: dict[str, dict[str, object]],
    calibration_trace: Path,
    plugin_root: Path,
    plugin_commit: str,
    comparison_contract_sha256: str,
    method: str = "one_shot_fbss_v7_matched_heldout",
    model: str = "DeepSeek-V2-Lite-Chat",
    compute_gate_on_attention: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "method": method,
        "arm": arm,
        "candidate_id": candidate_id,
        "model": model,
        "evaluation_split": "heldout",
        "selection_split": "calibration",
        "calibration_trace": str(calibration_trace.resolve()),
        "calibration_trace_sha256": sha256(calibration_trace),
        "plugin": {
            "root": str(plugin_root.resolve()),
            "commit": plugin_commit,
            "worktree_clean": True,
        },
        "topology": {
            "attention_dp": 2,
            "ffn_ep": 2,
            "attention_tp": 1,
            "expert_tp": 1,
        },
        "placement": {
            "enabled": False,
            "mode": "native-linear",
            "selection_split": "architecture-default",
        },
        "operating_points": {
            "selection_split": "calibration",
            "candidate_id": candidate_id,
            "calibration_trace_sha256": sha256(calibration_trace),
            "comparison_contract_sha256": comparison_contract_sha256,
            "by_rps": points,
        },
        "runtime_contract": runtime_contract(
            compute_gate_on_attention=compute_gate_on_attention,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_dir", type=Path)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument(
        "--plugin-root",
        type=Path,
        default=Path("third_party/afd-plugin-ecodep-v026-dvfs-v6-four-stage"),
    )
    parser.add_argument(
        "--sif",
        type=Path,
        default=Path("environment/native-runtime.json"),
    )
    parser.add_argument("--combined-controller", type=Path)
    parser.add_argument("--reference-controller", type=Path, required=True)
    parser.add_argument("--model", default="DeepSeek-V2-Lite-Chat")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=3072)
    parser.add_argument("--compute-gate-on-attention", action="store_true")
    args = parser.parse_args()
    protocol = args.protocol_dir.resolve()
    trace_root = args.trace_root.resolve()
    plugin_root = args.plugin_root.resolve()
    sif = args.sif.resolve()
    calibration_trace = trace_root / "calibration-200.jsonl"
    heldout_trace = trace_root / "heldout-400.jsonl"
    warmup_trace = trace_root / "warmup-8.jsonl"
    warmup_manifest = trace_root / "warmup-manifest.json"
    decision_path = protocol / "fbss-decision-v6.json"
    profile_path = protocol / "expanded-profile-v6.json"
    calibration_freeze_path = protocol / "FREEZE.json"
    combined_controller_path = (
        args.combined_controller.resolve()
        if args.combined_controller is not None
        else None
    )
    for path in (
        calibration_trace,
        heldout_trace,
        warmup_trace,
        warmup_manifest,
        decision_path,
        profile_path,
        calibration_freeze_path,
        sif,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if combined_controller_path is not None and not combined_controller_path.is_file():
        raise FileNotFoundError(combined_controller_path)

    reference_controller = json.loads(args.reference_controller.read_text())
    if reference_controller.get("selection_split") != "calibration" or reference_controller.get("method") != "fbss_constrained_causal_stage_routing_joint_frequency_power_v9":
        raise ValueError("reference must be the frozen v9 calibration controller")
    calibration_freeze = json.loads(calibration_freeze_path.read_text())
    decision = json.loads(decision_path.read_text())
    profile = json.loads(profile_path.read_text())
    if calibration_freeze.get("status") != "frozen_calibration_policy":
        raise ValueError("calibration policy is not frozen")
    if calibration_freeze.get("heldout_requests_used_for_tuning") != 0:
        raise ValueError("held-out outcomes were used for tuning")
    if calibration_freeze.get("slo_contract") != (
        "p90_ttft_and_p90_tpot_each_at_most_1.05x_and_"
        "output_tps_at_least_0.95x_matched_max"
    ):
        raise ValueError("unexpected SLO contract")
    if decision.get("absolute_capacity_gate") is not None:
        raise ValueError("primary relative-SLO run must not add a capacity gate")
    if decision.get("evaluated_cheap_predictions") != 2500 * len(RATES):
        raise ValueError("FBSS did not score 2,500 candidates at every rate")
    if decision.get("additional_physical_validation_runs") != 0:
        raise ValueError("one-shot policy performed candidate validation")
    if sha256(heldout_trace) != calibration_freeze.get("heldout_trace_sha256"):
        raise ValueError("held-out trace differs from calibration freeze")
    if sha256(decision_path) != calibration_freeze.get("fbss_decision_sha256"):
        raise ValueError("FBSS decision differs from calibration freeze")

    combined_controller = None
    if combined_controller_path is not None:
        combined_controller = json.loads(combined_controller_path.read_text())
        if combined_controller.get("method") not in {
            "fbss_constrained_causal_stage_routing_joint_frequency_power_v10",
        }:
            raise ValueError("combined controller method mismatch")
        if combined_controller.get("controller_revision") != 10:
            raise ValueError("v10 revision marker missing")
        binding = combined_controller.get("predictor", {}).get("fbss_binding", {})
        if binding.get("decision_sha256") != sha256(decision_path):
            raise ValueError("combined controller is not bound to this FBSS decision")
        if combined_controller.get("selection_split") != "calibration":
            raise ValueError("combined controller is not calibration-only")

    plugin_status = subprocess.check_output(
        ["git", "-C", str(plugin_root), "status", "--short"], text=True
    ).strip()
    if plugin_status:
        raise ValueError("plugin worktree is not clean")
    plugin_commit = subprocess.check_output(
        ["git", "-C", str(plugin_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if plugin_commit != calibration_freeze.get("plugin_commit"):
        raise ValueError("plugin revision differs from calibration freeze")

    by_id = {str(row["id"]): row for row in profile["candidates"]}
    selected = {
        int(row["rps"]): str(row["candidate_id"])
        for row in decision["deployment_candidates"]
    }
    if set(selected) != set(RATES):
        raise ValueError("FBSS must freeze one point for each declared rate")
    for candidate_id in selected.values():
        topology = by_id[candidate_id]["topology"]
        if (
            topology.get("attention_dp") != 2
            or topology.get("expert_ep") != 2
            or topology.get("attention_tp") != 1
            or topology.get("expert_tp") != 1
        ):
            raise ValueError(
                "current held-out sweep supports the selected 2A2E topology only"
            )

    image_id = f"native:{sha256(sif)}"
    model_config = Path("artifacts/models") / args.model / "config.json"
    model_config = model_config.resolve()
    if not model_config.is_file():
        raise FileNotFoundError(model_config)
    warmup_info = json.loads(warmup_manifest.read_text())
    comparison_contract = {
        "schema_version": 1,
        "data_plane": "official_vllm_afd_v0.26_role_pruned_attention_ffn",
        "container": {
            "image": "native-vllm-0.26.0-cu130",
            "image_id": image_id,
        },
        "model": {
            "name": args.model,
            "config_sha256": sha256(model_config),
        },
        "topology": {
            "attention_dp": 2,
            "ffn_ep": 2,
            "attention_tp": 1,
            "expert_tp": 1,
        },
        "server": {
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "prefix_caching": True,
            "cuda_graph_full_decode_only": False,
            "ffn_cudagraph": False,
            "enforce_eager": True,
            "cudagraph_capture_size": 32,
            "dbo": True,
            "dbo_decode_token_threshold": 2,
            "dbo_prefill_token_threshold": 12,
            "compute_gate_on_attention": args.compute_gate_on_attention,
        },
        "generation": {
            "api": "v1/completions",
            "temperature": 0.0,
            "ignore_eos": True,
            "max_output_tokens": 128,
        },
        "measurement": {
            "gpu_ids": [int(x) for x in (os.environ.get("ECODEP_ATTENTION_GPUS", "0,1") + "," + os.environ.get("ECODEP_EXPERT_GPUS", "2,3")).split(",")],
            "sample_interval_ms": 10,
            "energy_window": "request_first_submit_to_last_finish",
            "ttft_slo_ms": 400.0,
            "tpot_slo_ms": 120.0,
        },
        "arrival_rate_scaling": (
            "(request_count-1)/(last_arrival-first_arrival)/target_rps"
        ),
        "warmup": {
            "trace_sha256": sha256(warmup_trace),
            "manifest_sha256": sha256(warmup_manifest),
            "max_output_tokens": 128,
            "measured": False,
            "prefix_cache_reset_after": True,
        },
    }
    if warmup_info.get("output_sha256") != sha256(warmup_trace):
        raise ValueError("warmup manifest hash mismatch")
    contract_sha = canonical_sha256(comparison_contract)

    max_points = {
        str(rate): {
            "candidate_id": "c00-max",
            "attention_mhz": [1410, 1410],
            "expert_mhz": [1410, 1410],
            "attention_power_w": [400, 400],
            "expert_power_w": [400, 400],
            "request_path_clock_transitions": 0,
            "policy": "matched_max",
        }
        for rate in RATES
    }
    fbss_points: dict[str, dict[str, object]] = {}
    for rate in RATES:
        candidate_id = selected[rate]
        candidate = by_id[candidate_id]
        knobs = candidate["knobs"]
        fbss_points[str(rate)] = {
            "candidate_id": candidate_id,
            "attention_mhz": [int(knobs["attention_mhz"])] * 2,
            "expert_mhz": [int(knobs["expert_mhz"])] * 2,
            "attention_power_w": [int(knobs["attention_power_w"])] * 2,
            "expert_power_w": [int(knobs["expert_power_w"])] * 2,
            "request_path_clock_transitions": 0,
            "policy": "one_shot_fbss_no_candidate_validation",
            "calibration_feasible": True,
            "selection_split": "calibration",
        }

    baseline_path = protocol / "heldout-max-deployment.json"
    fbss_path = protocol / "heldout-fbss-deployment.json"
    schedule_path = protocol / "heldout-schedule.json"
    evaluation_freeze_path = protocol / "EVALUATION_FREEZE.json"
    write_new(
        baseline_path,
        deployment(
            arm="B2",
            candidate_id="c00-max",
            points=max_points,
            calibration_trace=calibration_trace,
            plugin_root=plugin_root,
            plugin_commit=plugin_commit,
            comparison_contract_sha256=contract_sha,
            model=args.model,
            compute_gate_on_attention=args.compute_gate_on_attention,
        ),
    )
    write_new(
        fbss_path,
        deployment(
            arm="ours",
            candidate_id=(
                "calibrated-dynamic-fbss-v10"
                if combined_controller is not None
                else "one-shot-fbss-v7"
            ),
            points=fbss_points,
            calibration_trace=calibration_trace,
            plugin_root=plugin_root,
            plugin_commit=plugin_commit,
            comparison_contract_sha256=contract_sha,
            method=(
                "calibrated_causal_routing_dvfs_v10_base"
                if combined_controller is not None
                else "one_shot_fbss_v7_matched_heldout"
            ),
            model=args.model,
            compute_gate_on_attention=args.compute_gate_on_attention,
        ),
    )
    write_new(
        schedule_path,
        {
            "schema_version": 1,
            "precommitted_before_heldout": True,
            "entries": [
                {"position": 1, "arm": "B2", "repetition": 1, "rates_rps": list(RATES)},
                {"position": 2, "arm": "causal-dvfs-v5-dynamic-ae", "controller_revision": 9,
                 "repetition": 1, "rates_rps": list(RATES)},
                {"position": 3, "arm": "causal-dvfs-v5-dynamic-ae", "controller_revision": 10,
                 "repetition": 1, "rates_rps": list(RATES)},
            ],
        },
    )
    write_new(
        evaluation_freeze_path,
        {
            "schema_version": 1,
            "status": "frozen_before_heldout",
            "frozen_at": datetime.now().astimezone().isoformat(),
            "selection_split": "calibration",
            "heldout_requests_used_for_tuning": 0,
            "heldout_outcomes_opened_before_freeze": 0,
            "slo_contract": calibration_freeze["slo_contract"],
            "latency_budget_ratio": 1.05,
            "output_token_throughput_min_ratio": 0.95,
            "absolute_capacity_gate": None,
            "rates_rps": list(RATES),
            "calibration_freeze_sha256": sha256(calibration_freeze_path),
            "decision_sha256": sha256(decision_path),
            "expanded_profile_sha256": sha256(profile_path),
            "calibration_trace_sha256": sha256(calibration_trace),
            "heldout_trace_sha256": sha256(heldout_trace),
            "warmup_trace_sha256": sha256(warmup_trace),
            "comparison_contract_sha256": contract_sha,
            "baseline_deployment_sha256": sha256(baseline_path),
            "fbss_deployment_sha256": sha256(fbss_path),
            "reference_controller_sha256": sha256(args.reference_controller),
            "shared_code_sha256": {
                str(path.resolve()): sha256(path)
                for path in [
                    *Path("migration").glob("*.py"),
                    *Path("environment").glob("native-platform.json"),
                    *Path("migration").glob("*.sh"),
                    Path("services/nvcontrold.py"),
                    Path("environment/PREPARED.json"),
                    Path("environment/native-runtime.json"),
                    Path("scripts/afd/start_server_native.sh"),
                    Path("scripts/afd/launch_pair.py"),
                    Path("scripts/afd/causal_dvfs/controller.py"),
                    Path("scripts/afd/run_v026_ours_rate_suite.sh"),
                    Path("scripts/afd/run_v026_causal_dvfs_rate_suite.sh"),
                    Path("scripts/afd/run_v026_causal_dvfs_v10_rate_suite.sh"),
                    Path("scripts/afd/causal_dvfs_v5/controller.py"),
                    Path("scripts/afd/causal_dvfs_v10/controller.py"),
                    Path("scripts/afd/causal_dvfs/replay_client.py"),
                ]
            },
            "reference_code_sha256": {
                "controller": sha256(Path("scripts/afd/causal_dvfs_v5/controller.py")),
                "runner": sha256(Path("scripts/afd/run_v026_causal_dvfs_rate_suite.sh")),
                "replay_client": sha256(Path("scripts/afd/causal_dvfs/replay_client.py")),
            },
            "combined_controller_sha256": (
                sha256(combined_controller_path)
                if combined_controller_path is not None
                else None
            ),
            "combined_code_sha256": (
                {
                    "controller": sha256(
                        Path("scripts/afd/causal_dvfs_v10/controller.py").resolve()
                    ),
                    "runner": sha256(
                        Path("scripts/afd/run_v026_causal_dvfs_v10_rate_suite.sh").resolve()
                    ),
                    "replay_client": sha256(
                        Path("scripts/afd/causal_dvfs/replay_client.py").resolve()
                    ),
                }
                if combined_controller is not None
                else None
            ),
            "schedule_sha256": sha256(schedule_path),
            "plugin_commit": plugin_commit,
            "additional_static_candidate_validation_runs": 0,
            "additional_dynamic_development_cells": 6,
            "additional_dynamic_development_requests": 1200,
        },
    )
    print(evaluation_freeze_path.read_text(), end="")


if __name__ == "__main__":
    main()

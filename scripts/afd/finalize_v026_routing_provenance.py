#!/usr/bin/env python3
"""Finalize a matched v0.26 routing-calibration run into placement provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from v026_routing_provenance import METHOD, sha256


def identity_permutation_by_layer(
    *, num_layers: int = 40, num_experts: int = 256
) -> str:
    """Return the canonical frozen B2 layerwise permutation contract."""
    row = ",".join(str(expert) for expert in range(num_experts))
    return ";".join(row for _ in range(num_layers))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True
    ).strip()


def plugin_source_sha256(plugin_root: Path) -> str:
    """Match the launcher's deterministic hash over imported Python sources."""
    lines = []
    for path in sorted((plugin_root / "afd_plugin").rglob("*.py")):
        lines.append(f"{sha256(path)}  {path}\n")
    if not lines:
        raise ValueError("routing plugin has no Python sources")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def inspect_sidecar(path: Path) -> dict[str, Any]:
    role: str | None = None
    rank: int | None = None
    layers: set[int] = set()
    experts: int | None = None
    top_k: int | None = None
    records = 0
    previous_timestamp = -1
    first_timestamp: int | None = None
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("schema_version", 0)) < 2 or row.get("event_type") != "routing":
            raise ValueError(f"invalid routing record {path}:{line_number}")
        if row.get("source_role") != "ffn" or row.get("availability") != "observed":
            raise ValueError(f"non-FFN observed record {path}:{line_number}")
        current_role = str(row["source_role"])
        current_rank = int(row["source_rank"])
        role = current_role if role is None else role
        rank = current_rank if rank is None else rank
        if role != current_role or rank != current_rank:
            raise ValueError(f"routing identity changes within {path}")
        current_experts = int(row["num_experts"])
        current_top_k = int(row["top_k"])
        experts = current_experts if experts is None else experts
        top_k = current_top_k if top_k is None else top_k
        if experts != current_experts or top_k != current_top_k:
            raise ValueError(f"routing shape changes within {path}")
        if int(row.get("expert_ep", 0)) != 2 or int(row.get("expert_tp", 0)) != 1:
            raise ValueError(f"routing topology changes within {path}")
        if row.get("expert_placement_mode") != "layerwise":
            raise ValueError("routing calibration must use frozen layerwise B2 placement")
        if row.get("expert_permutation") != list(range(current_experts)):
            raise ValueError("routing calibration is not B2 linear placement")
        timestamp = int(row["timestamp_ns"])
        if timestamp < previous_timestamp:
            raise ValueError(f"routing timestamps regress within {path}")
        previous_timestamp = timestamp
        first_timestamp = timestamp if first_timestamp is None else first_timestamp
        layers.add(int(row["layer_idx"]))
        records += 1
    if records == 0:
        raise ValueError(f"empty routing sidecar {path}")
    if layers != set(range(40)) or experts != 256 or top_k != 8:
        raise ValueError(f"routing sidecar has the wrong model shape: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "source_role": role,
        "source_rank": rank,
        "records": records,
        "layers": len(layers),
        "experts": experts,
        "top_k": top_k,
        "first_timestamp_ns": first_timestamp,
        "last_timestamp_ns": previous_timestamp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--operating-point-ack", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--collection-rate-rps", type=float, required=True)
    parser.add_argument(
        "--smoke-limit",
        type=int,
        help=(
            "Validate only the first N trace requests as a sidecar smoke. "
            "Smoke output is explicitly not formal placement provenance."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    trace_path = args.calibration_trace.resolve()
    launch = load(run_dir / "launch_config.json")
    summary = load(args.replay_summary.resolve())
    ack = load(args.operating_point_ack.resolve())
    plugin_root = args.plugin_root.resolve()

    expected_launch = {
        "model": "Qwen3.6-35B-A3B",
        "attention_ranks": 2,
        "expert_ranks": 2,
        "attention_tp": 1,
        "expert_tp": 1,
        "max_model_len": 8192,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 3072,
        "prefix_caching_enabled": 1,
        "cuda_graph_full_decode_only": False,
        "cudagraph_capture_size": 32,
        "dbo_enabled": True,
        "compute_gate_on_attention": False,
        "dbo_decode_token_threshold": 2,
        "dbo_prefill_token_threshold": 12,
        "routing_sidecar_enabled": 1,
        "routing_ffn_sidecar_enabled": 1,
        "routing_source_role": "ffn",
        "routing_async_copy_enabled": 1,
        "routing_active_marker": "/results/.routing-active",
        "ffn_cudagraph_enabled": 0,
        "stage_trace_enabled": 0,
        "expert_epoch_trace_enabled": 0,
        "expert_boundary_action_enabled": 0,
        "expert_permutation": "",
        "expert_permutation_by_layer": identity_permutation_by_layer(),
        "expert_placement_mode": "layerwise",
        "image": "ecodep-vllm-afd:v0.26.0",
    }
    for key, expected in expected_launch.items():
        if launch.get(key) != expected:
            raise ValueError(f"routing launch contract mismatch: {key}")
    if str(launch.get("plugin_host_root", "")) != str(plugin_root):
        raise ValueError("routing launch used a different plugin root")
    if git(plugin_root, "status", "--short"):
        raise ValueError("routing plugin worktree is dirty")
    plugin_source_hash = launch.get("plugin_source_sha256")
    plugin_source_hash_origin = "launch_config"
    if not plugin_source_hash:
        # Older Apptainer launch manifests omitted this field. Reconstruction
        # is valid only because the finalizer has already required the exact
        # read-only plugin root and a clean frozen worktree.
        plugin_source_hash = plugin_source_sha256(plugin_root)
        plugin_source_hash_origin = "reconstructed_from_clean_frozen_worktree"
    if not args.image_id.startswith("sha256:"):
        raise ValueError("immutable v0.26 image ID is required")
    if args.collection_rate_rps <= 0:
        raise ValueError("routing collection rate must be positive")
    if args.smoke_limit is not None and args.smoke_limit <= 0:
        raise ValueError("routing smoke limit must be positive")

    trace_requests = sum(1 for line in trace_path.read_text().splitlines() if line.strip())
    if args.smoke_limit is None and trace_requests < 200:
        raise ValueError("routing calibration requires at least 200 requests")
    expected_requests = (
        trace_requests
        if args.smoke_limit is None
        else min(args.smoke_limit, trace_requests)
    )
    if int(summary.get("schema_version", 0)) < 5:
        raise ValueError("routing replay summary schema is too old")
    if int(summary.get("requests", -1)) != expected_requests:
        raise ValueError("routing replay request count differs from its contract")
    if int(summary.get("completed_requests", -1)) != expected_requests or int(
        summary.get("failed_requests", -1)
    ) != 0:
        raise ValueError("routing replay did not complete cleanly")
    measurement_start = int(summary.get("measurement_start_wall_ns", 0))
    measurement_end = int(summary.get("measurement_end_wall_ns", 0))
    if measurement_start <= 0 or measurement_end < measurement_start:
        raise ValueError("routing replay lacks a valid measurement window")
    if ack.get("verified") is not True or ack.get("verification_errors"):
        raise ValueError("routing operating point was not verified")
    expected_ack = {
        "requested_attention_mhz": [1410, 1410],
        "requested_expert_mhz": [1410, 1410],
        "requested_attention_power_w": [400, 400],
        "requested_expert_power_w": [400, 400],
    }
    for key, expected in expected_ack.items():
        if ack.get(key) != expected:
            raise ValueError(f"routing operating-point mismatch: {key}")

    sidecars = sorted(run_dir.glob("routing-ffn-*.jsonl"))
    inspected = [inspect_sidecar(path) for path in sidecars]
    if {row["source_rank"] for row in inspected} != {0, 1} or len(inspected) != 2:
        raise ValueError("routing run lacks exactly two FFN-rank sidecars")
    if list(run_dir.glob("routing-attention-*.jsonl")):
        raise ValueError("routing run emitted a forbidden Attention sidecar")
    if any(
        int(row["first_timestamp_ns"]) < measurement_start
        or int(row["last_timestamp_ns"]) > measurement_end
        for row in inspected
    ):
        raise ValueError("routing sidecar contains startup or post-replay records")
    if list(run_dir.glob("stage-*.jsonl")) or list(run_dir.glob("expert-epoch-*.jsonl")):
        raise ValueError("routing run emitted forbidden instrumentation")

    trace_hash = sha256(trace_path)
    smoke_only = args.smoke_limit is not None
    payload = {
        "schema_version": 1,
        "method": (
            "ecodep_v026_routing_sidecar_smoke" if smoke_only else METHOD
        ),
        "selection_split": "calibration",
        "measurement_eligible": False,
        "formal_provenance": not smoke_only,
        "smoke_only": smoke_only,
        "runtime": {
            "vllm_afd_version": "0.26.0",
            "image": launch["image"],
            "image_id": args.image_id,
            "plugin_root": str(plugin_root),
            "plugin_commit": git(plugin_root, "rev-parse", "HEAD"),
            "plugin_source_sha256": plugin_source_hash,
            "plugin_source_sha256_origin": plugin_source_hash_origin,
        },
        "topology": {
            "attention_dp": 2,
            "ffn_ep": 2,
            "attention_tp": 1,
            "expert_tp": 1,
        },
        "collection_contract": {
            "routing_sidecar": True,
            "routing_source_role": "ffn",
            "routing_ffn_sidecar": True,
            "routing_async_copy": True,
            "routing_window_gate": "replay_client_marker",
            "ffn_cudagraph": False,
            "compute_gate_on_attention": False,
            "stage_trace": False,
            "expert_epoch_trace": False,
            "request_path_clock_transitions": 0,
            "attention_mhz": [1410, 1410],
            "expert_mhz": [1410, 1410],
            "attention_power_w": [400, 400],
            "expert_power_w": [400, 400],
            "placement_variant": "B2-linear-equal-capacity",
            "enable_dbo": True,
            "cuda_graph_full_decode_only": False,
            "prefix_caching": True,
            "offered_rps": args.collection_rate_rps,
            "max_model_len": 8192,
            "max_num_seqs": 32,
            "max_num_batched_tokens": 3072,
            "cudagraph_capture_size": 32,
            "dbo_decode_token_threshold": 2,
            "dbo_prefill_token_threshold": 12,
            "smoke_request_limit": args.smoke_limit,
        },
        "trace": {
            "path": str(trace_path),
            "sha256": trace_hash,
            "split": "calibration",
            "requests": expected_requests,
            "source_trace_requests": trace_requests,
        },
        "replay": {
            "summary_path": str(args.replay_summary.resolve()),
            "summary_sha256": sha256(args.replay_summary.resolve()),
            "trace_sha256": trace_hash,
            "completed_requests": int(summary["completed_requests"]),
            "failed_requests": int(summary["failed_requests"]),
        },
        "operating_point_ack": {
            "path": str(args.operating_point_ack.resolve()),
            "sha256": sha256(args.operating_point_ack.resolve()),
        },
        "launch_config": {
            "path": str((run_dir / "launch_config.json").resolve()),
            "sha256": sha256(run_dir / "launch_config.json"),
        },
        "sidecars": inspected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "sidecars": len(inspected)}))


if __name__ == "__main__":
    main()

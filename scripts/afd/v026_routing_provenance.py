#!/usr/bin/env python3
"""Fail-closed provenance checks for v0.26 routing calibration inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


METHOD = "ecodep_v026_matched_routing_calibration"
RUNTIME_VERSION = "0.26.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_routing_provenance(
    manifest_path: Path,
    sidecar_paths: list[Path],
) -> dict[str, Any]:
    """Validate that sidecars came from the frozen, non-measurement v0.26 run."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("schema_version", 0)) < 1:
        raise ValueError("routing provenance schema is unsupported")
    if manifest.get("method") != METHOD:
        raise ValueError("routing provenance method is not matched v0.26 calibration")
    if manifest.get("selection_split") != "calibration":
        raise ValueError("routing provenance is not calibration-only")
    if manifest.get("measurement_eligible") is not False:
        raise ValueError("instrumented routing run must be excluded from measurement")

    runtime = manifest.get("runtime", {})
    if runtime.get("vllm_afd_version") != RUNTIME_VERSION:
        raise ValueError("routing provenance is not from vLLM AFD v0.26.0")
    if not str(runtime.get("image_id", "")).startswith("sha256:"):
        raise ValueError("routing provenance lacks an immutable image ID")
    if not runtime.get("plugin_commit") or not runtime.get("plugin_source_sha256"):
        raise ValueError("routing provenance lacks frozen plugin identity")

    topology = manifest.get("topology", {})
    expected_topology = {
        "attention_dp": 2,
        "ffn_ep": 2,
        "attention_tp": 1,
        "expert_tp": 1,
    }
    if topology != expected_topology:
        raise ValueError("routing provenance topology is not matched DP2/EP2")

    collection = manifest.get("collection_contract", {})
    required_collection = {
        "routing_sidecar": True,
        "routing_source_role": "ffn",
        "routing_ffn_sidecar": True,
        "routing_async_copy": True,
        "routing_window_gate": "replay_client_marker",
        "ffn_cudagraph": False,
        "cuda_graph_full_decode_only": False,
        "compute_gate_on_attention": False,
        "stage_trace": False,
        "expert_epoch_trace": False,
        "request_path_clock_transitions": 0,
        "attention_mhz": [1410, 1410],
        "expert_mhz": [1410, 1410],
        "attention_power_w": [400, 400],
        "expert_power_w": [400, 400],
        "placement_variant": "B2-linear-equal-capacity",
    }
    for key, expected in required_collection.items():
        if collection.get(key) != expected:
            raise ValueError(f"routing collection contract mismatch: {key}")
    if collection.get("enable_dbo") is not True:
        raise ValueError("routing collection must use the matched DBO setting")
    if collection.get("cuda_graph_full_decode_only") is not False:
        raise ValueError("routing collection must use the frozen eager setting")
    if collection.get("prefix_caching") is not True:
        raise ValueError("routing collection must use the matched prefix-cache setting")
    offered_rps = float(collection.get("offered_rps", 0.0))
    if offered_rps <= 0:
        raise ValueError("routing collection lacks a positive offered rate")

    trace = manifest.get("trace", {})
    if trace.get("split") != "calibration":
        raise ValueError("routing trace is not marked calibration")
    if not trace.get("sha256") or int(trace.get("requests", 0)) < 200:
        raise ValueError("routing calibration trace evidence is insufficient")
    replay = manifest.get("replay", {})
    if replay.get("trace_sha256") != trace.get("sha256"):
        raise ValueError("routing replay trace hash differs from frozen trace")
    if int(replay.get("completed_requests", -1)) != int(trace["requests"]):
        raise ValueError("routing replay did not complete every calibration request")
    if int(replay.get("failed_requests", -1)) != 0:
        raise ValueError("routing replay contains failed requests")

    expected_files: dict[Path, dict[str, Any]] = {}
    for row in manifest.get("sidecars", []):
        path = Path(row.get("path", "")).resolve()
        if path in expected_files:
            raise ValueError("routing provenance duplicates a sidecar path")
        expected_files[path] = row
    actual_files = {path.resolve() for path in sidecar_paths}
    if actual_files != set(expected_files):
        raise ValueError("provided sidecars differ from routing provenance")
    if len(expected_files) != topology["ffn_ep"]:
        raise ValueError("routing provenance must contain one FFN sidecar per rank")

    ranks: set[int] = set()
    for path, row in expected_files.items():
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise ValueError(f"routing sidecar hash mismatch: {path}")
        if row.get("source_role") != "ffn":
            raise ValueError("placement may only use FFN routing sidecars")
        rank = int(row.get("source_rank", -1))
        ranks.add(rank)
        if int(row.get("records", 0)) <= 0:
            raise ValueError("routing sidecar has no recorded events")
        if int(row.get("layers", 0)) != 40 or int(row.get("experts", 0)) != 256:
            raise ValueError("routing sidecar model shape is not Qwen3.6-35B-A3B")
        if int(row.get("top_k", 0)) != 8:
            raise ValueError("routing sidecar top-k is not matched")
    if ranks != set(range(topology["ffn_ep"])):
        raise ValueError("routing provenance does not cover every FFN rank")
    return manifest

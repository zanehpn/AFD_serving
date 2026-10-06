#!/usr/bin/env python3
"""Build a leakage-free four-stage AFD profile from calibration runs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from four_stage_dse_v6.model import STAGES, pipeline_time_ms
from four_stage_dse_v6.attention_workload import (
    validate_context, rank_summary, reference_context, fit_rank_model,
)


_AUXILIARY_ATTENTION_EVENTS = (
    "attention_layer_total",
    "remote_ffn_roundtrip",
)
_TRACE_EVENTS = frozenset((*STAGES, *_AUXILIARY_ATTENTION_EVENTS))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def nnls_small(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Small active-set NNLS used without a scipy dependency."""
    best = np.zeros(design.shape[1], dtype=float)
    best_error = float(np.sum(target**2))
    for mask in range(1, 1 << design.shape[1]):
        columns = [idx for idx in range(design.shape[1]) if mask & (1 << idx)]
        fitted = np.linalg.lstsq(design[:, columns], target, rcond=None)[0]
        if np.any(fitted < 0):
            continue
        candidate = np.zeros(design.shape[1], dtype=float)
        candidate[columns] = fitted
        error = float(np.sum((target - design @ candidate) ** 2))
        if error < best_error:
            best, best_error = candidate, error
    return best


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    return float(np.quantile(np.asarray(values, dtype=float), quantile, method="higher"))


def summarize_reference_workload(
    groups: list[dict[str, Any]], routing_reference: float
) -> dict[str, float]:
    """Describe the workload unit consumed by the four-stage regressions.

    Stage models are fitted per logical DBO microbatch, so their reference
    features must come from those same microbatches.  Per-request input and
    output lengths are not interchangeable with concurrent-batch token counts.
    """
    if not groups:
        raise ValueError("reference workload requires four-stage groups")
    return {
        "prefill_tokens_per_microbatch": float(
            np.mean([float(row["prefill_tokens"]) for row in groups])
        ),
        "decode_tokens_per_microbatch": float(
            np.mean([float(row["decode_tokens"]) for row in groups])
        ),
        "communication_bytes_scale": 1.0,
        "routing_imbalance": float(routing_reference),
        **reference_context(groups),
    }


def deduplicate_attention_tp(ranks, tp):
    """Count a logical query once while retaining the slowest TP shard service."""
    workers = defaultdict(list)
    for key, row in ranks.items():
        mapping = row.get('parallel_layout')
        if not mapping or mapping['tp_size'] != tp:
            raise ValueError('Attention TP requires explicit rank mapping on every trace')
        local = mapping['local_role_rank']
        workers[(mapping['replica'], local // tp)].append((local % tp, row))
    output = {}
    for (replica, dp), members in workers.items():
        if sorted(rank for rank, _ in members) != list(range(tp)):
            raise ValueError('Incomplete or duplicated Attention TP group')
        first = members[0][1]
        for _, row in members[1:]:
            for field in ('prefill_tokens', 'decode_tokens', 'context', 'decode_request_spans', 'event_layers'):
                if row[field] != first[field]:
                    raise ValueError(f'TP peers disagree on logical {field}')
            if row['layer_ms'].keys() != first['layer_ms'].keys():
                raise ValueError('TP peers disagree on layer coverage')
        layer_ms = {k: max(r['layer_ms'][k] for _, r in members) for k in first['layer_ms']}
        output[f'replica-{replica}-dp-{dp}'] = {
            **first, 'layer_ms': layer_ms,
            'duration_ms': sum(layer_ms.values()) if layer_ms else max(r['duration_ms'] for _, r in members)}
    return output


def load_four_stage_groups(
    paths: list[Path], *, minimum_complete_coverage: float, token_scope: str = "rank_max",
    attention_tp: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join A/F rank traces by causal transaction and DBO microbatch."""
    if token_scope not in {"rank_max", "attention_dp_total"}:
        raise ValueError("Unknown four-stage token scope")
    nested: dict[
        tuple[str, int], dict[str, dict[str, dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    all_keys: set[tuple[str, int]] = set()
    event_rows = 0
    for path in paths:
        rank_key = path.name
        for row in read_jsonl(path):
            event = str(row.get("event"))
            if event not in _TRACE_EVENTS:
                continue
            transaction = row.get("transaction_id")
            duration_us = row.get("duration_us")
            if not transaction or duration_us is None or float(duration_us) <= 0:
                continue
            key = (str(transaction), int(row.get("stage_idx") or 0))
            all_keys.add(key)
            rank = nested[key][event].setdefault(
                rank_key,
                {
                    "duration_ms": 0.0,
                    "bytes": 0,
                    "prefill_tokens": 0,
                    "decode_tokens": 0,
                    "events": 0,
                    "event_layers": [],
                    "context": None,
                    "context_rows": 0,
                    "layer_ms": {},
                    "decode_request_spans": [],
                    "parallel_layout": row.get("parallel_layout"),
                },
            )
            if rank['parallel_layout'] != row.get('parallel_layout'):
                raise ValueError('parallel layout changed inside a causal transaction')
            rank["duration_ms"] += float(duration_us) / 1000.0
            rank["bytes"] += int(row.get("bytes") or 0)
            rank["prefill_tokens"] = max(
                int(rank["prefill_tokens"]), int(row.get("prefill_tokens") or 0)
            )
            rank["decode_tokens"] = max(
                int(rank["decode_tokens"]), int(row.get("decode_tokens") or 0)
            )
            rank["events"] += 1
            spans = row.get("decode_request_spans", [])
            if not isinstance(spans, list):
                raise ValueError("decode request spans must be a list")
            for span in spans:
                if not isinstance(span, dict) or not isinstance(span.get("request_id"), str) or not span["request_id"]:
                    raise ValueError("invalid decode request identity")
                for field in ("prompt_tokens", "first_token_position", "token_count"):
                    if type(span.get(field)) is not int or span[field] < (1 if field == "token_count" else 0):
                        raise ValueError("invalid decode request span")
                if span["first_token_position"] < span["prompt_tokens"]:
                    raise ValueError("prefill token in decode request span")
            if spans and (row.get("layer_idx") != 0 or sum(s["token_count"] for s in spans) != int(row.get("decode_tokens") or 0)):
                raise ValueError("decode request span layer/token mismatch")
            rank["decode_request_spans"].extend(spans)
            rank["event_layers"].append(row.get("layer_idx"))
            layer = row.get("layer_idx")
            if type(layer) is int and layer >= 0:
                rank["layer_ms"][layer] = rank["layer_ms"].get(layer, 0.) + float(duration_us) / 1000.
            if event in {"attention_compute", *_AUXILIARY_ATTENTION_EVENTS}:
                context = validate_context(row.get("attention_workload"),
                                           int(row.get("prefill_tokens") or 0),
                                           int(row.get("decode_tokens") or 0))
                if context is not None:
                    if spans and sum(s["token_count"] * (s["first_token_position"] + 1)
                                     + s["token_count"] * (s["token_count"] - 1) // 2 for s in spans) != context["decode_context_tokens"]:
                        raise ValueError("decode span/context workload mismatch")
                    if rank["context"] is not None and rank["context"] != context:
                        raise ValueError("context changed within a causal transaction/microbatch")
                    rank["context"] = context
                    rank["context_rows"] += 1
            event_rows += 1

    complete: list[dict[str, Any]] = []
    attention_sources: dict[str, int] = defaultdict(int)
    missing_counts = {stage: 0 for stage in STAGES}
    for transaction, stage_idx in sorted(all_keys):
        by_stage = nested[(transaction, stage_idx)]
        attention_source = "direct_attention_compute"
        if not by_stage.get("attention_compute"):
            totals = by_stage.get("attention_layer_total", {})
            remotes = by_stage.get("remote_ffn_roundtrip", {})
            derived: dict[str, dict[str, Any]] = {}
            for rank_key, total in totals.items():
                remote = remotes.get(rank_key)
                if remote is None:
                    continue
                direct_ms = float(total["duration_ms"]) - float(remote["duration_ms"])
                if direct_ms <= 0:
                    continue
                if total["context"] != remote["context"]:
                    raise ValueError("Attention total/remote context mismatch")
                layer_ms = {}
                if total["layer_ms"].keys() == remote["layer_ms"].keys():
                    layer_ms = {k: v - remote["layer_ms"][k] for k, v in total["layer_ms"].items()}
                    if any(v <= 0 for v in layer_ms.values()):
                        raise ValueError("nonpositive per-layer Attention total minus remote service")
                derived[rank_key] = {
                    "duration_ms": direct_ms,
                    "bytes": 0,
                    "prefill_tokens": max(
                        int(total["prefill_tokens"]),
                        int(remote["prefill_tokens"]),
                    ),
                    "decode_tokens": max(
                        int(total["decode_tokens"]),
                        int(remote["decode_tokens"]),
                    ),
                    "events": int(total["events"]),
                    "event_layers": total["event_layers"],
                    "context": total["context"],
                    "context_rows": min(total["context_rows"], remote["context_rows"]),
                    "layer_ms": layer_ms,
                    "decode_request_spans": total["decode_request_spans"],
                    "parallel_layout": total['parallel_layout'],
                }
            if derived:
                by_stage["attention_compute"] = derived
                attention_source = "attention_layer_total_minus_remote_ffn_roundtrip"
        missing = [stage for stage in STAGES if not by_stage.get(stage)]
        if missing:
            for stage in missing:
                missing_counts[stage] += 1
            continue
        durations: dict[str, float] = {}
        bytes_by_stage: dict[str, int] = {}
        prefill_tokens = 0
        decode_tokens = 0
        ranks_by_stage: dict[str, int] = {}
        for stage in STAGES:
            ranks = list(by_stage[stage].values())
            # Parallel ranks process the same logical microbatch.  End-to-end
            # progress is set by the slowest rank, not their summed GPU time.
            durations[stage] = max(float(rank["duration_ms"]) for rank in ranks)
            bytes_by_stage[stage] = sum(int(rank["bytes"]) for rank in ranks)
            prefill_tokens = max(
                prefill_tokens, max(int(rank["prefill_tokens"]) for rank in ranks)
            )
            decode_tokens = max(
                decode_tokens, max(int(rank["decode_tokens"]) for rank in ranks)
            )
            ranks_by_stage[stage] = len(ranks)
        logical_attention = by_stage['attention_compute']
        sizes = {r['parallel_layout']['tp_size'] for r in logical_attention.values() if r.get('parallel_layout')}
        tp = attention_tp if attention_tp is not None else (next(iter(sizes)) if len(sizes) == 1 else 1)
        if sizes and sizes != {tp}:
            raise ValueError('Attention trace TP differs from candidate')
        if tp > 1:
            logical_attention = deduplicate_attention_tp(logical_attention, tp)
        if token_scope == "attention_dp_total":
            # Count real tokens across Attention DP ranks exactly once. Expert
            # ranks contain either one peer (2A2E) or a concatenation (2A1E), so
            # max-over-A/E would silently change the workload unit with topology.
            attention_ranks = logical_attention.values()
            prefill_tokens = sum(int(rank["prefill_tokens"]) for rank in attention_ranks)
            attention_ranks = logical_attention.values()
            decode_tokens = sum(int(rank["decode_tokens"]) for rank in attention_ranks)
        attention_ranks = logical_attention
        event_sets = [sorted(row["event_layers"], key=lambda value: (value is None, value))
                      for row in attention_ranks.values()]
        observations = {key: {"duration_ms": row["duration_ms"],
                              "prefill_tokens": row["prefill_tokens"],
                              "decode_tokens": row["decode_tokens"],
                              "decode_request_spans": row["decode_request_spans"],
                              "context": row["context"] if row["context_rows"] == row["events"] else None}
                        for key, row in attention_ranks.items()}
        rank_audit = rank_summary(observations, comparable_events=all(item == event_sets[0] for item in event_sets))
        layer_sets = [set(row["layer_ms"]) for ranks in by_stage.values() for row in ranks.values()]
        layer_complete = bool(layer_sets and layer_sets[0]) and all(s == layer_sets[0] for s in layer_sets)
        layer_rows = []
        if layer_complete:
            for layer in sorted(layer_sets[0]):
                rank_times = {stage: {key: row["layer_ms"][layer] for key, row in by_stage[stage].items()}
                              for stage in STAGES}
                layer_rows.append({"layer_idx": layer, "rank_service_ms": rank_times,
                                   "stage_ms": {s: max(rank_times[s].values()) for s in STAGES}})
            if token_scope == "attention_dp_total":
                rank_audit["max_of_rank_layer_sums_ms"] = rank_audit["rank_max_ms"]
                rank_audit["rank_mean_ms"] = sum(float(np.mean(list(row["rank_service_ms"]["attention_compute"].values()))) for row in layer_rows)
                rank_audit["rank_max_ms"] = sum(row["stage_ms"]["attention_compute"] for row in layer_rows)
                rank_audit["service_barrier_uplift_ms"] = rank_audit["rank_max_ms"] - rank_audit["rank_mean_ms"]
                rank_audit["potential_wait_gpu_ms"] = sum(sum(row["stage_ms"]["attention_compute"] - v for v in row["rank_service_ms"]["attention_compute"].values()) for row in layer_rows)
                rank_audit["scope"] = "sum_of_per_layer_causal_participant_service_barriers"
                durations = {s: sum(row["stage_ms"][s] for row in layer_rows) for s in STAGES}
        complete.append(
            {
                "transaction_id": transaction,
                "stage_idx": stage_idx,
                "stage_times_ms": durations,
                "bytes_by_stage": bytes_by_stage,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "ranks_by_stage": ranks_by_stage,
                "attention_compute_source": attention_source,
                "attention_rank_observations": observations,
                "attention_rank_summary": rank_audit,
                "token_scope": token_scope,
                "attention_tp_deduplicated": tp,
                "layer_observations": layer_rows,
                "layer_barrier_available": layer_complete,
            }
        )
        attention_sources[attention_source] += 1
    coverage = len(complete) / max(len(all_keys), 1)
    if coverage < minimum_complete_coverage:
        raise ValueError(
            "incomplete four-stage trace coverage: "
            f"{coverage:.4f} < {minimum_complete_coverage:.4f}; "
            f"missing={missing_counts}"
        )
    if not complete:
        raise ValueError("no complete four-stage transaction/microbatch groups")
    return complete, {
        "logical_microbatches_seen": len(all_keys),
        "complete_logical_microbatches": len(complete),
        "complete_coverage": coverage,
        "minimum_complete_coverage": minimum_complete_coverage,
        "missing_counts": missing_counts,
        "four_stage_event_rows": event_rows,
        "token_scope": token_scope,
        "attention_compute_sources": dict(sorted(attention_sources.items())),
    }


def fit_stage_model(
    groups: list[dict[str, Any]],
    stage: str,
    routing_reference: float,
    *,
    routing_enabled: bool = True,
) -> dict[str, Any]:
    design = np.asarray(
        [
            [
                1.0,
                float(row["prefill_tokens"]),
                float(row["decode_tokens"]),
            ]
            for row in groups
        ],
        dtype=float,
    )
    target = np.asarray(
        [float(row["stage_times_ms"][stage]) for row in groups], dtype=float
    )
    coefficients = nnls_small(design, target)
    prediction = design @ coefficients
    positive_residual = np.maximum(target - prediction, 0.0)
    byte_values = [int(row["bytes_by_stage"][stage]) for row in groups]
    result = {
        "intercept_ms": float(coefficients[0]),
        "prefill_ms_per_token": float(coefficients[1]),
        "decode_ms_per_token": float(coefficients[2]),
        "p90_residual_ms": percentile(positive_residual.tolist(), 0.90),
        "p95_residual_ms": percentile(positive_residual.tolist(), 0.95),
        "rmse_ms": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "samples": len(groups),
        "measured_p50_ms": percentile(target.tolist(), 0.50),
        "measured_p90_ms": percentile(target.tolist(), 0.90),
        "reference_bytes_p50": percentile([float(v) for v in byte_values], 0.50),
        "routing_imbalance_reference": routing_reference,
        # A conservative first-order term.  It is used only for FFN and is
        # replaced by a fitted value once routing-window coverage is adequate.
        "routing_sensitivity": (
            1.0 if routing_enabled and stage == "ffn_compute" else 0.0
        ),
    }
    if stage == "attention_compute":
        # Context totals have the DP-total unit. Legacy rank-max profiles retain
        # their original prediction semantics and report rank diagnostics only.
        if all(row.get("token_scope") == "attention_dp_total" for row in groups):
            result["attention_workload_model"] = fit_rank_model(groups, nnls_small)
    return result


def load_routing_reference(paths: list[Path]) -> tuple[float, dict[str, Any]]:
    imbalances: list[float] = []
    for path in paths:
        for row in read_jsonl(path):
            counts = [int(value) for value in row.get("domain_counts", [])]
            if not counts or sum(counts) <= 0:
                continue
            mean = sum(counts) / len(counts)
            imbalances.append(max(counts) / mean)
    if not imbalances:
        raise ValueError("routing distribution was not observed in calibration")
    return float(np.median(imbalances)), {
        "windows": len(imbalances),
        "median": float(np.median(imbalances)),
        "p95": percentile(imbalances, 0.95),
        "maximum": max(imbalances),
    }


def fit_total_power_model(
    summary: dict[str, Any],
    telemetry: dict[str, Any],
    power_samples: list[dict[str, Any]],
    total_power_cap_w: float,
) -> dict[str, float]:
    duration_s = float(telemetry.get("duration_s") or summary["duration_s"])
    average_power = float(summary["energy_j"]) / duration_s
    completed_rate = float(summary["completed_requests"]) / max(
        float(summary["duration_s"]), 1e-9
    )
    offered_rate = max(float(summary.get("observed_submission_rate_rps", completed_rate)), 1e-9)
    saturated = completed_rate < 0.95 * offered_rate
    sample_totals = [
        sum(float(value) for value in row.get("power_w", []))
        for row in power_samples
        if row.get("power_w")
    ]
    if not sample_totals:
        raise ValueError("power model requires non-empty per-GPU samples")
    # At a single load, utilization and dynamic power are not independently
    # identifiable.  Anchor the empirical load at utilization=1 and use the
    # lower five-percent power envelope as the non-scaling floor.  In
    # particular, completed/offered RPS is not utilization: for a saturated
    # finite replay it is depressed by terminal drain time.
    observed_utilization = 1.0
    idle = min(
        float(np.quantile(np.asarray(sample_totals, dtype=float), 0.05, method="linear")),
        average_power,
    )
    slope = average_power - idle
    return {
        "idle_intercept_w": idle,
        "dynamic_slope_w": max(slope, 0.0),
        "total_power_cap_w": float(total_power_cap_w),
        "measured_average_power_w": average_power,
        "observed_utilization": observed_utilization,
        "offered_rate_rps": offered_rate,
        "completed_makespan_rate_rps": completed_rate,
        "calibration_saturated": saturated,
        "fit_status": "single_load_p05_floor_empirical_full_load_anchor",
    }


def gpu_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else int(value)


def validate_topology(topology: dict[str, Any]) -> bool:
    attention_gpus = gpu_count(topology["attention_gpus"])
    expert_gpus = gpu_count(topology["expert_gpus"])
    return (
        attention_gpus
        == int(topology.get("attention_dp", 1))
        * int(topology.get("attention_tp", 1))
        and expert_gpus
        == int(topology.get("expert_dp", 1))
        * int(topology.get("expert_ep", 1))
        * int(topology.get("expert_tp", 1))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("grid", type=Path)
    parser.add_argument("calibration_trace", type=Path)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-complete-coverage", type=float, default=0.95)
    parser.add_argument(
        "--routing-role", choices=("attention", "ffn", "none"), default="attention"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("v6 calibration profiles are immutable")

    manifest = json.loads(args.calibration_manifest.read_text())
    if manifest.get("evaluation_split") != "calibration":
        raise ValueError("profile builder accepts only a calibration manifest")
    if sha256(args.calibration_trace) != manifest.get("trace_sha256"):
        raise ValueError("calibration trace hash does not match its manifest")
    if Path(manifest["trace"]).resolve() != args.calibration_trace.resolve():
        raise ValueError("calibration trace path does not match its manifest")

    grid = json.loads(args.grid.read_text())
    if grid.get("selection_split") != "calibration":
        raise ValueError("DSE grid is not marked calibration-only")
    topology = dict(grid["topology"])
    if not validate_topology(topology):
        raise ValueError(f"illegal base topology: {topology}")
    guard_id = str(grid.get("guard_cell", "c00-max"))
    guard_dir = args.calibration_root / "cells" / guard_id
    guard_groups, _ = load_four_stage_groups(
        sorted((guard_dir / "traces").glob("stage-*.jsonl")),
        minimum_complete_coverage=args.minimum_complete_coverage,
    )
    routing_enabled = args.routing_role != "none"
    if routing_enabled:
        guard_routing_reference, _ = load_routing_reference(
            sorted(
                (guard_dir / "traces").glob(
                    f"routing-{args.routing_role}-*.jsonl"
                )
            )
        )
    else:
        guard_routing_reference = 1.0
    reference_workload = summarize_reference_workload(
        guard_groups, guard_routing_reference
    )

    cells: dict[str, dict[str, Any]] = {}
    for cell in grid["cells"]:
        cell_dir = args.calibration_root / "cells" / str(cell["id"])
        summary_path = cell_dir / "summary.json"
        telemetry_path = cell_dir / "telemetry.json"
        summary = json.loads(summary_path.read_text())
        telemetry = json.loads(telemetry_path.read_text())
        if int(summary["completed_requests"]) != int(summary["requests"]):
            raise ValueError(f"{cell['id']}: incomplete calibration requests")
        if int(summary.get("failed_requests", 0)) != 0:
            raise ValueError(f"{cell['id']}: failed calibration requests")
        stage_paths = sorted((cell_dir / "traces").glob("stage-*.jsonl"))
        groups, coverage = load_four_stage_groups(
            stage_paths,
            minimum_complete_coverage=args.minimum_complete_coverage,
        )
        if routing_enabled:
            routing_reference, routing_stats = load_routing_reference(
                sorted(
                    (cell_dir / "traces").glob(
                        f"routing-{args.routing_role}-*.jsonl"
                    )
                )
            )
        else:
            routing_reference = 1.0
            routing_stats = {
                "status": "disabled_static_no_routing",
                "windows": 0,
            }
        stage_models = {
            stage: fit_stage_model(
                groups,
                stage,
                routing_reference,
                routing_enabled=routing_enabled,
            )
            for stage in STAGES
        }
        reference_stage_times = {
            stage: (
                float(model["intercept_ms"])
                + float(model["prefill_ms_per_token"])
                * reference_workload["prefill_tokens_per_microbatch"]
                + float(model["decode_ms_per_token"])
                * reference_workload["decode_tokens_per_microbatch"]
                + float(model["p90_residual_ms"])
            )
            for stage, model in stage_models.items()
        }
        observed_microbatches = max(int(row["stage_idx"]) for row in groups) + 1
        layers = max(
            1,
            int(cell.get("layers", grid.get("layers", 28))),
        )
        reference_pipeline = pipeline_time_ms(
            reference_stage_times,
            microbatches=observed_microbatches,
            layers=layers,
        )
        attention_count = gpu_count(topology["attention_gpus"])
        expert_count = gpu_count(topology["expert_gpus"])
        total_cap = (
            attention_count * float(cell["attention_power_w"])
            + expert_count * float(cell["expert_power_w"])
        )
        cells[str(cell["id"])] = {
            "stage_models": stage_models,
            "reference_stage_times_ms": reference_stage_times,
            "reference_pipeline_time_ms": reference_pipeline,
            "microbatches": observed_microbatches,
            "layers": layers,
            "trace_coverage": coverage,
            "routing": routing_stats,
            "power_model": fit_total_power_model(
                summary,
                telemetry,
                read_jsonl(cell_dir / "power-samples.jsonl"),
                total_cap,
            ),
            "summary": {
                "energy_j": float(summary["energy_j"]),
                "duration_s": float(summary["duration_s"]),
                "p90_ttft_ms": float(summary["ttft_ms"]["p90"]),
                "p90_tpot_ms": float(summary["tpot_ms"]["p90"]),
                "completed_requests": int(summary["completed_requests"]),
            },
            "summary_sha256": sha256(summary_path),
            "telemetry_sha256": sha256(telemetry_path),
        }

    if guard_id not in cells:
        raise ValueError(f"guard cell {guard_id!r} is absent")
    guard = cells[guard_id]
    guard_ttft = float(guard["summary"]["p90_ttft_ms"])
    guard_tpot = float(guard["summary"]["p90_tpot_ms"])
    guard_pipeline = float(guard["reference_pipeline_time_ms"])
    candidates = []
    for cell in grid["cells"]:
        cell_id = str(cell["id"])
        measured = cells[cell_id]
        measured_capacity = float(measured["summary"]["completed_requests"]) / float(
            measured["summary"]["duration_s"]
        )
        requests_per_pipeline = (
            measured_capacity * float(measured["reference_pipeline_time_ms"]) / 1000.0
        )
        candidate_topology = {**topology, **dict(cell.get("topology", {}))}
        candidates.append(
            {
                "id": cell_id,
                "selection_split": "calibration",
                "validation_status": "physically_measured_anchor",
                "parallelism_legal": validate_topology(candidate_topology),
                "memory_feasible": True,
                "topology": candidate_topology,
                "knobs": {
                    "attention_mhz": int(cell["attention_mhz"]),
                    "attention_power_w": int(cell["attention_power_w"]),
                    "expert_mhz": int(cell["expert_mhz"]),
                    "expert_power_w": int(cell["expert_power_w"]),
                    **dict(cell.get("knobs", {})),
                },
                "microbatches": int(measured["microbatches"]),
                "layers": int(measured["layers"]),
                "requests_per_pipeline": requests_per_pipeline,
                "capacity_estimation": "conservative_completed_requests_per_full_makespan",
                "guard_pipeline_time_ms": guard_pipeline,
                "stage_models": measured["stage_models"],
                "power_model": measured["power_model"],
                "calibration_latency_ratios": {
                    "p90_ttft": float(measured["summary"]["p90_ttft_ms"]) / guard_ttft,
                    "p90_tpot": float(measured["summary"]["p90_tpot_ms"]) / guard_tpot,
                },
            }
        )

    plugin_commit = subprocess.check_output(
        ["git", "-C", str(args.plugin_root), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "schema_version": 1,
        "method": "four_stage_bottleneck_guided_energy_dse_v6",
        "selection_split": "calibration",
        "formal_evaluation_eligible": False,
        "model": grid["model"],
        "gpu_model": grid["gpu_model"],
        "guard_candidate_id": guard_id,
        "latency_budget_ratio": 1.05,
        "routing_input": (
            "disabled_static_no_routing" if not routing_enabled else args.routing_role
        ),
        "stage_order": list(STAGES),
        "reference_workload": reference_workload,
        "candidates": candidates,
        "cells": cells,
        "provenance": {
            "calibration_trace": str(args.calibration_trace.resolve()),
            "calibration_trace_sha256": sha256(args.calibration_trace),
            "calibration_manifest": str(args.calibration_manifest.resolve()),
            "calibration_manifest_sha256": sha256(args.calibration_manifest),
            "grid": str(args.grid.resolve()),
            "grid_sha256": sha256(args.grid),
            "plugin_root": str(args.plugin_root.resolve()),
            "plugin_commit": plugin_commit,
            "heldout_artifacts_opened_by_builder": 0,
        },
        "limitations": [
            "single-load power fits remain conservative until multi-load calibration",
            *(
                ["routing sensitivity starts conservative and requires window-level fitting"]
                if routing_enabled
                else ["routing histograms and routing-aware features are disabled"]
            ),
            "finite-replay completed/makespan capacity is a conservative lower bound",
            "one-shot shortlisted points receive no extra physical validation before held-out evaluation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidates": len(candidates),
                "stage_order": list(STAGES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

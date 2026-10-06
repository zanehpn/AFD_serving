#!/usr/bin/env python3
"""Summarize request-level replay logs without dropping failed requests."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = quantile * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-j", type=float, default=None)
    parser.add_argument("--ttft-slo-ms", type=float, default=400.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=120.0)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as source:
        records: list[dict[str, Any]] = [json.loads(line) for line in source]
    completed = [record for record in records if record.get("error") is None]
    good = [record for record in records if record.get("slo_good")]
    ttft = [float(record["ttft_ms"]) for record in completed]
    tpot = [float(record["tpot_ms"]) for record in completed]
    e2e = [float(record["e2e_ms"]) for record in completed]
    good_tokens = sum(int(record["actual_output_tokens"]) for record in good)
    output_tokens = sum(int(record.get("actual_output_tokens", 0)) for record in completed)
    input_tokens = sum(int(record.get("input_tokens", 0)) for record in records)
    has_wall_boundaries = bool(records) and all(
        record.get("submit_wall_ns") is not None
        and record.get("finish_wall_ns") is not None
        for record in records
    )
    if has_wall_boundaries:
        measurement_start_wall_ns = min(
            int(record["submit_wall_ns"]) for record in records
        )
        measurement_end_wall_ns = max(
            int(record["finish_wall_ns"]) for record in records
        )
        duration_s = max(
            (measurement_end_wall_ns - measurement_start_wall_ns) / 1e9, 0.0
        )
        duration_source = "request_first_submit_to_last_finish"
    else:
        starts = [float(record["submit_s"]) for record in records]
        finishes = [
            float(record["submit_s"]) + float(record["e2e_ms"]) / 1000.0
            for record in records
        ]
        duration_s = (
            max(finishes) - min(starts)
            if starts and finishes and max(finishes) >= min(starts)
            else 0.0
        )
        measurement_start_wall_ns = None
        measurement_end_wall_ns = None
        duration_source = "legacy_monotonic_submit_to_finish"
    p90_ttft_ms = percentile(ttft, 0.90)
    p90_tpot_ms = percentile(tpot, 0.90)
    paper_slo_compliant = (
        bool(records)
        and len(completed) == len(records)
        and p90_ttft_ms <= args.ttft_slo_ms
        and p90_tpot_ms <= args.tpot_slo_ms
    )
    completed_request_throughput_rps = (
        len(completed) / duration_s if duration_s > 0 else 0.0
    )
    output_token_throughput_tps = (
        output_tokens / duration_s if duration_s > 0 else 0.0
    )
    total_token_throughput_tps = (
        (input_tokens + output_tokens) / duration_s if duration_s > 0 else 0.0
    )
    submit_times = [float(record["submit_s"]) for record in records]
    submission_span_s = (
        max(submit_times) - min(submit_times) if len(submit_times) >= 2 else 0.0
    )
    observed_submission_rate_rps = (
        (len(submit_times) - 1) / submission_span_s if submission_span_s > 0 else 0.0
    )
    summary = {
        "schema_version": 5,
        "requests": len(records),
        "completed_requests": len(completed),
        "failed_requests": len(records) - len(completed),
        "slo_good_requests": len(good),
        "slo_attainment": len(good) / len(records) if records else 0.0,
        "paper_slo": {
            "quantile": "p90",
            "ttft_limit_ms": args.ttft_slo_ms,
            "tpot_limit_ms": args.tpot_slo_ms,
            "compliant": paper_slo_compliant,
            "source": "AFlex arXiv:2608.01891v1 Section VI-A",
        },
        "slo_good_output_tokens": good_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_s": duration_s,
        "duration_source": duration_source,
        "measurement_start_wall_ns": measurement_start_wall_ns,
        "measurement_end_wall_ns": measurement_end_wall_ns,
        "submission_span_s": submission_span_s,
        "observed_submission_rate_rps": observed_submission_rate_rps,
        "completed_request_throughput_rps": completed_request_throughput_rps,
        "output_token_throughput_tps": output_token_throughput_tps,
        "total_token_throughput_tps": total_token_throughput_tps,
        "request_goodput_rps": len(good) / duration_s if duration_s > 0 else 0.0,
        "slo_good_output_tokens_per_s": (
            good_tokens / duration_s if duration_s > 0 else 0.0
        ),
        "ttft_ms": {
            "mean": statistics.fmean(ttft) if ttft else math.nan,
            "p50": percentile(ttft, 0.50),
            "p90": p90_ttft_ms,
            "p95": percentile(ttft, 0.95),
            "p99": percentile(ttft, 0.99),
        },
        "tpot_ms": {
            "mean": statistics.fmean(tpot) if tpot else math.nan,
            "p50": percentile(tpot, 0.50),
            "p90": p90_tpot_ms,
            "p95": percentile(tpot, 0.95),
            "p99": percentile(tpot, 0.99),
        },
        "e2e_ms": {
            "mean": statistics.fmean(e2e) if e2e else math.nan,
            "p50": percentile(e2e, 0.50),
            "p90": percentile(e2e, 0.90),
            "p95": percentile(e2e, 0.95),
            "p99": percentile(e2e, 0.99),
        },
        "energy_j": args.energy_j,
        "energy_per_slo_good_token_j": (
            args.energy_j / good_tokens
            if args.energy_j is not None and good_tokens > 0
            else None
        ),
        "aflex_energy_per_total_token_j": (
            args.energy_j / (input_tokens + output_tokens)
            if args.energy_j is not None and input_tokens + output_tokens > 0
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()

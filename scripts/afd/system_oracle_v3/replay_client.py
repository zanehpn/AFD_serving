#!/usr/bin/env python3
"""Replay a trace through endpoints selected by a frozen Oracle v3 plan."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

CAUSAL_DIR = Path(__file__).resolve().parents[1] / "causal_dvfs"
sys.path.insert(0, str(CAUSAL_DIR))
import replay_client as causal_replay  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_trace import load_manifest  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=100.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--progress-event-interval-ms", type=float, default=250.0)
    parser.add_argument(
        "--progress-event-mode",
        choices=("per-request", "batched"),
        default="batched",
    )
    parser.add_argument("--submit-signal-fifo", type=Path, default=None)
    return parser.parse_args()


def _validate_plan(
    manifest: Path,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    if plan.get("method") != "afd_system_oracle_v3":
        raise ValueError("plan method must be afd_system_oracle_v3")
    if plan.get("formal_evaluation_eligible") is not False:
        raise ValueError("Oracle v3 replay accepts calibration-only plans")
    if plan["trace"]["sha256"] != _sha256(manifest):
        raise ValueError("plan trace hash does not match replay manifest")
    rows = {int(row["source_index"]): row for row in plan["requests"]}
    identities = [int(record["source_index"]) for record in records]
    if len(rows) != len(plan["requests"]):
        raise ValueError("plan contains duplicate source_index values")
    if set(rows) != set(identities):
        raise ValueError("plan and replay request identities do not match exactly")
    return rows


async def _replay_planned_one(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    record: dict[str, Any],
    assignment: dict[str, Any],
    experiment_start: float,
    events: causal_replay.EventWriter,
) -> dict[str, Any]:
    request_args = copy.copy(args)
    request_args.endpoint = assignment["endpoint"]
    request_args.model = assignment["model"]
    request_args.time_scale = 1.0
    request_args.routing_active_marker = None
    planned_record = dict(record)
    planned_record["arrival_s"] = float(assignment["planned_admit_s"])
    result = await causal_replay.replay_one(
        session,
        request_args,
        planned_record,
        experiment_start,
        events,
    )
    result["arrival_s"] = record["arrival_s"]
    result["source_arrival_s"] = record["arrival_s"]
    result["planned_admit_s"] = assignment["planned_admit_s"]
    result["intentional_delay_ms"] = assignment["intentional_delay_ms"]
    result["replica_id"] = assignment["replica_id"]
    result["endpoint"] = assignment["endpoint"]
    result["topology"] = assignment["topology"]
    result["operating_point"] = assignment["operating_point"]
    result["communication_mode"] = assignment["communication_mode"]
    return result


async def run(args: argparse.Namespace) -> int:
    records = load_manifest(args.manifest, args.limit)
    plan = json.loads(args.plan.read_text())
    assignments = _validate_plan(args.manifest, records, plan)
    events = causal_replay.EventWriter(
        args.events,
        progress_mode=args.progress_event_mode,
        progress_interval_ms=args.progress_event_interval_ms,
        submit_signal_fifo=args.submit_signal_fifo,
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    results: list[dict[str, Any]] = []
    try:
        await events.emit(
            "replay_start",
            policy="afd_system_oracle_v3",
            plan_sha256=_sha256(args.plan),
        )
        experiment_start = time.perf_counter() + 1.0
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                asyncio.create_task(
                    _replay_planned_one(
                        session,
                        args,
                        record,
                        assignments[int(record["source_index"])],
                        experiment_start,
                        events,
                    )
                )
                for record in records
            ]
            results = await asyncio.gather(*tasks)
        await events.flush_progress()
        await events.emit(
            "replay_end",
            requests=len(results),
            failures=sum(result["error"] is not None for result in results),
        )
    finally:
        events.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for result in sorted(results, key=lambda item: item["request_id"]):
            output.write(json.dumps(result, separators=(",", ":"), allow_nan=True))
            output.write("\n")
    return 1 if any(result["error"] is not None for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))

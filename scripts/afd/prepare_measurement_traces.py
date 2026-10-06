#!/usr/bin/env python3
"""Remove server-warmup trace records immediately before measured replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--sidecar-mode", choices=("sidecar-on", "sidecar-off"), required=True)
    parser.add_argument(
        "--expert-epoch-mode",
        choices=("epoch-on", "epoch-off"),
        required=True,
    )
    parser.add_argument(
        "--attention-ranks",
        type=int,
        help="Legacy alias for --routing-ranks when routing-role=attention.",
    )
    parser.add_argument(
        "--routing-role",
        choices=("attention", "ffn"),
        default="attention",
    )
    parser.add_argument("--routing-ranks", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-ms", type=float, default=250.0)
    parser.add_argument("--settle-timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    if args.settle_ms <= 0 or args.settle_timeout_s <= 0:
        raise ValueError("trace settle intervals must be positive")

    routing_ranks = args.routing_ranks
    if routing_ranks is None:
        routing_ranks = args.attention_ranks
    if routing_ranks is None or routing_ranks <= 0:
        raise ValueError("a positive --routing-ranks value is required")
    if (
        args.attention_ranks is not None
        and args.routing_role != "attention"
        and args.routing_ranks is None
    ):
        raise ValueError(
            "--attention-ranks cannot describe FFN sidecars; use --routing-ranks"
        )

    routing = sorted(
        args.result_root.glob(f"routing-{args.routing_role}-*.jsonl")
    )
    all_routing = sorted(args.result_root.glob("routing-*.jsonl"))
    stage = sorted(args.result_root.glob("stage-*.jsonl"))
    expert_epochs = sorted(args.result_root.glob("expert-epoch-*.jsonl"))
    if args.sidecar_mode == "sidecar-on" and len(routing) != routing_ranks:
        raise RuntimeError(
            f"expected {routing_ranks} {args.routing_role} sidecars before replay, "
            f"found {len(routing)}"
        )
    if args.sidecar_mode == "sidecar-off" and all_routing:
        raise RuntimeError("sidecar-off server created routing trace files")
    if args.expert_epoch_mode == "epoch-on" and not expert_epochs:
        raise RuntimeError("Expert-epoch tracing enabled but no trace files exist")
    if args.expert_epoch_mode == "epoch-off" and expert_epochs:
        raise RuntimeError("Expert-epoch tracing disabled but trace files exist")
    paths = all_routing + stage + expert_epochs
    # Async routing copies may still be draining after the warmup HTTP request
    # has finished.  A single truncate races that writer and lets warmup rows
    # reappear in the measured trace.  Wait for a stable idle interval, then
    # truncate and require another stable empty interval; retry if necessary.
    settle_s = args.settle_ms / 1000.0
    deadline = time.monotonic() + args.settle_timeout_s
    last_sizes: tuple[int, ...] | None = None
    stable_since = time.monotonic()
    while True:
        sizes = tuple(path.stat().st_size for path in paths)
        now = time.monotonic()
        if sizes != last_sizes:
            last_sizes = sizes
            stable_since = now
        if now - stable_since >= settle_s:
            break
        if now >= deadline:
            raise RuntimeError("warmup trace writers did not become idle")
        time.sleep(min(0.05, settle_s))

    attempts = 0
    while True:
        attempts += 1
        reset_wall_ns = time.time_ns()
        for path in paths:
            path.write_text("")
        time.sleep(settle_s)
        if all(path.stat().st_size == 0 for path in paths):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("warmup rows reappeared after trace reset")
    payload = {
        "schema_version": 2,
        "sidecar_mode": args.sidecar_mode,
        "routing_role": args.routing_role,
        "routing_ranks": routing_ranks,
        "expert_epoch_mode": args.expert_epoch_mode,
        "reset_wall_ns": reset_wall_ns,
        "routing_files": [str(path.resolve()) for path in routing],
        "all_routing_files": [str(path.resolve()) for path in all_routing],
        "stage_files": [str(path.resolve()) for path in stage],
        "expert_epoch_files": [str(path.resolve()) for path in expert_epochs],
        "settle_ms": args.settle_ms,
        "reset_attempts": attempts,
        "all_files_empty_after_reset": all(path.stat().st_size == 0 for path in paths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

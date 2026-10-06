#!/usr/bin/env python3
"""Verify that experiment GPUs are idle and back at their default limits."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time
from pathlib import Path


QUERY_FIELDS = (
    "index,power.limit,power.default_limit,clocks.current.sm,clocks.max.sm,"
    "memory.used,utilization.gpu"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU IDs")
    parser.add_argument("--reset-ack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-idle-memory-mib", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    requested_gpus = [int(value) for value in args.gpus.split(",")]
    reset_ack = json.loads(args.reset_ack.read_text())
    errors: list[str] = []
    if reset_ack.get("verified") is not True:
        errors.append("reset controller acknowledgement is not verified")
    acknowledged_gpus = sorted(int(row["gpu"]) for row in reset_ack.get("acknowledgements", []))
    reset_requested_gpus = sorted(
        reset_ack.get(
            "requested_gpus",
            list(
                set(reset_ack.get("attention_gpus", []))
                | set(reset_ack.get("expert_gpus", []))
            ),
        )
    )
    if acknowledged_gpus != reset_requested_gpus:
        errors.append("reset acknowledgements do not cover reset request")
    if reset_requested_gpus != sorted(requested_gpus):
        errors.append("reset request does not cover the complete experiment allocation")

    command = [
        "nvidia-smi",
        f"--query-gpu={QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(map(str, requested_gpus)),
    ]
    deadline = time.monotonic() + args.timeout_s
    attempts = 0
    rows: list[dict[str, int | float]] = []
    dynamic_errors: list[str] = []
    while True:
        attempts += 1
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        rows = []
        dynamic_errors = []
        for values in csv.reader(io.StringIO(completed.stdout)):
            gpu, power, default_power, sm, maximum_sm, memory, utilization = (
                value.strip() for value in values
            )
            row = {
                "gpu": int(gpu),
                "power_limit_w": float(power),
                "default_power_limit_w": float(default_power),
                "current_sm_mhz": int(sm),
                "maximum_sm_mhz": int(maximum_sm),
                "memory_used_mib": int(memory),
                "utilization_percent": int(utilization),
            }
            rows.append(row)
            if abs(row["power_limit_w"] - row["default_power_limit_w"]) > 0.01:
                dynamic_errors.append(f"GPU {row['gpu']} power limit is not the default")
            if row["memory_used_mib"] > args.maximum_idle_memory_mib:
                dynamic_errors.append(f"GPU {row['gpu']} still has material memory allocated")
            if row["utilization_percent"] != 0:
                dynamic_errors.append(f"GPU {row['gpu']} is still active")
            # After -rgc an idle A100 drops to its 210 MHz floor. This catches a
            # leaked application clock in addition to checking the controller ack.
            if row["current_sm_mhz"] != 210:
                dynamic_errors.append(
                    f"GPU {row['gpu']} SM clock did not return to idle floor"
                )
        if sorted(row["gpu"] for row in rows) != sorted(requested_gpus):
            dynamic_errors.append("nvidia-smi did not return every requested GPU")
        if not dynamic_errors or time.monotonic() >= deadline:
            break
        time.sleep(2)
    errors.extend(dynamic_errors)
    payload = {
        "schema_version": 1,
        "requested_gpus": requested_gpus,
        "reset_ack": str(args.reset_ack.resolve()),
        "rows": rows,
        "poll_attempts": attempts,
        "verified": not errors,
        "verification_errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

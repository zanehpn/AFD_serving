#!/usr/bin/env python3
"""Wait until the requested GPUs have sufficient free memory.

By default, an existing compute/graphics process is also treated as a conflict.
Shared exploratory runs can opt into memory-only admission with
``--allow-active-processes``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pynvml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--select-count",
        type=int,
        default=0,
        help="select this many available GPUs instead of requiring the full list",
    )
    parser.add_argument(
        "--selection-groups",
        default="",
        help=(
            "semicolon-separated disjoint GPU groups in IDS=COUNT form; "
            "for example 0,1,2=2;4,5,6,7=2"
        ),
    )
    parser.add_argument("--minimum-free-gib", type=float, default=70.0)
    parser.add_argument(
        "--allow-active-processes",
        action="store_true",
        help="admit GPUs based only on free memory, even when a process is active",
    )
    parser.add_argument("--wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument(
        "--stable-for-s",
        type=float,
        default=0.0,
        help="require sufficient free memory continuously for this duration",
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        help="atomically update this JSON file on every poll",
    )
    args = parser.parse_args()
    if args.wait_timeout_s < 0:
        parser.error("--wait-timeout-s must be non-negative")
    if args.poll_interval_s <= 0:
        parser.error("--poll-interval-s must be positive")
    if args.stable_for_s < 0:
        parser.error("--stable-for-s must be non-negative")
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    if len(gpu_ids) != len(set(gpu_ids)):
        parser.error("--gpus must contain unique indices")
    if args.select_count < 0 or args.select_count > len(gpu_ids):
        parser.error("--select-count must be between zero and the GPU-list length")
    if args.select_count and args.selection_groups:
        parser.error("--select-count and --selection-groups are mutually exclusive")
    selection_groups: list[tuple[list[int], int]] = []
    if args.selection_groups:
        grouped_ids: list[int] = []
        try:
            for specification in args.selection_groups.split(";"):
                values, count = specification.rsplit("=", 1)
                group = [int(value) for value in values.split(",")]
                required = int(count)
                if not group or required <= 0 or required > len(group):
                    raise ValueError
                selection_groups.append((group, required))
                grouped_ids.extend(group)
        except ValueError:
            parser.error("invalid --selection-groups specification")
        if len(grouped_ids) != len(set(grouped_ids)):
            parser.error("selection groups must be disjoint")
        if not set(grouped_ids).issubset(gpu_ids):
            parser.error("selection groups must be subsets of --gpus")
    pynvml.nvmlInit()
    started = time.monotonic()
    stable_since: float | None = None
    stable_selection: tuple[int, ...] | None = None
    polls = 0
    while True:
        polls += 1
        rows = []
        unavailable = []
        for gpu in gpu_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gib = memory.free / 2**30
            processes = []
            for getter in (
                pynvml.nvmlDeviceGetComputeRunningProcesses,
                pynvml.nvmlDeviceGetGraphicsRunningProcesses,
            ):
                try:
                    processes.extend(int(process.pid) for process in getter(handle))
                except pynvml.NVMLError:
                    pass
            row = {
                "gpu": gpu,
                "free_gib": free_gib,
                "process_pids": sorted(set(processes)),
            }
            rows.append(row)
            if free_gib < args.minimum_free_gib or (
                processes and not args.allow_active_processes
            ):
                unavailable.append(row)
        elapsed_s = time.monotonic() - started
        available_ids = [
            int(row["gpu"])
            for row in rows
            if row["free_gib"] >= args.minimum_free_gib
            and (args.allow_active_processes or not row["process_pids"])
        ]
        selected_groups: list[list[int]] = []
        if selection_groups:
            for group, required in selection_groups:
                selected_groups.append(
                    [gpu for gpu in group if gpu in available_ids][:required]
                )
            selection = tuple(gpu for group in selected_groups for gpu in group)
            required_count = sum(required for _, required in selection_groups)
            selection_complete = all(
                len(selected) == required
                for selected, (_, required) in zip(
                    selected_groups, selection_groups, strict=True
                )
            )
        else:
            required_count = args.select_count or len(gpu_ids)
            selection = tuple(available_ids[:required_count])
            selection_complete = len(selection) == required_count
        if not selection_complete:
            stable_since = None
            stable_selection = None
        elif stable_selection != selection:
            stable_selection = selection
            stable_since = time.monotonic()
        stable_s = 0.0 if stable_since is None else time.monotonic() - stable_since
        ready = (
            stable_selection is not None
            and len(stable_selection) == required_count
            and stable_s >= args.stable_for_s
        )
        result = {
            "gpus": rows,
            "available": ready,
            "available_gpu_ids": available_ids,
            "selected_gpus": list(stable_selection or ()),
            "selected_groups": selected_groups if stable_selection else [],
            "select_count": required_count,
            "allow_active_processes": args.allow_active_processes,
            "minimum_free_gib": args.minimum_free_gib,
            "required_stable_s": args.stable_for_s,
            "observed_stable_s": stable_s,
            "waited_s": elapsed_s,
            "polls": polls,
        }
        if args.status_output is not None:
            args.status_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.status_output.with_name(
                f".{args.status_output.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(temporary, args.status_output)
        if ready or elapsed_s >= args.wait_timeout_s:
            print(json.dumps(result, indent=2))
            pynvml.nvmlShutdown()
            if not ready:
                raise SystemExit(1)
            return
        time.sleep(min(args.poll_interval_s, args.wait_timeout_s - elapsed_s))


if __name__ == "__main__":
    main()

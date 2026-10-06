#!/usr/bin/env python3
"""Passively audit shared-GPU contamination with fail-closed coverage checks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, TextIO

import pynvml


def run_checked(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return completed.stdout


def exact_container_id(name: str) -> str:
    output = run_checked(
        [
            "docker",
            "ps",
            "--no-trunc",
            "--filter",
            f"name=^/{name}$",
            "--format",
            "{{.ID}}",
        ]
    )
    ids = [line.strip() for line in output.splitlines() if line.strip()]
    if len(ids) > 1:
        raise RuntimeError(f"exact container name {name!r} matched multiple IDs")
    return ids[0] if ids else ""


def gpu_uuids(gpus: set[int]) -> dict[str, int]:
    output = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    mapping: dict[str, int] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            raise RuntimeError(f"malformed nvidia-smi GPU row: {line!r}")
        index = int(fields[0])
        if index in gpus:
            mapping[fields[1]] = index
    if set(mapping.values()) != gpus:
        raise RuntimeError("nvidia-smi did not report every monitored GPU")
    return mapping


def gpu_utilization(gpus: set[int]) -> dict[int, int]:
    output = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    utilization: dict[int, int] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            raise RuntimeError(f"malformed nvidia-smi utilization row: {line!r}")
        index = int(fields[0])
        if index in gpus:
            utilization[index] = int(fields[1])
    if set(utilization) != gpus:
        raise RuntimeError("nvidia-smi did not report utilization for every monitored GPU")
    return utilization


def gpu_power_watts(gpus: set[int]) -> dict[int, float]:
    """Read board power, which remains meaningful for polling Expert workers.

    The AFD Expert connector can keep an idle NCCL receive kernel resident.  NVML
    then reports 100% SM utilization even though no routed work is executing.
    Idle board power still separates that polling state from active inference.
    """
    output = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    power: dict[int, float] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            raise RuntimeError(f"malformed nvidia-smi power row: {line!r}")
        index = int(fields[0])
        if index in gpus:
            try:
                power[index] = float(fields[1])
            except ValueError as error:
                raise RuntimeError(
                    f"nvidia-smi did not report numeric power for GPU {index}: "
                    f"{fields[1]!r}"
                ) from error
    if set(power) != gpus:
        raise RuntimeError("nvidia-smi did not report power for every monitored GPU")
    return power


def process_details(pid: int) -> tuple[str, str, bool]:
    proc = Path("/proc") / str(pid)
    try:
        cgroup = (proc / "cgroup").read_text()
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        ).strip()
        return cgroup, command, True
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        # A process observed by NVML but gone before /proc inspection still
        # overlapped this sample. Keep it as unowned instead of dropping it.
        return "", "<exited before inspection>", False


def process_is_descendant(pid: int, root_pid: int) -> bool:
    """Return whether *pid* belongs to a host process tree rooted at *root_pid*.

    Docker ownership is visible in /proc cgroups. Apptainer deliberately uses
    the host PID namespace in our baseline, so process ancestry is the stable
    ownership boundary instead. A vanished PID remains unowned and therefore
    fails closed as foreign work.
    """
    current = pid
    visited: set[int] = set()
    while current > 1 and current not in visited:
        if current == root_pid:
            return True
        visited.add(current)
        try:
            stat = (Path("/proc") / str(current) / "stat").read_text()
            closing_parenthesis = stat.rfind(")")
            if closing_parenthesis < 0:
                return False
            fields = stat[closing_parenthesis + 1 :].split()
            current = int(fields[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            return False
    return current == root_pid


def owned_process(cgroup: str, container_id: str, pid: int, root_pid: int | None) -> bool:
    if root_pid is not None:
        return process_is_descendant(pid, root_pid)
    return bool(container_id and container_id in cgroup)


def compute_processes(
    uuid_to_gpu: dict[str, int], container_id: str, root_pid: int | None = None
) -> list[dict[str, Any]]:
    output = run_checked(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip() or "No running processes found" in line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or fields[0] not in uuid_to_gpu:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        cgroup, command, proc_readable = process_details(pid)
        rows.append(
            {
                "gpu": uuid_to_gpu[fields[0]],
                "gpu_uuid": fields[0],
                "pid": pid,
                "used_memory_mib": fields[2],
                "command": command,
                "proc_readable": proc_readable,
                "owned_by_container": owned_process(
                    cgroup, container_id, pid, root_pid
                ),
            }
        )
    return rows


def nvml_gpu_handles(gpus: set[int]) -> dict[int, Any]:
    """Resolve persistent NVML handles once for the long-running watcher."""
    handles: dict[int, Any] = {}
    for gpu in sorted(gpus):
        handles[gpu] = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    return handles


def nvml_compute_processes(
    handles: dict[int, Any], container_id: str, root_pid: int | None = None
) -> list[dict[str, Any]]:
    """Read CUDA processes without spawning nvidia-smi for every sample.

    Under a saturated AFD workload, an nvidia-smi subprocess can occasionally
    block for many seconds even while direct NVML power sampling remains
    healthy. Persistent handles avoid that avoidable monitoring hole while
    preserving the same PID/cgroup ownership check.
    """
    def read_gpu(item: tuple[int, Any]) -> list[dict[str, Any]]:
        gpu, handle = item
        gpu_rows: list[dict[str, Any]] = []
        gpu_uuid = str(pynvml.nvmlDeviceGetUUID(handle))
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            pid = int(process.pid)
            cgroup, command, proc_readable = process_details(pid)
            used_bytes = getattr(process, "usedGpuMemory", None)
            unavailable = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", 2**64 - 1)
            used_memory_mib = (
                "N/A"
                if used_bytes in (None, unavailable)
                else str(int(used_bytes) // (1024 * 1024))
            )
            gpu_rows.append(
                {
                    "gpu": gpu,
                    "gpu_uuid": gpu_uuid,
                    "pid": pid,
                    "used_memory_mib": used_memory_mib,
                    "command": command,
                    "proc_readable": proc_readable,
                    "owned_by_container": owned_process(
                        cgroup, container_id, pid, root_pid
                    ),
                }
            )
        return gpu_rows

    # NVML is thread-safe. Querying the four handles serially occasionally
    # turns one driver scheduling delay into a >2 s monitoring hole under a
    # saturated AFD workload. Concurrent per-GPU reads preserve the exact same
    # ownership evidence while bounding the collection time by the slowest
    # handle instead of the sum of all four.
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(handles)
    ) as executor:
        for gpu_rows in executor.map(read_gpu, sorted(handles.items())):
            rows.extend(gpu_rows)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


class JsonlAppender:
    """Keep the hot watcher journal open without synchronous NFS commits."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.output: TextIO = path.open("a", encoding="utf-8", buffering=1)

    def append(self, row: dict[str, Any]) -> None:
        self.output.write(json.dumps(row, separators=(",", ":")) + "\n")
        # Line-buffering makes the row visible to the concurrent validator.
        # Durability is established when the runner archives the local spool;
        # fsync must not sit in the 500 ms sampling critical path.
        self.output.flush()

    def close(self) -> None:
        self.output.close()


def write_failure(
    marker: Path,
    output: Path,
    error: BaseException,
    sequence: int,
    appender: JsonlAppender | None = None,
) -> None:
    row = {
        "schema_version": 1,
        "timestamp_ns": time.time_ns(),
        "sequence": sequence,
        "healthy": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if appender is None:
        append_jsonl(output, row)
    else:
        appender.append(row)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(row, indent=2) + "\n")


def read_live_owner_pid(path: Path) -> int:
    try:
        pid = int(path.read_text().strip())
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(f"invalid Apptainer owner PID file: {path}") from error
    if pid <= 1 or not (Path("/proc") / str(pid)).exists():
        raise RuntimeError(f"Apptainer owner process is not running: {pid}")
    return pid


def watch(args: argparse.Namespace) -> None:
    gpus = {int(value) for value in args.gpus.split(",")}
    if not gpus or args.interval_s <= 0:
        raise ValueError("GPU set must be nonempty and interval must be positive")
    sequence = 0
    container_id = ""
    pynvml.nvmlInit()
    handles = nvml_gpu_handles(gpus)
    appender = JsonlAppender(args.output)
    previous_write_duration_ms = 0.0
    try:
        while True:
            started = time.monotonic()
            sample_started_wall_ns = time.time_ns()
            try:
                owner_root_pid = None
                if args.owner_pid_file is not None:
                    owner_root_pid = read_live_owner_pid(args.owner_pid_file)
                    container_id = f"pid:{owner_root_pid}"
                else:
                    # The runner creates one immutable, uniquely named Docker
                    # container. Resolve it once and retain its exact ID.
                    if not container_id:
                        container_id = exact_container_id(args.container)
                processes = nvml_compute_processes(
                    handles, container_id, owner_root_pid
                )
                owned = [row for row in processes if row["owned_by_container"]]
                foreign = [row for row in processes if not row["owned_by_container"]]
                row = {
                    "schema_version": 1,
                    # Timestamp the beginning of the ownership observation.
                    # Using the completion timestamp incorrectly adds the
                    # previous sleep interval to a slow NVML collection when
                    # computing coverage gaps.
                    "timestamp_ns": sample_started_wall_ns,
                    "collection_finished_wall_ns": time.time_ns(),
                    "sequence": sequence,
                    "healthy": True,
                    "container_name": args.container,
                    "container_id": container_id or None,
                    "gpu_ids": sorted(gpus),
                    "owned_processes": owned,
                    "foreign_processes": foreign,
                    "contamination_free": not foreign,
                    "collection_duration_ms": (time.monotonic() - started) * 1000,
                    "previous_write_duration_ms": previous_write_duration_ms,
                }
                write_started = time.monotonic()
                appender.append(row)
                previous_write_duration_ms = (
                    time.monotonic() - write_started
                ) * 1000
            except BaseException as error:
                write_failure(
                    args.failure_marker,
                    args.output,
                    error,
                    sequence,
                    appender,
                )
                raise SystemExit(1) from error
            sequence += 1
            elapsed = time.monotonic() - started
            time.sleep(max(args.interval_s - elapsed, 0.01))
    finally:
        appender.close()
        pynvml.nvmlShutdown()


def wait_clean(args: argparse.Namespace) -> None:
    """Wait for exclusive CUDA ownership and a quiescent persistent service."""
    gpus = {int(value) for value in args.gpus.split(",")}
    if (
        not gpus
        or args.interval_s <= 0
        or args.clean_duration_s <= 0
        or not 0 <= args.max_utilization_percent <= 100
        or args.max_power_watts < 0
    ):
        raise ValueError("GPU set and durations must be positive")
    started = time.monotonic()
    clean_started: float | None = None
    samples = 0
    while True:
        sample_started = time.monotonic()
        container_id = exact_container_id(args.container)
        if not container_id:
            raise RuntimeError(f"persistent container is not running: {args.container}")
        mapping = gpu_uuids(gpus)
        processes = compute_processes(mapping, container_id)
        foreign = [row for row in processes if not row["owned_by_container"]]
        utilization = gpu_utilization(gpus)
        power_watts = gpu_power_watts(gpus)
        utilization_ready = all(
            value <= args.max_utilization_percent for value in utilization.values()
        )
        power_ready = args.max_power_watts == 0 or all(
            value <= args.max_power_watts for value in power_watts.values()
        )
        now = time.monotonic()
        if foreign or not utilization_ready or not power_ready:
            clean_started = None
        elif clean_started is None:
            clean_started = now
        clean_for_s = 0.0 if clean_started is None else now - clean_started
        samples += 1
        report = {
            "schema_version": 1,
            "status": "ready" if clean_for_s >= args.clean_duration_s else "waiting",
            "timestamp_ns": time.time_ns(),
            "container_name": args.container,
            "container_id": container_id,
            "gpu_ids": sorted(gpus),
            "required_clean_duration_s": args.clean_duration_s,
            "maximum_utilization_percent": args.max_utilization_percent,
            "gpu_utilization_percent": utilization,
            "maximum_power_watts": args.max_power_watts or None,
            "gpu_power_watts": power_watts,
            "clean_for_s": clean_for_s,
            "samples": samples,
            "foreign_processes": foreign,
        }
        if args.status_output is not None:
            args.status_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.status_output.with_name(
                f".{args.status_output.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            os.replace(temporary, args.status_output)
        if report["status"] == "ready":
            print(json.dumps(report, separators=(",", ":")))
            return
        if args.timeout_s > 0 and now - started >= args.timeout_s:
            raise TimeoutError(
                f"target GPUs did not remain clean for {args.clean_duration_s}s"
            )
        elapsed = time.monotonic() - sample_started
        time.sleep(max(args.interval_s - elapsed, 0.01))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid monitor JSONL at line {number}") from error
    return rows


def validate(args: argparse.Namespace) -> None:
    telemetry = json.loads(args.telemetry.read_text())
    start_ns = int(telemetry["started_wall_ns"])
    end_ns = int(telemetry["finished_wall_ns"])
    deadline = time.monotonic() + args.wait_timeout_s
    rows: list[dict[str, Any]] = []
    while True:
        if args.failure_marker.exists():
            raise RuntimeError("shared-GPU monitor reported a collection failure")
        if args.output.exists():
            rows = load_rows(args.output)
            if rows and int(rows[-1].get("timestamp_ns", -1)) >= end_ns:
                break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    if not rows:
        raise RuntimeError("shared-GPU monitor produced no health records")
    timestamps = [int(row.get("timestamp_ns", -1)) for row in rows]
    before = [index for index, stamp in enumerate(timestamps) if stamp <= start_ns]
    after = [index for index, stamp in enumerate(timestamps) if stamp >= end_ns]
    if not before or not after:
        raise RuntimeError("shared-GPU monitor does not bracket the measurement window")
    left, right = before[-1], after[0]
    covered = rows[left : right + 1]
    if any(row.get("healthy") is not True for row in covered):
        raise RuntimeError("shared-GPU monitor was unhealthy during measurement")
    if any(row.get("container_name") != args.container for row in covered):
        raise RuntimeError("shared-GPU monitor was bound to the wrong container")
    if any(not row.get("container_id") for row in covered):
        raise RuntimeError("measured container was not visible throughout monitoring")
    gaps = [
        (int(b["timestamp_ns"]) - int(a["timestamp_ns"])) / 1e9
        for a, b in zip(covered, covered[1:])
    ]
    max_gap_s = max(gaps, default=0.0)
    if max_gap_s > args.max_gap_s:
        raise RuntimeError(
            f"shared-GPU monitoring gap {max_gap_s:.3f}s exceeds {args.max_gap_s:.3f}s"
        )
    max_collection_s = max(
        (float(row.get("collection_duration_ms", 0.0)) / 1000.0 for row in covered),
        default=0.0,
    )
    if max_collection_s > args.max_gap_s:
        raise RuntimeError(
            "shared-GPU ownership collection "
            f"{max_collection_s:.3f}s exceeds {args.max_gap_s:.3f}s"
        )
    contaminated = [row for row in covered if row.get("foreign_processes")]
    report = {
        "schema_version": 1,
        "verified": not contaminated,
        "container_name": args.container,
        "measurement_started_wall_ns": start_ns,
        "measurement_finished_wall_ns": end_ns,
        "first_monitor_timestamp_ns": int(covered[0]["timestamp_ns"]),
        "last_monitor_timestamp_ns": int(covered[-1]["timestamp_ns"]),
        "samples": len(covered),
        "max_gap_s": max_gap_s,
        "max_collection_s": max_collection_s,
        "foreign_process_samples": len(contaminated),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if contaminated:
        raise RuntimeError("foreign CUDA context overlapped the measurement window")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    watcher = subparsers.add_parser("watch")
    watcher.add_argument("--output", type=Path, required=True)
    watcher.add_argument("--failure-marker", type=Path, required=True)
    watcher.add_argument("--gpus", required=True)
    watcher.add_argument("--container", required=True)
    watcher.add_argument(
        "--owner-pid-file",
        type=Path,
        help="attribute CUDA processes by ancestry for an Apptainer service",
    )
    watcher.add_argument("--interval-s", type=float, default=0.5)
    watcher.set_defaults(function=watch)
    clean_waiter = subparsers.add_parser("wait-clean")
    clean_waiter.add_argument("--gpus", required=True)
    clean_waiter.add_argument("--container", required=True)
    clean_waiter.add_argument("--clean-duration-s", type=float, default=10.0)
    clean_waiter.add_argument("--max-utilization-percent", type=int, default=100)
    clean_waiter.add_argument(
        "--max-power-watts",
        type=float,
        default=0.0,
        help="require every GPU to stay at or below this board power; 0 disables",
    )
    clean_waiter.add_argument("--interval-s", type=float, default=0.5)
    clean_waiter.add_argument("--timeout-s", type=float, default=0.0)
    clean_waiter.add_argument("--status-output", type=Path)
    clean_waiter.set_defaults(function=wait_clean)
    validator = subparsers.add_parser("validate")
    validator.add_argument("--output", type=Path, required=True)
    validator.add_argument("--failure-marker", type=Path, required=True)
    validator.add_argument("--telemetry", type=Path, required=True)
    validator.add_argument("--container", required=True)
    validator.add_argument("--report", type=Path, required=True)
    # The watcher is a host process, so an otherwise healthy measurement can
    # occasionally miss a few 500 ms heartbeats while the CPU is scheduled
    # heavily.  Keep the CUDA-process/visibility checks strict, but allow a
    # bounded scheduling gap.  The campaign runner passes the same value
    # explicitly; the environment hook also protects already-running shells
    # that launch a fresh validator after this file is updated.
    validator.add_argument(
        "--max-gap-s",
        type=float,
        default=float(os.environ.get("ECODEP_GPU_MONITOR_MAX_GAP_S", "5.0")),
    )
    validator.add_argument("--wait-timeout-s", type=float, default=3.0)
    validator.set_defaults(function=validate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

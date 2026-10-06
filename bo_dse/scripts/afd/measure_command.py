#!/usr/bin/env python3
"""Integrate per-GPU NVML power while executing a replay command."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pynvml


def load_request_window(path: Path) -> tuple[int, int, int]:
    """Return the first-submit/last-finish wall-clock interval in a replay log."""
    records: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text().splitlines() if line
    ]
    if not records:
        raise ValueError(f"request log is empty: {path}")
    if any(
        record.get("submit_wall_ns") is None or record.get("finish_wall_ns") is None
        for record in records
    ):
        raise ValueError(f"request log lacks wall-clock boundaries: {path}")
    start_ns = min(int(record["submit_wall_ns"]) for record in records)
    end_ns = max(int(record["finish_wall_ns"]) for record in records)
    if end_ns <= start_ns:
        raise ValueError(f"invalid request measurement window: {start_ns}..{end_ns}")
    return start_ns, end_ns, len(records)


def resolve_measurement_window(
    request_log: Path | None,
    *,
    returncode: int,
    command_start_ns: int,
    command_end_ns: int,
) -> tuple[int, int, int | None, str, str | None]:
    """Resolve the exact request window without masking a failed replay.

    A replay process can die before it atomically publishes its JSONL output.
    In that case, retain command-window diagnostics and the original nonzero
    return code. A successful command still fails closed if its requested log
    is absent or malformed.
    """
    if request_log is None:
        return (
            command_start_ns,
            command_end_ns,
            None,
            "command_start_to_finish",
            None,
        )
    try:
        start_ns, end_ns, count = load_request_window(request_log)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if returncode == 0:
            raise
        return (
            command_start_ns,
            command_end_ns,
            None,
            "command_start_to_failure",
            f"{type(error).__name__}: {error}",
        )
    return start_ns, end_ns, count, "request_first_submit_to_last_finish", None


def integrate_window_energy(
    samples: list[tuple[float, int, list[float]]],
    *,
    start_ns: int,
    end_ns: int,
    gpu_count: int,
) -> tuple[list[float], float]:
    """Trapezoid-integrate power after clipping to an exact wall-clock window."""
    if end_ns <= start_ns:
        raise ValueError("energy window must have positive duration")
    if len(samples) < 2:
        raise ValueError("at least two power samples are required")
    if samples[0][1] > start_ns or samples[-1][1] < end_ns:
        raise ValueError(
            "power samples do not bracket the request measurement window: "
            f"samples={samples[0][1]}..{samples[-1][1]}, "
            f"requests={start_ns}..{end_ns}"
        )
    energy = [0.0] * gpu_count
    covered_ns = 0
    for (_, left_ns, left_power), (_, right_ns, right_power) in zip(
        samples, samples[1:]
    ):
        if right_ns <= left_ns or right_ns <= start_ns or left_ns >= end_ns:
            continue
        clipped_left = max(left_ns, start_ns)
        clipped_right = min(right_ns, end_ns)
        if clipped_right <= clipped_left:
            continue
        segment_ns = right_ns - left_ns
        left_fraction = (clipped_left - left_ns) / segment_ns
        right_fraction = (clipped_right - left_ns) / segment_ns
        duration_s = (clipped_right - clipped_left) / 1e9
        for index in range(gpu_count):
            power_at_left = left_power[index] + left_fraction * (
                right_power[index] - left_power[index]
            )
            power_at_right = left_power[index] + right_fraction * (
                right_power[index] - left_power[index]
            )
            energy[index] += duration_s * (power_at_left + power_at_right) / 2
        covered_ns += clipped_right - clipped_left
    return energy, min(covered_ns / (end_ns - start_ns), 1.0)


def samples_bracketing_window(
    samples: list[tuple[float, int, list[float]]], start_ns: int, end_ns: int
) -> list[tuple[float, int, list[float]]]:
    """Keep in-window samples plus the adjacent samples needed for interpolation."""
    first = next(index for index, sample in enumerate(samples) if sample[1] >= start_ns)
    last = max(index for index, sample in enumerate(samples) if sample[1] <= end_ns)
    return samples[max(first - 1, 0) : min(last + 2, len(samples))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--interval-ms", type=float, default=10.0)
    parser.add_argument("--nvml-retries", type=int, default=2)
    parser.add_argument("--nvml-retry-delay-ms", type=float, default=1.0)
    parser.add_argument("--record-operating-state", action="store_true",
                        help="Record actual SM clocks, enforced caps, utilization and throttle reasons with power samples.")
    parser.add_argument(
        "--request-log",
        type=Path,
        help=(
            "Replay JSONL written by the measured command. When supplied, energy "
            "and duration are clipped to first submit_wall_ns through last "
            "finish_wall_ns."
        ),
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        help="Optional JSONL output for wall-clock-aligned per-GPU power samples.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("a command is required")
    if args.nvml_retries < 0 or args.nvml_retry_delay_ms < 0:
        raise ValueError("NVML retry settings must be nonnegative")
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in gpu_ids]
    samples: list[tuple[float, int, list[float]]] = []
    sample_errors: list[dict[str, object]] = []
    sample_retry_errors: list[dict[str, object]] = []
    sample_retry_count = 0
    operating_states = {}
    stop = threading.Event()

    def sample() -> bool:
        nonlocal sample_retry_count
        last_error: dict[str, object] | None = None
        for attempt in range(args.nvml_retries + 1):
            try:
                powers = []
                states = []
                for gpu_id, handle in zip(gpu_ids, handles, strict=True):
                    try:
                        powers.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                        if args.record_operating_state:
                            states.append({
                                "graphics_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                                "power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0,
                                "gpu_utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                                "throttle_reasons": pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
                            })
                    except pynvml.NVMLError as error:
                        last_error = {
                            "timestamp_ns": time.time_ns(),
                            "gpu": gpu_id,
                            "attempt": attempt + 1,
                            "error": type(error).__name__,
                            "message": str(error),
                        }
                        raise
            except pynvml.NVMLError:
                if attempt >= args.nvml_retries:
                    if last_error is not None:
                        sample_errors.append(last_error)
                    return False
                sample_retry_count += 1
                if last_error is not None:
                    sample_retry_errors.append(last_error)
                time.sleep(args.nvml_retry_delay_ms / 1000.0)
                continue
            timestamp_ns = time.time_ns()
            samples.append((time.perf_counter(), timestamp_ns, powers))
            if args.record_operating_state:
                operating_states[timestamp_ns] = states
            return True
        raise AssertionError("unreachable NVML retry state")

    def sampler() -> None:
        while not stop.is_set():
            sample()
            stop.wait(args.interval_ms / 1000.0)

    if not sample():
        time.sleep(args.interval_ms / 1000.0)
        if not sample():
            raise RuntimeError("NVML power sampling unavailable before command start")
    thread = threading.Thread(target=sampler, daemon=True)
    started_wall_ns = time.time_ns()
    started = time.perf_counter()
    thread.start()
    process = subprocess.run(command, check=False)
    stop.set()
    thread.join()
    command_finished = time.perf_counter()
    command_finished_wall_ns = time.time_ns()
    sample()
    command_duration_s = command_finished - started
    (
        measurement_start_ns,
        measurement_end_ns,
        request_count,
        measurement_source,
        request_window_error,
    ) = resolve_measurement_window(
        args.request_log,
        returncode=process.returncode,
        command_start_ns=started_wall_ns,
        command_end_ns=command_finished_wall_ns,
    )
    per_gpu_energy, sample_coverage = integrate_window_energy(
        samples,
        start_ns=measurement_start_ns,
        end_ns=measurement_end_ns,
        gpu_count=len(handles),
    )
    if measurement_source == "request_first_submit_to_last_finish":
        reported_samples = samples_bracketing_window(
            samples, measurement_start_ns, measurement_end_ns
        )
    else:
        reported_samples = samples
    duration_s = (measurement_end_ns - measurement_start_ns) / 1e9
    payload = {
        "schema_version": 2,
        "command": command,
        "returncode": process.returncode,
        "measurement_window_source": measurement_source,
        "request_log": str(args.request_log.resolve()) if args.request_log else None,
        "request_count": request_count,
        "request_window_error": request_window_error,
        "started_wall_ns": measurement_start_ns,
        "finished_wall_ns": measurement_end_ns,
        "duration_s": duration_s,
        "command_started_wall_ns": started_wall_ns,
        "command_finished_wall_ns": command_finished_wall_ns,
        "command_duration_s": command_duration_s,
        "sample_interval_ms": args.interval_ms,
        "sample_count": len(reported_samples),
        "command_sample_count": len(samples),
        "sample_time_coverage": sample_coverage,
        "sample_error_count": len(sample_errors),
        "sample_errors": sample_errors[:20],
        "sample_retry_count": sample_retry_count,
        "sample_retry_error_count": len(sample_retry_errors),
        "sample_retry_errors": sample_retry_errors[:20],
        "gpu_ids": gpu_ids,
        "per_gpu_energy_j": per_gpu_energy,
        "energy_j": sum(per_gpu_energy),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.samples_output is not None:
        args.samples_output.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_output.open("w", encoding="utf-8") as output:
            for _, timestamp_ns, powers in reported_samples:
                output.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "timestamp_ns": timestamp_ns,
                            "gpu_ids": gpu_ids,
                            "power_w": powers,
                            **({"operating_state": operating_states[timestamp_ns]}
                               if args.record_operating_state else {}),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    print(json.dumps(payload, indent=2))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()

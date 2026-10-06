#!/usr/bin/env python3
"""Trace replay client with causal request-lifecycle events for DVFS control."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_trace import load_manifest, request_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18000/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=100.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--progress-event-interval-ms", type=float, default=50.0)
    parser.add_argument(
        "--progress-event-mode",
        choices=("per-request", "batched"),
        default="per-request",
    )
    parser.add_argument("--submit-signal-fifo", type=Path, default=None)
    parser.add_argument("--routing-active-marker", type=Path, default=None)
    return parser.parse_args()


@contextmanager
def routing_collection_window(marker: Path | None) -> Iterator[None]:
    if marker is None:
        yield
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=False)
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


class EventWriter:
    def __init__(
        self,
        path: Path,
        progress_mode: str = "per-request",
        progress_interval_ms: float = 50.0,
        submit_signal_fifo: Path | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        self.stream = path.open("a", encoding="utf-8", buffering=1)
        self.lock = asyncio.Lock()
        self.sequence = 0
        self.progress_mode = progress_mode
        self.progress_interval_s = progress_interval_ms / 1000.0
        self.last_batch_flush_time = time.perf_counter()
        self.pending_progress: dict[str, int] = {}
        self.signal_fd: int | None = None
        if submit_signal_fifo is not None:
            self.signal_fd = os.open(submit_signal_fifo, os.O_WRONLY | os.O_NONBLOCK)

    def _write_locked(self, event: str, fields: dict[str, Any]) -> None:
        self.sequence += 1
        row = {
            "schema_version": 1,
            "sequence": self.sequence,
            "event": event,
            "wall_ns": time.time_ns(),
            **fields,
        }
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.stream.flush()
        if event == "submit" and self.signal_fd is not None:
            os.write(self.signal_fd, b"S")

    async def emit(self, event: str, **fields: Any) -> None:
        async with self.lock:
            self._write_locked(event, fields)

    async def progress(self, request_id: str, output_chunks: int) -> None:
        if self.progress_mode == "per-request":
            await self.emit(
                "progress",
                request_id=request_id,
                output_chunks=output_chunks,
            )
            return
        async with self.lock:
            self.pending_progress[request_id] = output_chunks
            now = time.perf_counter()
            if now - self.last_batch_flush_time >= self.progress_interval_s:
                self._flush_progress_locked()

    def _flush_progress_locked(self, request_id: str | None = None) -> None:
        if request_id is None:
            selected = dict(self.pending_progress)
            self.pending_progress.clear()
        elif request_id in self.pending_progress:
            selected = {request_id: self.pending_progress.pop(request_id)}
        else:
            selected = {}
        if not selected:
            return
        self._write_locked(
            "progress_batch",
            {
                "requests": [
                    {"request_id": key, "output_chunks": selected[key]}
                    for key in sorted(selected)
                ]
            },
        )
        self.last_batch_flush_time = time.perf_counter()

    async def flush_progress(self, request_id: str | None = None) -> None:
        async with self.lock:
            self._flush_progress_locked(request_id)

    def close(self) -> None:
        if self.pending_progress:
            self._flush_progress_locked()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if self.signal_fd is not None:
            os.close(self.signal_fd)


async def replay_one(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    record: dict[str, Any],
    experiment_start: float,
    events: EventWriter,
) -> dict[str, Any]:
    scheduled_s = float(record["arrival_s"]) * args.time_scale
    await asyncio.sleep(max(0.0, experiment_start + scheduled_s - time.perf_counter()))
    submit_time = time.perf_counter()
    submit_wall_ns = time.time_ns()
    requested_output = int(record["output_tokens"])
    if args.max_output_tokens is not None:
        requested_output = min(requested_output, args.max_output_tokens)
    request_identity = f"ecodep-{record['request_id']}"
    await events.emit(
        "submit",
        request_id=request_identity,
        input_tokens=int(record["input_tokens"]),
        requested_output_tokens=requested_output,
    )
    payload = {
        "model": args.model,
        "prompt": request_prompt(record),
        "max_tokens": requested_output,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    first_token_time: float | None = None
    first_token_wall_ns: int | None = None
    last_progress_time: float | None = None
    output_chunks = 0
    usage: dict[str, Any] = {}
    error: str | None = None
    status = 0
    try:
        async with session.post(
            args.endpoint,
            json=payload,
            headers={"X-Request-Id": request_identity},
        ) as response:
            status = response.status
            if response.status != 200:
                error = await response.text()
            else:
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices or not choices[0].get("text"):
                        continue
                    output_chunks += 1
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        first_token_wall_ns = time.time_ns()
                        last_progress_time = now
                        await events.emit(
                            "first_token",
                            request_id=request_identity,
                            output_chunks=output_chunks,
                        )
                    elif args.progress_event_mode == "batched":
                        await events.progress(request_identity, output_chunks)
                    elif (
                        last_progress_time is None
                        or (now - last_progress_time) * 1000.0
                        >= args.progress_event_interval_ms
                    ):
                        last_progress_time = now
                        await events.progress(request_identity, output_chunks)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finish_time = time.perf_counter()
    finish_wall_ns = time.time_ns()
    if args.progress_event_mode == "batched":
        await events.flush_progress(request_identity)
    await events.emit(
        "finish",
        request_id=request_identity,
        error=error,
        output_chunks=output_chunks,
    )
    output_tokens = int(usage.get("completion_tokens", output_chunks))
    ttft_ms = (
        (first_token_time - submit_time) * 1000.0
        if first_token_time is not None
        else math.inf
    )
    tpot_ms = (
        (finish_time - first_token_time) * 1000.0 / max(output_tokens - 1, 1)
        if first_token_time is not None and output_tokens > 0
        else math.inf
    )
    slo_good = error is None and ttft_ms <= args.ttft_slo_ms and tpot_ms <= args.tpot_slo_ms
    return {
        **record,
        "scheduled_s": scheduled_s,
        "submit_s": submit_time - experiment_start,
        "submit_wall_ns": submit_wall_ns,
        "first_token_wall_ns": first_token_wall_ns,
        "finish_wall_ns": finish_wall_ns,
        "queue_lag_ms": (submit_time - experiment_start - scheduled_s) * 1000.0,
        "http_status": status,
        "requested_output_tokens": requested_output,
        "runtime_request_id": request_identity,
        "actual_output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": (finish_time - submit_time) * 1000.0,
        "slo_good": slo_good,
        "error": error,
    }


async def run(args: argparse.Namespace) -> int:
    records = load_manifest(args.manifest, args.limit)
    events = EventWriter(
        args.events,
        progress_mode=args.progress_event_mode,
        progress_interval_ms=args.progress_event_interval_ms,
        submit_signal_fifo=args.submit_signal_fifo,
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    results: list[dict[str, Any]] = []
    try:
        await events.emit("replay_start")
        with routing_collection_window(args.routing_active_marker):
            experiment_start = time.perf_counter() + 1.0
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = [
                    asyncio.create_task(
                        replay_one(session, args, record, experiment_start, events)
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

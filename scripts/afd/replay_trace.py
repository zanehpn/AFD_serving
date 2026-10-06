#!/usr/bin/env python3
"""Replay an Azure length/arrival trace against an OpenAI completions endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18000/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=100.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--routing-active-marker", type=Path, default=None)
    parser.add_argument("--record-output-token-ids", action="store_true")
    return parser.parse_args()


@contextmanager
def routing_collection_window(marker: Path | None) -> Iterator[None]:
    """Expose an explicit sidecar collection window owned by this replay."""
    if marker is None:
        yield
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=False)
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def make_input(request_id: int, length: int) -> list[int]:
    """Create deterministic valid token IDs while preserving public trace length."""
    return [1000 + ((request_id * 131 + position * 17) % 28000) for position in range(length)]


def request_seed(request_id: Any) -> int:
    """Map numeric or descriptive request IDs to a stable prompt seed."""
    try:
        return int(request_id)
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(request_id).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")


def request_prompt(record: dict[str, Any]) -> str | list[int]:
    """Preserve supplied task content; use shape-only tokens only when necessary."""
    if isinstance(record.get("prompt"), str):
        return str(record["prompt"])
    if isinstance(record.get("token_ids"), list):
        tokens = [int(value) for value in record["token_ids"]]
        if len(tokens) != int(record["input_tokens"]):
            raise ValueError("token_ids length differs from input_tokens")
        return tokens
    return make_input(request_seed(record["request_id"]), int(record["input_tokens"]))


def load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if limit is not None and len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


async def replay_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    record: dict[str, Any],
    experiment_start: float,
    time_scale: float,
    max_output_tokens: int | None,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    record_output_token_ids: bool = False,
) -> dict[str, Any]:
    scheduled_s = float(record["arrival_s"]) * time_scale
    await asyncio.sleep(max(0.0, experiment_start + scheduled_s - time.perf_counter()))
    submit_time = time.perf_counter()
    submit_wall_ns = time.time_ns()
    requested_output = int(record["output_tokens"])
    if max_output_tokens is not None:
        requested_output = min(requested_output, max_output_tokens)
    payload = {
        "model": model,
        "prompt": request_prompt(record),
        "max_tokens": requested_output,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    if record_output_token_ids:
        payload.update(return_token_ids=True, seed=0)
    output_token_ids = []
    token_arrival_s = []
    token_event_sizes = []
    first_token_time: float | None = None
    first_token_wall_ns: int | None = None
    output_chunks = 0
    usage: dict[str, Any] = {}
    error: str | None = None
    status = 0
    try:
        request_identity = f"ecodep-{record['request_id']}"
        async with session.post(
            endpoint,
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
                    if record_output_token_ids and choices:
                        delta = choices[0].get('token_ids')
                        if delta is not None:
                            if not isinstance(delta, list) or any(type(t) is not int or t < 0 for t in delta):
                                raise ValueError('Malformed streamed output token IDs')
                            output_token_ids.extend(delta)
                            if delta:
                                arrival = time.perf_counter()
                                token_arrival_s.extend([arrival] * len(delta))
                                token_event_sizes.append(len(delta))
                    if choices and (choices[0].get("text") or (record_output_token_ids and choices[0].get('token_ids'))):
                        output_chunks += 1
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            first_token_wall_ns = time.time_ns()
    except Exception as exc:  # Preserve failures in request-level logs.
        error = f"{type(exc).__name__}: {exc}"
    finish_time = time.perf_counter()
    finish_wall_ns = time.time_ns()
    output_tokens = int(usage.get("completion_tokens", output_chunks))
    if record_output_token_ids and len(output_token_ids) != output_tokens:
        error = error or 'Missing or incomplete streamed output token IDs'
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
    slo_good = error is None and ttft_ms <= ttft_slo_ms and tpot_ms <= tpot_slo_ms
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
        "runtime_request_id": f"ecodep-{record['request_id']}",
        "actual_output_tokens": output_tokens,
        **({'actual_output_token_ids': output_token_ids} if record_output_token_ids else {}),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        **({"token_arrival_s": [t-submit_time for t in token_arrival_s],
            "tbt_ms": [(b-a)*1000 for a, b in zip(token_arrival_s, token_arrival_s[1:])],
            "token_event_sizes": token_event_sizes,
            "tbt_observation": "client SSE token-ID arrival; tokens in one event share a timestamp"}
           if record_output_token_ids else {}),
        "e2e_ms": (finish_time - submit_time) * 1000.0,
        "slo_good": slo_good,
        "error": error,
    }


async def run(args: argparse.Namespace) -> int:
    records = load_manifest(args.manifest, args.limit)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    with routing_collection_window(args.routing_active_marker):
        experiment_start = time.perf_counter() + 1.0
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                asyncio.create_task(
                    replay_one(
                        session,
                        args.endpoint,
                        args.model,
                        record,
                        experiment_start,
                        args.time_scale,
                        args.max_output_tokens,
                        args.ttft_slo_ms,
                        args.tpot_slo_ms,
                        getattr(args, 'record_output_token_ids', False),
                    )
                )
                for record in records
            ]
            results = await asyncio.gather(*tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for result in sorted(results, key=lambda item: item["request_id"]):
            output.write(json.dumps(result, separators=(",", ":"), allow_nan=True))
            output.write("\n")
    return 1 if any(result["error"] is not None for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))

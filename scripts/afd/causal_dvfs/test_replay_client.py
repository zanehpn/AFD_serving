#!/usr/bin/env python3
"""Unit checks for submit wakeups and batched progress events."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from replay_client import EventWriter


async def check() -> None:
    with tempfile.TemporaryDirectory(prefix="causal-dvfs-events-") as directory:
        root = Path(directory)
        fifo = root / "submit.fifo"
        events_path = root / "events.jsonl"
        os.mkfifo(fifo, 0o600)
        read_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        keepalive_fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        writer = EventWriter(
            events_path,
            progress_mode="batched",
            progress_interval_ms=250,
            submit_signal_fifo=fifo,
        )
        try:
            await writer.emit("submit", request_id="r0", input_tokens=64)
            assert os.read(read_fd, 1) == b"S"
            await writer.progress("r0", 2)
            await writer.progress("r0", 3)
            await writer.progress("r1", 1)
            await writer.flush_progress()
        finally:
            writer.close()
            os.close(keepalive_fd)
            os.close(read_fd)

        rows = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert [row["event"] for row in rows] == ["submit", "progress_batch"]
        assert rows[1]["requests"] == [
            {"request_id": "r0", "output_chunks": 3},
            {"request_id": "r1", "output_chunks": 1},
        ]


asyncio.run(check())
print("causal DVFS replay client tests passed")

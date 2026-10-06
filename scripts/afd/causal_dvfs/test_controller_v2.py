#!/usr/bin/env python3
"""Unit checks for the backlog-driven causal DVFS v2 policy."""

from __future__ import annotations

import json
from pathlib import Path

from controller import CausalController


config = json.loads((Path(__file__).parent / "safe-v2.json").read_text())
t0 = 1_000_000_000

controller = CausalController(config, "dynamic-ae")
assert controller.current == {"attention": "eco", "expert": "normal"}
targets, signals = controller.desired(t0)
assert targets == {"attention": "eco", "expert": "normal"}
assert signals["outstanding"] == 0

# A submit immediately boosts Attention, while Expert remains at its 1290 MHz default.
controller.consume(
    {"event": "submit", "wall_ns": t0, "request_id": "r0", "input_tokens": 1024}
)
targets, signals = controller.desired(t0 + 1)
assert targets == {"attention": "boost", "expert": "normal"}
assert signals["attention_policy_reason"] == "prefill_present"
assert signals["expert_policy_reason"] == "expert_default_normal"

# Once prefill completes, Attention leaves boost and batched progress is accepted.
controller.consume(
    {
        "event": "progress_batch",
        "wall_ns": t0 + 10_000_000,
        "requests": [{"request_id": "r0", "output_chunks": 1}],
    }
)
targets, _ = controller.desired(t0 + 20_000_000)
assert targets == {"attention": "normal", "expert": "normal"}

# A large but stable decode population must not boost Expert by absolute count.
stable = CausalController(config, "dynamic-ae")
stable.consume({"event": "replay_start", "wall_ns": t0})
for request in range(20):
    event_ns = t0 + (request + 1) * 1_000_000
    request_id = f"stable-{request}"
    stable.consume(
        {
            "event": "submit",
            "wall_ns": event_ns,
            "request_id": request_id,
            "input_tokens": 64,
        }
    )
    stable.consume(
        {"event": "first_token", "wall_ns": event_ns + 1, "request_id": request_id}
    )
refresh_ns = t0 + 2_000_000_000
stable.consume(
    {
        "event": "progress_batch",
        "wall_ns": refresh_ns,
        "requests": [
            {"request_id": f"stable-{request}", "output_chunks": 8}
            for request in range(20)
        ],
    }
)
targets, signals = stable.desired(refresh_ns + 1)
assert targets["expert"] == "normal"
assert signals["queue_growth_rps"] == 0.0

# Sustained positive queue growth under continuing arrivals boosts Expert.
growing = CausalController(config, "dynamic-ae")
growing.consume({"event": "replay_start", "wall_ns": t0})
for request in range(4):
    event_ns = t0 + (request + 1) * 100_000_000
    request_id = f"growing-{request}"
    growing.consume(
        {
            "event": "submit",
            "wall_ns": event_ns,
            "request_id": request_id,
            "input_tokens": 64,
        }
    )
    growing.consume(
        {"event": "first_token", "wall_ns": event_ns + 1, "request_id": request_id}
    )
targets, _ = growing.desired(t0 + 400_000_001)
assert targets["expert"] == "normal"
growing.consume(
    {
        "event": "progress_batch",
        "wall_ns": t0 + 990_000_000,
        "requests": [
            {"request_id": f"growing-{request}", "output_chunks": 4}
            for request in range(4)
        ],
    }
)
growing.consume(
    {
        "event": "submit",
        "wall_ns": t0 + 1_000_000_000,
        "request_id": "growing-4",
        "input_tokens": 64,
    }
)
growing.consume(
    {"event": "first_token", "wall_ns": t0 + 1_000_000_001, "request_id": "growing-4"}
)
targets, signals = growing.desired(t0 + 1_000_000_002)
assert targets["expert"] == "boost"
assert signals["expert_policy_reason"] == "sustained_queue_growth"

# A decode progress stall is an independent Expert boost trigger.
stall = CausalController(config, "dynamic-ae")
stall.consume(
    {"event": "submit", "wall_ns": t0, "request_id": "stall", "input_tokens": 64}
)
stall.consume({"event": "first_token", "wall_ns": t0 + 1, "request_id": "stall"})
targets, signals = stall.desired(t0 + 1_600_000_000)
assert targets["expert"] == "boost"
assert signals["expert_policy_reason"] == "decode_stall"

print("causal DVFS v2 controller tests passed")

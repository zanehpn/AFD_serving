#!/usr/bin/env python3
"""Unit checks for deterministic causal DVFS decisions and hysteresis."""

from __future__ import annotations

import json
from pathlib import Path

from controller import CausalController


config = json.loads((Path(__file__).parent / "safe-v1.json").read_text())
controller = CausalController(config, "dynamic-ae")
t0 = 1_000_000_000

targets, signals = controller.desired(t0)
assert targets == {"attention": "eco", "expert": "eco"}
assert signals["outstanding"] == 0

controller.consume(
    {
        "event": "submit",
        "wall_ns": t0,
        "request_id": "r0",
        "input_tokens": 1024,
    }
)
targets, _ = controller.desired(t0 + 1)
assert targets["attention"] == "boost"
assert targets["expert"] == "normal"

controller.consume(
    {"event": "first_token", "wall_ns": t0 + 10_000_000, "request_id": "r0"}
)
targets, _ = controller.desired(t0 + 20_000_000)
assert targets == {"attention": "eco", "expert": "eco"}

for request in range(1, 12):
    controller.consume(
        {
            "event": "submit",
            "wall_ns": t0 + 30_000_000 + request,
            "request_id": f"r{request}",
            "input_tokens": 64,
        }
    )
targets, _ = controller.desired(t0 + 40_000_000)
assert targets == {"attention": "boost", "expert": "boost"}

static = CausalController(config, "static-e1290")
targets, _ = static.desired(t0)
assert targets == {"attention": "boost", "expert": "normal"}

allowed, reason = controller.transition_allowed("attention", "eco", t0)
assert not allowed and reason == "downshift_candidate_started"
allowed, reason = controller.transition_allowed("attention", "eco", t0 + 2_000_000_000)
assert allowed and reason == "delayed_downshift"

print("causal DVFS controller tests passed")

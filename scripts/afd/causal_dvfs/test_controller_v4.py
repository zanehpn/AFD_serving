#!/usr/bin/env python3
"""Unit checks for the joint frequency/power-cap policy."""

from __future__ import annotations

import json
from pathlib import Path

from controller import CausalController


config = json.loads((Path(__file__).parent / "safe-v4-joint.json").read_text())
t0 = 1_000_000_000

assert config["predictor"]["future_trace_access"] is False
assert config["predictor"]["static_dse_online"] is False
assert config["power_caps_w"]["attention"]["guard"] == 400
assert config["power_caps_w"]["expert"]["guard"] == 400

baseline = CausalController(config, "placement-max")
assert baseline.current == {"attention": "guard", "expert": "guard"}
targets, _ = baseline.desired(t0)
assert targets == {"attention": "guard", "expert": "guard"}

candidate = CausalController(config, "dynamic-ae")
targets, _ = candidate.desired(t0)
assert targets == {"attention": "eco", "expert": "eco"}
candidate.consume(
    {
        "event": "submit",
        "wall_ns": t0,
        "request_id": "r0",
        "input_tokens": 911,
        "requested_output_tokens": 128,
    }
)
targets, _ = candidate.desired(t0 + 1)
assert targets["attention"] == "normal"

# Missing lifecycle updates while work is outstanding must select the uncapped
# 1410 MHz/400 W guard state, not the energy-oriented 1410 MHz/350 W boost.
targets, signals = candidate.desired(t0 + 1_100_000_000)
assert targets == {"attention": "guard", "expert": "guard"}
assert signals["safety_fallback"] == 1

print("causal DVFS v4 joint-actuator tests passed")

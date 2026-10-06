#!/usr/bin/env python3
"""Unit checks for the leakage-free AFD stage-capacity predictor."""

from __future__ import annotations

import json
from pathlib import Path

from controller import CausalController


config = json.loads((Path(__file__).parent / "safe-v3-predictor.json").read_text())
t0 = 1_000_000_000

assert config["predictor"]["future_trace_access"] is False
assert config["predictor"]["static_dse_online"] is False
assert "evaluation_trace_file" in config["predictor"]["forbidden_inputs"]

controller = CausalController(config, "dynamic-ae")
targets, signals = controller.desired(t0)
assert targets == {"attention": "eco", "expert": "eco"}
assert signals["outstanding"] == 0

# A request is unknown until submit.  Once submitted, its current input size is
# legal and triggers an event-driven Attention reserve without future output use.
controller.consume(
    {
        "event": "submit",
        "wall_ns": t0 + 1,
        "request_id": "r0",
        "input_tokens": 911,
        "requested_output_tokens": 128,
    }
)
targets, signals = controller.desired(t0 + 2)
assert targets["attention"] == "normal"
assert signals["attention_policy_reason"] == "v3_submit_prefill_reserve"
assert signals["submitted_input_tokens_ewma"] == 911

# A causal 2 RPS stream is outside the guarded eco capacity and goes fail-high.
loaded = CausalController(config, "dynamic-ae")
for index in range(6):
    event_ns = t0 + index * 500_000_000
    loaded.consume(
        {
            "event": "submit",
            "wall_ns": event_ns,
            "request_id": f"r{index}",
            "input_tokens": 911,
            "requested_output_tokens": 112,
        }
    )
    loaded.consume(
        {"event": "first_token", "wall_ns": event_ns + 1, "request_id": f"r{index}"}
    )
targets, signals = loaded.desired(t0 + 2_500_000_002)
assert targets == {"attention": "boost", "expert": "boost"}
assert signals["attention_predicted_reference_work_per_s"] > 2.0

# Old requests and missing progress independently override the model.
stall = CausalController(config, "dynamic-ae")
stall.consume(
    {
        "event": "submit",
        "wall_ns": t0,
        "request_id": "stall",
        "input_tokens": 64,
        "requested_output_tokens": 128,
    }
)
stall.consume({"event": "first_token", "wall_ns": t0 + 1, "request_id": "stall"})
targets, signals = stall.desired(t0 + 1_100_000_000)
assert targets == {"attention": "boost", "expert": "boost"}
assert signals["safety_fallback"] == 1

print("causal DVFS v3 predictor tests passed")

#!/usr/bin/env python3
"""Unit checks for causal routing and measured operating-point selection."""

from __future__ import annotations

import copy

from controller import CausalControllerV5


def point(state: str, frequency: int, cap: int, capacity: float) -> dict:
    return {
        "state": state,
        "frequency_mhz": frequency,
        "power_cap_w": cap,
        "capacity_reference_work_per_s": capacity,
    }


power_model = {
    "states": {
        "low": {"idle_intercept_w": 50.0, "dynamic_slope_w": 50.0, "power_cap_w": 200},
        "high": {"idle_intercept_w": 80.0, "dynamic_slope_w": 100.0, "power_cap_w": 400},
    }
}
config = {
    "policy_version": 5,
    "initial_states": {"attention": "low", "expert": "low"},
    "signals": {
        "queue_growth_window_ms": 1000,
        "arrival_ewma_alpha": 0.2,
        "arrival_ewma_decay_ms": 2000,
        "max_instantaneous_arrival_rps": 8,
        "work_ewma_alpha": 0.2,
        "routing_ewma_alpha": 0.25,
        "routing_stale_ms": 750,
    },
    "predictor": {
        "reference_workload": {"input_tokens_mean": 100, "output_tokens_mean": 20},
        "work_weights": {
            "attention": {"prefill": 0.8, "decode": 0.2},
            "expert": {"prefill": 0.3, "decode": 0.7},
        },
        "routing_model": {
            "reference_imbalance_median": 1.0,
            "calibration_imbalance_p95": 1.5,
        },
        "risk_margin": 1.0,
        "prediction_horizon_ms": 500,
        "decode_progress_ewma_alpha": 0.25,
        "queue_growth_gain": 0.0,
        "active_prefill_reserve": 0.0,
        "utilization_limit": 0.75,
        "role_models": {
            role: {
                "guard_state": "high",
                "power_model": power_model,
                "operating_points": [
                    point("low", 1050, 200, 1.0),
                    point("high", 1410, 400, 4.0),
                ],
            }
            for role in ("attention", "expert")
        },
        "joint_operating_points": [
            {
                "id": "low-pair",
                "attention_state": "low",
                "expert_state": "low",
                "calibration_latency_eligible": True,
            },
            {
                "id": "c00-max",
                "attention_state": "high",
                "expert_state": "high",
                "calibration_latency_eligible": True,
            },
        ],
        "urgent_thresholds": {
            role: {"boost_prefill_age_ms": 350, "boost_progress_gap_ms": 300}
            for role in ("attention", "expert")
        },
    },
    "safety": {
        "downshift_hold_ms": {"attention": 500, "expert": 750},
        "minimum_transition_interval_ms": {"attention": 250, "expert": 500},
        "event_stale_ms": 1000,
    },
}

t0 = 1_000_000_000
controller = CausalControllerV5(config, "dynamic-ae")
controller.consume(
    {
        "event": "submit",
        "wall_ns": t0,
        "request_id": "r0",
        "input_tokens": 100,
        "requested_output_tokens": 20,
    }
)
update = controller.consume_routing(
    [
        {
            "timestamp_ns": t0,
            "tokens": 10,
            "domain_counts": [16, 4],
        }
    ],
    t0 + 1_000_000,
)
assert update is not None
assert update["imbalance"] == 1.6
signals = controller.signals(t0 + 2_000_000)
assert signals["routing_imbalance_source"] == "online_observed"
assert signals["routing_imbalance"] == 1.6

stale = controller.signals(t0 + 2_000_000_000)
assert stale["routing_imbalance_source"] == "calibration_p95_stale_fallback"
assert stale["routing_imbalance"] == 1.5

controller.consume(
    {
        "event": "submit",
        "wall_ns": t0 + 100_000_000,
        "request_id": "r1",
        "input_tokens": 100,
        "requested_output_tokens": 20,
    }
)
targets, predicted = controller.desired(t0 + 100_000_001)
assert targets["attention"] == "high"
assert targets["expert"] == "high"
assert predicted["expert_routing_factor"] > 1.0

# Calibration latency pruning is fail-closed even when a low point has capacity.
slo_config = copy.deepcopy(config)
slo_config["predictor"]["joint_operating_points"][0][
    "calibration_latency_eligible"
] = False
slo_controller = CausalControllerV5(slo_config, "dynamic-ae")
slo_targets, _ = slo_controller.desired(t0)
assert slo_targets == {"attention": "high", "expert": "high"}

# The combined latency/throughput gate is fail-closed when present.
throughput_config = copy.deepcopy(config)
throughput_config["predictor"]["joint_operating_points"][0][
    "calibration_slo_eligible"
] = False
throughput_controller = CausalControllerV5(throughput_config, "dynamic-ae")
throughput_targets, _ = throughput_controller.desired(t0)
assert throughput_targets == {"attention": "high", "expert": "high"}

# Progress events contribute decode work in reference-requests/second.
decode_controller = CausalControllerV5(copy.deepcopy(config), "dynamic-ae")
decode_controller.consume(
    {
        "event": "submit",
        "wall_ns": t0,
        "request_id": "decode",
        "input_tokens": 100,
        "requested_output_tokens": 20,
    }
)
decode_controller.consume(
    {
        "event": "first_token",
        "wall_ns": t0 + 100_000_000,
        "request_id": "decode",
        "output_chunks": 1,
    }
)
decode_controller.consume(
    {
        "event": "progress",
        "wall_ns": t0 + 200_000_000,
        "request_id": "decode",
        "output_chunks": 6,
    }
)
decode_signals = decode_controller.signals(t0 + 200_000_001)
assert decode_signals["decode_output_token_rate_ewma"] == 50.0
assert decode_controller._role_demand_v5("expert", decode_signals) > 0.0

print("causal DVFS v5 controller tests passed")

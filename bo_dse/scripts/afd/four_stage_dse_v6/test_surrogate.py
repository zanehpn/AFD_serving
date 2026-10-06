from __future__ import annotations

import copy

from .surrogate import expand_candidate_space, fit_elasticity_surrogate
from .test_model import WORKLOAD, _candidate


def _anchors() -> list[dict]:
    guard = _candidate("guard")
    guard["topology"].update(
        {
            "attention_dp": 2,
            "attention_tp": 1,
            "expert_ep": 2,
            "expert_tp": 1,
        }
    )
    guard["knobs"].update(
        {
            "attention_mhz": 1410,
            "expert_mhz": 1410,
            "attention_power_w": 400,
            "expert_power_w": 400,
        }
    )
    guard["validation_status"] = "physically_measured_anchor"
    lower_expert = copy.deepcopy(guard)
    lower_expert["id"] = "e1290"
    lower_expert["knobs"]["expert_mhz"] = 1290
    lower_expert["stage_models"]["ffn_compute"]["intercept_ms"] = 8.6
    return [guard, lower_expert]


def test_sparse_anchor_surrogate_expands_unmeasured_points() -> None:
    anchors = _anchors()
    surrogate = fit_elasticity_surrogate(
        anchors, WORKLOAD, guard_candidate_id="guard"
    )
    specification = {
        "topologies": [
            {
                "attention_gpus": 2,
                "expert_gpus": 2,
                "attention_dp": 2,
                "attention_tp": 1,
                "expert_ep": 2,
                "expert_tp": 1,
                "memory_feasible": True,
            }
        ],
        "attention_frequencies_mhz": [1410],
        "expert_frequencies_mhz": [1290, 1410],
        "attention_power_caps_w": [400],
        "expert_power_caps_w": [300, 400],
        "microbatches": [2],
    }
    expanded = expand_candidate_space(anchors, surrogate, specification, WORKLOAD)
    assert len(expanded) == 4
    assert sum(row["validation_status"] == "physically_measured_anchor" for row in expanded) == 2
    assert sum(row["validation_status"] == "surrogate_unvalidated" for row in expanded) == 2


def test_less_expert_frequency_predicts_no_faster_ffn() -> None:
    anchors = _anchors()
    surrogate = fit_elasticity_surrogate(
        anchors, WORKLOAD, guard_candidate_id="guard"
    )
    assert surrogate["stage_fits"]["ffn_compute"]["elasticity"]["expert_mhz"] >= 0

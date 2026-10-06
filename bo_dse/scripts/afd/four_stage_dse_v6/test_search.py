from __future__ import annotations

import copy

from .search import (
    bottleneck_guided_search,
    criticality_weights,
    recommended_controls,
)
from .test_model import WORKLOAD, _candidate


def test_criticality_exposes_pipeline_asymmetry() -> None:
    weights = criticality_weights(
        {
            "attention_compute": 10.0,
            "a2f_dispatch": 2.0,
            "ffn_compute": 8.0,
            "f2a_combine": 1.0,
        },
        microbatches=4,
        layers=28,
    )
    assert weights["attention_compute"] == 4.0
    assert weights["ffn_compute"] == 1.0 / 28.0


def test_control_mapping_is_stage_specific() -> None:
    assert "attention_mhz" in recommended_controls("attention_compute")
    assert "expert_ep" in recommended_controls("ffn_compute")
    assert "microbatches" in recommended_controls("a2f_dispatch")


def test_search_path_bound_does_not_change_model_selected_optimum() -> None:
    guard = _candidate("guard")
    guard["knobs"] = {"attention_mhz": 1410, "expert_mhz": 1410}
    middle = copy.deepcopy(guard)
    middle["id"] = "middle"
    middle["knobs"]["expert_mhz"] = 1290
    middle["power_model"]["dynamic_slope_w"] = 500.0
    low = copy.deepcopy(middle)
    low["id"] = "low"
    low["knobs"]["attention_mhz"] = 1290
    low["power_model"]["dynamic_slope_w"] = 400.0
    result = bottleneck_guided_search(
        [guard, middle, low],
        [WORKLOAD],
        guard_candidate_id="guard",
        max_search_path_points=2,
    )
    assert result.search_path == ("guard", "middle")
    assert result.predicted_best_by_workload == ("low",)
    assert result.evaluated_predictions == 3


def test_search_never_shortlists_heldout_profile_as_optimum() -> None:
    guard = _candidate("guard")
    leaked = _candidate("leaked")
    leaked["selection_split"] = "heldout"
    leaked["power_model"]["dynamic_slope_w"] = 1.0
    result = bottleneck_guided_search(
        [guard, leaked],
        [WORKLOAD],
        guard_candidate_id="guard",
        max_search_path_points=1,
    )
    assert result.predicted_best_by_workload == ("guard",)

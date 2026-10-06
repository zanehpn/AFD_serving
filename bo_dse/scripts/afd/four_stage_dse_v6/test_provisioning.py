import copy

import numpy as np
import pytest

from static_dse.provisioning import cycle_time, recommend_ratios, workload_moments


def profile():
    return {"selection_split": "calibration", "ratio_scope": "attention_workers_per_ffn_service_group",
            "batch_size_per_attention": 16, "ratios": list(range(1, 8)),
            "latency_coefficients_ms": {"attention": {"slope": .01, "intercept": 1.},
                                        "ffn": {"slope": .2, "intercept": 2.},
                                        "communication_roundtrip": {"slope": .02, "intercept": .1}}}


def requests():
    return [{"source_index": i, "prompt_tokens": p, "decode_steps": d}
            for i, (p, d) in enumerate([(10, 2), (40, 8), (10, 1), (30, 12)])]


def power_model(attention=200., ffn=200.):
    return {"type": "constant_role_power_v1", "selection_split": "calibration",
            "assumption": "ratio_independent_cycle_average_role_power",
            "attention_worker_w": attention, "ffn_service_group_w": ffn}


def test_equal_role_power_makes_energy_inverse_to_throughput_per_instance():
    p = profile()
    p["power_model"] = power_model()
    result = recommend_ratios(p, requests())
    energy = result["energy_advice"]
    assert energy["recommendation"] == result["recommendation"]
    throughput = {row["ratio"]: row["tokens_per_second_per_instance"] for row in result["ranked_ratios"]}
    for row in energy["ranked_ratios"]:
        for method, joules in row["predicted_j_per_decode_token"].items():
            assert joules == pytest.approx(200 / throughput[row["ratio"]][method])


def test_unequal_role_power_changes_stationary_ratio_and_discrete_recommendation():
    p = profile()
    p["latency_coefficients_ms"] = {"attention": {"slope": 0., "intercept": 1.},
                                     "ffn": {"slope": 1 / 16, "intercept": 9.},
                                     "communication_roundtrip": {"slope": 0., "intercept": .1}}
    p["power_model"] = power_model(100., 400.)
    result = recommend_ratios(p, requests())
    assert result["recommendation"]["mean_field"] == 3
    assert result["energy_advice"]["recommendation"] == {"mean_field": 6, "gaussian_barrier": 6}
    assert 6. in result["energy_advice"]["mean_field_candidate_ratios"]
    best = result["energy_advice"]["ranked_ratios"][0]
    assert best["predicted_j_per_decode_token"]["mean_field"] == pytest.approx(.15625)
    assert result["execution_eligibility_changed"] is False
    assert result["slo_feasibility_established"] is False


def test_no_role_power_keeps_energy_advice_unavailable():
    assert recommend_ratios(profile(), requests())["energy_advice"] == {
        "available": False, "reason": "missing_calibrated_role_power"}


def test_constant_latency_energy_optimum_is_feasible_upper_boundary():
    p = profile()
    p["latency_coefficients_ms"] = {key: {"slope": 0., "intercept": 2.}
                                     for key in p["latency_coefficients_ms"]}
    p["power_model"] = power_model()
    result = recommend_ratios(p, requests())["energy_advice"]
    assert result["mean_field_candidate_ratios"] == [1., 7.]
    assert result["recommendation"] == {"mean_field": 7, "gaussian_barrier": 7}


@pytest.mark.parametrize("mutation", ["heldout", "cap", "varying_power", "missing", "zero", "negative", "nan", "boolean"])
def test_invalid_role_power_contracts_fail(mutation):
    p = profile()
    p["power_model"] = power_model()
    power = p["power_model"]
    if mutation == "heldout":
        power["selection_split"] = "heldout"
    elif mutation == "cap":
        power["type"] = "power_cap_w"
    elif mutation == "varying_power":
        power["assumption"] = "power_changes_with_ratio"
    elif mutation == "missing":
        power.pop("ffn_service_group_w")
    else:
        power["attention_worker_w"] = {"zero": 0, "negative": -1, "nan": float("nan"), "boolean": True}[mutation]
    with pytest.raises(ValueError):
        recommend_ratios(p, requests())


def test_renewal_statistics_match_explicit_slot_history_with_correlated_lengths():
    rows = requests()
    history = [r["prompt_tokens"] + i for r in rows for i in range(r["decode_steps"])]
    moments = workload_moments(rows)
    assert moments["theta"] == pytest.approx(np.mean(history))
    assert moments["variance"] == pytest.approx(np.var(history))
    assert moments["theta"] != np.mean([r["prompt_tokens"] for r in rows])


def test_gaussian_cycle_integrates_max_before_expectation():
    p = profile()
    moments = workload_moments(requests())
    rng = np.random.default_rng(41)
    ratio, batch = 4, 16
    mean = .01 * batch * moments["theta"] + 1.
    sigma = .01 * np.sqrt(batch * moments["variance"])
    other = max(.2 * ratio * batch + 2, .02 * ratio * batch + .1)
    empirical = np.maximum(other, mean + sigma * rng.normal(size=(150000, ratio)).max(axis=1)).mean()
    predicted = cycle_time(p["latency_coefficients_ms"], moments, batch, ratio, barrier=True)
    assert predicted == pytest.approx(empirical, rel=.01)
    assert predicted >= cycle_time(p["latency_coefficients_ms"], moments, batch, ratio, barrier=False)


def test_zero_variance_reduces_to_mean_field_and_returns_only_advice():
    rows = [{"source_index": i, "prompt_tokens": 100, "decode_steps": 1} for i in range(5)]
    result = recommend_ratios(profile(), rows)
    assert result["recommendation"]["mean_field"] == result["recommendation"]["gaussian_barrier"]
    for row in result["ranked_ratios"]:
        assert row["cycle_ms"]["mean_field"] == row["cycle_ms"]["gaussian_barrier"]
    assert result["execution_eligibility_changed"] is False
    assert result["slo_feasibility_established"] is False


@pytest.mark.parametrize("mutation", ["heldout", "ep_rank", "negative", "nan", "duplicate_ratio", "fractional_ratio", "missing_lifetime"])
def test_invalid_provisioning_contracts_fail(mutation):
    p, rows = profile(), requests()
    if mutation == "heldout":
        p["selection_split"] = "heldout"
    elif mutation == "ep_rank":
        p["ratio_scope"] = "attention_gpus_per_ep_rank"
    elif mutation in ("negative", "nan"):
        p["latency_coefficients_ms"]["attention"]["slope"] = -1 if mutation == "negative" else float("nan")
    elif mutation == "duplicate_ratio":
        p["ratios"] = [1, 1]
    elif mutation == "fractional_ratio":
        p["ratios"] = [1, 1.5]
    else:
        rows[0].pop("decode_steps")
        rows[0]["output_tokens"] = 10
    with pytest.raises((ValueError, KeyError)):
        recommend_ratios(p, rows)

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from static_dse.campaign import (ask, create_campaign, freeze, isolation, read_json,
                                 status, summary_result, tell, write_json)
from static_dse.demo import make_demo, run_demo, synthetic_result
from static_dse.mechanism import build_mechanism, inform_candidates
from static_dse.optimizer import best_measured, lognormal_ei, posterior, propose
from static_dse.space import (audit, configuration, enumerate_candidates, hard_filter,
                              hardware_from_snapshot, layout, launch_environment, model_priors)


@pytest.fixture
def campaign(tmp_path):
    config = make_demo(tmp_path / "inputs", evaluations=6)
    directory = tmp_path / "campaign"
    create_campaign(config, directory)
    return directory


def test_enumeration_counts_and_distinguishes_expert_tp(tmp_path):
    make_demo(tmp_path / "inputs")
    spec = read_json(tmp_path / "inputs/spec.json")
    spec["enumerate_parallelism"] = True
    spec["microbatches"] = [1, 2, 4]
    rows = enumerate_candidates(spec)
    assert len(rows) == 6 * 3 * 144
    assert len({r["id"] for r in rows}) == len(rows)
    assert any(c["topology"]["expert_tp"] == 2 for c in rows)
    spec["topologies"] *= 2
    assert len(enumerate_candidates(spec)) == len(rows)


def test_hardware_vs_unknown_vs_unsupported(campaign):
    bundle = read_json(campaign / "bundle.json")
    c = copy.deepcopy(bundle["candidates"][0])
    hw, runtime = bundle["hardware"], bundle["runtime"]
    assert hard_filter(c, hw, runtime)["status"] == "eligible"
    # Low power + high requested frequency is still legal.
    c["knobs"].update(expert_mhz=1410, expert_power_w=200)
    assert hard_filter(c, hw, runtime)["status"] == "eligible"
    c["knobs"]["expert_power_w"] = 450
    assert hard_filter(c, hw, runtime)["status"] == "hard_rejected"
    c["knobs"]["expert_power_w"] = 400
    c["microbatches"] = 4
    assert hard_filter(c, hw, runtime)["status"] == "pending_validation"
    runtime["adapter"] = "v026_launch_pair"
    assert "launcher_microbatch_count_unrepresentable" in hard_filter(c, hw, runtime)["reasons"]


def test_current_adapter_rejects_independent_expert_tp(campaign):
    b = read_json(campaign / "bundle.json")
    c = b["candidates"][0]
    b["runtime"]["adapter"] = "v026_launch_pair"
    c["topology"].update(expert_tp=2, expert_ep=1)
    assert "launcher_expert_parallelism_unrepresentable" in hard_filter(c, b["hardware"], b["runtime"])["reasons"]
    with pytest.raises(ValueError):
        launch_environment(c, b["runtime"])


def test_no_unverified_memory_flag_promotion(campaign):
    b = read_json(campaign / "bundle.json")
    b["runtime"]["structures"] = []
    c = b["candidates"][0]
    c["memory_feasible"] = True
    assert hard_filter(c, b["hardware"], b["runtime"])["status"] == "pending_validation"


@pytest.mark.parametrize("use_prior", [True, False])
def test_analytical_selection_respects_eligibility_and_generic_bo(campaign, use_prior):
    req = ask(campaign)
    tell(campaign, synthetic_result(req))
    bundle, state = read_json(campaign / "bundle.json"), read_json(campaign / "state.json")
    candidates = bundle["candidates"]
    reference = bundle["settings"]["reference_candidate_id"]
    alternatives = [c for c in candidates if c["id"] != reference]
    wanted, forbidden = alternatives[:2]
    for c in candidates:
        c["mechanism"] = {"stage_ms": dict.fromkeys(("attention_compute", "a2f_dispatch", "ffn_compute", "f2a_combine"), 1.),
                          "unidentified_knobs": [], "analytical_provisioning": {
                              "available": True, "pipeline_ms": 5., "layer_barrier_coverage": 1.}}
        c["prior"]["log_energy"] = 7.
    wanted["prior"]["log_energy"] = 5.
    forbidden["prior"]["log_energy"] = 1.
    bundle["settings"]["bo"].update(use_model_prior=use_prior, initial_parameter_probes=0, initial_joint_probes=0)
    eligible = {c["id"] for c in candidates} - {forbidden["id"]}
    selected, decision = propose(candidates, state["observations"], bundle["settings"], eligible, 10.)
    assert selected["id"] != forbidden["id"]
    if use_prior:
        assert selected["id"] == wanted["id"]
        assert decision["reason"] == "analytical_provisioning_probe"
        assert not decision["execution_eligibility_changed"]
    else:
        assert decision["reason"] != "analytical_provisioning_probe"


def test_trace_overlap_and_missing_identity(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"source_index":1,"source_timestamp":"x"}\n')
    b.write_text('{"source_index":1.0,"source_timestamp":"y"}\n')
    with pytest.raises(ValueError, match="overlap"):
        isolation(a, b, ["source_index", "source_timestamp"])
    b.write_text('{"source_index":2}\n')
    with pytest.raises(ValueError, match="identity"):
        isolation(a, b, ["source_index", "source_timestamp"])


def test_ask_is_resumable_and_single_pending(campaign):
    first = ask(campaign)
    assert first == ask(campaign)
    assert first["decision"]["reason"] == "measure_reference"
    assert status(campaign)["cost"]["evaluations"] == 0
    tell(campaign, synthetic_result(first))
    assert ask(campaign)["trial_id"] != first["trial_id"]


@pytest.mark.parametrize("field,value", [("trace_sha256", "heldout"), ("selection_split", "heldout"),
                                         ("configuration_sha256", "wrong"), ("origin", "model_prediction"),
                                         ("context_sha256", "wrong"), ("mode", "physical")])
def test_receipt_identity_and_provenance_enforced(campaign, field, value):
    request = ask(campaign)
    result = synthetic_result(request)
    result[field] = value
    with pytest.raises(ValueError):
        tell(campaign, result)
    assert status(campaign)["pending"] == request


def test_failure_is_charged_without_fake_objective(campaign):
    req = ask(campaign)
    result = synthetic_result(req)
    result.update(status="failed", failure_reason="executor timeout")
    result.pop("metrics")
    reply = tell(campaign, result)
    assert reply["cost"]["evaluations"] == 1
    assert reply["cost"]["gpu_hours"] > 0
    assert reply["best"] is None
    with pytest.raises(ValueError, match="no measured feasible"):
        freeze(campaign)
    assert not ask(campaign).get("stopped")


def test_budget_and_freeze(campaign):
    for _ in range(6):
        request = ask(campaign)
        assert not request.get("stopped")
        tell(campaign, synthetic_result(request))
    assert ask(campaign)["reason"] == "budget_exhausted"
    frozen = freeze(campaign)
    assert frozen["best"]["feasible"]
    assert frozen == freeze(campaign)
    with pytest.raises(ValueError, match="frozen"):
        ask(campaign)


def test_input_tamper_detected(campaign):
    bundle = read_json(campaign / "bundle.json")
    from pathlib import Path
    Path(bundle["sources"]["runtime"]["path"]).write_text('{}')
    with pytest.raises(ValueError, match="input changed"):
        ask(campaign)


def test_slo_infeasible_energy_winner_is_not_selected():
    metrics = dict(energy_j=100, ttft_ms=10, tpot_ms=10, output_tps=100)
    rows = [{"candidate_id": "good", "status": "ok", "metrics": metrics},
            {"candidate_id": "bad", "status": "ok", "metrics": {**metrics, "energy_j": 1, "output_tps": 1}}]
    limits = {"ttft_ms": 20, "tpot_ms": 20, "min_output_tps": 90}
    assert best_measured(rows, limits)["candidate_id"] == "good"


def test_posterior_updates_and_finite_lognormal_ei():
    x = np.array([[0.], [1.]])
    mean, std = posterior(x, [0], [math.log(2)], np.zeros(2), np.ones(2), .01)
    assert mean[0] == pytest.approx(math.log(2), abs=.001)
    assert std[0] < std[1]
    assert np.all(np.isfinite(lognormal_ei(3, mean, std)))
    assert lognormal_ei(3, np.array([math.log(2)]), np.array([1e-10]))[0] == pytest.approx(1)


def test_summary_bridge(campaign, tmp_path):
    req = ask(campaign)
    receipt = synthetic_result(req)
    summary = {"requests": 8, "completed_requests": 8, "failed_requests": 0,
               "energy_j": 1200, "ttft_ms": {"p90": 100}, "tpot_ms": {"p90": 100},
               "output_token_throughput_tps": 100}
    write_json(tmp_path / "summary.json", summary)
    result = summary_result(req, receipt, tmp_path / "summary.json")
    assert tell(campaign, result)["best"]["feasible"]


def test_nonfinite_measurement_rejected(campaign):
    req = ask(campaign)
    result = synthetic_result(req)
    result["metrics"]["energy_j"] = float("nan")
    with pytest.raises(ValueError):
        tell(campaign, result)


def anchors_for_mechanism(candidate):
    base = {**copy.deepcopy(candidate), "id": "max", "selection_split": "calibration",
            "validation_status": "physically_measured_anchor", "layers": 28, "requests_per_pipeline": 1,
            "stage_models": {s: {"intercept_ms": t} for s, t in zip(
                ("attention_compute", "a2f_dispatch", "ffn_compute", "f2a_combine"), (10, 2, 20, 1))},
            "power_model": {"idle_intercept_w": 1000, "dynamic_slope_w": 0, "total_power_cap_w": 1600}}
    base["knobs"] = {"attention_mhz": 1410, "expert_mhz": 1410, "attention_power_w": 400, "expert_power_w": 400}
    low = copy.deepcopy(base)
    low["id"] = "low"
    low["knobs"]["expert_mhz"] = 1050
    low["stage_models"]["ffn_compute"]["intercept_ms"] = 28
    low["power_model"]["idle_intercept_w"] = 750
    return [base, low]


def test_parameter_effects_use_only_physical_anchors(campaign):
    c = read_json(campaign / "bundle.json")["candidates"][0]
    anchors = anchors_for_mechanism(c)
    invented = copy.deepcopy(anchors[1])
    invented["validation_status"] = "surrogate_unvalidated"
    invented["stage_models"]["ffn_compute"]["intercept_ms"] = 10000
    report = build_mechanism({"selection_split": "calibration", "candidates": anchors + [invented]}, {"arrival_rate_rps": 1})
    group = next(iter(report["groups"].values()))
    effect = group["effects"]["expert_mhz"]
    assert report["physical_anchor_count"] == 2
    assert effect["time_elasticity"]["ffn_compute"] == pytest.approx(math.log(1.4) / math.log(1410 / 1050))
    assert effect["time_elasticity"]["attention_compute"] == 0
    assert "attention_mhz" not in group["effects"]
    c["knobs"] = {**anchors[0]["knobs"], "expert_mhz": 1170}
    informed = inform_candidates([c], report, {"arrival_rate_rps": 1}, 8)[0]
    assert informed["mechanism"]["bottleneck"] == "ffn_compute"
    assert informed["prior"]["source"] == "measured_parameter_elasticity"


def test_synthetic_full_loop(tmp_path):
    result = run_demo(tmp_path / "demo", evaluations=5)
    assert result["mode"] == "synthetic"
    assert result["cost"]["evaluations"] == 5
    assert result["heldout_evaluation_completed"] is False


def test_actual_cost_overrun_stops_next_trial(campaign):
    req = ask(campaign)
    result = synthetic_result(req)
    result["cost"]["gpu_hours"] = 11
    tell(campaign, result)
    assert ask(campaign)["reason"] == "budget_exhausted"


def test_all_soft_deferred_points_can_be_recovered(campaign):
    req = ask(campaign)
    tell(campaign, synthetic_result(req))
    b = read_json(campaign / "bundle.json")
    state = read_json(campaign / "state.json")
    b["settings"]["bo"].update(exploration_policy='legacy', max_repeats=1, initial_parameter_probes=0, initial_joint_probes=0)
    for c in b["candidates"]:
        if c["id"] != req["candidate_id"]:
            c["prior"].update(log_energy=math.log(1e8), energy_sd=.01)
    eligible = {c["id"] for c in b["candidates"]}
    candidate, decision = propose(b["candidates"], state["observations"], b["settings"], eligible, 10)
    assert candidate is not None
    assert decision["reason"] == "deferred_rescue"
    assert decision["deferred_count"] == len(b["candidates"]) - 1


def test_real_stage_feedback_updates_mechanism(campaign):
    from static_dse.campaign import updated_candidates
    req = ask(campaign)
    result = synthetic_result(req)
    result["four_stage"] = {"workload_sha256": req["model_workload_sha256"], "layers": 28,
                            "requests_per_pipeline": 1, "power_w": 600,
                            "stage_ms": {"attention_compute": 10, "a2f_dispatch": 2,
                                         "ffn_compute": 20, "f2a_combine": 1}}
    tell(campaign, result)
    rows = updated_candidates(read_json(campaign / "bundle.json"), read_json(campaign / "state.json"))
    c = next(c for c in rows if c["id"] == req["candidate_id"])
    assert c["mechanism"]["bottleneck"] == "ffn_compute"
    assert c["prior"]["source"] == "measured_parameter_elasticity"


def test_executor_bridge_and_failure_remain_pending(campaign, tmp_path):
    import sys
    from static_dse.executor import run_one
    script = tmp_path / "executor.py"
    # This external process is a synthetic fixture, with no GPU access.
    import static_dse
    from pathlib import Path
    module_root = str(Path(static_dse.__file__).resolve().parent.parent)
    script.write_text('import sys,json,argparse\n' + f'sys.path.insert(0,{module_root!r})\n' +
                      'from static_dse.demo import synthetic_result\n' +
                      'p=argparse.ArgumentParser();p.add_argument("--request");p.add_argument("--result");a=p.parse_args()\n' +
                      'json.dump(synthetic_result(json.load(open(a.request))),open(a.result,"w"))\n')
    assert run_one(campaign, [sys.executable, str(script)])["best"]["feasible"]
    with pytest.raises(RuntimeError, match="record a charged failure"):
        run_one(campaign, [sys.executable, "-c", "raise SystemExit(3)"])
    assert status(campaign)["pending"] is not None


def test_initial_probes_change_one_parameter(tmp_path):
    config = make_demo(tmp_path / 'inputs')
    settings = read_json(config)
    settings.setdefault('bo', {})['exploration_policy'] = 'legacy'
    write_json(config, settings)
    campaign = tmp_path / 'campaign'
    create_campaign(config, campaign)
    req = ask(campaign)
    tell(campaign, synthetic_result(req))
    probe = ask(campaign)
    assert probe["decision"]["reason"] == "single_parameter_probe"
    assert [k for k in probe["configuration"] if probe["configuration"][k] != req["configuration"][k]] == ["expert_mhz"]

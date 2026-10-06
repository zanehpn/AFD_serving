"""Mechanism identification, raw feedback, structural probes and fair comparisons."""
from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from static_dse.campaign import (ask, create_campaign, file_hash, read_json, status,
                                 tell, updated_candidates, write_json)
from static_dse.calibration import calibration_plan
from static_dse.comparison import create_comparison, comparison_round, comparison_report
from static_dse.demo import make_demo, synthetic_result
from static_dse.feedback import attach_feedback, make_manifest
from static_dse.response import (fit_joint, joint_features, joint_predict, fit_structural,
                                 structural_features, structural_predict, normalized_workload)
from static_dse.space import configuration, enumerate_candidates, digest, layout, structure
from four_stage_dse_v6.model import STAGES


def factorial_anchors():
    high = dict(attention_mhz=1400, expert_mhz=1400, attention_power_w=400, expert_power_w=400)
    points = [high]
    for role in ("attention", "expert"):
        f, p = role + "_mhz", role + "_power_w"
        points += [{**high, f: 1000}, {**high, p: 200}, {**high, f: 1000, p: 200}]
    anchors = []
    for i, knobs in enumerate(points):
        x = np.asarray(joint_features(knobs, high))
        stage = math.exp(x @ [2., -.6, -.3, -.1, -.2, 1.2, -.8])
        power = math.exp(x @ [6., .2, .3, .4, .5, -.3, .2])
        anchors.append({"id": str(i), "knobs": knobs, "stage_ms": {s: stage for s in STAGES},
                        "power_w": power, "repetitions": 2,
                        "operating_state": {r: {"effective_mhz": knobs[r + "_mhz"] * .8,
                                                "power_cap_active_fraction": .5} for r in ("attention", "expert")}})
    return anchors


def test_joint_interactions_require_rank_and_predict_unmeasured_interior():
    rows = factorial_anchors()
    insufficient = fit_joint(rows[:3] * 5)
    assert not insufficient["identified"]
    assert insufficient["rank"] == 3
    model = fit_joint(rows)
    assert model["rank"] == 7
    knobs = dict(attention_mhz=1200, expert_mhz=1300, attention_power_w=300, expert_power_w=250)
    expected = math.exp(np.asarray(joint_features(knobs, rows[0]["knobs"])) @ [2., -.6, -.3, -.1, -.2, 1.2, -.8])
    assert joint_predict(model, knobs)[STAGES[0]] == pytest.approx(expected)
    assert joint_predict(model, knobs)["attention_effective_mhz"] == pytest.approx(960)


@pytest.fixture
def raw_feedback(tmp_path):
    config = make_demo(tmp_path / "inputs")
    settings = read_json(config)
    settings["workload"].update(prefill_tokens_per_microbatch=0, decode_tokens_per_microbatch=4)
    write_json(config, settings)
    campaign = tmp_path / "campaign"
    create_campaign(config, campaign)
    request = ask(campaign)
    receipt = synthetic_result(request)
    receipt.pop("four_stage")
    cell = tmp_path / "cell"
    (cell / "traces").mkdir(parents=True)
    start, end = 10_000_000_000, 12_000_000_000
    write_json(cell / "measurement-trace-reset.json", {"all_files_empty_after_reset": True, "reset_wall_ns": start - 100})
    write_json(cell / "summary.json", {"requests": 8, "completed_requests": 8, "failed_requests": 0,
                                       "energy_j": 800, "ttft_ms": {"p90": 100}, "tpot_ms": {"p90": 100},
                                       "output_token_throughput_tps": 100})
    write_json(cell / "telemetry.json", {"returncode": 0, "sample_error_count": 0, "sample_time_coverage": 1,
                                         "request_count": 8, "started_wall_ns": start, "finished_wall_ns": end,
                                         "gpu_ids": [0, 1, 2, 3]})
    for role in ("attention", "ffn"):
        for rank in range(2):
            events = ["ffn_compute"] if role == "ffn" else [s for s in STAGES if s != "ffn_compute"]
            rows = []
            for microbatch in range(2):
                for event in events:
                    rows.append({"event": event, "transaction_id": "txn1", "stage_idx": microbatch,
                                 "start_wall_ns": start + 100, "end_wall_ns": end - 100,
                                 "duration_us": 1000 * (rank + 1), "prefill_tokens": 0, "decode_tokens": 4})
            # Warmup event must not pollute regression or transaction coverage.
            rows.append({**rows[0], "transaction_id": "warmup", "start_wall_ns": start - 1000, "end_wall_ns": start - 500, "duration_us": 1e9})
            (cell / "traces" / f"stage-{role}-{rank}-123.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    samples = [{"timestamp_ns": t, "gpu_ids": [0, 1, 2, 3], "power_w": [100] * 4,
                "operating_state": [{"graphics_mhz": 1050, "power_limit_w": 400,
                                     "gpu_utilization": 50, "throttle_reasons": 4}] * 4}
               for t in [start, start + 1_000_000_000, end]]
    (cell / "power-samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    manifest = cell / "feedback.json"
    write_json(manifest, make_manifest(request, cell, 28))
    return campaign, request, receipt, cell, manifest


def test_raw_feedback_reintegrates_energy_excludes_warmup_and_updates_prior(raw_feedback):
    campaign, request, receipt, cell, manifest = raw_feedback
    result = attach_feedback(request, receipt, manifest)
    assert result["four_stage"]["stage_ms"]["attention_compute"] == pytest.approx(2.)
    assert result["four_stage"]["power_w"] == pytest.approx(400.)
    assert result["four_stage"]["operating_state"]["expert"]["effective_mhz"] == pytest.approx(1050.)
    assert result["four_stage"]["operating_state"]["expert"]["power_cap_active_fraction"] == 1
    tell(campaign, result)
    candidates = updated_candidates(read_json(campaign / "bundle.json"), read_json(campaign / "state.json"))
    anchor = next(c for c in candidates if c["id"] == request["candidate_id"])
    assert anchor["mechanism"]["stage_ms"]["attention_compute"] == pytest.approx(2.)


@pytest.mark.parametrize("mutation,error", [("missing_clocks", "actual clocks"), ("cap", "power limit"),
                                           ("shape", "not identifiable"), ("warmup", "reset evidence"),
                                           ("hash", "hash mismatch"), ("energy", "reintegrated energy")])
def test_feedback_rejects_invalid_measurement(raw_feedback, mutation, error):
    _, request, receipt, cell, manifest_path = raw_feedback
    manifest = read_json(manifest_path)
    if mutation in ("missing_clocks", "cap"):
        path = cell / "power-samples.jsonl"
        samples = [json.loads(r) for r in path.read_text().splitlines()]
        for sample in samples:
            if mutation == "missing_clocks":
                sample.pop("operating_state")
            else:
                sample["operating_state"][0]["power_limit_w"] = 200
        path.write_text("".join(json.dumps(r) + "\n" for r in samples))
        manifest["power_samples"]["sha256"] = file_hash(path)
    elif mutation == "shape":
        request["model_workload"]["decode_tokens_per_microbatch"] = 8
    elif mutation == "warmup":
        path = cell / "measurement-trace-reset.json"
        write_json(path, {"all_files_empty_after_reset": False, "reset_wall_ns": 1})
        manifest["trace_reset"]["sha256"] = file_hash(path)
    elif mutation == "hash":
        manifest["summary"]["sha256"] = "wrong"
    else:
        path = cell / "summary.json"
        data = read_json(path)
        data["energy_j"] = 2
        write_json(path, data)
        manifest["summary"]["sha256"] = file_hash(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match=error):
        attach_feedback(request, receipt, manifest_path)


def test_structural_model_checks_workload_rank_and_support(tmp_path):
    make_demo(tmp_path / "inputs")
    candidate = enumerate_candidates(read_json(tmp_path / "inputs/spec.json"))[0]
    groups = {}
    # Two placements with identical mechanism features supply no independent
    # TP information. Even a low training residual must not authorize TP=2.
    for i in (0, 1):
        c = copy.deepcopy(candidate)
        c["topology"]["attention_gpus"] = [i * 4, i * 4 + 1]
        c["topology"]["expert_gpus"] = [i * 4 + 2, i * 4 + 3]
        anchor = {"id": str(i), "knobs": c["knobs"], "stage_ms": {s: 10 for s in STAGES},
                  "power_w": 400, "requests_per_pipeline": 1, "layers": 28}
        groups[str(i)] = {"structure": structure(c), "anchors": [anchor]}
    assert not fit_structural(groups, {})["enabled"]
    report = fit_structural(groups, {"prefill_tokens_per_step": 0, "decode_tokens_per_step": 16})
    assert structural_predict(report, candidate)["values"][STAGES[0]] == pytest.approx(10)
    candidate["topology"].update(attention_dp=1, attention_tp=2)
    assert structural_predict(report, candidate) is None
    assert next(iter(report["models"].values()))["responses"][STAGES[0]]["identified_coefficients"] is None
    assert normalized_workload({"decode_tokens_per_step": 16}, candidate)["decode_tokens_per_microbatch"] == 8


def test_structural_model_recovers_identifiable_costs_and_unmeasured_layout():
    groups = {}
    target_degrees = (1, 2, 2, 2, 1, 2)
    expected_weights = np.array([1., 30., 50., 2., 3., 4., .5])
    target = None
    for degrees in itertools.product((1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2, 4)):
        ad, at, ed, ep, et, m = degrees
        candidate = {"topology": {"attention_gpus": list(range(ad * at)),
                                  "expert_gpus": list(range(ad * at, ad * at + ed * ep * et)),
                                  "attention_dp": ad, "attention_tp": at, "expert_dp": ed, "expert_ep": ep, "expert_tp": et},
                     "microbatches": m, "execution_mode": "eager", "knobs": factorial_anchors()[0]["knobs"]}
        if degrees == target_degrees:
            target = candidate
            continue
        time = float(np.asarray(structural_features(configuration(candidate))) @ expected_weights)
        key = digest(structure(candidate))
        groups[key] = {"structure": structure(candidate), "anchors": [{"id": key, "knobs": candidate["knobs"],
                       "stage_ms": {s: time for s in STAGES}, "power_w": 400, "requests_per_pipeline": 1, "layers": 28}]}
    report = fit_structural(groups, {"prefill_tokens_per_step": 0, "decode_tokens_per_step": 32})
    prediction = structural_predict(report, target)
    expected = np.asarray(structural_features(configuration(target))) @ expected_weights
    assert prediction["values"][STAGES[0]] == pytest.approx(expected)
    model = next(iter(report["models"].values()))
    assert model["rank"] == 7
    assert model["responses"][STAGES[0]]["identified_coefficients"] == pytest.approx(expected_weights)


def test_joint_probe_then_bo_fits_interaction(tmp_path):
    config = make_demo(tmp_path / "inputs", evaluations=9)
    settings = read_json(config)
    settings.setdefault('bo', {})['exploration_policy'] = 'legacy'
    write_json(config, settings)
    campaign = tmp_path / "campaign"
    create_campaign(config, campaign)
    reasons = []
    for _ in range(8):
        request = ask(campaign)
        reasons.append(request["decision"]["reason"])
        tell(campaign, synthetic_result(request))
    assert reasons[:7] == ["measure_reference"] + ["single_parameter_probe"] * 4 + ["joint_parameter_probe"] * 2
    candidates = updated_candidates(read_json(campaign / "bundle.json"), read_json(campaign / "state.json"))
    assert any(c["mechanism"]["joint_response_identified"] for c in candidates)


@pytest.mark.parametrize("failure", [False, True])
def test_structure_plan_promotion_or_charged_failure(tmp_path, failure):
    config = make_demo(tmp_path / "inputs", evaluations=30)
    spec = read_json(config.parent / "spec.json")
    spec["microbatches"] = [1, 2]
    write_json(config.parent / "spec.json", spec)
    candidates = enumerate_candidates(spec)
    plan = calibration_plan(candidates, read_json(config.parent / "hardware.json"), read_json(config.parent / "runtime.json"), repetitions=1)
    assert plan["planned_evaluations"] == 14
    write_json(config.parent / "plan.json", plan)
    settings = read_json(config)
    settings.update(calibration_plan="plan.json", allow_structure_probes=True, require_four_stage=True)
    write_json(config, settings)
    campaign = tmp_path / "campaign"
    create_campaign(config, campaign)
    req = ask(campaign)
    tell(campaign, synthetic_result(req))
    probe = ask(campaign)
    assert probe["structure_validation_trial"]
    assert probe["configuration"]["microbatches"] == 1
    result = synthetic_result(probe)
    if failure:
        result.update(status="runtime_incompatible", failure_reason="fixture unsupported")
        result.pop("metrics")
        result.pop("four_stage")
    tell(campaign, result)
    seen_new = False
    for _ in range(12):
        req = ask(campaign)
        if req.get("stopped"):
            break
        seen_new |= req["configuration"]["microbatches"] == 1
        tell(campaign, synthetic_result(req))
    assert seen_new != failure
    assert status(campaign)["cost"]["evaluations"] >= 3


def test_comparison_has_matched_budgets_independent_states_and_explicit_mode(tmp_path):
    config = make_demo(tmp_path / "inputs", evaluations=3)
    directory = tmp_path / "comparison"
    manifest = create_comparison(config, directory, seeds=[3, 7])
    assert len(manifest["campaigns"]) == 8
    with pytest.raises(ValueError, match="mode"):
        comparison_round(directory)
    for _ in range(4):
        outcome = comparison_round(directory, synthetic=True)
    assert outcome["all_stopped"]
    report = comparison_report(directory)
    assert all(a["cost"]["evaluations"] == 3 for a in report["arms"])
    assert all(len(a["curve"]) == 3 and not a["pending"] for a in report["arms"])
    assert report["reference_scope"] == "observed_union_only_not_exhaustive_oracle"
    for arm in manifest["campaigns"]:
        b = read_json(Path(arm["directory"]) / "bundle.json")
        assert b["settings"]["bo"]["use_model_prior"] == (arm["method"] == "v2")
    referenced = comparison_report(directory, reference_campaign=manifest["campaigns"][0]["directory"])
    assert referenced["reference_scope"] == "partially_measured_reference"
    assert referenced["reference_evidence"]["cost"]["evaluations"] == 3


def test_executor_automatically_attaches_native_feedback(raw_feedback):
    from static_dse.executor import run_one
    campaign, request, receipt, _, manifest = raw_feedback
    receipt["feedback_manifest"] = {"path": str(manifest), "sha256": file_hash(manifest)}
    trial = campaign / "trials" / request["trial_id"]
    trial.mkdir(parents=True)
    write_json(trial / "result.json", receipt)
    assert run_one(campaign, [])["best"]["feasible"]
    assert status(campaign)["four_stage_feedback_count"] == 1


def test_required_feedback_cannot_silently_degrade_to_bo_only(tmp_path):
    config = make_demo(tmp_path / "inputs")
    settings = read_json(config)
    settings["require_four_stage"] = True
    write_json(config, settings)
    campaign = tmp_path / "campaign"
    create_campaign(config, campaign)
    request = ask(campaign)
    result = synthetic_result(request)
    result.pop("four_stage")
    with pytest.raises(ValueError, match="requires measured four-stage"):
        tell(campaign, result)
    assert status(campaign)["pending"]["trial_id"] == request["trial_id"]


@pytest.mark.parametrize("clock_error", [False, True])
def test_operating_state_collector_with_mock_nvml(tmp_path, monkeypatch, clock_error):
    import sys
    from types import SimpleNamespace
    import measure_command
    nvml = measure_command.pynvml
    monkeypatch.setattr(nvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(nvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(nvml, "nvmlDeviceGetPowerUsage", lambda handle: 100_000)
    def clock(handle, kind):
        if clock_error:
            raise nvml.NVMLError_NotSupported()
        return 1050
    monkeypatch.setattr(nvml, "nvmlDeviceGetClockInfo", clock)
    monkeypatch.setattr(nvml, "nvmlDeviceGetEnforcedPowerLimit", lambda handle: 200_000)
    monkeypatch.setattr(nvml, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=80))
    monkeypatch.setattr(nvml, "nvmlDeviceGetCurrentClocksThrottleReasons", lambda handle: 4)
    output, samples = tmp_path / "telemetry.json", tmp_path / "samples.jsonl"
    monkeypatch.setattr(sys, "argv", ["measure_command", "--gpus", "0", "--output", str(output),
                                     "--samples-output", str(samples), "--record-operating-state", "--nvml-retries", "0",
                                     "--", sys.executable, "-c", "pass"])
    if clock_error:
        with pytest.raises(RuntimeError, match="sampling unavailable"):
            measure_command.main()
        assert not output.exists()
    else:
        with pytest.raises(SystemExit) as error:
            measure_command.main()
        assert error.value.code == 0
        rows = [json.loads(r) for r in samples.read_text().splitlines()]
        assert all(r["operating_state"][0] == {"graphics_mhz": 1050, "power_limit_w": 200,
                                              "gpu_utilization": 80, "throttle_reasons": 4} for r in rows)
        assert read_json(output)["sample_error_count"] == 0

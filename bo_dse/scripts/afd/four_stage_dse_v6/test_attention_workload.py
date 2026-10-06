from __future__ import annotations

import copy
import json

import pytest

from build_deepseek_v6_four_stage_profile import fit_stage_model, load_four_stage_groups, nnls_small
from four_stage_dse_v6.attention_workload import (
    common_context_target, context_totals, fit_rank_model, predict_rank_model,
    rank_summary, validate_context,
)
from four_stage_dse_v6.model import STAGES, predict_stage_times


def context(tokens, total):
    return dict(schema_version=1, source="request_token_spans", active_requests=tokens,
                prefill_tokens=0, decode_tokens=tokens, prefill_context_tokens=0,
                decode_context_tokens=total, kv_sequence_tokens=total)


def group(total, *, tokens=4, skew=.25, ranks=2):
    mean = 1. + total * .01
    observations = {str(i): dict(duration_ms=mean + (skew if i == ranks - 1 else -skew / (ranks - 1)),
                                prefill_tokens=0, decode_tokens=tokens // ranks,
                                context=context(tokens // ranks, total // ranks))
                    for i in range(ranks)}
    return dict(transaction_id=str(total), stage_idx=0, token_scope="attention_dp_total",
                prefill_tokens=0, decode_tokens=tokens,
                attention_rank_observations=observations,
                attention_rank_summary=rank_summary(observations, comparable_events=True),
                stage_times_ms={s: mean + skew if s == "attention_compute" else 2. for s in STAGES},
                bytes_by_stage={s: 128 for s in STAGES})


def workload(total, tokens=4):
    return dict(prefill_tokens_per_microbatch=0, decode_tokens_per_microbatch=tokens,
                prefill_context_tokens_per_microbatch=0, decode_context_tokens_per_microbatch=total)


def test_equal_decode_tokens_different_kv_load_changes_prediction_without_double_barrier():
    groups = [group(total) for total in (100, 200, 400, 800)]
    model = fit_stage_model(groups, "attention_compute", 1.)
    models = {s: model if s == "attention_compute" else {"intercept_ms": 2.} for s in STAGES}
    short = predict_stage_times(models, workload(200), risk_quantile="nominal_no_margin")
    long = predict_stage_times(models, workload(600), risk_quantile="nominal_no_margin")
    assert short["attention_compute"] == pytest.approx(3.25)
    assert long["attention_compute"] == pytest.approx(7.25)
    assert short["ffn_compute"] == long["ffn_compute"]
    report = predict_rank_model(model["attention_workload_model"], workload(600))
    assert report["mean_service"] == pytest.approx(7.)
    assert report["barrier_uplift"] == pytest.approx(.25)
    assert not model["attention_workload_model"]["coefficients_individually_identified"]


def test_rank_imbalance_changes_uplift_at_same_global_context():
    balanced = fit_rank_model([group(200, skew=0), group(400, skew=0)], nnls_small)
    skewed = fit_rank_model([group(200, skew=1), group(400, skew=1)], nnls_small)
    assert predict_rank_model(balanced, workload(300))["duration_ms"] == pytest.approx(4.)
    assert predict_rank_model(skewed, workload(300))["duration_ms"] == pytest.approx(5.)


@pytest.mark.parametrize("target,reason", [(workload(1000), "outside_calibration_range"),
                                           ({"prefill_tokens_per_microbatch": 0, "decode_tokens_per_microbatch": 4}, "missing_context_target"),
                                           (workload(200, tokens=8), "unidentifiable_workload")])
def test_unsupported_context_falls_back_without_claiming_zero_kv(target, reason):
    model = fit_stage_model([group(100), group(400)], "attention_compute", 1.)
    report = predict_rank_model(model["attention_workload_model"], target)
    assert not report["available"] and report["reason"] == reason
    models = {s: copy.deepcopy(model) for s in STAGES}
    enhanced = predict_stage_times(models, target)
    for m in models.values():
        m.pop("attention_workload_model")
    assert enhanced == predict_stage_times(models, target)


def test_incomplete_context_uses_explicit_rank_only_fit():
    groups = [group(100), group(200)]
    groups[1]["attention_rank_observations"]["1"]["context"] = None
    model = fit_rank_model(groups, nnls_small)
    assert model["enabled"] and not model["context_enabled"]
    assert context_totals(groups[1]) is None
    assert predict_rank_model(model, workload(100))["available"]


def test_mismatched_layer_coverage_disables_rank_prediction():
    groups = [group(100), group(200)]
    groups[0]["attention_rank_summary"]["comparable_events"] = False
    assert not fit_rank_model(groups, nnls_small)["enabled"]


def test_non_calibration_model_rejected():
    model = fit_rank_model([group(100), group(200)], nnls_small)
    model["selection_split"] = "heldout"
    with pytest.raises(ValueError, match="non-calibration"):
        predict_rank_model(model, workload(150))


def test_common_context_target_requires_all_topology_support():
    rows = [[group(100), group(300)], [group(200), group(400, ranks=4)]]
    target, status = common_context_target(rows, workload(0))
    assert 200 <= target["decode_context_tokens_per_microbatch"] <= 300
    assert status == "shared_observed_context_supported"
    target, status = common_context_target([[group(100)], [group(400)]], workload(0))
    assert target == {} and status == "no_common_context_support"


def test_trace_context_is_counted_once_across_layers_and_causal_ranks(tmp_path):
    paths = []
    for rank, total in enumerate((100, 300)):
        path = tmp_path / f"attention-{rank}.jsonl"
        rows = []
        for layer in range(3):
            for event, duration in (("attention_layer_total", 5000 + rank * 1000),
                                    ("remote_ffn_roundtrip", 2000), ("a2f_dispatch", 100)):
                rows.append(dict(event=event, transaction_id="t", stage_idx=0, layer_idx=layer,
                                 duration_us=duration, prefill_tokens=0, decode_tokens=2,
                                 attention_workload=context(2, total) if event != "a2f_dispatch" else None))
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)
    path = tmp_path / "expert.jsonl"
    path.write_text("".join(json.dumps(dict(event=s, transaction_id="t", stage_idx=0,
                                           duration_us=100, prefill_tokens=0, decode_tokens=4)) + "\n"
                            for s in ("ffn_compute", "f2a_combine")))
    paths.append(path)
    groups, _ = load_four_stage_groups(paths, minimum_complete_coverage=1., token_scope="attention_dp_total")
    row = groups[0]
    assert context_totals(row)["decode_context_tokens"] == 400
    assert row["stage_times_ms"]["attention_compute"] == 12
    assert row["attention_rank_summary"]["rank_mean_ms"] == 10.5
    assert row["attention_rank_summary"]["service_barrier_uplift_ms"] == 1.5


@pytest.mark.parametrize("mutation", ["negative", "nan", "token_mismatch", "missing_requests", "wrong_source"])
def test_malformed_context_cannot_become_training_data(mutation):
    row = context(4, 200)
    if mutation == "negative":
        row["decode_context_tokens"] = -1
    elif mutation == "nan":
        row["decode_context_tokens"] = float("nan")
    elif mutation == "token_mismatch":
        row["decode_tokens"] = 5
    elif mutation == "missing_requests":
        row["active_requests"] = 0
    else:
        row["source"] = "heldout_request_lengths"
    with pytest.raises(ValueError):
        validate_context(row, 0, 4)

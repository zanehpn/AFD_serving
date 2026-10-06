from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_deepseek_v6_four_stage_profile import (
    fit_total_power_model,
    fit_stage_model,
    load_four_stage_groups,
    summarize_reference_workload,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(event: str, transaction: str, duration_us: float) -> dict:
    return {
        "event": event,
        "transaction_id": transaction,
        "stage_idx": 0,
        "duration_us": duration_us,
        "prefill_tokens": 4,
        "decode_tokens": 1,
        "bytes": 128,
    }


def test_four_stage_join_uses_slowest_parallel_rank(tmp_path: Path) -> None:
    attention0 = []
    attention1 = []
    expert0 = []
    for index in range(4):
        tx = f"tx-{index}"
        attention0.extend(
            [_row("attention_compute", tx, 1000), _row("a2f_dispatch", tx, 200)]
        )
        attention1.extend(
            [_row("attention_compute", tx, 1200), _row("a2f_dispatch", tx, 180)]
        )
        expert0.extend(
            [_row("ffn_compute", tx, 800), _row("f2a_combine", tx, 100)]
        )
    paths = [
        tmp_path / "stage-attention-0.jsonl",
        tmp_path / "stage-attention-1.jsonl",
        tmp_path / "stage-ffn-0.jsonl",
    ]
    for path, rows in zip(paths, (attention0, attention1, expert0), strict=True):
        _write_rows(path, rows)
    groups, audit = load_four_stage_groups(paths, minimum_complete_coverage=1.0)
    assert audit["complete_coverage"] == 1.0
    assert groups[0]["stage_times_ms"]["attention_compute"] == pytest.approx(1.2)
    assert groups[0]["stage_times_ms"]["a2f_dispatch"] == pytest.approx(0.2)
    model = fit_stage_model(groups, "ffn_compute", routing_reference=1.1)
    assert model["samples"] == 4
    assert model["measured_p90_ms"] == pytest.approx(0.8)


def test_incomplete_four_stage_coverage_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "stage-attention-0.jsonl"
    _write_rows(path, [_row("attention_compute", "tx", 1000)])
    with pytest.raises(ValueError, match="incomplete four-stage trace coverage"):
        load_four_stage_groups([path], minimum_complete_coverage=0.95)


def test_cuda_attention_is_derived_without_remote_wait(tmp_path: Path) -> None:
    attention = tmp_path / "stage-attention-0.jsonl"
    expert = tmp_path / "stage-ffn-0.jsonl"
    _write_rows(
        attention,
        [
            _row("attention_layer_total", "tx", 3000),
            _row("remote_ffn_roundtrip", "tx", 1800),
            _row("a2f_dispatch", "tx", 200),
        ],
    )
    _write_rows(
        expert,
        [_row("ffn_compute", "tx", 800), _row("f2a_combine", "tx", 100)],
    )
    groups, audit = load_four_stage_groups(
        [attention, expert], minimum_complete_coverage=1.0
    )
    assert groups[0]["stage_times_ms"]["attention_compute"] == pytest.approx(1.2)
    assert groups[0]["attention_compute_source"] == (
        "attention_layer_total_minus_remote_ffn_roundtrip"
    )
    assert audit["attention_compute_sources"] == {
        "attention_layer_total_minus_remote_ffn_roundtrip": 1
    }


def test_reference_workload_uses_microbatch_tokens_not_request_lengths() -> None:
    groups = [
        {"prefill_tokens": 0, "decode_tokens": 8},
        {"prefill_tokens": 128, "decode_tokens": 4},
    ]
    workload = summarize_reference_workload(groups, routing_reference=1.25)
    assert workload == {
        "prefill_tokens_per_microbatch": 64.0,
        "decode_tokens_per_microbatch": 6.0,
        "communication_bytes_scale": 1.0,
        "routing_imbalance": 1.25,
    }


def test_power_fit_does_not_treat_terminal_drain_as_low_utilization() -> None:
    summary = {
        "energy_j": 4000.0,
        "duration_s": 10.0,
        "completed_requests": 5,
        "observed_submission_rate_rps": 1.0,
    }
    telemetry = {"duration_s": 10.0}
    samples = [
        {"power_w": [80.0, 80.0, 80.0, 80.0]},
        {"power_w": [100.0, 100.0, 100.0, 100.0]},
    ]
    model = fit_total_power_model(summary, telemetry, samples, 1600.0)
    assert model["observed_utilization"] == 1.0
    assert model["calibration_saturated"] is True
    assert model["idle_intercept_w"] == pytest.approx(324.0)
    assert model["dynamic_slope_w"] == pytest.approx(76.0)


def test_workload_counts_attention_dp_tokens_once_for_both_topologies(tmp_path):
    a0=tmp_path/'stage-attention-0.jsonl'
    a1=tmp_path/'stage-attention-1.jsonl'
    e0=tmp_path/'stage-ffn-0.jsonl'
    e1=tmp_path/'stage-ffn-1.jsonl'
    def rows(events, prefill, decode, duration):
        return [dict(_row(event,'tx',duration),prefill_tokens=prefill,decode_tokens=decode) for event in events]
    _write_rows(a0,rows(['attention_compute','a2f_dispatch'],3,0,1000))
    _write_rows(a1,rows(['attention_compute','a2f_dispatch'],0,5,1200))
    _write_rows(e0,rows(['ffn_compute','f2a_combine'],3,0,800))
    _write_rows(e1,rows(['ffn_compute','f2a_combine'],0,5,900))
    parallel,_=load_four_stage_groups([a0,a1,e0,e1],minimum_complete_coverage=1,token_scope="attention_dp_total")
    _write_rows(e0,rows(['ffn_compute','f2a_combine'],3,5,1700))
    combined,_=load_four_stage_groups([a0,a1,e0],minimum_complete_coverage=1,token_scope="attention_dp_total")
    assert (combined[0]['prefill_tokens'],combined[0]['decode_tokens'])==(3,5)
    assert (parallel[0]['prefill_tokens'],parallel[0]['decode_tokens'])==(3,5)
    assert combined[0]['stage_times_ms']['ffn_compute']==pytest.approx(1.7)
    assert parallel[0]['stage_times_ms']['ffn_compute']==pytest.approx(.9)

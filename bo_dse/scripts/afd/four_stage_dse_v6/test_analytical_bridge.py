import copy
import json

import numpy as np
import pytest

from build_deepseek_v6_four_stage_profile import load_four_stage_groups
from four_stage_dse_v6.model import STAGES, pipeline_time_ms, bottleneck_pipeline_time_ms
from static_dse.analytical import calibrated_model, predict


def schedule(shared=True):
    return {"type": "finite_microbatch_fifo_v1",
            "communication": "shared_roundtrip" if shared else "independent_directions"}


def test_finite_population_obeys_serial_path_and_single_batch_exactness():
    stages = dict.fromkeys(STAGES, 10.)
    assert pipeline_time_ms(stages, microbatches=1, layers=10, schedule_model=schedule()) == 40.
    assert pipeline_time_ms(stages, microbatches=2, layers=10, schedule_model=schedule()) >= 40.
    # Keep historical profiles numerically stable when no schedule was frozen.
    assert bottleneck_pipeline_time_ms(stages, microbatches=2, layers=10) == 23.
    assert pipeline_time_ms(stages, microbatches=2, layers=10) >= 40.


def test_independent_pipeline_matches_classic_fill_drain_and_shared_link_serializes():
    stages = dict.fromkeys(STAGES, 1.)
    assert pipeline_time_ms(stages, microbatches=3, layers=1, schedule_model=schedule(False)) == 6.
    assert pipeline_time_ms(stages, microbatches=3, layers=1, schedule_model=schedule(True)) == 7.


def test_layer_weights_conserve_work_and_do_not_hide_heavy_layers():
    model = {**schedule(), "layer_weights": [[.9, .1, .9, .1], [.1, .9, .1, .9]]}
    assert pipeline_time_ms(dict.fromkeys(STAGES, 10.), microbatches=1, layers=2, schedule_model=model) == pytest.approx(40.)
    model["layer_weights"][0][0] = 1.
    with pytest.raises(ValueError, match="conserve"):
        pipeline_time_ms(dict.fromkeys(STAGES, 10.), microbatches=2, layers=2, schedule_model=model)


def test_nonfinite_schedule_input_rejected_before_event_loop():
    with pytest.raises(ValueError, match="finite"):
        pipeline_time_ms(dict.fromkeys(STAGES, float("nan")), microbatches=2, layers=2, schedule_model=schedule())


def raw_groups(tmp_path, *, missing_layer=False, derived=False):
    paths = []
    for role, count, stages in (("attention", 2, STAGES[:1] + STAGES[1:2] + STAGES[3:]), ("ffn", 1, STAGES[2:3])):
        for rank in range(count):
            events = []
            for transaction, tokens in enumerate((2, 4, 6)):
                for layer in range(2):
                    if missing_layer and rank == 1 and layer == 1:
                        continue
                    for stage in stages:
                        n = tokens // 2 if role == "attention" else tokens
                        duration = (9 if rank == layer else 1) if stage == "attention_compute" else 2 + tokens
                        context = {"schema_version": 1, "source": "request_token_spans", "prefill_tokens": 0,
                                   "decode_tokens": n, "active_requests": n, "kv_sequence_tokens": 100 * n,
                                   "prefill_context_tokens": 0, "decode_context_tokens": 100 * n}
                        row = {"event": stage, "transaction_id": str(transaction), "stage_idx": 0,
                               "layer_idx": layer, "prefill_tokens": 0, "decode_tokens": n,
                               "duration_us": duration * 1000, "attention_workload": context}
                        if derived and stage == "attention_compute":
                            events += [{**row, "event": "attention_layer_total", "duration_us": (duration + 20) * 1000},
                                       {**row, "event": "remote_ffn_roundtrip", "duration_us": 20000}]
                        else:
                            events.append(row)
            path = tmp_path / f"stage-{role}-{rank}-1.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in events))
            paths.append(path)
    return load_four_stage_groups(paths, minimum_complete_coverage=1., token_scope="attention_dp_total")[0]


@pytest.mark.parametrize("derived", [False, True])
def test_per_layer_barrier_handles_alternating_stragglers(tmp_path, derived):
    groups = raw_groups(tmp_path, derived=derived)
    for g in groups:
        assert g["stage_times_ms"]["attention_compute"] == 18.
        assert g["attention_rank_summary"]["max_of_rank_layer_sums_ms"] == 10.
        assert g["attention_rank_summary"]["service_barrier_uplift_ms"] == 8.
        assert len(g["layer_observations"]) == 2


def test_incomplete_layer_evidence_does_not_claim_layer_barriers(tmp_path):
    assert not raw_groups(tmp_path, missing_layer=True)[0]["layer_barrier_available"]


def cfg():
    return {"attention_dp": 2, "attention_tp": 1, "expert_dp": 1, "expert_ep": 1, "expert_tp": 1,
            "attention_gpus": [0, 1], "expert_gpus": [2], "microbatches": 2}


def test_automatic_coefficients_power_mapping_and_finite_schedule(tmp_path):
    groups = raw_groups(tmp_path)
    work = {"prefill_tokens_per_microbatch": 0, "decode_tokens_per_microbatch": 4}
    model = calibrated_model(groups, cfg(), 2, work, {"attention": 200., "expert": 100.}, {"completed_requests": 2})
    assert model["mapping"]["global_fan_in"] == 2
    assert model["mapping"]["ffn_service_groups"] == 1
    assert model["layer_barrier_coverage"] == 1.
    assert model["coefficient_fits"]["ffn"]["coefficients"] == pytest.approx([2., 1.])
    result = predict(model, groups[1]["stage_times_ms"], cfg(), 2, 300.)
    assert result["available"] and result["gaussian_cycle_ms"] is None
    assert result["predicted_decode_j_per_token"] == pytest.approx(300 * result["pipeline_ms"] / 8000)
    assert result["execution_eligibility_changed"] is False
    other = {**cfg(), "expert_ep": 2, "expert_gpus": [2, 3]}
    assert predict(model, groups[1]["stage_times_ms"], other, 2, 400.)["available"] is False


def test_request_lifetimes_deduplicate_layers_and_reject_partial_cohorts(tmp_path):
    groups = raw_groups(tmp_path)
    for i, g in enumerate(groups):
        for rank, row in enumerate(g["attention_rank_observations"].values()):
            row["decode_request_spans"] = [{"request_id": str(rank), "prompt_tokens": 99,
                                            "first_token_position": 99 + i, "token_count": 1}]
    work = {"prefill_tokens_per_microbatch": 0, "decode_tokens_per_microbatch": 4}
    evidence = {"completed_requests": 2, "expected_decode_requests": 2, "expected_decode_queries": 6}
    model = calibrated_model(groups + copy.deepcopy(groups), cfg(), 2, work, {}, evidence)
    moments = model["workload_moments"]
    assert moments["available"] and moments["decode_slot_steps"] == 6
    assert moments["theta"] == 101.
    assert moments["variance"] == pytest.approx(np.var([100, 101, 102]))
    assert predict(model, groups[1]["stage_times_ms"], cfg(), 2, 300.)["gaussian_cycle_ms"] is not None
    incomplete = calibrated_model(groups, cfg(), 2, work, {}, {**evidence, "expected_decode_queries": 7})
    assert not incomplete["workload_moments"]["available"]

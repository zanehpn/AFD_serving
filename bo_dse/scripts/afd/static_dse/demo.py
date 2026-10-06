"""Synthetic executor for testing the full ask/tell protocol; never a GPU result."""
from __future__ import annotations

import json
from pathlib import Path

from .campaign import ask, create_campaign, file_hash, freeze, tell, write_json
from .space import configuration, layout
from four_stage_dse_v6.model import pipeline_time_ms


def make_demo(directory, evaluations=10):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    topology = {"attention_gpus": [0, 1], "expert_gpus": [2, 3], "attention_dp": 2,
                "attention_tp": 1, "expert_dp": 1, "expert_ep": 2, "expert_tp": 1}
    spec = {"selection_split": "calibration", "topologies": [topology],
            "attention_frequencies_mhz": [1050, 1170, 1290, 1410],
            "expert_frequencies_mhz": [1050, 1170, 1290, 1410],
            "attention_power_caps_w": [200, 300, 400], "expert_power_caps_w": [200, 300, 400],
            "microbatches": [2], "execution_mode": "eager"}
    write_json(directory / "spec.json", spec)
    (directory / "calibration.jsonl").write_text(''.join(json.dumps({"source_index": i, "source_timestamp": f"cal-{i}"}) + '\n' for i in range(8)))
    (directory / "heldout.jsonl").write_text(''.join(json.dumps({"source_index": i + 100, "source_timestamp": f"test-{i}"}) + '\n' for i in range(8)))
    write_json(directory / "evidence.json", {"mode": "synthetic", "note": "Test fixture, not execution evidence on physical GPUs."})
    candidate = {"topology": topology, "knobs": {"attention_mhz": 1410, "expert_mhz": 1410,
                                               "attention_power_w": 400, "expert_power_w": 400},
                 "microbatches": 2, "execution_mode": "eager"}
    runtime = {"adapter": "external", "gpu_budget": 4, "allowed_gpus": [0, 1, 2, 3],
               "memory_clock_mhz": 1593, "execution_modes": ["eager"],
               "structures": [{"layout": layout(candidate), "status": "verified",
                               "evidence": [{"path": "evidence.json", "sha256": file_hash(directory / "evidence.json")}]}]}
    write_json(directory / "runtime.json", runtime)
    devices = {str(i): {"memory_mib": 81920, "min_power_w": 100, "max_power_w": 400,
                       "clock_pairs": [{"memory_mhz": 1593, "graphics_mhz": f} for f in spec["attention_frequencies_mhz"]]} for i in range(4)}
    write_json(directory / "hardware.json", {"host": "synthetic", "devices": devices})
    write_json(directory / "profile.json", {"selection_split": "calibration", "candidates": []})
    config = {"selection_split": "calibration", "mode": "synthetic", "specification": "spec.json",
              "hardware": "hardware.json", "runtime": "runtime.json", "profile": "profile.json",
              "calibration_trace": "calibration.jsonl", "heldout_trace": "heldout.jsonl",
              "workload": {"arrival_rate_rps": 1}, "reference_configuration": configuration(candidate),
              "limits": {"ttft_ms": 110, "tpot_ms": 110, "min_output_tps": 90},
              "default_energy_j": 1200, "setup_cost": {"evaluations": 0, "gpu_hours": 0,
                                                         "wall_seconds": 0, "tuning_energy_j": 0},
              "budget": {"evaluations": evaluations, "gpu_hours": 10},
              "bo": {"seed": 7, "evaluation_seconds": 2, "structure_switch_seconds": 1,
                     "knob_switch_seconds": .1, "max_repeats": 2}}
    write_json(directory / "config.json", config)
    return directory / "config.json"


def synthetic_result(request):
    c = request["configuration"]
    # Known nonlinear response is deliberately outside the optimizer.
    fa = min(c["attention_mhz"] / 1410, (c["attention_power_w"] / 400) ** .25)
    fe = min(c["expert_mhz"] / 1410, (c["expert_power_w"] / 400) ** .25)
    result = {k: request[k] for k in ("trial_id", "candidate_id", "configuration_sha256", "context_sha256",
                                     "request_sha256", "mode", "selection_split")}
    result.update(origin="synthetic_test", trace_sha256=request["trace"]["sha256"], status="ok",
                  requests=request["trace"]["requests"], completed_requests=request["trace"]["requests"],
                  failed_requests=0, execution_verified=True, telemetry_valid=True,
                  metrics={"energy_j": 600 + 250 * fa ** 3 + 350 * fe ** 3,
                           "ttft_ms": 100 * (.4 + .6 / fa), "tpot_ms": 100 * (.3 + .7 / fe),
                           "output_tps": 100 * min(fa, fe) ** .5},
                  cost={"gpu_hours": request["decision"]["expected_gpu_hours"], "wall_seconds": 2.1,
                        "tuning_energy_j": 1500})
    stages = {"attention_compute": 8 / fa, "a2f_dispatch": 2 + .5 / fa,
              "ffn_compute": 12 / fe, "f2a_combine": 1 + .5 / fe}
    duration = request["trace"]["requests"] / request["workload"]["arrival_rate_rps"]
    pipeline = pipeline_time_ms(stages, microbatches=c["microbatches"], layers=28)
    result["four_stage"] = {"workload_sha256": request["model_workload_sha256"], "stage_ms": stages,
                            "power_w": result["metrics"]["energy_j"] / duration,
                            "layers": 28, "requests_per_pipeline": request["trace"]["requests"] / duration * pipeline / 1000,
                            "operating_state": {role: {"effective_mhz": 1410 * f,
                                                       "power_cap_active_fraction": float(f < c[role + "_mhz"] / 1410)}
                                                for role, f in (("attention", fa), ("expert", fe))},
                            "provenance": "synthetic_test"}
    return result


def run_demo(directory, evaluations=10):
    config = make_demo(directory, evaluations)
    campaign = Path(directory) / "campaign"
    create_campaign(config, campaign)
    while True:
        request = ask(campaign)
        if request.get("stopped"):
            break
        tell(campaign, synthetic_result(request))
    deployment = freeze(campaign)
    write_json(Path(directory) / "synthetic-deployment.json", deployment)
    return deployment

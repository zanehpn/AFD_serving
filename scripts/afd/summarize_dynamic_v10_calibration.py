#!/usr/bin/env python3
"""Compare valid calibration MAX/dynamic cells before opening fresh held-out."""
import argparse
import hashlib
import json
from pathlib import Path
from dynamic_v10_provenance import effective_calibration_hashes
from inherited_max import validate_reference


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("baseline_suite", type=Path)
    parser.add_argument("dynamic_suite", type=Path)
    args = parser.parse_args()
    suites = [args.baseline_suite.resolve(), args.dynamic_suite.resolve()]
    pre, _ = effective_calibration_hashes(args.protocol)
    reference = pre.get('migration', {}).get('inherited_max_reference')
    if reference:
        suites[0] = Path(reference).resolve()
        expected = pre['migration']['inherited_max_provenance_sha256']
        if hashlib.sha256((suites[0] / 'PROVENANCE.json').read_bytes()).hexdigest() != expected:
            raise ValueError('inherited calibration provenance changed since preparation')
        validate_reference(suites[0])
    manifests = []
    for index, suite in enumerate(suites):
        if not (reference and index == 0) and not (suite / "COMPLETE").is_file():
            raise ValueError(f"incomplete calibration: {suite}")
        manifest = load(suite / "manifest.json")
        if manifest["evaluation_split"] != "calibration":
            raise ValueError("development report cannot read held-out measurements")
        manifests.append(manifest)
    if manifests[0]["trace_sha256"] != manifests[1]["trace_sha256"]:
        raise ValueError("calibration traces differ")
    if manifests[0]["model"] != manifests[1]["model"]:
        raise ValueError("models differ")
    metadata = manifests[1]['controller']
    if hashlib.sha256(Path(metadata['source_config']).read_bytes()).hexdigest() != hashlib.sha256((args.protocol / 'combined-controller-v10.json').read_bytes()).hexdigest():
        raise ValueError('measured controller differs from development artifact')
    pre, expected_hashes = effective_calibration_hashes(args.protocol)
    for path, expected in expected_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'policy implementation changed: {path}')
    contracts = [manifest["comparison_contract"] for manifest in manifests]
    if "compute_gate_on_attention" not in contracts[1]["server"]:
        suite = suites[1]
        launch_path = suite.parents[2] / "afd_serving" / suite.parent.name / suite.name / "launch_config.json"
        launch = load(launch_path)
        if launch["compute_gate_on_attention"] is not False or manifests[1]["runtime_contract"]["compute_gate_on_attention"] is not False:
            raise ValueError("gate placement differs")
        contracts[1]["server"]["compute_gate_on_attention"] = False
    if reference:
        # The old machine/runtime is deliberately retained as an inherited reference.
        # Check workload/serving semantics; never label the energy ratio as same-server.
        for key in ['model', 'topology', 'server', 'generation', 'arrival_rate_scaling']:
            if contracts[0][key] != contracts[1][key]:
                raise ValueError(f'inherited calibration {key} differs')
    elif contracts[0] != contracts[1]:
        raise ValueError("calibration comparison contracts differ")
    rows = []
    for rate in [1, 2, 4]:
        summaries = []
        request_sets = []
        expert_low_fraction = 0.0
        transitions = 0
        for arm, suite in enumerate(suites):
            case = (suite / f"rps-{rate}" if reference and arm == 0 else
                    suite.parents[2] / "afd_serving" / suite.parent.name / suite.name / f"rps-{rate}")
            summary = load(case / "summary.json")
            telemetry = load(case / "telemetry.json")
            ownership = load(case / "gpu-contamination-validation.json")
            if summary['requests'] != 200 or summary["completed_requests"] != 200 or summary["failed_requests"]:
                raise ValueError("expected 200 completed calibration requests")
            if telemetry["returncode"] or telemetry["sample_error_count"] or telemetry["sample_time_coverage"] < .99:
                raise ValueError("invalid telemetry")
            if not ownership["verified"] or ownership["foreign_process_samples"]:
                raise ValueError("invalid GPU ownership")
            if ownership['max_gap_s'] > 1.5 or ownership.get('max_collection_s', 0) > 1.5:
                raise ValueError('invalid GPU monitoring timing')
            request_path = case / "requests.jsonl" if reference and arm == 0 else Path(telemetry["request_log"])
            requests = [json.loads(line) for line in request_path.read_text().splitlines() if line]
            request_sets.append({(r["source_index"], r["source_timestamp"], r["input_tokens"], r["output_tokens"]) for r in requests})
            if arm == 1:
                controller = load(case / "controller-summary.json")
                if controller["status"] != "complete" or not controller["restored_to_guard"] or controller["routing_updates"] <= 0:
                    raise ValueError("controller validity failed")
                actions = [json.loads(line) for line in (case / "controller-actions.jsonl").read_text().splitlines() if line]
                start = telemetry["started_wall_ns"]
                finish = telemetry["finished_wall_ns"]
                events = sorted([r for r in actions if r["event"] == "controller_start" or (r["event"] == "transition" and r["role"] == "expert")], key=lambda r: r["wall_ns"])
                state = "f1410-p400"
                cursor = start
                low_ns = 0
                for event in events:
                    stamp = min(max(event["wall_ns"], start), finish)
                    if state == "f1050-p400":
                        low_ns += max(stamp - cursor, 0)
                    state = event["states"]["expert"] if event["event"] == "controller_start" else event["to_state"]
                    cursor = stamp
                    transitions += event["event"] == "transition"
                if state == "f1050-p400":
                    low_ns += max(finish - cursor, 0)
                expert_low_fraction = low_ns / (finish - start)
            summaries.append(summary)
        if len(request_sets[0]) != 200 or request_sets[0] != request_sets[1]:
            raise ValueError("calibration request identities or shapes differ")
        baseline, dynamic = summaries
        if any(baseline[key] != dynamic[key] for key in ('input_tokens', 'output_tokens')):
            raise ValueError('matched token counts differ')
        ratios = {"ttft_ratio": dynamic["ttft_ms"]["p90"] / baseline["ttft_ms"]["p90"],
                  "tpot_ratio": dynamic["tpot_ms"]["p90"] / baseline["tpot_ms"]["p90"],
                  "output_tps_ratio": dynamic["output_token_throughput_tps"] / baseline["output_token_throughput_tps"]}
        passed = ratios["ttft_ratio"] <= 1.05 and ratios["tpot_ratio"] <= 1.05 and ratios["output_tps_ratio"] >= .95
        rows.append({"rps": rate, **ratios, "gate_pass": passed,
                     "baseline_energy_j": baseline["energy_j"], "dynamic_energy_j": dynamic["energy_j"],
                     "raw_saving": 1 - dynamic["energy_j"] / baseline["energy_j"],
                     "expert_low_commanded_time_fraction": expert_low_fraction,
                     "expert_transitions": transitions})
    result = {"status": "complete_calibration_only", "model": manifests[0]["model"],
              "selection_split": "calibration", "formal_evaluation_eligible": False,
              "heldout_outcomes_read": 0, "rows": rows,
              "all_calibration_slo_pass": all(r["gate_pass"] for r in rows),
              "pooled_raw_saving": 1 - sum(r["dynamic_energy_j"] for r in rows) / sum(r["baseline_energy_j"] for r in rows),
              "controller_sha256": hashlib.sha256((args.protocol / "combined-controller-v10.json").read_bytes()).hexdigest()}
    if reference:
        result['calibration_comparison_scope'] = 'cross_server_inherited_max_reference'
        result['destination_max_calibration_run'] = False
        result['same_server_saving_measured'] = False
        result['inherited_max_reference'] = reference
        result['all_calibration_reference_gates_pass'] = result.pop('all_calibration_slo_pass')
        result['pooled_reference_raw_saving'] = result.pop('pooled_raw_saving')
        for row in result['rows']:
            row['reference_raw_saving'] = row.pop('raw_saving')
    (args.protocol / "CALIBRATION_RESULTS_V10.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

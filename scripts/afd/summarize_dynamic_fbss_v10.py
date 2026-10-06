#!/usr/bin/env python3
"""Report a complete, frozen MAX/dynamic FBSS pair with the registered SLO gates."""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('baseline_suite', type=Path)
    parser.add_argument('dynamic_suite', type=Path)
    parser.add_argument('--revision', type=int, choices=[9, 10], default=10)
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    freeze = load(protocol / 'EVALUATION_FREEZE.json')
    for path, expected in freeze['shared_code_sha256'].items():
        require(digest(path) == expected, f'frozen shared code differs: {path}')
    code_hashes = dict(freeze['combined_code_sha256' if args.revision == 10 else 'reference_code_sha256'])
    amendment = protocol / 'DYNAMIC_LAUNCH_TECHNICAL_AMENDMENT.json'
    if amendment.exists():
        correction = load(amendment)
        require(correction['original_evaluation_freeze_sha256'] == digest(protocol / 'EVALUATION_FREEZE.json'), 'amendment freeze mismatch')
        require(correction['original_runner_sha256'] == code_hashes['runner'], 'amendment runner mismatch')
        require(correction['status'] == 'frozen_before_first_dynamic_launch' and not correction['controller_policy_changed'] and not correction['heldout_performance_or_energy_results_inspected'], 'invalid technical amendment')
        code_hashes['runner'] = correction['corrected_runner_sha256']
    require(freeze['heldout_requests_used_for_tuning'] == 0, 'heldout was used for tuning')
    require(freeze['latency_budget_ratio'] == 1.05, 'unexpected latency gate')
    require(freeze['output_token_throughput_min_ratio'] == 0.95, 'unexpected throughput gate')
    require(freeze['rates_rps'] == [1, 2, 4], 'unexpected rate schedule')
    suites = [args.baseline_suite.resolve(), args.dynamic_suite.resolve()]
    manifests = []
    contract_audit = []
    for arm_index, suite in enumerate(suites):
        require((suite / 'COMPLETE').is_file(), f'incomplete suite: {suite}')
        manifest = load(suite / 'manifest.json')
        require(manifest['evaluation_split'] == 'heldout', 'not heldout')
        require(manifest['measurement_eligible'], 'ineligible measurement')
        require(manifest['trace_sha256'] == freeze['heldout_trace_sha256'], 'trace differs from freeze')
        require(digest(manifest['trace']) == freeze['heldout_trace_sha256'], 'trace contents changed')
        contract = manifest['comparison_contract']
        canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(canonical(contract) == manifest['comparison_contract_sha256'], 'manifest contract hash is corrupt')
        reconstruction = None
        if canonical(contract) != freeze['comparison_contract_sha256']:
            # The causal runner omits this field from its comparison summary.
            # Recover only this omission from independent recorded launch evidence.
            require(arm_index == 1 and 'compute_gate_on_attention' not in contract['server'], 'contract differs')
            launch_path = suite.parents[2] / 'afd_serving' / suite.parent.name / suite.name / 'launch_config.json'
            launch = load(launch_path)
            require(launch['compute_gate_on_attention'] is False and manifest['runtime_contract']['compute_gate_on_attention'] is False, 'gate placement differs')
            contract['server']['compute_gate_on_attention'] = False
            require(canonical(contract) == freeze['comparison_contract_sha256'], 'additional contract differences')
            reconstruction = {'field': 'server.compute_gate_on_attention', 'value': False, 'evidence': str(launch_path), 'evidence_sha256': digest(launch_path)}
        contract_audit.append({'suite': str(suite), 'recorded_hash': manifest['comparison_contract_sha256'], 'verified_contract_hash': canonical(contract), 'missing_field_reconstruction': reconstruction})
        require(manifest['schedule']['sha256'] == freeze['schedule_sha256'], 'schedule differs')
        require(manifest['plugin']['commit'] == freeze['plugin_commit'], 'plugin differs')
        deployment_key = 'baseline_deployment_sha256' if arm_index == 0 else 'fbss_deployment_sha256'
        require(manifest['deployment_sha256'] == freeze[deployment_key] == digest(manifest['deployment']), 'deployment differs')
        require(manifest['rates_rps'] == [1, 2, 4] and manifest['repetition'] == 1, 'run schedule differs')
        if arm_index == 1:
            metadata = manifest['controller']
            require(digest(metadata['source_config']) == freeze['combined_controller_sha256' if args.revision == 10 else 'reference_controller_sha256'], 'controller config differs')
            for name, field in [('controller', 'script_sha256'), ('runner', 'runner_script_sha256'), ('replay_client', 'replay_script_sha256')]:
                require(metadata[field] == code_hashes[name], f'{name} differs')
        reset_path = suite / ('clock-reset.json' if arm_index == 0 else 'gpu-reset-check.json')
        reset = load(reset_path)
        require(reset.get('verified', reset.get('status') == 'PASS'), 'clock cleanup failed')
        manifests.append(manifest)
    require(manifests[0]['model'] == manifests[1]['model'], 'models differ')
    require(manifests[0]['schedule']['position'] == 1 and manifests[1]['schedule']['position'] == (3 if args.revision == 10 else 2), 'arm order differs')
    trace = [json.loads(line) for line in Path(manifests[0]['trace']).read_text().splitlines() if line]
    identities = {(r['source_index'], r['source_timestamp']) for r in trace}
    require(len(trace) == len(identities) == 400, 'expected 400 unique heldout requests')
    split_audit = load(protocol / 'freeze-trace-isolation-audit.json')
    require(split_audit['status'] == 'PASS' and all(v == 0 for v in split_audit['overlap_counts'].values()), 'trace isolation failed')
    serving_roots = [s.parents[2] / 'afd_serving' / s.parent.name / s.name for s in suites]
    for arm_index, serving in enumerate(serving_roots):
        launch = load(serving / 'launch_config.json')
        require(launch['runtime'] == 'native' and launch['launch_mode'] == 'process_group', 'runtime differs')
        require(launch['attention_gpus'] == os.environ.get('ECODEP_ATTENTION_GPUS', '0,1') and launch['expert_gpus'] == os.environ.get('ECODEP_EXPERT_GPUS', '2,3'), 'launch GPUs differ')
        require(launch['memory_limit_bytes'] is None, 'memory limit differs')
        require(not launch['compute_gate_on_attention'], 'routing gate moved to attention')
        if arm_index == 1:
            require(launch['routing_source_role'] == 'ffn' and launch['routing_ffn_sidecar_enabled'] == 1, 'FFN routing collector missing')
            require(launch['routing_prefill_only'] == 1 and launch['routing_layer_stride'] == 4 and launch['routing_request_detail'] == 0, 'routing collection differs from preregistration')
    rows = []
    for rate in freeze['rates_rps']:
        summaries = []
        controller = None
        transition_count = 0
        for i, serving in enumerate(serving_roots):
            case = serving / f'rps-{rate}'
            summary = load(case / 'summary.json')
            telemetry = load(case / 'telemetry.json')
            ownership = load(case / 'gpu-contamination-validation.json')
            require(summary['requests'] == summary['completed_requests'] == 400 and summary['failed_requests'] == 0, 'incomplete requests')
            require(telemetry['returncode'] == 0 and telemetry['sample_error_count'] == 0 and telemetry['sample_time_coverage'] >= .99, 'invalid telemetry')
            require(telemetry['measurement_window_source'] == 'request_first_submit_to_last_finish', 'wrong energy window')
            require(telemetry['gpu_ids'] == [int(x) for x in (os.environ.get('ECODEP_ATTENTION_GPUS', '0,1') + ',' + os.environ.get('ECODEP_EXPERT_GPUS', '2,3')).split(',')], 'wrong GPUs')
            require(ownership['verified'] and ownership['foreign_process_samples'] == 0, 'GPU contamination')
            require(ownership['max_gap_s'] <= 1.5 and ownership.get('max_collection_s', 0) <= 1.5, 'monitor timing failed')
            requests = [json.loads(line) for line in Path(telemetry['request_log']).read_text().splitlines() if line]
            require(len(requests) == 400 and {(r['source_index'], r['source_timestamp']) for r in requests} == identities, 'replayed identities differ')
            expected_by_id = {r['source_index']: r for r in trace}
            for request in requests:
                expected = expected_by_id[request['source_index']]
                require(all(request[k] == expected[k] for k in ['input_tokens', 'output_tokens', 'arrival_s']), 'request shape or arrival schedule differs')
            require(all(r['error'] is None for r in requests), 'request error')
            require(math.isclose(summary['energy_j'], telemetry['energy_j'], rel_tol=1e-9), 'energy mismatch')
            if i == 1:
                controller = load(case / 'controller-summary.json')
                require(controller['status'] == 'complete' and controller['replay_started'] and controller['replay_ended'], 'controller incomplete')
                require(controller['outstanding_at_exit'] == 0 and controller['restored_to_guard'] and controller['routing_updates'] > 0, 'controller validity gate failed')
                actions = [json.loads(line) for line in (case / 'controller-actions.jsonl').read_text().splitlines() if line]
                transition_count = sum(row.get('event') == 'transition' for row in actions)
            summaries.append(summary)
        baseline, dynamic = summaries
        require(baseline['input_tokens'] == dynamic['input_tokens'] and baseline['output_tokens'] == dynamic['output_tokens'], 'matched token counts differ')
        ratios = {
            'ttft_ratio': dynamic['ttft_ms']['p90'] / baseline['ttft_ms']['p90'],
            'tpot_ratio': dynamic['tpot_ms']['p90'] / baseline['tpot_ms']['p90'],
            'output_tps_ratio': dynamic['output_token_throughput_tps'] / baseline['output_token_throughput_tps'],
        }
        require(all(math.isfinite(v) and v > 0 for v in ratios.values()), 'invalid metric')
        passed = ratios['ttft_ratio'] <= 1.05 and ratios['tpot_ratio'] <= 1.05 and ratios['output_tps_ratio'] >= .95
        effective = dynamic['energy_j'] if passed else baseline['energy_j']
        rows.append(dict(rps=rate, baseline_energy_j=baseline['energy_j'], dynamic_energy_j=dynamic['energy_j'],
                         effective_energy_j=effective, gate_pass=passed,
                         raw_saving=1-dynamic['energy_j']/baseline['energy_j'],
                         effective_saving=1-effective/baseline['energy_j'],
                         routing_updates=controller['routing_updates'], transition_events=transition_count, **ratios))
    baseline_total = sum(r['baseline_energy_j'] for r in rows)
    result = dict(status='complete_valid_pair', model=manifests[0]['model'], repetition=1, controller_revision=args.revision,
                  heldout_requests_per_cell=400, suites=[str(s) for s in suites],
                  freeze_sha256=digest(protocol / 'EVALUATION_FREEZE.json'), rows=rows,
                  technical_amendment_sha256=digest(amendment) if amendment.exists() else None,
                  comparison_contract_audit=contract_audit,
                  pooled_raw_saving=1-sum(r['dynamic_energy_j'] for r in rows)/baseline_total,
                  pooled_effective_saving=1-sum(r['effective_energy_j'] for r in rows)/baseline_total)
    (protocol / f'DYNAMIC_RESULTS_V{args.revision}.json').write_text(json.dumps(result, indent=2)+'\n')
    (protocol / f'COMPARISON_CONTRACT_AUDIT_V{args.revision}.json').write_text(json.dumps(contract_audit, indent=2)+'\n')
    lines = [f"# {result['model']}: dynamic FBSS v{args.revision} held-out result", '',
             'Native Python; fixed 2A+2E on the registered GPU IDs. Registered order: MAX, v9, v10; 400 matched requests per rate and arm.',
             'One repetition; descriptive results without a confidence interval.', '',
             '| RPS | MAX kJ | Dynamic kJ | Raw saving | Effective saving | TTFT ratio | TPOT ratio | Output TPS ratio | Gate |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
    for r in rows:
        lines.append(f"| {r['rps']} | {r['baseline_energy_j']/1000:.2f} | {r['dynamic_energy_j']/1000:.2f} | {r['raw_saving']:.2%} | {r['effective_saving']:.2%} | {r['ttft_ratio']:.3f} | {r['tpot_ratio']:.3f} | {r['output_tps_ratio']:.3f} | {'PASS' if r['gate_pass'] else 'FAIL'} |")
    lines += ['', f"Pooled effective energy saving: **{result['pooled_effective_saving']:.2%}**.",
              'Failed relative-SLO cells use matched MAX energy under the preregistered fallback rule.',
              'Every accepted cell passed request identity/count, telemetry, GPU ownership and controller checks.', '',
              f'The dynamic manifest omits `server.compute_gate_on_attention` from its comparison hash; `COMPARISON_CONTRACT_AUDIT_V{args.revision}.json` reconstructs this field from the recorded launch configuration. The reconstructed contract exactly matches the frozen MAX contract; original manifests are retained.', '',
              'See the revision-specific result JSON, `EVALUATION_FREEZE.json` and `freeze-trace-isolation-audit.json` for provenance.']
    (protocol / f'DYNAMIC_RESULTS_V{args.revision}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

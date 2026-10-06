"""Import the user-selected 200-request RPS4 calibration MAX as historical evidence.

This adapter does not claim runtime equivalence or invent a missing TBT limit.
All normalized metrics and role power integrals retain their original evidence.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def artifact(path):
    return dict(path=str(path), sha256=sha(path))


def validate(prior, config, inputs):
    prior, inputs = Path(prior), Path(inputs)
    sys.path.insert(0, str(ROOT / 'scripts/afd'))
    from inherited_max import validate_reference
    provenance = validate_reference(prior)
    manifest = read(prior / 'manifest.json')
    summary = read(prior / 'rps-4/summary.json')
    telemetry = read(prior / 'rps-4/telemetry.json')
    ownership = read(prior / 'rps-4/gpu-contamination-validation.json')
    if (config['rps'] != 4 or config['evaluation_requests'] != 200
            or len(config['gpus']) != 4 or config['tbt_slo']
            or not config['direct_search'] or config['slo_mode'] != 'relative_max'):
        raise ValueError('Legacy reference requires RPS4, 200 requests, four GPUs and three relative SLOs')
    expected_name = {'qwen36': 'Qwen3.6-35B-A3B', 'deepseek-v2-lite': 'DeepSeek-V2-Lite-Chat'}[config['model']]
    if manifest['model'] != expected_name:
        raise ValueError('Historical model differs')
    if sha(inputs / 'calibration-source.jsonl') != manifest['trace_sha256']:
        raise ValueError('Historical calibration source differs')
    contract = manifest['comparison_contract']
    if sha(Path(config['model_path']) / 'config.json') != contract['model']['config_sha256']:
        raise ValueError('Historical model configuration differs')
    c = config['reference_configuration']
    if (len(c['attention_gpus']), len(c['expert_gpus']), c['attention_tp'], c['expert_tp'], c['microbatches']) != (2, 2, 1, 1, 2):
        raise ValueError('Historical reference must be balanced 2A2F with DBO enabled')
    if any(c[k] != v for k, v in dict(attention_mhz=1410, expert_mhz=1410, attention_power_w=400, expert_power_w=400).items()):
        raise ValueError('Historical reference operating controls differ')
    if summary['requests'] != 200 or summary['completed_requests'] != 200 or summary['failed_requests']:
        raise ValueError('Historical reference did not complete 200 requests')
    rows = [json.loads(x) for x in (prior / 'rps-4/requests.jsonl').read_text().splitlines() if x]
    selected = [json.loads(x) for x in (inputs / 'calibration.jsonl').read_text().splitlines() if x]
    keys = ('source_index', 'source_timestamp', 'input_tokens', 'output_tokens')
    if len(rows) != 200 or [tuple(r[k] for k in keys) for r in rows] != [tuple(r[k] for k in keys) for r in selected]:
        raise ValueError('Historical request identities or shapes differ')
    if any(r['http_status'] != 200 or r['error']
           or r['requested_output_tokens'] != min(r['output_tokens'], config['max_output_tokens'])
           or r['actual_output_tokens'] != r['requested_output_tokens'] for r in rows):
        raise ValueError('Historical request completion evidence failed')
    if (telemetry['returncode'] or telemetry['sample_error_count'] or telemetry['sample_time_coverage'] < .99
            or not ownership['verified'] or ownership['foreign_process_samples']):
        raise ValueError('Historical telemetry/ownership evidence failed')
    if telemetry['energy_j'] != summary['energy_j'] or telemetry['duration_s'] != summary['duration_s']:
        raise ValueError('Historical energy/time summaries disagree')
    if not math.isclose(sum(telemetry['per_gpu_energy_j']), summary['energy_j'], rel_tol=1e-9):
        raise ValueError('Historical per-GPU integrals disagree')
    if telemetry['gpu_ids'] != manifest['topology']['attention_dp2'] + manifest['topology']['ffn_ep2']:
        raise ValueError('Historical GPU role mapping differs')
    return provenance, manifest, summary, telemetry


def inherit(config, prior, inputs):
    prior, inputs = Path(prior).resolve(), Path(inputs)
    provenance, manifest, summary, telemetry = validate(prior, config, inputs)
    # Copy immutable source evidence. Its provenance hashes are validated above.
    evidence = inputs / 'legacy-max-source'
    shutil.copytree(prior, evidence)
    duration = summary['duration_s']
    metrics = dict(energy_j=summary['energy_j'], ttft_ms=summary['ttft_ms']['p90'],
                   tpot_ms=summary['tpot_ms']['p90'], output_tps=summary['output_token_throughput_tps'])
    limits = {k: metrics['output_tps' if k == 'min_output_tps' else k] * ratio
              for k, ratio in config['slo_ratios'].items()}
    receipt_path = inputs / 'inherited-max-receipt.json'
    receipt = dict(status='ok', requests=200, completed_requests=200, failed_requests=0,
                   metrics=metrics, execution_verified=True, local_execution_verified=False,
                   telemetry_valid=True, selection_split='calibration',
                   origin='historical_max_reuse', normalized_from_legacy_schema=True,
                   source_summary=artifact(evidence / 'rps-4/summary.json'),
                   source_telemetry=artifact(evidence / 'rps-4/telemetry.json'),
                   output_correctness=dict(protocol='request_completion_only_v1', verified=False,
                                           request_completion_verified=True, exact_tokens_checked=False,
                                           semantic_correctness_checked=False, requests=200, selection_split='calibration'),
                   external_observables=dict(provenance='public_replay_and_nvml', duration_s=duration,
                       role_mean_power_w=dict(attention=sum(telemetry['per_gpu_energy_j'][:2])/duration,
                                              expert=sum(telemetry['per_gpu_energy_j'][2:])/duration),
                       sample_time_coverage=telemetry['sample_time_coverage'],
                       derivation='role energy integral divided by measured serving duration; no utilization/stage labels'),
                   cost=dict(wall_seconds=duration, gpu_hours=duration*4/3600, tuning_energy_j=summary['energy_j']),
                   cost_energy_complete=False, cost_is_lower_bound=True,
                   cost_scope='Only archived serving-window cost is available; historical loading/warmup/cleanup costs are unknown')
    write(receipt_path, receipt)
    source_reference = inputs / 'inherited-max-reference.json'
    write(source_reference, dict(protocol='legacy_rps4_200_three_relative_slos_v1',
          configuration=config['reference_configuration'], trace=config['trace'], metrics=metrics,
          ratios=config['slo_ratios'], limits=limits, receipt=artifact(receipt_path),
          source_manifest=artifact(evidence / 'manifest.json'), tbt_available=False,
          selection_split='calibration', heldout_evaluated=False))
    differences = dict(source_plugin=manifest['plugin']['commit'],
                       current_plugin=read(ROOT / 'environment/official-runtime.lock.json')['plugin_commit'],
                       source_prefix_caching=True, current_prefix_caching=False,
                       source_gpu_ids=telemetry['gpu_ids'], current_gpu_ids=config['gpus'],
                       source_hardware=provenance['source_hardware'],
                       same_runtime_comparison=False, same_server_saving_measured=False,
                       tbt_limit_available=False, historical_full_setup_cost_available=False)
    write(inputs / 'reference-comparison-scope.json', differences)
    record = dict(origin='historical_max_reuse', source_directory=str(prior),
                  receipt=artifact(receipt_path), reference=artifact(source_reference),
                  source_hardware=provenance['source_hardware'], current_hardware=config['hardware'],
                  local_measurement_performed=False, local_execution_verified=False,
                  limits=limits, comparison_scope='cross_runtime_historical_reference', differences=differences,
                  cost_scope=receipt['cost_scope'])
    write(inputs / 'inherited-max.json', record)
    config['inherited_max'] = record
    config['comparison_scope'] = 'cross_runtime_historical_reference'

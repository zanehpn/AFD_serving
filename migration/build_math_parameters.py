#!/usr/bin/env python3
"""Generate equivalent physical parameters by arithmetic on frozen calibration probes.

No least squares, learned elasticity, or RPS/performance regression is used.
The rates are effective service equivalents, not independently measured chip peaks.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import re
import statistics
import struct

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration import static_dse as campaign
from static_dse.analytical import validate_parameters
from summarize_replay import percentile


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def weight_ledger(model_dir):
    """Read only safetensors headers; do not load weights or create CUDA contexts."""
    model_dir = Path(model_dir)
    raw = campaign.read(model_dir / 'config.json')
    cfg = raw.get('text_config', raw)
    layers, hidden = cfg['num_hidden_layers'], cfg['hidden_size']
    kind = cfg['model_type']
    campaign.check(kind in ('qwen3_5_moe_text', 'deepseek_v2'), 'Unsupported model arithmetic ledger')
    experts = cfg.get('num_experts', cfg.get('n_routed_experts'))
    index = campaign.read(model_dir / 'model.safetensors.index.json')['weight_map']
    totals = defaultdict(int)
    counts = defaultdict(int)
    headers = {}
    for shard in sorted(set(index.values())):
        with (model_dir / shard).open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            campaign.check(0 < size < 100_000_000, 'Invalid safetensors header')
            header_bytes = f.read(size)
            header = json.loads(header_bytes)
        import hashlib
        headers[shard] = hashlib.sha256(header_bytes).hexdigest()
        for name, meta in header.items():
            if name == '__metadata__' or name not in index:
                continue
            shape = meta['shape']
            numel = math.prod(shape)
            if '.layers.' in name and not any(x in name for x in ('visual.', 'vision.', 'mtp.')):
                match = re.search(r'\.layers\.(\d+)\.', name)
                if not match or int(match[1]) >= layers:
                    continue
                if '.mlp.experts.' in name:
                    role = 'routed'
                elif '.mlp.' in name:
                    role = 'shared_and_gate'
                else:
                    role = 'attention'
            elif name.endswith('lm_head.weight'):
                role = 'attention'  # amortize the output projection over the layer ledger
            else:
                continue
            # Deployed BF16 inference bytes, not quantized/on-disk storage bytes.
            totals[role + '_bytes'] += 2 * numel
            if len(shape) >= 2:
                totals[role + '_flops'] += 2 * numel
            counts[role] += 1
    campaign.check(all(counts[k] for k in ('attention', 'routed', 'shared_and_gate')), 'Incomplete model weight ledger')
    if kind == 'qwen3_5_moe_text':
        full_fraction = cfg['layer_types'].count('full_attention') / layers
        context_flops = full_fraction * 4 * cfg['num_attention_heads'] * cfg['head_dim']
        kv_bytes = full_fraction * 4 * cfg['num_key_value_heads'] * cfg['head_dim']
        linear_flops = ((1 - full_fraction) * 12 * cfg['linear_num_value_heads']
                        * cfg['linear_key_head_dim'] * cfg['linear_value_head_dim'])
    else:
        context_flops = 2 * cfg['num_attention_heads'] * (cfg['qk_nope_head_dim'] + cfg['qk_rope_head_dim'] + cfg['v_head_dim'])
        kv_bytes = 2 * (cfg['kv_lora_rank'] + cfg['qk_rope_head_dim'])
        linear_flops = 0
    architecture = dict(layers=layers, hidden_size=hidden, experts=experts, top_k=cfg['num_experts_per_tok'],
        attention_flops_per_token=totals['attention_flops'] / layers + linear_flops,
        attention_weight_bytes_per_layer=totals['attention_bytes'] / layers,
        attention_context_flops_per_pair=context_flops, kv_bytes_per_context_token=kv_bytes,
        attention_activation_bytes_per_token=4 * hidden,
        expert_flops_per_routed_token=totals['routed_flops'] / (layers * experts),
        expert_weight_bytes_per_layer=totals['routed_bytes'] / (layers * experts),
        expert_activation_bytes_per_routed_token=4 * hidden,
        shared_expert_flops_per_token=totals['shared_and_gate_flops'] / layers,
        shared_expert_weight_bytes_per_layer=totals['shared_and_gate_bytes'] / layers)
    return architecture, dict(model_config_sha256=campaign.sha(model_dir / 'config.json'),
        weight_index_sha256=campaign.sha(model_dir / 'model.safetensors.index.json'),
        header_sha256=headers, model_type=kind, tensor_counts=dict(counts),
        arithmetic_scope='per-layer averages; BF16 matrix work plus explicit context/linear-attention terms')


def stage_groups(files, start_ns, finish_ns):
    """Pair CUDA events per transaction, layer and microbatch; exclude warmup."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for path in files:
        for row in jsonl(path):
            if not (start_ns <= row.get('start_wall_ns', -1) <= row.get('end_wall_ns', -1) <= finish_ns):
                continue
            duration = row.get('duration_us')
            if not row.get('transaction_id') or duration is None or not math.isfinite(duration) or duration <= 0:
                continue
            if row.get('layer_idx') is None:
                continue
            key = (row['transaction_id'], row.get('stage_idx') or 0, row['layer_idx'])
            rank = grouped[key][str(path)].setdefault(row['event'], dict(time=0., bytes=0, prefill=0, decode=0))
            rank['time'] += duration / 1e6
            rank['bytes'] += int(row.get('bytes') or 0)
            rank['prefill'] = max(rank['prefill'], int(row.get('prefill_tokens') or 0))
            rank['decode'] = max(rank['decode'], int(row.get('decode_tokens') or 0))
    output, complete = [], 0
    stages = ['attention_compute', 'a2f_dispatch', 'ffn_compute', 'f2a_combine']
    for key, ranks in grouped.items():
        for events in ranks.values():
            if 'attention_compute' not in events and all(k in events for k in ('attention_layer_total', 'remote_ffn_roundtrip')):
                total, remote = events['attention_layer_total'], events['remote_ffn_roundtrip']
                t = total['time'] - remote['time']
                if t > 0:
                    events['attention_compute'] = dict(total, time=t)
        present = {stage: [r[stage] for r in ranks.values() if stage in r] for stage in stages}
        if not all(present.values()):
            continue
        complete += 1
        for stage, rows in present.items():
            slow = max(rows, key=lambda r: r['time'])
            if slow['prefill'] + slow['decode']:
                output.append(dict(stage=stage, layer=key[2], **slow))
    coverage = complete / max(len(grouped), 1)
    campaign.check(complete > 0 and coverage >= .90, f'Insufficient complete four-stage probe coverage: {coverage:.3f}')
    return output, dict(groups=len(grouped), complete_groups=complete, coverage=coverage)


def work(arch, role, row, ne, mean_context):
    n = row['prefill'] + row['decode']
    context = row['prefill'] * mean_context / 2 + row['decode'] * mean_context
    if role == 'attention':
        flops = (n * arch['attention_flops_per_token'] + context * arch['attention_context_flops_per_pair']) / 2
        volume = arch['attention_weight_bytes_per_layer'] + (n * arch['attention_activation_bytes_per_token'] + context * arch['kv_bytes_per_context_token']) / 2
    else:
        active = arch['experts'] * (1 - (1 - arch['top_k'] / arch['experts']) ** n)
        flops = n * (arch['top_k'] * arch['expert_flops_per_routed_token'] + arch['shared_expert_flops_per_token']) / ne
        volume = (active * arch['expert_weight_bytes_per_layer'] + n * arch['top_k'] * arch['expert_activation_bytes_per_routed_token']) / ne + arch['shared_expert_weight_bytes_per_layer']
    return flops, volume


def communication(rows):
    # Two payload sizes identify alpha/beta directly when the observations allow it.
    by_bytes = defaultdict(list)
    for r in rows:
        if r['bytes'] > 0:
            by_bytes[r['bytes']].append(r['time'])
    campaign.check(by_bytes, 'Communication events lack payload byte counts')
    sizes = sorted(by_bytes)
    lo, hi = sizes[0], sizes[-1]
    low_t, high_t = statistics.median(by_bytes[lo]), statistics.median(by_bytes[hi])
    slope = (high_t - low_t) / (hi - lo) if hi != lo else 0
    intercept = low_t - slope * lo
    identified = slope > 0 and intercept >= 0
    if not identified:
        # A single bandwidth-equivalent service term is identifiable; alpha is not.
        slope = max(statistics.median(values) / size for size, values in by_bytes.items())
        intercept = 0.
    return dict(messages_per_layer=1, latency_s=intercept,
                bytes_per_token=statistics.median(r['bytes'] / (r['prefill'] + r['decode']) for r in rows if r['bytes'] > 0),
                effective_bytes_per_s=1 / slope,
                identification='two_payload_direct_alpha_beta' if identified else 'bandwidth_equivalent_zero_alpha_assumption')


def build(plan_path, output):
    plan = campaign.verify_plan(plan_path)
    campaign.check(plan['phase'] == 'parameter_probe', 'Parameters may only consume sparse calibration probes')
    campaign.collect(plan)
    model_dir = campaign.ROOT / 'artifacts/models' / plan['model_tag']
    arch, ledger = weight_ledger(model_dir)
    trace = jsonl(campaign.ROOT / 'inputs/traces/calibration-200.jsonl')
    context = statistics.mean(r['input_tokens'] + min(r['output_tokens'], 128) / 2 for r in trace)
    hardware_by_topology, topologies, diagnostics, sources = {}, {}, {}, []
    for path in (model_dir / 'config.json', model_dir / 'model.safetensors.index.json'):
        sources.append(dict(path=str(path.resolve()), sha256=campaign.sha(path), selection_split='calibration'))
    for entry in plan['schedule']:
        topology = next(c['topology'] for c in plan['candidates'] if c['id'] == entry['arm'])
        suite, serving = campaign.paths(plan, entry)
        telemetry = campaign.read(serving / f'rps-{entry["rates_rps"][0]}/telemetry.json')
        sidecars = sorted(serving.glob('stage-*.jsonl'))
        rows, coverage = stage_groups(sidecars, telemetry['started_wall_ns'], telemetry['finished_wall_ns'])
        power_file = serving / f'rps-{entry["rates_rps"][0]}/power-samples.jsonl'
        powers = [r['power_w'] for r in jsonl(power_file)
                  if telemetry['started_wall_ns'] <= r['timestamp_ns'] <= telemetry['finished_wall_ns']]
        campaign.check(powers and all(len(r) == 4 for r in powers), 'Missing four-GPU power samples')
        ne = 1 if topology == '2a1e' else 2
        hw = {}
        for role, stage, positions in [('attention', 'attention_compute', [0, 1]), ('expert', 'ffn_compute', list(range(2, 2 + ne)))]:
            values = [r[i] for r in powers for i in positions]
            floor, high = percentile(values, .05), percentile(values, .95)
            campaign.check(floor > 0 and high >= floor, 'Invalid observed power distribution')
            domain = dict(reference_mhz=1410, idle_w=floor, dynamic_w_at_reference=max(high - floor, .001),
                voltage_ratio_by_mhz={str(f): 1. for f in ([810, 1050, 1290, 1410] if role == 'attention' else [1050, 1290, 1410])},
                power_identification='observed_p05_floor_plus_p95_excursion; constant_voltage_frequency_scaling_assumption')
            for phase in ('prefill', 'decode'):
                phase_rows = [r for r in rows if r['stage'] == stage and (r['prefill'] > 0) == (phase == 'prefill')]
                campaign.check(phase_rows, f'No {phase} events for {topology}/{role}; rerun a new probe plan with a suitable calibration rate')
                equivalents = [(f / r['time'], b / r['time']) for r in phase_rows for f, b in [work(arch, role, r, ne, context)]]
                domain[phase] = dict(effective_flops_per_s=statistics.median(v[0] for v in equivalents),
                                     effective_bytes_per_s=statistics.median(v[1] for v in equivalents), launch_s=0.,
                                     samples=len(equivalents), identification='direct_work_over_CUDA_time_equivalents',
                                     independent_compute_bandwidth_identified=False)
            hw[role] = domain
        # The reserved inactive device is directly observed in the 2A1E probe.
        hw['inactive_gpu_w'] = statistics.median(r[3] for r in powers) if ne == 1 else min(hw['attention']['idle_w'], hw['expert']['idle_w'])
        hardware_by_topology[topology] = hw
        topologies[topology] = dict(routing_load_factor=1., routing_assumption='balanced; observed rank imbalance is absorbed in stage equivalent rates',
            a2f=communication([r for r in rows if r['stage'] == 'a2f_dispatch']),
            f2a=communication([r for r in rows if r['stage'] == 'f2a_combine']),
            expert_collective=dict(messages_per_layer=0, latency_s=0, bytes_per_token=0, effective_bytes_per_s=1.,
                                   accounting='already_inside_measured_FFN_CUDA_region; do_not_double_count'))
        diagnostics[topology] = coverage
        for path in [*sidecars, power_file, suite / 'manifest.json', serving / f'rps-{entry["rates_rps"][0]}/telemetry.json']:
            sources.append(dict(path=str(path.resolve()), sha256=campaign.sha(path), selection_split='calibration'))
    campaign.check(set(hardware_by_topology) == {'2a1e', '2a2e'}, 'Both topology probes are required')
    inactive = hardware_by_topology['2a1e']['inactive_gpu_w']
    for hw in hardware_by_topology.values():
        hw['inactive_gpu_w'] = inactive
    parameters = dict(schema_version=2, model_kind='roofline_alpha_beta_reentrant_pipeline',
        parameter_status='calibration_service_equivalents_with_explicit_assumptions', selection_split='calibration',
        model=plan['model_tag'], calibration_trace_sha256=plan['workload']['trace_sha256'],
        measurement_sources=sources, architecture=arch, architecture_ledger=ledger,
        hardware=hardware_by_topology['2a2e'], hardware_by_topology=hardware_by_topology, topologies=topologies,
        scheduler=dict(max_num_seqs=32, max_num_batched_tokens=3072, microbatches=2, max_output_tokens=128),
        probe_plan=str(Path(plan_path).resolve()), probe_plan_sha256=campaign.sha(plan_path), diagnostics=diagnostics,
        assumptions=['compute and bandwidth equivalents are not independently identified chip constants',
                     'zero separate kernel launch term: included in measured stage service',
                     'constant voltage ratio at lower frequency is a declared approximation, not an NVML voltage measurement',
                     'per-layer average weight/context arithmetic; MLA and mixed-attention terms are explicit',
                     'online/current request shapes drive the same mathematical model at every RPS'],
        deployment_authorized=False)
    validate_parameters(parameters)
    campaign.write(output, parameters)
    return parameters


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('plan', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); build(args.plan, args.output); print(args.output)

if __name__ == '__main__':
    main()

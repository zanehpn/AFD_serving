"""Physical roofline costs and deterministic queue/batch recurrences.

This module contains no regressions, learned topology priors or fitted RPS curves.
Predictions are hypotheses for a bounded calibration validation, not SLO proofs.
All times are seconds, volumes bytes, rates per second, frequencies MHz.
"""
from __future__ import annotations

import math


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def quantile(values, q=.9):
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    low, high = math.floor(k), math.ceil(k)
    return ordered[low] * (high - k) + ordered[high] * (k - low) if high != low else ordered[low]


def validate_parameters(p):
    if p.get('selection_split') != 'calibration':
        raise ValueError('Physical parameters must come from calibration measurements')
    if p.get('model_kind') != 'roofline_alpha_beta_reentrant_pipeline':
        raise ValueError('Unsupported mathematical model')
    for key in ('layers', 'hidden_size', 'experts', 'top_k'):
        value = positive(p['architecture'][key], key)
        if int(value) != value:
            raise ValueError(f'{key} must be an integer')
    if p['architecture']['top_k'] > p['architecture']['experts']:
        raise ValueError('top_k exceeds expert count')
    for key in ('attention_flops_per_token', 'attention_weight_bytes_per_layer',
                'expert_flops_per_routed_token', 'expert_weight_bytes_per_layer'):
        positive(p['architecture'][key], key)
    for key in ('attention_context_flops_per_pair', 'kv_bytes_per_context_token',
                'attention_activation_bytes_per_token', 'expert_activation_bytes_per_routed_token',
                'shared_expert_flops_per_token', 'shared_expert_weight_bytes_per_layer'):
        nonnegative(p['architecture'][key], key)
    for role in ('attention', 'expert'):
        domain = p['hardware'][role]
        positive(domain['reference_mhz'], 'reference_mhz')
        positive(domain['idle_w'], 'idle_w')
        positive(domain['dynamic_w_at_reference'], 'dynamic_w_at_reference')
        for phase in ('prefill', 'decode'):
            kernel = domain[phase]
            positive(kernel['effective_flops_per_s'], 'effective_flops_per_s')
            positive(kernel['effective_bytes_per_s'], 'effective_bytes_per_s')
            nonnegative(kernel['launch_s'], 'launch_s')
        for f, ratio in domain['voltage_ratio_by_mhz'].items():
            positive(float(f), 'frequency')
            positive(ratio, 'voltage_ratio')
    positive(p['hardware']['inactive_gpu_w'], 'inactive_gpu_w')
    for name, topology in p['topologies'].items():
        if name not in ('2a1e', '2a2e'):
            raise ValueError('Only 2A1E/2A2E are currently modeled')
        n_e = 1 if name == '2a1e' else 2
        positive(topology['routing_load_factor'], 'routing_load_factor')
        if not 1 <= topology['routing_load_factor'] <= n_e:
            raise ValueError('Routing imbalance must lie between balanced and all-to-one')
        for stage in ('a2f', 'f2a', 'expert_collective'):
            link = topology[stage]
            nonnegative(link['messages_per_layer'], 'messages_per_layer')
            nonnegative(link['latency_s'], 'latency_s')
            nonnegative(link['bytes_per_token'], 'bytes_per_token')
            positive(link['effective_bytes_per_s'], 'effective_bytes_per_s')
    for key in ('max_num_seqs', 'max_num_batched_tokens', 'max_output_tokens'):
        value = positive(p['scheduler'][key], key)
        if value != int(value):
            raise ValueError(f'{key} must be an integer')
    if p['scheduler']['microbatches'] not in (1, 2):
        raise ValueError('Modeled DBO microbatch count must be one or two')
    if 'hardware_by_topology' in p:
        for hardware in p['hardware_by_topology'].values():
            base = {k: v for k, v in p.items() if k != 'hardware_by_topology'}
            validate_parameters(dict(base, hardware=hardware))
    return p


def point_id(topology, attention_mhz, expert_mhz):
    return f'{topology}-a{attention_mhz}-e{expert_mhz}'


def compute_s(hardware, phase, flops, nbytes, mhz):
    kernel = hardware[phase]
    rate = kernel['effective_flops_per_s'] * mhz / hardware['reference_mhz']
    return kernel['launch_s'] + max(flops / rate, nbytes / kernel['effective_bytes_per_s'])


def transfer_s(link, tokens):
    return link['messages_per_layer'] * link['latency_s'] + tokens * link['bytes_per_token'] / link['effective_bytes_per_s']


def layer_stages(p, point, chunks):
    """Service demand for a logical microbatch across both A ranks.

    chunks: (prefill_tokens, decode_tokens, attention_context_pairs).
    Architecture counts are per transformer layer and per token, not model totals.
    A throughput parameter is per physical GPU. EP uses the expected number of
    active experts under a uniform-routing approximation plus measured imbalance.
    """
    if 'hardware_by_topology' in p:
        p = dict(p, hardware=p['hardware_by_topology'][point['topology']])
    a = p['architecture']
    prefill, decode, contexts = (sum(c[k] for c in chunks) for k in range(3))
    tokens = prefill + decode
    if tokens <= 0:
        return [0., 0., 0., 0.]
    phase = 'prefill' if prefill else 'decode'
    ne = 1 if point['topology'] == '2a1e' else 2
    topo = p['topologies'][point['topology']]
    # The roofline explicitly retains the weight/KV bandwidth floor at low batch.
    f_a = tokens * a['attention_flops_per_token'] + contexts * a['attention_context_flops_per_pair']
    b_a = (tokens * a['attention_activation_bytes_per_token'] + contexts * a['kv_bytes_per_context_token']) / 2 + a['attention_weight_bytes_per_layer']
    t_a = compute_s(p['hardware']['attention'], phase, f_a / 2, b_a, point['attention_mhz'])
    touched = a['experts'] * (1 - (1 - a['top_k'] / a['experts']) ** tokens)
    imbalance = topo['routing_load_factor']
    f_e = (tokens * a['top_k'] * a['expert_flops_per_routed_token'] * imbalance / ne
           + tokens * a['shared_expert_flops_per_token'] / ne)
    b_e = (touched * a['expert_weight_bytes_per_layer'] * imbalance / ne
           + tokens * a['top_k'] * a['expert_activation_bytes_per_routed_token'] * imbalance / ne
           + a['shared_expert_weight_bytes_per_layer'])
    t_e = compute_s(p['hardware']['expert'], phase, f_e, b_e, point['expert_mhz'])
    t_e += transfer_s(topo['expert_collective'], tokens)
    return [t_a, transfer_s(topo['a2f'], tokens), t_e, transfer_s(topo['f2a'], tokens)]


def pipeline(p, point, microbatches):
    """Max-plus recurrence on four shared stage resources, revisited every layer.

    C[l,m,s] = max(C[predecessor], availability[s]) + service[l,m,s].
    The explicit schedule interleaves microbatches in a fixed layer-major order.
    It is an approximation to vLLM DBO, not an assertion of identical scheduling.
    """
    availability = [0.] * 4
    done = [0.] * len(microbatches)
    busy = [0.] * 4
    costs = [layer_stages(p, point, batch) for batch in microbatches]
    for _ in range(p['architecture']['layers']):
        for m, stages in enumerate(costs):
            for stage, cost in enumerate(stages):
                done[m] = max(done[m], availability[stage]) + cost
                availability[stage] = done[m]
                busy[stage] += cost
    return max(done), done, busy


def instantaneous_power(domain, mhz):
    key = str(mhz)
    if key not in domain['voltage_ratio_by_mhz']:
        raise ValueError('Unknown voltage/frequency point requires a physical parameter probe')
    voltage = domain['voltage_ratio_by_mhz'][key]
    return domain['idle_w'] + domain['dynamic_w_at_reference'] * voltage ** 2 * mhz / domain['reference_mhz']


def simulate(p, point, trace, rate=None):
    """Replay request arrival/length metadata through queue and batching recurrences.

    No model weights, GPU inference, regression, or heldout outcomes are accessed.
    Rate scaling preserves the calibration trace's interarrival burst pattern.
    """
    validate_parameters(p)
    if point['topology'] not in p['topologies']:
        return {'status': 'needs_parameter_measurement', 'reason': 'unmeasured_topology'}
    if not trace or any(r.get('evaluation_split') != 'calibration' for r in trace):
        raise ValueError('Only calibration request metadata may drive DSE')
    if len({r['source_index'] for r in trace}) != len(trace):
        raise ValueError('Duplicate source identities')
    if point['attention_mhz'] <= 0 or point['expert_mhz'] <= 0:
        raise ValueError('Invalid frequency')
    if 'hardware_by_topology' in p:
        p = dict(p, hardware=p['hardware_by_topology'][point['topology']])
    try:
        pa = instantaneous_power(p['hardware']['attention'], point['attention_mhz'])
        pe = instantaneous_power(p['hardware']['expert'], point['expert_mhz'])
    except ValueError as error:
        return {'status': 'needs_parameter_measurement', 'reason': str(error)}
    # A cap is not free energy saving: without a throttling model it cannot clip P.
    if pa > point['attention_power_w'] or pe > point['expert_power_w']:
        return {'status': 'needs_parameter_measurement', 'reason': 'power_cap_may_throttle_requested_frequency'}
    ordered = sorted(trace, key=lambda r: (r['arrival_s'], r['source_index']))
    first, last = ordered[0]['arrival_s'], ordered[-1]['arrival_s']
    if len(ordered) < 2 or last <= first:
        raise ValueError('At least two distinct arrivals are needed')
    scale = 1. if rate is None else (len(ordered) - 1) / (last - first) / positive(rate, 'rate')
    jobs = []
    for r in ordered:
        input_n, output_n = int(r['input_tokens']), min(int(r['output_tokens']), p['scheduler']['max_output_tokens'])
        if input_n < 1 or output_n < 2:
            raise ValueError('Model requires positive prefill and at least two output tokens')
        jobs.append(dict(arrival=(r['arrival_s'] - first) * scale, input=input_n, remaining=input_n,
                         output=output_n, produced=0, first=None, last=None, source_index=r['source_index']))
    ne = 1 if point['topology'] == '2a1e' else 2
    base_power = (2 * p['hardware']['attention']['idle_w'] + ne * p['hardware']['expert']['idle_w']
                  + (4 - 2 - ne) * p['hardware']['inactive_gpu_w'])
    energy_dynamic, stage_busy, batch_sizes = 0., [0.] * 4, []
    clock, index, active = 0., 0, []
    budget_max, seq_max = p['scheduler']['max_num_batched_tokens'], p['scheduler']['max_num_seqs']
    while index < len(jobs) or active:
        if not active and index < len(jobs):
            clock = max(clock, jobs[index]['arrival'])
        while index < len(jobs) and jobs[index]['arrival'] <= clock and len(active) < seq_max:
            active.append(jobs[index]); index += 1
        budget, selected = budget_max, []
        # FCFS chunked prefill/decode service; emitted assumptions identify its limits.
        for job in active:
            if not budget:
                break
            n = min(job['remaining'], budget) if job['remaining'] else 1
            contexts = (n * (job['input'] - job['remaining']) + n * (n + 1) / 2
                        if job['remaining'] else job['input'] + job['produced'])
            chunk = (n, 0, contexts) if job['remaining'] else (0, 1, contexts)
            selected.append((job, chunk)); budget -= n
        count = min(p['scheduler']['microbatches'], len(selected))
        partitions = [selected[k::count] for k in range(count)]
        elapsed, finish, busy = pipeline(p, point, [[chunk for _, chunk in batch] for batch in partitions])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Invalid service time')
        # Compute and its adjacent transfer share each role's power domain.
        duty_a = min(elapsed, busy[0] + busy[1])
        duty_e = min(elapsed, busy[2] + busy[3])
        energy_dynamic += (2 * (pa - p['hardware']['attention']['idle_w']) * duty_a
                           + ne * (pe - p['hardware']['expert']['idle_w']) * duty_e)
        stage_busy = [x + y for x, y in zip(stage_busy, busy)]
        batch_sizes.append(sum(sum(chunk[:2]) for _, chunk in selected))
        for m, batch in enumerate(partitions):
            for job, chunk in batch:
                if job['remaining']:
                    job['remaining'] -= chunk[0]
                    if not job['remaining']:
                        job['produced'] = 1
                        job['first'] = clock + finish[m]
                else:
                    job['produced'] += 1
                if job['produced'] == job['output']:
                    job['last'] = clock + finish[m]
        active = [job for job in active if job['last'] is None]
        clock += elapsed
    makespan = max(job['last'] for job in jobs)
    ttft = [(job['first'] - job['arrival']) * 1000 for job in jobs]
    tpot = [(job['last'] - job['first']) * 1000 / (job['output'] - 1) for job in jobs]
    return {'status': 'predicted_requires_validation', 'p90_ttft_ms': quantile(ttft),
            'p90_tpot_ms': quantile(tpot), 'output_tps': sum(j['output'] for j in jobs) / makespan,
            'energy_j_per_request': (base_power * makespan + energy_dynamic) / len(jobs),
            'makespan_s': makespan, 'mean_batch_tokens': sum(batch_sizes) / len(batch_sizes),
            'stage_utilization': [min(x / makespan, 1.) for x in stage_busy],
            'requests': len(jobs), 'allocation_gpus_count': 4,
            'tail_semantics': 'empirical_quantile_of_analytical_trace_recurrence_not_a_statistical_bound'}


def search(p, points, trace, rates, weights=None):
    weights = weights or [1 / len(rates)] * len(rates)
    if (not rates or len(weights) != len(rates) or any(not math.isfinite(w) or w <= 0 for w in weights)
        or not math.isclose(sum(weights), 1)):
        raise ValueError('Positive request weights summing to one are required')
    by_id = {point['id']: point for point in points}
    if len(by_id) != len(points) or '2a2e-max' not in by_id:
        raise ValueError('Unique candidates and a matched 2A2E MAX baseline are required')
    rows, choices = [], {}
    for rate in rates:
        baseline = simulate(p, by_id['2a2e-max'], trace, rate)
        if baseline['status'] != 'predicted_requires_validation':
            raise ValueError('MAX physical parameters are incomplete')
        eligible = []
        for point in points:
            pred = simulate(p, point, trace, rate)
            row = dict(candidate_id=point['id'], rate=rate, prediction=pred, measured_feasible=False)
            if pred['status'] == 'predicted_requires_validation':
                row['predicted_gate_pass'] = (pred['p90_ttft_ms'] <= 1.05 * baseline['p90_ttft_ms']
                                              and pred['p90_tpot_ms'] <= 1.05 * baseline['p90_tpot_ms']
                                              and pred['output_tps'] >= .95 * baseline['output_tps'])
                if row['predicted_gate_pass']:
                    eligible.append(row)
            rows.append(row)
        best = min(eligible, key=lambda r: (r['prediction']['energy_j_per_request'], r['candidate_id']))
        choices[str(rate)] = best['candidate_id']
    feasible = []
    for point in points:
        scored = [r for r in rows if r['candidate_id'] == point['id']]
        if all(r.get('predicted_gate_pass', False) for r in scored):
            feasible.append((sum(w * r['prediction']['energy_j_per_request'] for w, r in zip(weights, scored)), point['id']))
    fixed = min(feasible)[1]
    return {'model_kind': p['model_kind'], 'selection_split': 'calibration',
            'predicted_best_by_rate': choices, 'predicted_fixed_configuration': fixed,
            'rows': rows, 'formal_evaluation_eligible': False,
            'next_action': 'validate_only_selected_candidates_and_matched_MAX_on_calibration',
            'deployment_authorized': False}

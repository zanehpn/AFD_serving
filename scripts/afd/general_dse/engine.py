"""Topology-parametric operator/communication/queue model.

No topology-specific service rates, per-model performance regression, or fixed
GPU count. Ring and balanced routing are explicit analytical assumptions, not
claims about a particular NCCL or vLLM execution. All predictions need validation.
"""
from __future__ import annotations
import itertools
import json
import math
from collections import defaultdict

from .model_ir import validate as validate_model, positive_int


class MissingMeasurement(ValueError):
    pass


def finite(x, name, zero=False):
    if not math.isfinite(x) or x < 0 or (x == 0 and not zero):
        raise ValueError(f'Invalid {name}')
    return x


def validate_hardware(h):
    if h.get('schema') != 'hardware_primitives_v1' or h.get('selection_split') != 'calibration':
        raise ValueError('A calibration hardware_primitives_v1 profile is required')
    if 'hardware_by_topology' in h or 'topologies' in h:
        raise ValueError('Topology-specific stage equivalents are not portable hardware primitives')
    ids = [g['id'] for g in h['gpus']]
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError('Invalid physical GPU identities')
    for g in h['gpus']:
        finite(g['memory_bytes'], 'memory_bytes')
    for f, v in h['device']['by_mhz'].items():
        positive_int(int(f), 'frequency')
        finite(v['idle_w'], 'idle_w'); finite(v['active_w'], 'active_w')
        if v['active_w'] < v['idle_w']:
            raise ValueError('Active power below idle power')
        finite(v['memory_bytes_per_s'], 'memory bandwidth')
        if not v['gemm']:
            raise ValueError('No GEMM primitives')
        for row in v['gemm']:
            for k in ('m', 'n', 'k'):
                positive_int(row[k], k)
            finite(row['flops_per_s'], 'GEMM service rate')
    if not h['device']['by_mhz']:
        raise ValueError('No measured frequency points')
    for key, link in h['links'].items():
        a, b = map(int, key.split(':'))
        if a == b or a not in ids or b not in ids:
            raise ValueError('Invalid physical link')
        finite(link['latency_s'], 'link latency', True)
        finite(link['bandwidth_bytes_per_s'], 'link bandwidth')
    finite(h['fabric_bandwidth_bytes_per_s'], 'fabric bandwidth')
    positive_int(h['dtype_bytes'], 'profile dtype')
    return h


def topology(point, h):
    alloc = point['allocation']
    a, e = point['attention_groups'], point['expert_groups']
    ids = {g['id'] for g in h['gpus']}
    if not alloc or len(set(alloc)) != len(alloc) or not set(alloc) <= ids:
        raise ValueError('Invalid allocation')
    for groups in (a, e):
        if not groups or not groups[0] or any(len(g) != len(groups[0]) for g in groups):
            raise ValueError('Empty or uneven TP groups')
        flat = sum(groups, [])
        if len(flat) != len(set(flat)) or not set(flat) <= set(alloc):
            raise ValueError('Duplicate or unallocated GPU in role')
    if point['placement'] == 'colocated':
        if a != e or point['attention_mhz'] != point['expert_mhz']:
            raise ValueError('Colocated roles must share the same TP groups and frequency')
        ep = 1
    elif point['placement'] == 'disaggregated':
        if set(sum(a, [])) & set(sum(e, [])):
            raise ValueError('Disaggregated roles overlap')
        ep = len(e)
    else:
        raise ValueError('Unknown placement')
    for role in ('attention', 'expert'):
        positive_int(point[role + '_mhz'], 'frequency')
        finite(point[role + '_power_w'], 'power cap')
    return dict(dp=len(a), atp=len(a[0]), ep=ep, etp=len(e[0]),
                used=set(sum(a, []) + sum(e, [])), allocation=len(alloc))


def execution_support(model, point):
    """Keep analytical support separate from the currently tested native launcher."""
    a, e = point['attention_groups'], point['expert_groups']
    supported = (model.get('architecture_type') in ('deepseek_v2', 'qwen3_5_moe_text')
        and point['placement'] == 'disaggregated' and len(a) == 2 and all(len(g) == 1 for g in a)
        and len(e) in (1, 2) and all(len(g) == 1 for g in e) and len(point['allocation']) == 4)
    return dict(native_case_shape_supported=supported, automatic_launch_authorized=False,
                reason='requires matched calibration and explicit deployment adapter; analytical prediction alone never launches GPUs')


def enumerate_points(h, allocation, attention_dp, attention_tp, expert_ep, expert_tp, frequencies, placements):
    points = []
    cap = h['device']['power_limit_w']
    for dp, tp in itertools.product(attention_dp, attention_tp):
        positive_int(dp, 'DP'); positive_int(tp, 'TP')
        na = dp * tp
        if na > len(allocation):
            continue
        ag = [allocation[i*tp:(i+1)*tp] for i in range(dp)]
        for placement in placements:
            tail = [(1, tp)] if placement == 'colocated' else itertools.product(expert_ep, expert_tp)
            for ep, etp in tail:
                positive_int(ep, 'EP'); positive_int(etp, 'expert TP')
                if placement == 'disaggregated' and na + ep * etp > len(allocation):
                    continue
                eg = ag if placement == 'colocated' else [allocation[na+i*etp:na+(i+1)*etp] for i in range(ep)]
                for af, ef in itertools.product(frequencies, frequencies):
                    if placement == 'colocated' and af != ef:
                        continue
                    point = dict(id=f'{placement}-adp{dp}-atp{tp}-ep{ep}-etp{etp}-a{af}-e{ef}',
                        placement=placement, attention_groups=ag, expert_groups=eg, allocation=allocation,
                        attention_mhz=af, expert_mhz=ef, attention_power_w=cap, expert_power_w=cap)
                    topology(point, h); points.append(point)
    return points


class Physics:
    def __init__(self, model, hardware, point):
        self.m, self.h, self.p = model, hardware, point
        self.t = topology(point, hardware)
        self.notes = {'analytical_ring_collectives', 'synchronized_microbatch_dispatch',
                      'GEMM_proxy_for_non_GEMM_attention_and_elementwise_work'}
        self.cache = {}
        self.block_keys = [json.dumps(block, sort_keys=True) for block in model['blocks']]
        self.block_templates = dict(zip(self.block_keys, model['blocks']))
        if model['dtype_bytes'] != hardware['dtype_bytes']:
            raise MissingMeasurement('Model dtype differs from measured primitives')
        for role in ('attention', 'expert'):
            f = str(point[role + '_mhz'])
            if f not in hardware['device']['by_mhz']:
                raise MissingMeasurement('Unmeasured frequency: ' + f)
            if hardware['device']['by_mhz'][f]['active_w'] > point[role + '_power_w']:
                raise MissingMeasurement('Power cap may throttle; no free clipping of predicted power')
        self.idle, self.dynamic = {}, {}
        for gpu in point['allocation']:
            role = 'attention' if gpu in sum(point['attention_groups'], []) else 'expert'
            f = str(point[role + '_mhz']) if gpu in self.t['used'] else str(hardware['device']['idle_reference_mhz'])
            v = hardware['device']['by_mhz'][f]
            self.idle[gpu] = v['idle_w']
            self.dynamic[gpu] = v['active_w'] - v['idle_w']

    def compute(self, role, flops, nbytes, shape):
        if flops == 0 and nbytes == 0:
            return 0.
        values = self.h['device']['by_mhz'][str(self.p[role + '_mhz'])]
        key = (role, *shape)
        if key not in self.cache:
            probes = values['gemm']
            nearest = min(probes, key=lambda r: sum(abs(math.log2(max(n, 1) / r[k])) for n, k in zip(shape, ('m', 'k', 'n'))))
            if any(not min(r[k] for r in probes) <= n <= max(r[k] for r in probes) for n, k in zip(shape, ('m', 'k', 'n'))):
                self.notes.add('GEMM_shape_outside_measured_envelope')
            self.cache[key] = nearest['flops_per_s']
        # Separate GEMM and streaming memory measurements; no F/T,D/T from one stage.
        return max(flops / self.cache[key], nbytes / values['memory_bytes_per_s'])

    def exchange(self, transfers):
        transfers = [(a, b, v) for a, b, v in transfers if a != b and v > 0]
        if not transfers:
            return 0.
        outgoing, incoming = defaultdict(float), defaultdict(float)
        total = 0.
        for a, b, volume in transfers:
            link = self.h['links'].get(f'{a}:{b}')
            if link is None:
                raise MissingMeasurement(f'No physical link primitive for {a}:{b}')
            cost = link['latency_s'] + volume / link['bandwidth_bytes_per_s']
            outgoing[a] += cost; incoming[b] += cost; total += volume
        # Endpoint serialization plus a shared-fabric bound. Placement uses actual link IDs.
        return max(max(outgoing.values()), max(incoming.values()), total / self.h['fabric_bandwidth_bytes_per_s'])

    def allreduce(self, group, volume):
        n = len(group)
        if n == 1 or volume == 0:
            return 0.
        # Two ring phases; each step moves 1/n of the reduced tensor.
        return 2 * (n - 1) * self.exchange([(group[i], group[(i+1) % n], volume / n) for i in range(n)])

    def memory(self, scheduler, requests):
        dp, atp, ep, etp = (self.t[k] for k in ('dp', 'atp', 'ep', 'etp'))
        max_context = max(r['input_tokens'] + min(r['output_tokens'], scheduler['max_output_tokens']) for r in requests)
        per_replica = scheduler['max_num_seqs']  # all live long requests may land on one replica
        residency = {g: scheduler['workspace_bytes_per_gpu'] for g in self.p['allocation']}
        extra = self.m['embedding_weight_bytes'] + self.m['output_weight_bytes']
        for group in self.p['attention_groups']:
            for g in group:
                residency[g] += extra / atp
        for block in self.m['blocks']:
            a, f = block['attention'], block['ffn']
            # Reject unsupported divisibility rather than invent fractional attention heads.
            if a['heads'] % atp:
                return {'feasible': False, 'reason': 'attention_heads_not_divisible_by_TP'}
            if f['projection_shape'][0] % etp or f['projection_shape'][1] % etp:
                return {'feasible': False, 'reason': 'FFN_matrix_not_divisible_by_TP'}
            kv_shards = min(a['kv_heads'], atp)
            if max(a['kv_heads'], atp) % kv_shards:
                return {'feasible': False, 'reason': 'KV_heads_not_evenly_shardable'}
            kv_divisor = atp if a['kind'] == 'linear_attention' else (1 if a['kind'] == 'mla' else kv_shards)
            for group in self.p['attention_groups']:
                for g in group:
                    residency[g] += a['weight_bytes'] / atp + per_replica * (max_context * a['kv_bytes'] + a['state_bytes']) / kv_divisor
            for rank, group in enumerate(self.p['expert_groups']):
                owned = f['experts'] if self.p['placement'] == 'colocated' else len(range(rank, f['experts'], ep))
                for g in group:
                    residency[g] += (owned * f['expert_weight_bytes'] + f['shared_weight_bytes'] + f['router_weight_bytes']) / etp
        capacities = {g['id']: g['memory_bytes'] * scheduler['memory_fraction'] for g in self.h['gpus']}
        used = self.t['used']
        return dict(feasible=all(residency[g] <= capacities[g] for g in used),
                    reason='capacity_bound', required_bytes_by_gpu=residency,
                    assumption='worst-context KV with all permitted live sequences on any DP replica; workspace is explicit')

    def layer(self, block, chunks):
        """Chunks contain (DP owner, prefill, decode, context pairs). DP never shards a request."""
        a, f = block['attention'], block['ffn']
        h, b = self.m['hidden_size'], self.m['dtype_bytes']
        groups = self.p['attention_groups']; eg = self.p['expert_groups']
        local = [[c for c in chunks if c[0] == rank] for rank in range(len(groups))]
        amounts = [sum(c[1] + c[2] for c in rows) for rows in local]
        total = sum(amounts)
        times_a, active_a = [], {}
        for group, rows, n in zip(groups, local, amounts):
            if not n:
                continue
            tp = len(group); contexts = sum(c[3] for c in rows)
            kv_div = 1 if a['kind'] == 'mla' else min(a['kv_heads'], tp)
            flops = (n * a['linear_flops'] + contexts * a['context_flops']) / tp
            volume = a['weight_bytes'] / tp + n * a['activation_bytes'] / tp + contexts * a['kv_bytes'] / kv_div
            volume += len(rows) * a['state_bytes'] * 2 / tp
            seconds = self.compute('attention', flops, volume, (n, h, max(1, h // tp)))
            seconds += self.allreduce(group, n * h * b)
            times_a.append(seconds)
            active_a.update({g: seconds for g in group})
        probs = f.get('routing_probabilities', [f['top_k'] / f['experts']] * f['experts'])
        if 'routing_probabilities' not in f and f['experts'] > 1:
            self.notes.add('uniform_expected_routing_not_a_tail_bound')
        times_e, active_e = [], {}
        for rank, group in enumerate(eg):
            if self.p['placement'] == 'colocated':
                n, experts = amounts[rank], list(range(f['experts']))
            else:
                n, experts = total, list(range(rank, f['experts'], self.t['ep']))
            if not n:
                continue
            # Dense blocks within MoE models live on rank 0; unused ranks stay idle.
            routed = n * sum(probs[i] for i in experts)
            touched = sum(1 - (1 - probs[i]) ** n for i in experts)
            share_n = n if self.p['placement'] == 'colocated' else n / self.t['ep']
            tp = len(group)
            flops = (routed * f['expert_flops'] + share_n * (f['shared_flops'] + f['router_flops'])) / tp
            volume = (touched * f['expert_weight_bytes'] + (f['shared_weight_bytes'] if share_n else 0)
                      + f['router_weight_bytes'] + 4 * b * h * routed) / tp
            if routed == 0 and f['shared_flops'] == 0:
                continue
            typical_m = max(1, routed / max(touched, 1))
            seconds = self.compute('expert', flops, volume, (typical_m, h, max(1, f['projection_shape'][1] // tp)))
            seconds += self.allreduce(group, (routed + (share_n if f['shared_flops'] else 0)) * h * b)
            times_e.append(seconds); active_e.update({g: seconds for g in group})
        dispatch = combine = 0.
        network_active = {}
        if self.p['placement'] == 'disaggregated':
            forward, backward = [], []
            for group, n in zip(groups, amounts):
                for rank, dest in enumerate(eg):
                    fraction = sum(probs[i] for i in range(rank, f['experts'], self.t['ep']))
                    # Model token dispatch after routing; shared experts add a balanced stream.
                    volume = n * h * b * (fraction + (1 / self.t['ep'] if f['shared_flops'] else 0))
                    forward.append((group[0], dest[0], volume))
                    backward.append((dest[0], group[0], volume))
            dispatch, combine = self.exchange(forward), self.exchange(backward)
            # Replicate dispatched activations to TP peers with a conservative ring allgather-equivalent.
            received = {g[0]: sum(v for _, dst, v in forward if dst == g[0]) for g in eg}
            returned = {g[0]: sum(v for _, dst, v in backward if dst == g[0]) for g in groups}
            dispatch += sum(self.allreduce(g, received[g[0]]) / 2 for g in eg)
            combine += sum(self.allreduce(g, returned[g[0]]) / 2 for g in groups)
            for g in eg:
                if received[g[0]]:
                    network_active.update({gpu: dispatch + combine for gpu in g})
            for g in groups:
                if returned[g[0]]:
                    network_active.update({gpu: dispatch + combine for gpu in g})
        return [max(times_a, default=0.), dispatch, max(times_e, default=0.), combine], [active_a, network_active, active_e, {}]

    def pipeline(self, batches):
        available, done, busy = [0.] * 4, [0.] * len(batches), defaultdict(float)
        colocated = self.p['placement'] == 'colocated'
        evaluated = {(key, i): self.layer(block, chunks) for key, block in self.block_templates.items()
                     for i, chunks in enumerate(batches)}
        for key in self.block_keys:
            for i, chunks in enumerate(batches):
                costs, activities = evaluated[key, i]
                for stage, cost in enumerate(costs):
                    resource = 0 if colocated and stage == 2 else stage
                    done[i] = max(done[i], available[resource]) + cost
                    available[resource] = done[i]
                    for gpu, value in activities[stage].items():
                        busy[gpu] += value
        # Output projection happens once per token, not once per Transformer layer.
        for i, chunks in enumerate(batches):
            head = 0.
            for rank, group in enumerate(self.p['attention_groups']):
                count = sum(1 for c in chunks if c[0] == rank and (c[2] or c[4]))
                if not count:
                    continue
                tp = len(group)
                sec = self.compute('attention', count * self.m['output_flops_per_token'] / tp,
                    (self.m['output_weight_bytes'] or self.m['embedding_weight_bytes']) / tp,
                    (count, self.m['hidden_size'], max(1, self.m['output_flops_per_token'] / (2*self.m['hidden_size']*tp))))
                sec += self.allreduce(group, count * self.m['dtype_bytes'])
                head = max(head, sec)
                for gpu in group:
                    busy[gpu] += sec
            done[i] = max(done[i], available[0]) + head
            available[0] = done[i]
        return max(done), done, busy


def validate_scheduler(s):
    for key in ('max_num_seqs', 'max_num_batched_tokens', 'max_output_tokens', 'microbatches'):
        positive_int(s[key], key)
    finite(s['workspace_bytes_per_gpu'], 'workspace', True)
    if not 0 < s['memory_fraction'] <= 1:
        raise ValueError('memory_fraction must lie in (0,1]')


def simulate(model, hardware, point, trace, scheduler, rate):
    validate_model(model); validate_hardware(hardware); validate_scheduler(scheduler)
    if not trace or any(r.get('evaluation_split') != 'calibration' for r in trace):
        raise ValueError('DSE may only consume calibration request shapes')
    for key in ('source_index', 'source_timestamp'):
        vals = [r[key] for r in trace]
        if len(vals) != len(set(vals)):
            raise ValueError('Duplicate calibration request identities')
    ordered = sorted(trace, key=lambda r: (r['arrival_s'], r['source_index']))
    for r in ordered:
        positive_int(r['input_tokens'], 'input_tokens'); positive_int(r['output_tokens'], 'output_tokens')
        finite(r['arrival_s'], 'arrival_s', True)
    if len(ordered) < 2 or ordered[-1]['arrival_s'] <= ordered[0]['arrival_s']:
        raise ValueError('Need distinct request arrival times')
    finite(rate, 'offered rate')
    try:
        physics = Physics(model, hardware, point)
        memory = physics.memory(scheduler, trace)
        if not memory['feasible']:
            return dict(status='analytically_infeasible', memory=memory)
        scale = (len(ordered)-1) / ((ordered[-1]['arrival_s']-ordered[0]['arrival_s']) * rate)
        jobs = [dict(arrival=(r['arrival_s']-ordered[0]['arrival_s'])*scale, input=r['input_tokens'], remaining=r['input_tokens'],
            output=min(r['output_tokens'], scheduler['max_output_tokens']), produced=0, first=None, last=None,
            owner=i % physics.t['dp']) for i, r in enumerate(ordered)]
        clock, index, active, dynamic_energy = 0., 0, [], 0.
        batch_sizes = []
        while index < len(jobs) or active:
            if not active:
                clock = max(clock, jobs[index]['arrival'])
            while index < len(jobs) and jobs[index]['arrival'] <= clock and len(active) < scheduler['max_num_seqs']:
                active.append(jobs[index]); index += 1
            budget, selected = scheduler['max_num_batched_tokens'], []
            for job in active:
                if not budget:
                    break
                n = min(job['remaining'], budget) if job['remaining'] else 1
                context = (n*(job['input']-job['remaining']) + n*(n+1)/2 if job['remaining'] else job['input']+job['produced'])
                selected.append((job, (job['owner'], n if job['remaining'] else 0, 0 if job['remaining'] else 1,
                                       context, job['remaining'] == n)))
                budget -= n
            count = min(scheduler['microbatches'], len(selected))
            partitions = [selected[i::count] for i in range(count)]
            elapsed, finish, busy = physics.pipeline([[chunk for _, chunk in part] for part in partitions])
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Invalid analytical service time')
            dynamic_energy += sum(physics.dynamic[g] * min(elapsed, seconds) for g, seconds in busy.items())
            batch_sizes.append(sum(c[1]+c[2] for _, c in selected))
            for i, part in enumerate(partitions):
                for job, chunk in part:
                    if job['remaining']:
                        job['remaining'] -= chunk[1]
                        if not job['remaining']:
                            job['produced'] = 1; job['first'] = clock + finish[i]
                    else:
                        job['produced'] += 1
                    if job['produced'] == job['output']:
                        job['last'] = clock + finish[i]
            active = [j for j in active if j['last'] is None]
            clock += elapsed
        from statistics import mean
        from static_dse.analytical import quantile
        makespan = max(j['last'] for j in jobs)
        ttft = [(j['first']-j['arrival'])*1000 for j in jobs]
        tpot = [(j['last']-j['first'])*1000/(j['output']-1) for j in jobs if j['output'] > 1]
        return dict(status='predicted_requires_validation', p90_ttft_ms=quantile(ttft), p90_tpot_ms=quantile(tpot) if tpot else 0,
            output_tps=sum(j['output'] for j in jobs)/makespan, energy_j_per_request=(sum(physics.idle.values())*makespan+dynamic_energy)/len(jobs),
            makespan_s=makespan, requests=len(jobs), mean_batch_tokens=mean(batch_sizes), allocation_gpus_count=len(point['allocation']),
            active_gpus_count=len(physics.t['used']), memory=memory, assumptions=sorted(physics.notes),
            execution=execution_support(model, point), measured_feasible=False)
    except MissingMeasurement as exc:
        return dict(status='needs_primitive_measurement', reason=str(exc), measured_feasible=False)


def search(model, hardware, points, trace, scheduler, rates, baseline_id, weights=None, latency_ratio=1.05, throughput_ratio=.95):
    if not points or not rates or len(set(rates)) != len(rates):
        raise ValueError('Nonempty points and unique rates required')
    by_id = {p['id']: p for p in points}
    if len(by_id) != len(points) or baseline_id not in by_id:
        raise ValueError('Unique candidate IDs and an explicit baseline are required')
    allocations = [p['allocation'] for p in points]
    if any(a != allocations[0] for a in allocations):
        raise ValueError('Use the same reserved allocation for energy comparison')
    weights = weights or [1/len(rates)]*len(rates)
    if len(weights) != len(rates) or any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1):
        raise ValueError('Positive weights summing to one are required')
    rows, best = [], {}
    for rate in rates:
        baseline = simulate(model, hardware, by_id[baseline_id], trace, scheduler, rate)
        for p in points:
            pred = baseline if p['id'] == baseline_id else simulate(model, hardware, p, trace, scheduler, rate)
            row = dict(candidate_id=p['id'], rate=rate, prediction=pred, predicted_gate_pass=None)
            if baseline['status'] == pred['status'] == 'predicted_requires_validation':
                row['predicted_gate_pass'] = (pred['p90_ttft_ms'] <= latency_ratio*baseline['p90_ttft_ms']
                    and pred['p90_tpot_ms'] <= latency_ratio*baseline['p90_tpot_ms'] and pred['output_tps'] >= throughput_ratio*baseline['output_tps'])
            rows.append(row)
        eligible = [r for r in rows if r['rate'] == rate and r['predicted_gate_pass']]
        best[str(rate)] = min(eligible, key=lambda r: (r['prediction']['energy_j_per_request'], r['candidate_id']))['candidate_id'] if eligible else None
    fixed = []
    for p in points:
        candidate = [r for r in rows if r['candidate_id'] == p['id']]
        if all(r['predicted_gate_pass'] for r in candidate):
            fixed.append((sum(w*r['prediction']['energy_j_per_request'] for w,r in zip(weights,candidate)), p['id']))
    return dict(schema='general_dse_prediction_v1', selection_split='calibration', model=model['name'], points=points,
        baseline_id=baseline_id, contract=dict(latency_ratio_max=latency_ratio, throughput_ratio_min=throughput_ratio),
        rates=rates, request_weights=weights, rows=rows, predicted_best_by_rate=best,
        predicted_fixed_configuration=min(fixed)[1] if fixed else None, deployment_authorized=False,
        formal_evaluation_eligible=False, topology_specific_calibration_required=False,
        unknown_candidates=[dict(candidate_id=r['candidate_id'], rate=r['rate'], reason=r['prediction'].get('reason'))
                            for r in rows if r['prediction']['status']=='needs_primitive_measurement'])

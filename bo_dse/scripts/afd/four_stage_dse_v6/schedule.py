"""Deterministic finite-population A/dispatch/F/combine scheduling prior.

Every microbatch returns to Attention at the next layer only after combine.
FIFO, nonpreemptive resources; the two communication directions share a resource
unless the frozen model explicitly describes independent resources. This is a
schedule prediction under declared assumptions, not measured GPU waiting.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import heapq
import math


@lru_cache(maxsize=16384)
def _simulate(costs, microbatches, shared_communication):
    resources = (0, 1, 2, 1 if shared_communication else 3)
    queues = [deque() for _ in range(4)]
    busy = [False] * 4
    for batch in range(microbatches):
        queues[0].append((batch, 0, 0))
    events, now, serial = [], 0., 0
    while True:
        for resource, queue in enumerate(queues):
            if queue and not busy[resource]:
                batch, layer, stage = queue.popleft()
                serial += 1
                heapq.heappush(events, (now + costs[layer][stage], serial, resource, batch, layer, stage))
                busy[resource] = True
        if not events:
            return now
        now = events[0][0]
        # All simultaneous completions become ready before resources dispatch.
        completed = []
        while events and events[0][0] == now:
            completed.append(heapq.heappop(events))
        for _, _, resource, batch, layer, stage in completed:
            busy[resource] = False
            stage += 1
            if stage == 4:
                layer, stage = layer + 1, 0
            if layer < len(costs):
                queues[resources[stage]].append((batch, layer, stage))


def schedule_time(stage_totals, microbatches, layers, model):
    if len(stage_totals) != 4 or any(not math.isfinite(v) or v <= 0 for v in stage_totals):
        raise ValueError("finite positive stage totals required")
    if model.get("type") != "finite_microbatch_fifo_v1":
        raise ValueError("unknown finite microbatch schedule")
    if model.get("communication") not in {"shared_roundtrip", "independent_directions"}:
        raise ValueError("explicit communication resource mapping required")
    if type(microbatches) is not int or microbatches < 1 or type(layers) is not int or layers < 1:
        raise ValueError("positive integer microbatches and layers required")
    weights = model.get("layer_weights")
    if weights is None:
        weights = [[1. / layers] * 4 for _ in range(layers)]
    if len(weights) != layers or any(len(row) != 4 for row in weights):
        raise ValueError("schedule layer shape mismatch")
    if any(not math.isfinite(v) or v < 0 for row in weights for v in row):
        raise ValueError("invalid layer weights")
    if any(not math.isclose(sum(row[k] for row in weights), 1., abs_tol=1e-7) for k in range(4)):
        raise ValueError("layer weights must conserve stage service")
    costs = tuple(tuple(total * row[k] for k, total in enumerate(stage_totals)) for row in weights)
    return _simulate(costs, microbatches, model["communication"] == "shared_roundtrip")

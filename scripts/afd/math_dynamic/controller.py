"""Causal mathematical DVFS; consumes lifecycle events, never a replay trace."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import select
import signal
import sys
import time
from collections import deque
from pathlib import Path

AFD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AFD))
from static_dse.analytical import pipeline, instantaneous_power, validate_parameters

spec = importlib.util.spec_from_file_location('math_dvfs_actuator', AFD / 'causal_dvfs/controller.py')
legacy = importlib.util.module_from_spec(spec); sys.modules[spec.name] = legacy; spec.loader.exec_module(legacy)


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config['selection_split'] != 'calibration' or config['method'] != 'static_bound_mathematical_causal_dvfs':
        raise ValueError('Invalid mathematical controller configuration')
    for name, expected in config['files_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen controller dependency changed: {name}')
    params = config['parameters']
    if params.get('model_kind') == 'general_operator_physics':
        from general_dse import engine, model_ir
        model_ir.validate(params['general_model']); engine.validate_hardware(params['general_hardware'])
        engine.validate_scheduler(params['scheduler'])
    else:
        validate_parameters(params)
    topology = config['static_configuration']['topology']
    if any(point['topology'] != topology for point in config['points']):
        raise ValueError('Dynamic control cannot change the frozen topology')
    if (len(config['attention_gpus']) != 2 or len(config['expert_gpus']) != (1 if topology == '2a1e' else 2)
        or len(config['allocation']) != 4 or len(set(config['allocation'])) != 4
        or len(set(config['attention_gpus'] + config['expert_gpus'])) != len(config['attention_gpus'] + config['expert_gpus'])
        or not set(config['attention_gpus'] + config['expert_gpus']) <= set(config['allocation'])):
        raise ValueError('Controller physical GPU mapping does not match static topology')
    points = {p['id']: p for p in config['points']}
    initial = config['static_configuration']
    if (len(points) != len(config['points']) or config['guard_id'] not in points
        or initial['id'] not in points
        or any(initial[k] != points[initial['id']][k] for k in ('topology', 'attention_mhz', 'expert_mhz', 'attention_power_w', 'expert_power_w'))):
        raise ValueError('Invalid guard or static starting point')
    curve = config['guard_by_rps']
    if not curve or [r['rps'] for r in curve] != sorted({r['rps'] for r in curve}):
        raise ValueError('Invalid inherited guard curve')
    if any(not math.isfinite(r[k]) or r[k] <= 0 for r in curve for k in ('prefill_age_guard_ms', 'progress_gap_guard_ms')):
        raise ValueError('Invalid inherited guard thresholds')
    return config


class Controller:
    def __init__(self, config):
        self.config = config
        self.params = config['parameters']
        self.active = {}
        self.next_owner = 0
        self.arrivals = deque()
        self.last_event_ns = None
        self.started_ns = None
        self.ended = False
        self.current = dict(config['static_configuration'])
        self.pending = None
        self.pending_since = 0
        self.last_transition = 0
        self.guard = next(p for p in config['points'] if p['id'] == config['guard_id'])
        self.general_physics = None
        if self.params.get('model_kind') == 'general_operator_physics':
            from general_dse.engine import Physics, MissingMeasurement
            from general_dse.case_adapter import point
            self.general_physics = {}
            for candidate in config['points']:
                try:
                    self.general_physics[candidate['id']] = Physics(self.params['general_model'], self.params['general_hardware'],
                                                                  point(candidate, config['allocation']))
                except MissingMeasurement:
                    pass
            if self.guard['id'] not in self.general_physics:
                raise ValueError('MAX guard lacks portable primitive coverage')

    def consume(self, event):
        kind, now = event['event'], int(event['wall_ns'])
        self.last_event_ns = now
        if kind == 'submit':
            key = str(event['request_id'])
            if key in self.active:
                raise ValueError('Duplicate live request id')
            self.active[key] = dict(phase='prefill', submitted=now, progress=now,
                                   input=int(event['input_tokens']), output=int(event['requested_output_tokens']), produced=0,
                                   owner=self.next_owner % len(self.config['attention_gpus']))
            self.next_owner += 1
            self.arrivals.append(now)
            self.started_ns = now if self.started_ns is None else self.started_ns
        elif kind in ('first_token', 'progress', 'finish'):
            key = str(event['request_id'])
            if key in self.active:
                row = self.active[key]
                if kind != 'finish':
                    row.update(phase='decode', progress=now, produced=max(row['produced'], int(event.get('output_chunks') or 1)))
                else:
                    self.active.pop(key)
        elif kind == 'progress_batch':
            for event_row in event['requests']:
                key = str(event_row['request_id'])
                if key in self.active:
                    r = self.active[key]
                    count = int(event_row.get('output_chunks') or 0)
                    if count > r['produced']:
                        r.update(phase='decode', progress=now, produced=count)
        elif kind == 'replay_end':
            self.ended = True

    def limits(self, rate):
        rows = self.config['guard_by_rps']
        left = right = rows[0]
        for row in rows:
            if rate >= row['rps']:
                left = right = row
            else:
                right = row; break
        fraction = min(max((rate - left['rps']) / (right['rps'] - left['rps']), 0), 1) if right['rps'] != left['rps'] else 0
        return [left[k] + fraction * (right[k] - left[k]) for k in ('prefill_age_guard_ms', 'progress_gap_guard_ms')]

    def choose(self, now):
        while self.arrivals and self.arrivals[0] <= now - 1_000_000_000:
            self.arrivals.popleft()
        rate = len(self.arrivals)
        prefill_limit, progress_limit = self.limits(rate)
        if self.active:
            if now - self.started_ns < 1_000_000_000:
                return self.guard, 'startup_guard'
            if self.last_event_ns is not None and now - self.last_event_ns >= self.config['event_stale_ms'] * 1e6:
                return self.guard, 'stale_events_guard'
            if any((now - r['submitted']) / 1e6 >= prefill_limit for r in self.active.values() if r['phase'] == 'prefill'):
                return self.guard, 'inherited_prefill_guard'
            if any((now - r['progress']) / 1e6 >= progress_limit for r in self.active.values() if r['phase'] == 'decode'):
                return self.guard, 'inherited_progress_guard'
        if not self.active:
            return min(self.config['points'], key=lambda p: (p['attention_mhz'] + p['expert_mhz'], p['id'])), 'idle'
        if self.general_physics is not None:
            return self.choose_general()
        chunks, budget = [], self.params['scheduler']['max_num_batched_tokens']
        for row in list(self.active.values())[:self.params['scheduler']['max_num_seqs']]:
            if budget <= 0:
                break
            if row['phase'] == 'prefill':
                n = min(row['input'], budget); chunks.append((n, 0, n * row['input'] / 2)); budget -= n
            else:
                chunks.append((0, 1, row['input'] + row['produced'])); budget -= 1
        count = min(self.params['scheduler']['microbatches'], len(chunks))
        batches = [chunks[k::count] for k in range(count)]
        guard_t, _, _ = pipeline(self.params, self.guard, batches)
        hw = self.params.get('hardware_by_topology', {}).get(self.guard['topology'], self.params['hardware'])
        ne = len(self.config['expert_gpus'])
        options = []
        for point in self.config['points']:
            service, _, busy = pipeline(self.params, point, batches)
            if service > self.config['relative_service_budget'] * guard_t:
                continue
            pa = instantaneous_power(hw['attention'], point['attention_mhz'])
            pe = instantaneous_power(hw['expert'], point['expert_mhz'])
            if pa > point['attention_power_w'] or pe > point['expert_power_w']:
                continue
            # Same allocation: idle energy remains in the objective.
            idle = 2 * hw['attention']['idle_w'] + ne * hw['expert']['idle_w'] + (2 - ne) * hw['inactive_gpu_w']
            energy = (idle * service + 2 * (pa - hw['attention']['idle_w']) * min(service, busy[0] + busy[1])
                      + ne * (pe - hw['expert']['idle_w']) * min(service, busy[2] + busy[3]))
            options.append((energy, point['id'], point))
        return (min(options, key=lambda x: x[:2])[2], 'minimum_predicted_batch_energy') if options else (self.guard, 'service_guard')

    def choose_general(self):
        from general_dse.engine import MissingMeasurement
        chunks, budget = [], self.params['scheduler']['max_num_batched_tokens']
        for r in list(self.active.values())[:self.params['scheduler']['max_num_seqs']]:
            if not budget:
                break
            if r['phase'] == 'prefill':
                n = min(r['input'], budget)
                chunks.append((r['owner'], n, 0, n*(n+1)/2, n == r['input']))
            else:
                n = 1
                chunks.append((r['owner'], 0, 1, r['input']+r['produced'], True))
            budget -= n
        count = min(self.params['scheduler']['microbatches'], len(chunks))
        batches = [chunks[k::count] for k in range(count)]
        reference, _, _ = self.general_physics[self.guard['id']].pipeline(batches)
        choices = []
        for candidate in self.config['points']:
            physics = self.general_physics.get(candidate['id'])
            if physics is None:
                continue
            try:
                elapsed, _, busy = physics.pipeline(batches)
            except MissingMeasurement:
                continue
            if elapsed <= self.config['relative_service_budget'] * reference:
                energy = sum(physics.idle.values())*elapsed + sum(physics.dynamic[g]*min(elapsed,t) for g,t in busy.items())
                choices.append((energy, candidate['id'], candidate))
        return (min(choices, key=lambda x:x[:2])[2], 'portable_operator_batch_energy') if choices else (self.guard, 'service_guard')

    def transition(self, desired, now):
        if desired['id'] == self.current['id']:
            self.pending = None; return False
        up = (desired['id'] == self.guard['id'] or desired['attention_mhz'] > self.current['attention_mhz']
              or desired['expert_mhz'] > self.current['expert_mhz']
              or desired['attention_power_w'] > self.current['attention_power_w']
              or desired['expert_power_w'] > self.current['expert_power_w'])
        if up:
            return True
        if self.pending != desired['id']:
            self.pending, self.pending_since = desired['id'], now
            return False
        return (now - self.pending_since >= self.config['downshift_hold_ms'] * 1e6
                and now - self.last_transition >= self.config['minimum_transition_interval_ms'] * 1e6)


def main():
    p = argparse.ArgumentParser(__doc__)
    for arg in ('config', 'events', 'actions', 'summary', 'stop', 'ready'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--clock-url', required=True)
    p.add_argument('--submit-fifo', type=Path)
    args = p.parse_args(); config = load_config(args.config); controller = Controller(config)
    actuator = legacy.ClockActuator(config['attention_gpus'], config['expert_gpus'], args.clock_url, args.clock_url)
    def apply(point):
        acks = {}
        for role in ('attention', 'expert'):
            current = controller.current
            up = point[role + '_mhz'] > current[role + '_mhz'] or point[role + '_power_w'] > current[role + '_power_w']
            acks[role] = actuator.set_role_operating_point(role, point[role + '_mhz'], point[role + '_power_w'], up)
        controller.current = dict(point); controller.last_transition = time.time_ns()
        return acks
    def stop_signal(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal); signal.signal(signal.SIGINT, stop_signal)
    status, error, transitions, cursor, buffer = 'complete', None, 0, 0, ''
    read_fd = keepalive_fd = None
    args.actions.parent.mkdir(parents=True, exist_ok=True)
    try:
        acks = apply(config['static_configuration'])
        with args.actions.open('a') as out:
            out.write(json.dumps(dict(event='initial', point=controller.current, acknowledgements=acks)) + '\n')
        if args.submit_fifo:
            os.mkfifo(args.submit_fifo, 0o600)
            read_fd = os.open(args.submit_fifo, os.O_RDONLY | os.O_NONBLOCK)
            keepalive_fd = os.open(args.submit_fifo, os.O_WRONLY | os.O_NONBLOCK)
        args.ready.touch()
        while not args.stop.exists():
            if args.events.exists():
                with args.events.open() as f:
                    f.seek(cursor); buffer += f.read(); cursor = f.tell()
                lines = buffer.split('\n'); buffer = lines.pop()
                for line in lines:
                    if line:
                        controller.consume(json.loads(line))
            now = time.time_ns()
            desired, reason = controller.choose(now)
            if controller.transition(desired, now):
                acks = apply(desired); transitions += 1
                with args.actions.open('a') as out:
                    out.write(json.dumps(dict(wall_ns=now, event='transition', point=desired, reason=reason, acknowledgements=acks)) + '\n')
            if controller.ended and not controller.active:
                break
            if read_fd is None:
                time.sleep(config['control_interval_ms'] / 1000)
            else:
                ready, _, _ = select.select([read_fd], [], [], config['control_interval_ms'] / 1000)
                if ready:
                    try:
                        while os.read(read_fd, 4096):
                            pass
                    except BlockingIOError:
                        pass
    except (Exception, KeyboardInterrupt) as exc:
        status, error = 'failed', str(exc)
    finally:
        for fd in (read_fd, keepalive_fd):
            if fd is not None:
                os.close(fd)
        if args.submit_fifo:
            args.submit_fifo.unlink(missing_ok=True)
        try:
            restore = apply(controller.guard)
        except Exception as exc:
            status, error, restore = 'failed', f'{error}; restore: {exc}', None
        args.summary.write_text(json.dumps(dict(status=status, error=error, replay_ended=controller.ended,
            outstanding=len(controller.active), transitions=transitions, restored_to_guard=restore is not None,
            restore_acknowledgements=restore)) + '\n')
    return 0 if status == 'complete' and controller.ended and not controller.active else 1

if __name__ == '__main__':
    raise SystemExit(main())

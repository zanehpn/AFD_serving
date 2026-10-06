#!/usr/bin/env python3
"""Deterministic causal Attention/Expert DVFS controller.

The controller never opens the replay trace. It consumes only lifecycle events
that are observable at or after request submission and applies role-level clock
actions through nvcontrold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARMS = {
    "placement-max",
    "static-e1290",
    "dynamic-a",
    "dynamic-e",
    "dynamic-ae",
}


@dataclass
class RequestState:
    phase: str
    submit_ns: int
    last_progress_ns: int
    input_tokens: int
    requested_output_tokens: int


def parse_csv_ints(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value]
    if not values:
        raise ValueError("GPU lists must be non-empty")
    return values


def post(base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    call = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(call, timeout=10) as response:
        return json.load(response)


class ClockActuator:
    def __init__(
        self,
        attention_gpus: list[int],
        expert_gpus: list[int],
        attention_url: str,
        expert_url: str,
    ) -> None:
        self.role_gpus = {"attention": attention_gpus, "expert": expert_gpus}
        self.role_urls = {"attention": attention_url, "expert": expert_url}

    def set_role(self, role: str, frequency_mhz: int) -> list[dict[str, Any]]:
        gpus = self.role_gpus[role]
        url = self.role_urls[role]
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            acknowledgements = list(
                executor.map(
                    lambda gpu: post(
                        url, "/set_clock", {"gpu": gpu, "sm_mhz": frequency_mhz}
                    ),
                    gpus,
                )
            )
        by_gpu = {int(row.get("gpu", -1)): row for row in acknowledgements}
        if set(by_gpu) != set(gpus):
            raise RuntimeError(f"{role} clock acknowledgements do not cover all GPUs")
        for gpu in gpus:
            row = by_gpu[gpu]
            if (
                int(row.get("requested_mhz", -1)) != frequency_mhz
                or int(row.get("applied_mhz", -1)) != frequency_mhz
                or row.get("clock_control") != "locked"
            ):
                raise RuntimeError(
                    f"GPU {gpu} did not lock requested {frequency_mhz} MHz: {row}"
                )
        return acknowledgements

    def set_role_operating_point(
        self,
        role: str,
        frequency_mhz: int,
        power_cap_w: float,
        power_first: bool,
    ) -> list[dict[str, Any]]:
        """Apply a joint operating point with a safe transition order.

        Upshifts raise the power ceiling before the clock. Downshifts lower the
        clock before the power ceiling. Each GPU must acknowledge both actions.
        """
        gpus = self.role_gpus[role]
        url = self.role_urls[role]

        def apply(gpu: int) -> dict[str, Any]:
            def set_clock() -> dict[str, Any]:
                return post(url, "/set_clock", {"gpu": gpu, "sm_mhz": frequency_mhz})

            def set_power() -> dict[str, Any]:
                return post(
                    url,
                    "/set_power_limit",
                    {"gpu": gpu, "watts": power_cap_w},
                )

            if power_first:
                power_ack = set_power()
                clock_ack = set_clock()
            else:
                clock_ack = set_clock()
                power_ack = set_power()
            return {"gpu": gpu, "clock": clock_ack, "power": power_ack}

        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            acknowledgements = list(executor.map(apply, gpus))
        by_gpu = {int(row.get("gpu", -1)): row for row in acknowledgements}
        if set(by_gpu) != set(gpus):
            raise RuntimeError(f"{role} operating-point acknowledgements incomplete")
        for gpu in gpus:
            clock_ack = by_gpu[gpu]["clock"]
            power_ack = by_gpu[gpu]["power"]
            if (
                int(clock_ack.get("requested_mhz", -1)) != frequency_mhz
                or int(clock_ack.get("applied_mhz", -1)) != frequency_mhz
                or clock_ack.get("clock_control") != "locked"
            ):
                raise RuntimeError(
                    f"GPU {gpu} did not lock requested {frequency_mhz} MHz: {clock_ack}"
                )
            if (
                float(power_ack.get("requested_w", -1)) != power_cap_w
                or float(power_ack.get("applied_w", -1)) != power_cap_w
                or power_ack.get("power_control") != "limited"
            ):
                raise RuntimeError(
                    f"GPU {gpu} did not apply requested {power_cap_w:g} W: {power_ack}"
                )
        return acknowledgements


class CausalController:
    def __init__(self, config: dict[str, Any], arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported ablation arm {arm!r}")
        self.config = config
        self.arm = arm
        self.policy_version = int(config.get("policy_version", 1))
        self.requests: dict[str, RequestState] = {}
        self.last_event_ns: int | None = None
        self.last_submit_ns: int | None = None
        self.arrival_rate_ewma_rps = 0.0
        self.input_tokens_ewma = 0.0
        self.requested_output_tokens_ewma = 0.0
        self.outstanding_history: deque[tuple[int, int]] = deque()
        self.pressure_since_ns: dict[str, int | None] = {
            "attention": None,
            "expert": None,
        }
        self.replay_started = False
        self.replay_ended = False
        maximum_state = "guard" if self.policy_version >= 4 else "boost"
        if arm == "static-e1290":
            self.current = {"attention": "boost", "expert": "normal"}
        elif arm == "placement-max":
            self.current = {"attention": maximum_state, "expert": maximum_state}
        else:
            self.current = dict(
                config.get(
                    "initial_states",
                    {"attention": "boost", "expert": "boost"},
                )
            )
        self.candidate = dict(self.current)
        self.candidate_since_ns = {"attention": 0, "expert": 0}
        self.last_transition_ns = {"attention": 0, "expert": 0}
        self.transition_count = {"attention": 0, "expert": 0}
        self.event_counts: dict[str, int] = {}

    def consume(self, event: dict[str, Any]) -> None:
        kind = str(event["event"])
        event_ns = int(event["wall_ns"])
        self.last_event_ns = event_ns
        self.event_counts[kind] = self.event_counts.get(kind, 0) + 1
        request_id = str(event.get("request_id", ""))
        if kind == "replay_start":
            self.replay_started = True
            self._record_outstanding(event_ns)
        elif kind == "submit":
            self._update_arrival_rate(event_ns)
            self.requests[request_id] = RequestState(
                phase="prefill",
                submit_ns=event_ns,
                last_progress_ns=event_ns,
                input_tokens=int(event.get("input_tokens", 0)),
                requested_output_tokens=int(event.get("requested_output_tokens", 0)),
            )
            self._update_work_ewma(
                int(event.get("input_tokens", 0)),
                int(event.get("requested_output_tokens", 0)),
            )
            self._record_outstanding(event_ns)
        elif kind in {"first_token", "progress"}:
            self._record_progress(request_id, event_ns)
        elif kind == "progress_batch":
            for row in event.get("requests", []):
                self._record_progress(str(row.get("request_id", "")), event_ns)
        elif kind == "finish":
            self.requests.pop(request_id, None)
            self._record_outstanding(event_ns)
        elif kind == "replay_end":
            self.replay_ended = True

    def _record_progress(self, request_id: str, event_ns: int) -> None:
        state = self.requests.get(request_id)
        if state is not None:
            state.phase = "decode"
            state.last_progress_ns = event_ns

    def _update_arrival_rate(self, event_ns: int) -> None:
        signal_config = self.config.get("signals", {})
        if self.last_submit_ns is not None and event_ns > self.last_submit_ns:
            instantaneous = 1e9 / (event_ns - self.last_submit_ns)
            maximum = float(signal_config.get("max_instantaneous_arrival_rps", 100.0))
            instantaneous = min(instantaneous, maximum)
            alpha = float(signal_config.get("arrival_ewma_alpha", 0.25))
            if self.arrival_rate_ewma_rps == 0.0:
                self.arrival_rate_ewma_rps = instantaneous
            else:
                self.arrival_rate_ewma_rps = (
                    alpha * instantaneous
                    + (1.0 - alpha) * self.arrival_rate_ewma_rps
                )
        self.last_submit_ns = event_ns

    def _update_work_ewma(self, input_tokens: int, output_tokens: int) -> None:
        signal_config = self.config.get("signals", {})
        alpha = float(signal_config.get("work_ewma_alpha", 0.2))
        if self.input_tokens_ewma == 0.0:
            self.input_tokens_ewma = float(input_tokens)
        else:
            self.input_tokens_ewma = (
                alpha * input_tokens + (1.0 - alpha) * self.input_tokens_ewma
            )
        if output_tokens > 0:
            if self.requested_output_tokens_ewma == 0.0:
                self.requested_output_tokens_ewma = float(output_tokens)
            else:
                self.requested_output_tokens_ewma = (
                    alpha * output_tokens
                    + (1.0 - alpha) * self.requested_output_tokens_ewma
                )

    def _record_outstanding(self, event_ns: int) -> None:
        value = (event_ns, len(self.requests))
        if self.outstanding_history and self.outstanding_history[-1][0] == event_ns:
            self.outstanding_history[-1] = value
        else:
            self.outstanding_history.append(value)

    def _queue_growth_rps(self, now_ns: int) -> float:
        window_ms = float(self.config.get("signals", {}).get("queue_growth_window_ms", 1000.0))
        cutoff_ns = now_ns - int(window_ms * 1e6)
        while (
            len(self.outstanding_history) >= 2
            and self.outstanding_history[1][0] <= cutoff_ns
        ):
            self.outstanding_history.popleft()
        if not self.outstanding_history:
            return 0.0
        base_ns, base_outstanding = self.outstanding_history[0]
        elapsed_s = (now_ns - base_ns) / 1e9
        if elapsed_s <= 0:
            return 0.0
        return (len(self.requests) - base_outstanding) / elapsed_s

    def _arrival_ewma_at(self, now_ns: int) -> float:
        if self.last_submit_ns is None:
            return 0.0
        decay_ms = float(self.config.get("signals", {}).get("arrival_ewma_decay_ms", 2000.0))
        elapsed_ms = max(0.0, (now_ns - self.last_submit_ns) / 1e6)
        return self.arrival_rate_ewma_rps * math.exp(-elapsed_ms / decay_ms)

    def signals(self, now_ns: int) -> dict[str, Any]:
        prefill = [state for state in self.requests.values() if state.phase == "prefill"]
        decode = [state for state in self.requests.values() if state.phase == "decode"]
        oldest_prefill_ms = max(
            ((now_ns - state.submit_ns) / 1e6 for state in prefill), default=0.0
        )
        oldest_progress_gap_ms = max(
            ((now_ns - state.last_progress_ns) / 1e6 for state in decode), default=0.0
        )
        return {
            "outstanding": len(self.requests),
            "prefill": len(prefill),
            "decode": len(decode),
            "prefill_input_tokens": sum(state.input_tokens for state in prefill),
            "oldest_prefill_ms": oldest_prefill_ms,
            "oldest_progress_gap_ms": oldest_progress_gap_ms,
            "queue_growth_rps": self._queue_growth_rps(now_ns),
            "arrival_rate_ewma_rps": self._arrival_ewma_at(now_ns),
            "submitted_input_tokens_ewma": self.input_tokens_ewma,
            "requested_output_tokens_ewma": self.requested_output_tokens_ewma,
        }

    def _dynamic_target_v3(
        self,
        role: str,
        signals: dict[str, Any],
        now_ns: int,
    ) -> tuple[str, str]:
        """Select capacity for the next window from causal stage demand.

        This policy deliberately predicts normalized stage work rather than an
        RPS-specific operating point.  Its only learned inputs are frozen
        calibration capacity curves; it never consumes future requests or the
        replay trace.
        """
        predictor = self.config["predictor"]
        reference = predictor["reference_workload"]
        weights = predictor["work_weights"][role]
        role_model = predictor["role_models"][role]
        urgent = predictor["urgent_thresholds"][role]

        reference_input = float(reference["input_tokens_mean"])
        reference_output = float(reference["output_tokens_mean"])
        input_tokens = float(signals["submitted_input_tokens_ewma"])
        output_tokens = float(signals["requested_output_tokens_ewma"])
        input_ratio = input_tokens / reference_input if input_tokens > 0 else 1.0
        output_ratio = output_tokens / reference_output if output_tokens > 0 else 1.0
        work_factor = (
            float(weights["prefill"]) * input_ratio
            + float(weights["decode"]) * output_ratio
        )
        arrival_work = float(signals["arrival_rate_ewma_rps"]) * work_factor
        growth_work = max(0.0, float(signals["queue_growth_rps"])) * float(
            predictor["queue_growth_gain"]
        )
        prefill_work = (
            float(signals["prefill_input_tokens"])
            / reference_input
            * float(predictor["active_prefill_reserve"])
        )
        predicted_work = (
            arrival_work + growth_work + prefill_work
        ) * float(predictor["risk_margin"])

        maximum_capacity = float(role_model["maximum_capacity_reference_work_per_s"])
        utilization_limit = float(predictor["utilization_limit"])
        selected = "boost"
        predicted_utilization = float("inf")
        for state in ("eco", "normal", "boost"):
            capacity = maximum_capacity * float(role_model["capacity_ratio_by_state"][state])
            utilization = predicted_work / capacity if capacity > 0 else float("inf")
            if utilization <= utilization_limit:
                selected = state
                predicted_utilization = utilization
                break
        if not math.isfinite(predicted_utilization):
            predicted_utilization = predicted_work / maximum_capacity

        signals[f"{role}_predicted_reference_work_per_s"] = predicted_work
        signals[f"{role}_predicted_utilization"] = predicted_utilization
        signals[f"{role}_work_factor"] = work_factor

        oldest_prefill_ms = float(signals["oldest_prefill_ms"])
        oldest_gap_ms = float(signals["oldest_progress_gap_ms"])
        fail_high_state = "guard" if self.policy_version >= 4 else "boost"
        if oldest_prefill_ms >= float(urgent["boost_prefill_age_ms"]):
            return fail_high_state, "v3_prefill_age_fail_high"
        if (
            int(signals["decode"]) > 0
            and oldest_gap_ms >= float(urgent["boost_progress_gap_ms"])
        ):
            return fail_high_state, "v3_progress_stall_fail_high"
        if (
            role == "attention"
            and int(signals["prefill"]) > 0
            and selected == "eco"
        ):
            return "normal", "v3_submit_prefill_reserve"
        return selected, "v3_predicted_stage_capacity"

    def _dynamic_target_v1(self, role: str, signals: dict[str, Any]) -> tuple[str, str]:
        thresholds = self.config["thresholds"][role]
        outstanding = int(signals["outstanding"])
        prefill = int(signals["prefill"])
        decode = int(signals["decode"])
        prefill_tokens = int(signals["prefill_input_tokens"])
        oldest_prefill_ms = float(signals["oldest_prefill_ms"])
        oldest_gap_ms = float(signals["oldest_progress_gap_ms"])

        if (
            prefill >= int(thresholds["boost_prefill"])
            or outstanding >= int(thresholds["boost_outstanding"])
            or prefill_tokens >= int(thresholds["boost_prefill_tokens"])
            or oldest_prefill_ms >= float(thresholds["boost_prefill_age_ms"])
            or (
                decode > 0
                and oldest_gap_ms >= float(thresholds["boost_progress_gap_ms"])
            )
        ):
            return "boost", "v1_absolute_or_age_boost"
        if (
            outstanding >= int(thresholds["normal_outstanding"])
            or decode >= int(thresholds["normal_decode"])
            or prefill > 0
        ):
            return "normal", "v1_active_normal"
        return "eco", "v1_idle_eco"

    def _dynamic_target_v2(
        self,
        role: str,
        signals: dict[str, Any],
        now_ns: int,
    ) -> tuple[str, str]:
        thresholds = self.config["thresholds"][role]
        outstanding = int(signals["outstanding"])
        prefill = int(signals["prefill"])
        decode = int(signals["decode"])
        oldest_prefill_ms = float(signals["oldest_prefill_ms"])
        oldest_gap_ms = float(signals["oldest_progress_gap_ms"])
        queue_growth_rps = float(signals["queue_growth_rps"])
        arrival_ewma_rps = float(signals["arrival_rate_ewma_rps"])

        if role == "attention":
            if prefill >= int(thresholds["boost_prefill"]):
                return "boost", "prefill_present"
            if oldest_prefill_ms >= float(thresholds["boost_prefill_age_ms"]):
                return "boost", "prefill_age"
            if (
                queue_growth_rps >= float(thresholds["boost_queue_growth_rps"])
                and arrival_ewma_rps >= float(thresholds["boost_arrival_ewma_rps"])
            ):
                return "boost", "queue_growth"
            if outstanding > 0 or arrival_ewma_rps >= float(
                thresholds["normal_arrival_ewma_rps"]
            ):
                return "normal", "active_decode_or_arrivals"
            return "eco", "idle"

        urgent_reason = None
        if prefill > 0 and oldest_prefill_ms >= float(
            thresholds["boost_prefill_age_ms"]
        ):
            urgent_reason = "severe_prefill_age"
        elif decode > 0 and oldest_gap_ms >= float(
            thresholds["boost_progress_gap_ms"]
        ):
            urgent_reason = "decode_stall"

        growth_pressure = (
            queue_growth_rps >= float(thresholds["boost_queue_growth_rps"])
            and arrival_ewma_rps >= float(thresholds["boost_arrival_ewma_rps"])
        )
        prefill_pressure = (
            prefill >= int(thresholds["boost_prefill"])
            and oldest_prefill_ms
            >= float(thresholds["boost_backlog_prefill_age_ms"])
            and arrival_ewma_rps >= float(thresholds["boost_arrival_ewma_rps"])
        )
        backlog_pressure = growth_pressure or prefill_pressure
        if backlog_pressure:
            if self.pressure_since_ns[role] is None:
                self.pressure_since_ns[role] = now_ns
        else:
            self.pressure_since_ns[role] = None
        pressure_since_ns = self.pressure_since_ns[role]
        growth_held = (
            pressure_since_ns is not None
            and (now_ns - pressure_since_ns) / 1e6
            >= float(thresholds["boost_queue_growth_hold_ms"])
        )
        if urgent_reason is not None:
            return "boost", urgent_reason
        if growth_held:
            if prefill_pressure:
                return "boost", "sustained_prefill_backlog"
            return "boost", "sustained_queue_growth"
        return "normal", "expert_default_normal"

    def _dynamic_target(
        self,
        role: str,
        signals: dict[str, Any],
        now_ns: int,
    ) -> tuple[str, str]:
        if self.policy_version >= 3:
            return self._dynamic_target_v3(role, signals, now_ns)
        if self.policy_version >= 2:
            return self._dynamic_target_v2(role, signals, now_ns)
        return self._dynamic_target_v1(role, signals)

    def desired(self, now_ns: int) -> tuple[dict[str, str], dict[str, Any]]:
        values = self.signals(now_ns)
        dynamic_attention, attention_reason = self._dynamic_target(
            "attention", values, now_ns
        )
        dynamic_expert, expert_reason = self._dynamic_target("expert", values, now_ns)
        values["attention_policy_reason"] = attention_reason
        values["expert_policy_reason"] = expert_reason
        maximum_state = "guard" if self.policy_version >= 4 else "boost"
        targets = {
            "placement-max": {"attention": maximum_state, "expert": maximum_state},
            "static-e1290": {"attention": "boost", "expert": "normal"},
            "dynamic-a": {"attention": dynamic_attention, "expert": maximum_state},
            "dynamic-e": {"attention": maximum_state, "expert": dynamic_expert},
            "dynamic-ae": {
                "attention": dynamic_attention,
                "expert": dynamic_expert,
            },
        }[self.arm]
        stale_ms = float(self.config["safety"]["event_stale_ms"])
        if (
            self.requests
            and self.last_event_ns is not None
            and (now_ns - self.last_event_ns) / 1e6 >= stale_ms
        ):
            targets = {"attention": maximum_state, "expert": maximum_state}
            values["safety_fallback"] = 1
        else:
            values["safety_fallback"] = 0
        return targets, values

    def transition_allowed(self, role: str, target: str, now_ns: int) -> tuple[bool, str]:
        current = self.current[role]
        levels = {"eco": 0, "normal": 1, "boost": 2, "guard": 3}
        if target == current:
            self.candidate[role] = target
            self.candidate_since_ns[role] = now_ns
            return False, "unchanged"
        if levels[target] > levels[current]:
            return True, "immediate_upshift"
        if self.candidate[role] != target:
            self.candidate[role] = target
            self.candidate_since_ns[role] = now_ns
            return False, "downshift_candidate_started"
        hold_config = self.config["safety"]["downshift_hold_ms"]
        interval_config = self.config["safety"]["minimum_transition_interval_ms"]
        hold_ms = float(hold_config[role] if isinstance(hold_config, dict) else hold_config)
        interval_ms = float(
            interval_config[role] if isinstance(interval_config, dict) else interval_config
        )
        if (now_ns - self.candidate_since_ns[role]) / 1e6 < hold_ms:
            return False, "downshift_hold"
        if (now_ns - self.last_transition_ns[role]) / 1e6 < interval_ms:
            return False, "minimum_transition_interval"
        return True, "delayed_downshift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stop-marker", type=Path, required=True)
    parser.add_argument("--submit-signal-fifo", type=Path, default=None)
    parser.add_argument("--attention-gpus", required=True)
    parser.add_argument("--expert-gpus", required=True)
    parser.add_argument("--attention-url", default="http://127.0.0.1:9096")
    parser.add_argument("--expert-url", default="http://127.0.0.1:9096")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    controller = CausalController(config, args.arm)
    actuator = ClockActuator(
        parse_csv_ints(args.attention_gpus),
        parse_csv_ints(args.expert_gpus),
        args.attention_url,
        args.expert_url,
    )
    poll_s = float(config["control_interval_ms"]) / 1000.0
    frequencies = config["frequencies_mhz"]
    power_caps = config.get("power_caps_w")
    stop_requested = False

    def apply_state(
        role: str,
        state: str,
        previous_state: str | None,
    ) -> list[dict[str, Any]]:
        frequency_mhz = int(frequencies[role][state])
        if power_caps is None:
            return actuator.set_role(role, frequency_mhz)
        levels = {"eco": 0, "normal": 1, "boost": 2, "guard": 3}
        power_first = previous_state is not None and (
            levels[state] > levels[previous_state]
        )
        return actuator.set_role_operating_point(
            role,
            frequency_mhz,
            float(power_caps[role][state]),
            power_first,
        )

    def request_stop(unused_signum: int, unused_frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    args.summary.unlink(missing_ok=True)
    signal_read_fd: int | None = None
    signal_keepalive_fd: int | None = None
    if args.submit_signal_fifo is not None:
        args.submit_signal_fifo.parent.mkdir(parents=True, exist_ok=True)
        args.submit_signal_fifo.unlink(missing_ok=True)
        os.mkfifo(args.submit_signal_fifo, 0o600)
        signal_read_fd = os.open(args.submit_signal_fifo, os.O_RDONLY | os.O_NONBLOCK)
        signal_keepalive_fd = os.open(args.submit_signal_fifo, os.O_WRONLY | os.O_NONBLOCK)
    offset = 0
    buffer = ""
    started_ns = time.time_ns()
    last_signals: dict[str, float | int] = {}
    status = "complete"
    error: str | None = None
    try:
        initial_acks = {}
        for role in ("attention", "expert"):
            initial_acks[role] = apply_state(role, controller.current[role], None)
        append_jsonl(
            args.output,
            {
                "event": "controller_start",
                "wall_ns": time.time_ns(),
                "arm": args.arm,
                "states": dict(controller.current),
                "acknowledgements": initial_acks,
            },
        )
        while not stop_requested and not args.stop_marker.exists():
            if args.events.exists():
                with args.events.open("r", encoding="utf-8") as stream:
                    stream.seek(offset)
                    buffer += stream.read()
                    offset = stream.tell()
                lines = buffer.split("\n")
                buffer = lines.pop()
                for line in lines:
                    if line:
                        controller.consume(json.loads(line))
            now_ns = time.time_ns()
            desired, last_signals = controller.desired(now_ns)
            for role in ("attention", "expert"):
                allowed, reason = controller.transition_allowed(role, desired[role], now_ns)
                if not allowed:
                    continue
                old_state = controller.current[role]
                target_state = desired[role]
                target_mhz = int(frequencies[role][target_state])
                try:
                    acknowledgements = apply_state(role, target_state, old_state)
                except Exception as exc:
                    # Fail high for both roles when any actuation is unverified.
                    fallback_state = (
                        "guard" if controller.policy_version >= 4 else "boost"
                    )
                    fallback_acks = {
                        fallback_role: apply_state(
                            fallback_role,
                            fallback_state,
                            controller.current[fallback_role],
                        )
                        for fallback_role in ("attention", "expert")
                    }
                    controller.current = {
                        "attention": fallback_state,
                        "expert": fallback_state,
                    }
                    append_jsonl(
                        args.output,
                        {
                            "event": "actuation_failure_fallback",
                            "wall_ns": time.time_ns(),
                            "role": role,
                            "error": f"{type(exc).__name__}: {exc}",
                            "acknowledgements": fallback_acks,
                            "signals": last_signals,
                        },
                    )
                    continue
                applied_ns = time.time_ns()
                controller.current[role] = target_state
                controller.candidate[role] = target_state
                controller.candidate_since_ns[role] = applied_ns
                controller.last_transition_ns[role] = applied_ns
                controller.transition_count[role] += 1
                append_jsonl(
                    args.output,
                    {
                        "event": "transition",
                        "wall_ns": applied_ns,
                        "role": role,
                        "from_state": old_state,
                        "to_state": target_state,
                        "frequency_mhz": target_mhz,
                        "power_cap_w": (
                            float(power_caps[role][target_state])
                            if power_caps is not None
                            else None
                        ),
                        "reason": reason,
                        "signals": last_signals,
                        "acknowledgements": acknowledgements,
                    },
                )
            if controller.replay_ended and not controller.requests:
                break
            if signal_read_fd is None:
                time.sleep(poll_s)
            else:
                ready, _, _ = select.select([signal_read_fd], [], [], poll_s)
                if ready:
                    while True:
                        try:
                            if not os.read(signal_read_fd, 4096):
                                break
                        except BlockingIOError:
                            break
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if signal_read_fd is not None:
            os.close(signal_read_fd)
        if signal_keepalive_fd is not None:
            os.close(signal_keepalive_fd)
        if args.submit_signal_fifo is not None:
            args.submit_signal_fifo.unlink(missing_ok=True)
        restore_errors = []
        restore_acks = {}
        restore_state = "guard" if controller.policy_version >= 4 else "boost"
        for role in ("attention", "expert"):
            try:
                restore_acks[role] = apply_state(
                    role,
                    restore_state,
                    controller.current[role],
                )
            except Exception as exc:
                restore_errors.append(f"{role}: {type(exc).__name__}: {exc}")
        if restore_errors:
            status = "failed"
            error = "; ".join(filter(None, [error, *restore_errors]))
        finished_ns = time.time_ns()
        summary = {
            "schema_version": 1,
            "method": "deterministic_causal_role_dvfs",
            "status": status,
            "error": error,
            "arm": args.arm,
            "policy_version": controller.policy_version,
            "submit_event_wakeup": args.submit_signal_fifo is not None,
            "started_wall_ns": started_ns,
            "finished_wall_ns": finished_ns,
            "duration_s": (finished_ns - started_ns) / 1e9,
            "replay_started": controller.replay_started,
            "replay_ended": controller.replay_ended,
            "outstanding_at_exit": len(controller.requests),
            "event_counts": controller.event_counts,
            "transition_counts": controller.transition_count,
            "final_signals": last_signals,
            "restored_to_max": not restore_errors,
            "restore_acknowledgements": restore_acks,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())

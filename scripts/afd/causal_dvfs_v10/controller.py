#!/usr/bin/env python3
"""Causal v10 control with time-window rates and relative MAX service guards."""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import math
import os
import select
import signal
import sys
import time
from pathlib import Path
from typing import Any


_LEGACY_PATH = Path(__file__).parents[1] / "causal_dvfs" / "controller.py"
_SPEC = importlib.util.spec_from_file_location("ecodep_causal_dvfs_legacy", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load legacy controller support from {_LEGACY_PATH}")
legacy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = legacy
_SPEC.loader.exec_module(legacy)

ClockActuator = legacy.ClockActuator
append_jsonl = legacy.append_jsonl
parse_csv_ints = legacy.parse_csv_ints


class RoutingTail:
    """Tail all current and future routing sidecars without opening the trace."""

    def __init__(self, directory: Path, pattern: str) -> None:
        self.directory = directory
        self.pattern = pattern
        self.offsets: dict[Path, int] = {}
        self.buffers: dict[Path, str] = {}

    def read(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob(self.pattern)):
            offset = self.offsets.get(path, 0)
            size = path.stat().st_size
            if size < offset:
                offset = 0
                self.buffers[path] = ""
            with path.open(encoding="utf-8") as stream:
                stream.seek(offset)
                chunk = stream.read()
                self.offsets[path] = stream.tell()
            buffer = self.buffers.get(path, "") + chunk
            lines = buffer.split("\n")
            self.buffers[path] = lines.pop()
            for line in lines:
                if line:
                    rows.append(json.loads(line))
        return rows


class CausalControllerV10(legacy.CausalController):
    def __init__(self, config: dict[str, Any], arm: str) -> None:
        super().__init__(config, arm)
        if self.policy_version != 5:
            raise ValueError("v10 uses the existing policy_version=5 event schema")
        if config.get("controller_revision") != 10:
            raise ValueError("v10 requires a revision-10 frozen configuration")
        budget = float(config["predictor"]["relative_service_budget_ratio"])
        if not 1.0 <= budget <= 1.05:
            raise ValueError("relative service budget must be in [1, 1.05]")
        if float(config["signals"]["rate_window_ms"]) < float(config["events"]["progress_event_interval_ms"]):
            raise ValueError("rate window must span a complete progress batch")
        curve = config["predictor"]["calibration_guard_by_rps"]
        rates = [float(row["rps"]) for row in curve]
        if not rates or rates != sorted(set(rates)):
            raise ValueError("guard rates must be unique and increasing")
        if any(not math.isfinite(float(row[key])) or float(row[key]) <= 0
               for row in curve for key in ("prefill_age_guard_ms", "progress_gap_guard_ms")):
            raise ValueError("guard thresholds must be finite and positive")
        self.arrival_samples: deque[tuple[int, int]] = deque()
        self.progress_samples: deque[tuple[int, int]] = deque()
        self.first_submit_ns: int | None = None
        self.routing_imbalance_ewma: float | None = None
        self.routing_assignments_per_token_ewma: float | None = None
        self.routing_last_observed_ns: int | None = None
        self.routing_last_ingested_ns: int | None = None
        self.routing_rows = 0
        self.routing_updates = 0
        self.routing_lag_ms_max = 0.0
        self.generated_tokens: dict[str, int] = {}
        self.decode_token_rate_ewma = 0.0
        self.state_capacity = {
            role: {
                str(point["state"]): float(point["capacity_reference_work_per_s"])
                for point in config["predictor"]["role_models"][role]["operating_points"]
            }
            for role in ("attention", "expert")
        }
        self.state_specs = {
            role: {
                str(point["state"]): (
                    int(point["frequency_mhz"]), int(point["power_cap_w"])
                )
                for point in config["predictor"]["role_models"][role]["operating_points"]
            }
            for role in ("attention", "expert")
        }

    def consume(self, event: dict[str, Any]) -> None:
        """Count progress once; never divide by inter-event batching gaps."""
        kind = str(event["event"])
        event_ns = int(event["wall_ns"])
        request_id = str(event.get("request_id", ""))
        delta = 0
        if kind == "submit":
            if self.first_submit_ns is None:
                self.first_submit_ns = event_ns
            self.generated_tokens[request_id] = 0
        elif kind in {"first_token", "progress", "finish"}:
            if request_id in self.generated_tokens:
                current = int(event.get("output_chunks") or 0)
                previous = self.generated_tokens[request_id]
                delta = max(current - previous, 0)
                self.generated_tokens[request_id] = max(current, previous)
        elif kind == "progress_batch":
            for row in event.get("requests", []):
                key = str(row.get("request_id", ""))
                if key not in self.generated_tokens:
                    continue
                current = int(row.get("output_chunks") or 0)
                previous = self.generated_tokens[key]
                delta += max(current - previous, 0)
                self.generated_tokens[key] = max(current, previous)
        if delta:
            self.progress_samples.append((event_ns, delta))
        super().consume(event)
        if kind == "finish":
            self.generated_tokens.pop(request_id, None)

    def _update_arrival_rate(self, event_ns: int) -> None:
        self.arrival_samples.append((event_ns, 1))
        self.last_submit_ns = event_ns

    def _window_rate(self, samples: deque[tuple[int, int]], now_ns: int) -> float:
        window_ns = int(float(self.config["signals"]["rate_window_ms"]) * 1e6)
        if window_ns <= 0:
            raise ValueError("rate window must be positive")
        cutoff = now_ns - window_ns
        while samples and samples[0][0] <= cutoff:
            samples.popleft()
        # Full-window denominator prevents partial-window startup spikes.
        return sum(count for stamp, count in samples if stamp <= now_ns) * 1e9 / window_ns

    def _arrival_ewma_at(self, now_ns: int) -> float:
        return self._window_rate(self.arrival_samples, now_ns)

    def _guard_limits(self, rate: float) -> tuple[float, float]:
        rows = self.config["predictor"]["calibration_guard_by_rps"]
        if not rows:
            raise ValueError("calibration guard curve is empty")
        ordered = sorted(rows, key=lambda row: float(row["rps"]))
        left = right = ordered[0]
        for row in ordered:
            if float(row["rps"]) <= rate:
                left = right = row
            else:
                right = row
                break
        span = float(right["rps"]) - float(left["rps"])
        fraction = min(max((rate - float(left["rps"])) / span, 0.0), 1.0) if span else 0.0
        return tuple(
            float(left[key]) + fraction * (float(right[key]) - float(left[key]))
            for key in ("prefill_age_guard_ms", "progress_gap_guard_ms")
        )

    def consume_routing(self, rows: list[dict[str, Any]], now_ns: int) -> dict[str, Any] | None:
        usable = []
        for row in rows:
            counts = [int(value) for value in row.get("domain_counts", [])]
            tokens = int(row.get("tokens") or 0)
            if counts and sum(counts) > 0 and tokens > 0:
                usable.append((row, counts, tokens))
        if not usable:
            return None
        totals = [0] * len(usable[0][1])
        assignments = 0
        tokens = 0
        newest = 0
        for row, counts, token_count in usable:
            if len(counts) != len(totals):
                continue
            totals = [left + right for left, right in zip(totals, counts, strict=True)]
            assignments += sum(counts)
            tokens += token_count
            newest = max(newest, int(row.get("timestamp_ns") or 0))
        if assignments <= 0 or tokens <= 0:
            return None
        imbalance = max(totals) / (assignments / len(totals))
        assignments_per_token = assignments / tokens
        alpha = float(self.config["signals"]["routing_ewma_alpha"])
        if self.routing_imbalance_ewma is None:
            self.routing_imbalance_ewma = imbalance
            self.routing_assignments_per_token_ewma = assignments_per_token
        else:
            self.routing_imbalance_ewma = (
                alpha * imbalance + (1.0 - alpha) * self.routing_imbalance_ewma
            )
            assert self.routing_assignments_per_token_ewma is not None
            self.routing_assignments_per_token_ewma = (
                alpha * assignments_per_token
                + (1.0 - alpha) * self.routing_assignments_per_token_ewma
            )
        self.routing_last_observed_ns = newest
        self.routing_last_ingested_ns = now_ns
        self.routing_rows += len(usable)
        self.routing_updates += 1
        lag_ms = max(0.0, (now_ns - newest) / 1e6)
        self.routing_lag_ms_max = max(self.routing_lag_ms_max, lag_ms)
        return {
            "event": "routing_update",
            "wall_ns": now_ns,
            "rows": len(usable),
            "domain_counts": totals,
            "imbalance": imbalance,
            "imbalance_ewma": self.routing_imbalance_ewma,
            "assignments_per_token_ewma": self.routing_assignments_per_token_ewma,
            "producer_to_controller_lag_ms": lag_ms,
        }

    def signals(self, now_ns: int) -> dict[str, Any]:
        self.decode_token_rate_ewma = self._window_rate(self.progress_samples, now_ns)
        values = super().signals(now_ns)
        values["rate_estimator"] = "count_over_fixed_time_window"
        values["startup_window_complete"] = (self.first_submit_ns is not None and (now_ns - self.first_submit_ns) / 1e6 >= float(self.config["signals"]["rate_window_ms"]))
        routing = self.config["predictor"]["routing_model"]
        stale_ms = float(self.config["signals"]["routing_stale_ms"])
        routing_age_ms = (
            math.inf
            if self.routing_last_observed_ns is None
            else max(0.0, (now_ns - self.routing_last_observed_ns) / 1e6)
        )
        if self.routing_imbalance_ewma is None or routing_age_ms > stale_ms:
            imbalance = float(routing["calibration_imbalance_p95"])
            source = "calibration_p95_stale_fallback"
        else:
            imbalance = float(self.routing_imbalance_ewma)
            source = "online_observed"
        values.update(
            {
                "routing_imbalance": imbalance,
                "routing_imbalance_source": source,
                "routing_age_ms": routing_age_ms,
                "routing_assignments_per_token_ewma": self.routing_assignments_per_token_ewma,
                "decode_output_token_rate_ewma": self.decode_token_rate_ewma,
            }
        )
        return values

    @staticmethod
    def _predict_power(model: dict[str, Any], state: str, utilization: float) -> float:
        fitted = model["states"][state]
        watts = float(fitted["idle_intercept_w"]) + float(
            fitted["dynamic_slope_w"]
        ) * utilization
        return min(max(watts, 0.0), float(fitted["power_cap_w"]))

    def _role_demand_v5(self, role: str, signals: dict[str, Any]) -> float:
        predictor = self.config["predictor"]
        reference = predictor["reference_workload"]
        weights = predictor["work_weights"][role]
        input_tokens = float(signals["submitted_input_tokens_ewma"])
        output_tokens = float(signals["requested_output_tokens_ewma"])
        input_ratio = input_tokens / float(reference["input_tokens_mean"]) if input_tokens > 0 else 1.0
        output_ratio = output_tokens / float(reference["output_tokens_mean"]) if output_tokens > 0 else 1.0
        prefill_weight = float(weights["prefill"])
        decode_weight = float(weights["decode"])
        work_factor = prefill_weight * input_ratio + decode_weight * output_ratio
        routing_factor = 1.0
        if role == "expert":
            routing_reference = float(predictor["routing_model"]["reference_imbalance_median"])
            routing_factor = max(float(signals["routing_imbalance"]) / routing_reference, 0.5)
        arrival_rate = float(signals["arrival_rate_ewma_rps"])
        prefill_arrival_work = arrival_rate * prefill_weight * input_ratio
        decode_arrival_work = arrival_rate * decode_weight * output_ratio
        observed_decode_work = (
            float(signals["decode_output_token_rate_ewma"])
            / float(reference["output_tokens_mean"])
            * decode_weight
        )
        arrival_work = prefill_arrival_work + max(
            decode_arrival_work, observed_decode_work
        )
        growth_work = (
            max(0.0, float(signals["queue_growth_rps"]))
            * work_factor
            * float(predictor["queue_growth_gain"])
        )
        horizon_s = float(predictor["prediction_horizon_ms"]) / 1000.0
        prefill_reserve = (
            float(signals["prefill_input_tokens"])
            / float(reference["input_tokens_mean"])
            * prefill_weight
            * float(predictor["active_prefill_reserve"])
            / horizon_s
        )
        predicted_work = (
            arrival_work + growth_work + prefill_reserve
        ) * float(predictor["risk_margin"]) * routing_factor
        signals[f"{role}_predicted_reference_work_per_s"] = predicted_work
        signals[f"{role}_work_factor"] = work_factor
        signals[f"{role}_routing_factor"] = routing_factor
        return predicted_work

    def _select_joint_v5(self, signals: dict[str, Any]) -> tuple[dict[str, str], str]:
        predictor = self.config["predictor"]
        guard = {
            role: str(predictor["role_models"][role]["guard_state"])
            for role in ("attention", "expert")
        }
        if int(signals["outstanding"]) > 0 and not signals["startup_window_complete"]:
            return guard, "v10_rate_window_startup_guard"
        prefill_limit, progress_limit = self._guard_limits(float(signals["arrival_rate_ewma_rps"]))
        signals["prefill_age_guard_ms"] = prefill_limit
        signals["progress_gap_guard_ms"] = progress_limit
        if float(signals["oldest_prefill_ms"]) >= prefill_limit:
            return guard, "v10_calibrated_prefill_age_guard"
        if int(signals["decode"]) > 0 and float(signals["oldest_progress_gap_ms"]) >= progress_limit:
            return guard, "v10_calibrated_progress_stall_guard"
        demand = {
            role: self._role_demand_v5(role, signals)
            for role in ("attention", "expert")
        }
        candidates = []
        for point in predictor["joint_operating_points"]:
            if not bool(
                point.get(
                    "calibration_slo_eligible",
                    point["calibration_latency_eligible"],
                )
            ):
                continue
            states = {
                "attention": str(point["attention_state"]),
                "expert": str(point["expert_state"]),
            }
            utilization = {
                role: demand[role] / self.state_capacity[role][states[role]]
                for role in ("attention", "expert")
            }
            # Match the relative-SLO objective when MAX is itself saturated.
            # A candidate may consume at most the calibrated relative service
            # budget above MAX; this is a heuristic, not a queueing guarantee.
            limits = {
                role: max(
                    float(predictor["utilization_limit"]),
                    demand[role] / self.state_capacity[role][guard[role]]
                    * float(predictor["relative_service_budget_ratio"]),
                )
                for role in ("attention", "expert")
            }
            if any(utilization[role] > limits[role] for role in ("attention", "expert")):
                continue
            power = {
                role: self._predict_power(
                    predictor["role_models"][role]["power_model"],
                    states[role],
                    min(max(utilization[role], 0.0), 1.0),
                )
                for role in ("attention", "expert")
            }
            total_power = sum(float(predictor["role_gpu_counts"][role]) * power[role] for role in ("attention", "expert"))
            candidates.append((total_power, str(point["id"]), states, utilization, power))
        if not candidates:
            return guard, "v10_relative_service_guard"
        total_power, point_id, states, utilization, power = min(candidates)
        signals["selected_joint_point"] = point_id
        signals["predicted_total_gpu_power_w"] = total_power
        for role in ("attention", "expert"):
            signals[f"{role}_predicted_utilization"] = utilization[role]
            signals[f"{role}_predicted_power_w"] = power[role]
        return states, "v10_min_power_relative_service_feasible_joint"

    def desired(self, now_ns: int) -> tuple[dict[str, str], dict[str, Any]]:
        values = self.signals(now_ns)
        dynamic, dynamic_reason = self._select_joint_v5(values)
        for role in ("attention", "expert"):
            values[f"{role}_policy_reason"] = dynamic_reason
        guard = {
            role: str(self.config["predictor"]["role_models"][role]["guard_state"])
            for role in ("attention", "expert")
        }
        if self.arm == "placement-max":
            targets = dict(guard)
        elif self.arm == "dynamic-a":
            targets = {"attention": dynamic["attention"], "expert": guard["expert"]}
        elif self.arm == "dynamic-e":
            targets = {"attention": guard["attention"], "expert": dynamic["expert"]}
        elif self.arm == "dynamic-ae":
            targets = dict(dynamic)
        else:
            raise ValueError(f"unsupported v5 arm {self.arm}")
        stale_ms = float(self.config["safety"]["event_stale_ms"])
        if self.requests and self.last_event_ns is not None and (now_ns - self.last_event_ns) / 1e6 >= stale_ms:
            targets = guard
            values["safety_fallback"] = 1
        else:
            values["safety_fallback"] = 0
        return targets, values

    def transition_allowed(self, role: str, target: str, now_ns: int) -> tuple[bool, str]:
        current = self.current[role]
        if target == current:
            self.candidate[role] = target
            self.candidate_since_ns[role] = now_ns
            return False, "unchanged"
        guard = str(self.config["predictor"]["role_models"][role]["guard_state"])
        if target == guard:
            return True, "immediate_guard"
        current_frequency, current_cap = self.state_specs[role][current]
        target_frequency, target_cap = self.state_specs[role][target]
        if target_frequency > current_frequency or target_cap > current_cap:
            return True, "immediate_resource_upshift"
        if self.state_capacity[role][target] > self.state_capacity[role][current] + 1e-9:
            return True, "immediate_capacity_upshift"
        if self.candidate[role] != target:
            self.candidate[role] = target
            self.candidate_since_ns[role] = now_ns
            return False, "downshift_candidate_started"
        hold = self.config["safety"]["downshift_hold_ms"]
        interval = self.config["safety"]["minimum_transition_interval_ms"]
        if (now_ns - self.candidate_since_ns[role]) / 1e6 < float(hold[role]):
            return False, "downshift_hold"
        if (now_ns - self.last_transition_ns[role]) / 1e6 < float(interval[role]):
            return False, "minimum_transition_interval"
        return True, "delayed_capacity_downshift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("placement-max", "dynamic-a", "dynamic-e", "dynamic-ae"), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--routing-pattern", default="routing-attention-*.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stop-marker", type=Path, required=True)
    parser.add_argument("--submit-signal-fifo", type=Path)
    parser.add_argument("--attention-gpus", required=True)
    parser.add_argument("--expert-gpus", required=True)
    parser.add_argument("--attention-url", default="http://127.0.0.1:9096")
    parser.add_argument("--expert-url", default="http://127.0.0.1:9096")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    controller = CausalControllerV10(config, args.arm)
    routing_tail = RoutingTail(args.routing_dir, args.routing_pattern)
    actuator = ClockActuator(
        parse_csv_ints(args.attention_gpus), parse_csv_ints(args.expert_gpus),
        args.attention_url, args.expert_url,
    )
    frequencies = config["frequencies_mhz"]
    power_caps = config["power_caps_w"]
    poll_s = float(config["control_interval_ms"]) / 1000.0
    stop_requested = False

    def state_level(role: str, state: str) -> float:
        return controller.state_capacity[role][state]

    def apply_state(role: str, state: str, previous: str | None) -> list[dict[str, Any]]:
        target_cap = float(power_caps[role][state])
        previous_cap = (
            float(power_caps[role][previous]) if previous is not None else target_cap
        )
        power_first = target_cap > previous_cap
        return actuator.set_role_operating_point(
            role, int(frequencies[role][state]), target_cap, power_first
        )

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    args.summary.unlink(missing_ok=True)
    read_fd: int | None = None
    keepalive_fd: int | None = None
    if args.submit_signal_fifo is not None:
        args.submit_signal_fifo.unlink(missing_ok=True)
        os.mkfifo(args.submit_signal_fifo, 0o600)
        read_fd = os.open(args.submit_signal_fifo, os.O_RDONLY | os.O_NONBLOCK)
        keepalive_fd = os.open(args.submit_signal_fifo, os.O_WRONLY | os.O_NONBLOCK)
    offset = 0
    buffer = ""
    started_ns = time.time_ns()
    status = "complete"
    error: str | None = None
    last_signals: dict[str, Any] = {}
    try:
        initial_acks = {
            role: apply_state(role, controller.current[role], None)
            for role in ("attention", "expert")
        }
        append_jsonl(args.output, {"event": "controller_start", "wall_ns": time.time_ns(), "states": dict(controller.current), "acknowledgements": initial_acks})
        while not stop_requested and not args.stop_marker.exists():
            if args.events.exists():
                with args.events.open(encoding="utf-8") as stream:
                    stream.seek(offset)
                    buffer += stream.read()
                    offset = stream.tell()
                lines = buffer.split("\n")
                buffer = lines.pop()
                for line in lines:
                    if line:
                        controller.consume(json.loads(line))
            now_ns = time.time_ns()
            routing_update = controller.consume_routing(routing_tail.read(), now_ns)
            if routing_update is not None:
                append_jsonl(args.output, routing_update)
            desired, last_signals = controller.desired(now_ns)
            for role in ("attention", "expert"):
                allowed, reason = controller.transition_allowed(role, desired[role], now_ns)
                if not allowed:
                    continue
                old = controller.current[role]
                target = desired[role]
                try:
                    acknowledgements = apply_state(role, target, old)
                except Exception as exc:
                    fallback = {
                        fallback_role: apply_state(
                            fallback_role,
                            str(config["predictor"]["role_models"][fallback_role]["guard_state"]),
                            controller.current[fallback_role],
                        )
                        for fallback_role in ("attention", "expert")
                    }
                    for fallback_role in ("attention", "expert"):
                        controller.current[fallback_role] = str(config["predictor"]["role_models"][fallback_role]["guard_state"])
                    append_jsonl(args.output, {"event": "actuation_failure_fallback", "wall_ns": time.time_ns(), "error": f"{type(exc).__name__}: {exc}", "acknowledgements": fallback})
                    continue
                applied_ns = time.time_ns()
                controller.current[role] = target
                controller.candidate[role] = target
                controller.candidate_since_ns[role] = applied_ns
                controller.last_transition_ns[role] = applied_ns
                controller.transition_count[role] += 1
                append_jsonl(args.output, {"event": "transition", "wall_ns": applied_ns, "role": role, "from_state": old, "to_state": target, "frequency_mhz": int(frequencies[role][target]), "power_cap_w": float(power_caps[role][target]), "reason": reason, "signals": last_signals, "acknowledgements": acknowledgements})
            if controller.replay_ended and not controller.requests:
                break
            if read_fd is None:
                time.sleep(poll_s)
            else:
                ready, _, _ = select.select([read_fd], [], [], poll_s)
                if ready:
                    try:
                        while os.read(read_fd, 4096):
                            pass
                    except BlockingIOError:
                        pass
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if keepalive_fd is not None:
            os.close(keepalive_fd)
        if args.submit_signal_fifo is not None:
            args.submit_signal_fifo.unlink(missing_ok=True)
        restore_errors = []
        restore_acks = {}
        for role in ("attention", "expert"):
            guard = str(config["predictor"]["role_models"][role]["guard_state"])
            try:
                restore_acks[role] = apply_state(role, guard, controller.current[role])
            except Exception as exc:
                restore_errors.append(f"{role}: {type(exc).__name__}: {exc}")
        if restore_errors:
            status = "failed"
            error = "; ".join(filter(None, [error, *restore_errors]))
        finished_ns = time.time_ns()
        summary = {
            "schema_version": 1,
            "method": config["method"],
            "status": status,
            "error": error,
            "arm": args.arm,
            "policy_version": 5,
            "duration_s": (finished_ns - started_ns) / 1e9,
            "replay_started": controller.replay_started,
            "replay_ended": controller.replay_ended,
            "outstanding_at_exit": len(controller.requests),
            "event_counts": controller.event_counts,
            "transition_counts": controller.transition_count,
            "routing_rows": controller.routing_rows,
            "routing_updates": controller.routing_updates,
            "routing_lag_ms_max": controller.routing_lag_ms_max,
            "final_signals": last_signals,
            "restored_to_guard": not restore_errors,
            "restore_acknowledgements": restore_acks,
        }
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

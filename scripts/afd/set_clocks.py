#!/usr/bin/env python3
"""Apply and verify Attention/Expert clocks and power caps via nvcontrold.

A power cap is a second, independent actuator.  It matters here because the
measured energy of an idle A100 on this node is ~64 W against ~96 W while an
Expert rank is serving: roughly two thirds of a domain's energy does not scale
with the SM clock at all, which is why dropping a domain from 1050 to 690 MHz
recovered only 7.7% of its energy.  Capping board power constrains the quantity
actually being measured and lets the hardware pick its own f/V operating point
inside the budget, instead of pinning a clock and hoping the voltage follows.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def request(base: str, path: str, payload: dict) -> dict:
    call = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(call, timeout=20) as response:
        return json.load(response)


def parse_gpu_url_map(text: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not text:
        return mapping
    for entry in text.split(","):
        try:
            gpu_text, url = entry.split("=", 1)
            gpu = int(gpu_text)
        except ValueError as error:
            raise ValueError(
                "GPU URL map entries must use GPU=http://host:port"
            ) from error
        if gpu in mapping or not url:
            raise ValueError(f"invalid or duplicate GPU URL mapping: {entry!r}")
        mapping[gpu] = url.rstrip("/")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-url", default="http://127.0.0.1:9093")
    parser.add_argument("--expert-url", default="http://127.0.0.1:9096")
    parser.add_argument(
        "--gpu-url-map",
        default="",
        help="optional per-GPU nvcontrold routes, e.g. 0=http://host:9095,5=http://host:9096",
    )
    parser.add_argument("--attention-gpus", default="0,1,2,3")
    parser.add_argument("--expert-gpus", default="4,5,6,7")
    parser.add_argument("--attention", required=True)
    parser.add_argument("--expert", required=True)
    # Power caps are optional so every existing caller keeps its behaviour.
    # An empty value leaves the board limit at whatever it currently is; the
    # --reset path always restores the default limit so a capped arm cannot
    # leak its budget into the next one.
    parser.add_argument("--attention-power-w", default="")
    parser.add_argument("--expert-power-w", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    attention = [int(value) for value in args.attention.split(",")]
    expert = [int(value) for value in args.expert.split(",")]
    attention_gpus = [int(value) for value in args.attention_gpus.split(",")]
    expert_gpus = [int(value) for value in args.expert_gpus.split(",")]
    gpu_url_map = parse_gpu_url_map(args.gpu_url_map)
    if len(attention) != len(attention_gpus) or len(expert) != len(expert_gpus):
        raise ValueError("clock counts must match their Attention/Expert GPU lists")

    def parse_power(text: str, gpus: list[int], label: str) -> list[float] | None:
        if not text:
            return None
        values = [float(value) for value in text.split(",")]
        if len(values) == 1:
            values = values * len(gpus)
        if len(values) != len(gpus):
            raise ValueError(f"{label} power caps must match their GPU list")
        return values

    attention_power = parse_power(args.attention_power_w, attention_gpus, "Attention")
    expert_power = parse_power(args.expert_power_w, expert_gpus, "Expert")

    acknowledgements = []
    power_acknowledgements = []
    for gpu, frequency in zip(attention_gpus, attention, strict=True):
        path = "/reset" if args.reset else "/set_clock"
        payload = {"gpu": gpu} if args.reset else {"gpu": gpu, "sm_mhz": frequency}
        acknowledgements.append(
            request(gpu_url_map.get(gpu, args.attention_url), path, payload)
        )
    for gpu, frequency in zip(expert_gpus, expert, strict=True):
        path = "/reset" if args.reset else "/set_clock"
        payload = {"gpu": gpu} if args.reset else {"gpu": gpu, "sm_mhz": frequency}
        acknowledgements.append(
            request(gpu_url_map.get(gpu, args.expert_url), path, payload)
        )
    if args.reset:
        for url, gpus in (
            (args.attention_url, attention_gpus),
            (args.expert_url, expert_gpus),
        ):
            for gpu in gpus:
                power_acknowledgements.append(
                    request(
                        gpu_url_map.get(gpu, url),
                        "/set_power_limit",
                        {"gpu": gpu, "reset": True},
                    )
                )
    else:
        for url, gpus, caps in (
            (args.attention_url, attention_gpus, attention_power),
            (args.expert_url, expert_gpus, expert_power),
        ):
            if caps is None:
                continue
            for gpu, watts in zip(gpus, caps, strict=True):
                power_acknowledgements.append(
                    request(
                        gpu_url_map.get(gpu, url),
                        "/set_power_limit",
                        {"gpu": gpu, "watts": watts},
                    )
                )
    verification_errors: list[str] = []
    expected_clocks = dict(
        zip(attention_gpus + expert_gpus, attention + expert, strict=True)
    )
    clock_by_gpu = {row.get("gpu"): row for row in acknowledgements}
    if set(clock_by_gpu) != set(expected_clocks):
        verification_errors.append("clock acknowledgements do not cover requested GPUs")
    for gpu, frequency in expected_clocks.items():
        acknowledgement = clock_by_gpu.get(gpu, {})
        if args.reset:
            if acknowledgement.get("clock_reset") is not True:
                verification_errors.append(f"GPU {gpu} clock reset was not acknowledged")
        elif (
            acknowledgement.get("requested_mhz") != frequency
            or acknowledgement.get("applied_mhz") != frequency
            or acknowledgement.get("clock_control") != "locked"
        ):
            verification_errors.append(
                f"GPU {gpu} did not lock requested {frequency} MHz"
            )

    expected_power: dict[int, float] = {}
    if args.reset:
        expected_power = {gpu: 0.0 for gpu in expected_clocks}
    else:
        if attention_power is not None:
            expected_power.update(zip(attention_gpus, attention_power, strict=True))
        if expert_power is not None:
            expected_power.update(zip(expert_gpus, expert_power, strict=True))
    power_by_gpu = {row.get("gpu"): row for row in power_acknowledgements}
    if set(power_by_gpu) != set(expected_power):
        verification_errors.append("power acknowledgements do not cover requested GPUs")
    for gpu, watts in expected_power.items():
        acknowledgement = power_by_gpu.get(gpu, {})
        if args.reset:
            if acknowledgement.get("power_control") != "default":
                verification_errors.append(
                    f"GPU {gpu} default power limit was not acknowledged"
                )
        elif (
            float(acknowledgement.get("requested_w", -1)) != watts
            or float(acknowledgement.get("applied_w", -1)) != watts
            or acknowledgement.get("power_control") != "limited"
        ):
            verification_errors.append(
                f"GPU {gpu} did not apply requested {watts:g} W limit"
            )
    payload = {
        "requested_attention_mhz": attention,
        "requested_expert_mhz": expert,
        "requested_attention_power_w": attention_power,
        "requested_expert_power_w": expert_power,
        "attention_gpus": attention_gpus,
        "expert_gpus": expert_gpus,
        "gpu_url_map": gpu_url_map,
        "acknowledgements": acknowledgements,
        "power_acknowledgements": power_acknowledgements,
        "verified": not verification_errors,
        "verification_errors": verification_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if verification_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

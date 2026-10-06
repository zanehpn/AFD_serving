#!/usr/bin/env python3
"""Reset clocks and power limits for an explicit physical-GPU allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from set_clocks import parse_gpu_url_map, request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9096")
    parser.add_argument("--gpu-url-map", default="")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU allocation must be non-empty and unique")
    gpu_url_map = parse_gpu_url_map(args.gpu_url_map)
    acknowledgements = []
    power_acknowledgements = []
    for gpu in gpus:
        url = gpu_url_map.get(gpu, args.url)
        acknowledgements.append(request(url, "/reset", {"gpu": gpu}))
        power_acknowledgements.append(
            request(url, "/set_power_limit", {"gpu": gpu, "reset": True})
        )

    errors: list[str] = []
    clock_by_gpu = {row.get("gpu"): row for row in acknowledgements}
    power_by_gpu = {row.get("gpu"): row for row in power_acknowledgements}
    if set(clock_by_gpu) != set(gpus):
        errors.append("clock reset acknowledgements do not cover allocation")
    if set(power_by_gpu) != set(gpus):
        errors.append("power reset acknowledgements do not cover allocation")
    for gpu in gpus:
        if clock_by_gpu.get(gpu, {}).get("clock_reset") is not True:
            errors.append(f"GPU {gpu} clock reset was not acknowledged")
        if power_by_gpu.get(gpu, {}).get("power_control") != "default":
            errors.append(f"GPU {gpu} default power limit was not acknowledged")

    payload = {
        "schema_version": 1,
        "requested_gpus": gpus,
        "gpu_url_map": gpu_url_map,
        "acknowledgements": acknowledgements,
        "power_acknowledgements": power_acknowledgements,
        "verified": not errors,
        "verification_errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read NVML capabilities into a DSE hardware input; never change GPU settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket


def capture(gpu_ids, nvml):
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
        raise ValueError("GPU indices must be nonnegative and distinct")
    devices = {}
    nvml.nvmlInit()
    try:
        for index in gpu_ids:
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            low, high = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            pairs = []
            for memory in nvml.nvmlDeviceGetSupportedMemoryClocks(handle):
                for graphics in nvml.nvmlDeviceGetSupportedGraphicsClocks(handle, memory):
                    pairs.append({"memory_mhz": int(memory), "graphics_mhz": int(graphics)})
            uuid = nvml.nvmlDeviceGetUUID(handle)
            name = nvml.nvmlDeviceGetName(handle)
            devices[str(index)] = {
                "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                "name": name.decode() if isinstance(name, bytes) else name,
                "memory_mib": nvml.nvmlDeviceGetMemoryInfo(handle).total / 2**20,
                "min_power_w": low / 1000, "max_power_w": high / 1000,
                "clock_pairs": pairs,
            }
    finally:
        nvml.nvmlShutdown()
    return {
        "schema_version": 1, "host": socket.gethostname(),
        "captured_at": datetime.now(timezone.utc).isoformat(), "devices": devices,
        "scope": "Advertised capabilities only; actuation, memory fit, execution and SLO are unverified.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Physical NVML indices, e.g. 0,1,2,3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new hardware snapshot path")
    import pynvml
    result = capture([int(value) for value in args.gpus.split(",")], pynvml)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

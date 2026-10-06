#!/usr/bin/env python3

import json
import tempfile
from pathlib import Path

from measure_command import (
    integrate_window_energy,
    load_request_window,
    resolve_measurement_window,
    samples_bracketing_window,
)


with tempfile.TemporaryDirectory() as directory:
    request_log = Path(directory) / "requests.jsonl"
    request_log.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"submit_wall_ns": 2_000_000_000, "finish_wall_ns": 2_500_000_000},
                {"submit_wall_ns": 2_200_000_000, "finish_wall_ns": 4_000_000_000},
            )
        )
    )
    assert load_request_window(request_log) == (2_000_000_000, 4_000_000_000, 2)

    missing_log = Path(directory) / "missing.jsonl"
    assert resolve_measurement_window(
        missing_log,
        returncode=17,
        command_start_ns=10,
        command_end_ns=20,
    )[:4] == (10, 20, None, "command_start_to_failure")
    try:
        resolve_measurement_window(
            missing_log,
            returncode=0,
            command_start_ns=10,
            command_end_ns=20,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("successful command must require its request log")

samples = [
    (0.0, 1_000_000_000, [100.0, 200.0]),
    (1.0, 2_000_000_000, [100.0, 200.0]),
    (2.0, 3_000_000_000, [200.0, 300.0]),
    (3.0, 4_000_000_000, [300.0, 400.0]),
    (4.0, 5_000_000_000, [300.0, 400.0]),
]
energy, coverage = integrate_window_energy(
    samples,
    start_ns=2_500_000_000,
    end_ns=4_500_000_000,
    gpu_count=2,
)
# GPU 0: 0.5 s at 175 W + 1.0 s at 250 W + 0.5 s at 300 W.
assert abs(energy[0] - 487.5) < 1e-9
# GPU 1: 0.5 s at 275 W + 1.0 s at 350 W + 0.5 s at 400 W.
assert abs(energy[1] - 687.5) < 1e-9
assert coverage == 1.0
assert samples_bracketing_window(samples, 2_500_000_000, 4_500_000_000) == samples[1:]

print("measure command window tests passed")

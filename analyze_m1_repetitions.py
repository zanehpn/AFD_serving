#!/usr/bin/env python3
"""Aggregate repeated M1 measurements and report 95% confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def ci95(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    critical = T95.get(len(values), 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    groups: dict[tuple, list[dict]] = defaultdict(list)
    per_repeat: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(args.result_dir.glob("r*_*.json")):
        row = json.loads(path.read_text())
        repetition = int(path.name.split("_", 1)[0][1:])
        key = (row["phase"], row["op"], row["applied_frequency_mhz"])
        groups[key].append(row)
        per_repeat[(repetition, row["phase"], row["op"])].append(row)

    aggregate = []
    for (phase, op, frequency), rows in sorted(groups.items()):
        latency = [float(row["latency_mean_ms"]) for row in rows]
        energy = [float(row["energy_per_iteration_j"]) for row in rows]
        aggregate.append(
            {
                "phase": phase,
                "op": op,
                "frequency_mhz": frequency,
                "n": len(rows),
                "latency_mean_ms": statistics.fmean(latency),
                "latency_ci95_ms": ci95(latency),
                "energy_mean_j": statistics.fmean(energy),
                "energy_ci95_j": ci95(energy),
            }
        )
    optima = []
    for (repetition, phase, op), rows in sorted(per_repeat.items()):
        best = min(rows, key=lambda row: row["energy_per_iteration_j"])
        optima.append(
            {
                "repetition": repetition,
                "phase": phase,
                "op": op,
                "optimal_frequency_mhz": best["applied_frequency_mhz"],
                "energy_j": best["energy_per_iteration_j"],
                "latency_ms": best["latency_mean_ms"],
            }
        )

    output = {"aggregate": aggregate, "per_repetition_optima": optima}
    (args.result_dir / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    with (args.result_dir / "aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

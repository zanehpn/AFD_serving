#!/usr/bin/env python3
"""Fail-closed audit for calibration/evaluation JSONL trace isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path, identity_fields: list[str]) -> tuple[int, dict[str, set[str]]]:
    identities = {field: set() for field in identity_fields}
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                record: Any = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record is not a JSON object")
            count += 1
            for field in identity_fields:
                if field not in record or record[field] is None:
                    raise ValueError(f"{path}:{line_number}: missing identity field {field!r}")
                value = json.dumps(
                    record[field], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                if value in identities[field]:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate {field}={record[field]!r} within split"
                    )
                identities[field].add(value)
    if count == 0:
        raise ValueError(f"{path}: trace is empty")
    return count, identities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument(
        "--identity-field",
        action="append",
        default=[],
        help="Immutable source identity; repeat to check more than one (default: source_index)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fields = list(dict.fromkeys(args.identity_field or ["source_index"]))
    calibration = args.calibration.expanduser().resolve(strict=True)
    evaluation = args.evaluation.expanduser().resolve(strict=True)

    if os.path.samefile(calibration, evaluation):
        raise ValueError("calibration and evaluation resolve to the same file")

    calibration_count, calibration_ids = load_jsonl(calibration, fields)
    evaluation_count, evaluation_ids = load_jsonl(evaluation, fields)
    overlaps: dict[str, int] = {}
    overlap_samples: dict[str, list[Any]] = {}
    for field in fields:
        shared = calibration_ids[field] & evaluation_ids[field]
        overlaps[field] = len(shared)
        if shared:
            overlap_samples[field] = [json.loads(value) for value in sorted(shared)[:10]]

    report: dict[str, Any] = {
        "status": "PASS" if not any(overlaps.values()) else "FAIL",
        "calibration": {
            "path": str(calibration),
            "sha256": sha256(calibration),
            "requests": calibration_count,
        },
        "evaluation": {
            "path": str(evaluation),
            "sha256": sha256(evaluation),
            "requests": evaluation_count,
        },
        "identity_fields": fields,
        "overlap_counts": overlaps,
    }
    if overlap_samples:
        report["overlap_samples"] = overlap_samples
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)

#!/usr/bin/env python3
"""Static DSE: inspect, initialize, ask, measure externally, tell, and freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from static_dse.campaign import (ask, create_campaign, freeze, read_json, status, summary_result, tell, write_json)
from static_dse.demo import run_demo
from static_dse.executor import run_one
from static_dse.mechanism import build_mechanism
from static_dse.space import audit, enumerate_candidates, hardware_from_snapshot
from static_dse.calibration import calibration_plan
from static_dse.comparison import create_comparison, comparison_round, comparison_report
from static_dse.feedback import attach_feedback, make_manifest
from static_dse.provisioning import recommend_ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit", help="enumerate and filter only; does not deploy")
    for key in ("specification", "hardware", "runtime", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p = sub.add_parser("init")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p = sub.add_parser("provision-ratios", help="analytical A/F throughput and optional energy advice; does not deploy or certify SLO")
    for key in ("profile", "calibration", "heldout", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p = sub.add_parser("analyze-model", help="learn parameter effects only from physical calibration anchors")
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--rps", type=float, required=True)
    p.add_argument("--offered-utilization", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plan-calibration", help="generate factorial and structure probes; never launches GPUs")
    for key in ("specification", "hardware", "runtime", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=2)
    p = sub.add_parser("build-feedback", help="normalize measured stage traces and actual operating state")
    for key in ("request", "receipt", "manifest", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p = sub.add_parser("index-feedback", help="index native AFD stage/power files after a calibration replay")
    for key in ("request", "cell-directory", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--layers", type=int, required=True)
    p = sub.add_parser("compare-init", help="freeze independent V2/BO/Random/GA campaigns with matched budgets")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", choices=['v2','generic_bo','random','ga'], default=None)
    p.add_argument("--resume", action="store_true", help="verify and retain existing arms and observations")
    p = sub.add_parser("compare-round", help="one interleaved trial per arm; physical execution requires an evaluator")
    p.add_argument("directory", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--evaluator", nargs=argparse.REMAINDER)
    p = sub.add_parser("compare-report")
    p.add_argument("directory", type=Path)
    p.add_argument("--tolerance", type=float, default=.02)
    p.add_argument("--reference-campaign", type=Path)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("run-one", help="explicitly invoke an external measurement executor for one trial")
    p.add_argument("directory", type=Path)
    p.add_argument("--timeout-seconds", type=float, default=1800)
    p.add_argument("--evaluator", nargs=argparse.REMAINDER, required=True)
    for name in ("ask", "status", "freeze", "tell", "record-summary"):
        p = sub.add_parser(name)
        p.add_argument("directory", type=Path)
        p.add_argument("--output", type=Path)
        if name == "tell":
            p.add_argument("--result", type=Path, required=True)
        if name == "record-summary":
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--summary", type=Path, required=True)
    p = sub.add_parser("demo", help="synthetic CPU integration test, not an experiment")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--evaluations", type=int, default=10)
    args = parser.parse_args()
    if getattr(args, "output", None) and args.output.exists():
        raise FileExistsError("output exists; use a new artifact path")
    if args.command in ("audit", "plan-calibration"):
        hw = read_json(args.hardware)
        if "gpu_metadata_csv" in hw:
            hw = hardware_from_snapshot(hw)
        candidates, runtime = enumerate_candidates(read_json(args.specification)), read_json(args.runtime)
        result = (audit(candidates, hw, runtime) if args.command == "audit" else
                  calibration_plan(candidates, hw, runtime, args.repetitions))
    elif args.command == "provision-ratios":
        from static_dse.campaign import isolation, file_hash
        split = isolation(args.calibration, args.heldout, ["source_index"])
        requests = [json.loads(line) for line in args.calibration.read_text().splitlines() if line.strip()]
        result = recommend_ratios(read_json(args.profile), requests)
        result.update(split_audit=split, profile_sha256=file_hash(args.profile))
    elif args.command == "build-feedback":
        result = attach_feedback(read_json(args.request), read_json(args.receipt), args.manifest)
    elif args.command == "index-feedback":
        result = make_manifest(read_json(args.request), args.cell_directory, args.layers)
    elif args.command == "compare-init":
        result = create_comparison(args.config, args.directory, args.seeds, resume=args.resume, methods=args.methods)
    elif args.command == "compare-round":
        result = comparison_round(args.directory, args.evaluator, args.synthetic)
    elif args.command == "compare-report":
        result = comparison_report(args.directory, args.tolerance, args.reference_campaign)
    elif args.command == "init":
        result = create_campaign(args.config, args.directory)
        result = {k: v for k, v in result.items() if k != "candidates"}
    elif args.command == "analyze-model":
        profile = read_json(args.profile)
        if args.rps <= 0 or not 0 <= args.offered_utilization <= 1:
            raise ValueError("invalid offered load/utilization")
        result = build_mechanism(profile, {**profile.get("reference_workload", {}),
                                           "arrival_rate_rps": args.rps, "offered_utilization": args.offered_utilization})
    elif args.command == "run-one":
        result = run_one(args.directory, args.evaluator, args.timeout_seconds)
    elif args.command == "ask":
        result = ask(args.directory)
    elif args.command == "tell":
        result = tell(args.directory, read_json(args.result))
    elif args.command == "record-summary":
        pending = status(args.directory)["pending"]
        if pending is None:
            raise ValueError("no pending trial")
        result = tell(args.directory, summary_result(pending, read_json(args.receipt), args.summary))
    elif args.command == "freeze":
        result = freeze(args.directory)
    elif args.command == "status":
        result = status(args.directory)
    else:
        result = run_demo(args.directory, args.evaluations)
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({"output": str(args.output)}, indent=2))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Reviewable factorial calibration and progressive structural validation plans."""
from __future__ import annotations

from collections import defaultdict
from .space import configuration, digest, hard_filter, structure, KNOBS


def calibration_plan(candidates, hardware, runtime, repetitions=2):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be positive integer")
    groups = defaultdict(list)
    excluded = defaultdict(int)
    for c in candidates:
        check = hard_filter(c, hardware, runtime)
        if check["status"] == "hard_rejected" or set(check["pending_reasons"]) - {"structure_execution_unverified"}:
            excluded[check["status"]] += 1
            continue
        groups[digest(structure(c))].append((c, check["status"]))
    trials, structures = [], []
    for key, rows in sorted(groups.items()):
        by_knobs = {digest(c["knobs"]): c for c, _ in rows}
        high = {k: max(c["knobs"][k] for c, _ in rows) for k in KNOBS}
        low = {k: min(c["knobs"][k] for c, _ in rows) for k in KNOBS}
        base = by_knobs[digest(high)]
        verified = rows[0][1] == "eligible"
        structures.append({"structure_id": key, "configuration": configuration(base),
                           "status": "verified" if verified else "needs_execution_probe"})
        points = [(base, "reference_anchor" if verified else "structure_probe")]
        # Each role's 2x2 grid holds the other role fixed. Shared high/high
        # baseline gives seven distinct points for four main + two joint terms.
        for role in ("attention", "expert"):
            freq, power = role + "_mhz", role + "_power_w"
            for controls in ((freq,), (power,), (freq, power)):
                knobs = {**high, **{k: low[k] for k in controls}}
                candidate = by_knobs.get(digest(knobs))
                if candidate and candidate["id"] not in {p[0]["id"] for p in points}:
                    points.append((candidate, "joint_parameter_probe" if len(controls) == 2 else "single_parameter_probe"))
        for repeat in range(repetitions):
            ordered = points if repeat % 2 == 0 else list(reversed(points))
            for c, purpose in ordered:
                trials.append({"candidate_id": c["id"], "purpose": purpose, "repetition": repeat + 1,
                               "structure_id": key,
                               "requires_structure_success": not verified and c["id"] != base["id"],
                               "structure_reference_id": base["id"], "configuration": configuration(c)})
    # Validate all new structures at MAX before spending on their response grids.
    trials.sort(key=lambda t: (t["purpose"] != "structure_probe", t["repetition"] > 1))
    return {"schema_version": 1, "selection_split": "calibration", "structures": structures,
            "trials": trials, "planned_evaluations": len(trials), "excluded": dict(excluded),
            "design": "two role-local 2x2 frequency/cap grids with shared MAX; reverse repeated order",
            "execution": "plan only; unverified structures need allow_structure_probes=true; failures remain charged",
            "identification": "cross-role interactions and unrepresented TP/DP combinations require further designs"}


def planned_candidate(plan, candidates, observations, audit_rows, allow_structure_probes):
    by_id = {c["id"]: c for c in candidates}
    checks = {r["id"]: r for r in audit_rows}
    counts, occurrences = defaultdict(int), defaultdict(int)
    successful_structures, failed_structures = set(), set()
    failed_candidates = set()
    for row in observations:
        counts[row["candidate_id"]] += 1
        key = digest(structure(by_id[row["candidate_id"]]))
        if row["status"] == "ok" and "four_stage" in row:
            successful_structures.add(key)
        if row["status"] == "runtime_incompatible":
            failed_candidates.add(row["candidate_id"])
            if row.get("structure_validation_trial"):
                failed_structures.add(key)
    for trial in plan.get("trials", []):
        candidate_id = trial["candidate_id"]
        occurrences[candidate_id] += 1
        if counts[candidate_id] >= occurrences[candidate_id]:
            continue
        c = by_id[candidate_id]
        check = checks[candidate_id]
        key = digest(structure(c))
        if key in failed_structures or candidate_id in failed_candidates:
            continue
        if check["status"] != "eligible" and key not in successful_structures:
            if not allow_structure_probes or trial["purpose"] != "structure_probe":
                continue
        return c, trial
    return None, None

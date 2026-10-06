"""Audit recorded search evidence and save an append-only checkpoint.

Does not launch trials, read heldout outcomes, or change search state.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    plan = read(BASE / 'PLAN.json')
    destination = BASE / 'continuation-audits'
    previous_files = sorted(destination.glob('*.json'))
    previous = read(previous_files[-1]) if previous_files else None
    errors, arms = [], {}
    checked_artifacts = 0
    for job in plan['jobs']:
        root = BASE / 'search' / job['id']
        cfg = read(root / 'official-config.json')
        if file_digest(cfg['trace']['path']) != cfg['trace']['sha256']:
            errors.append(f"{job['id']}: calibration trace changed")
        for method in plan['methods']:
            key = f"{job['id']}/{method}"
            arm = root / 'comparison' / f'{method}-seed0'
            observations = read(arm / 'state.json')['observations']
            fingerprints = [digest(row) for row in observations]
            if previous:
                old = previous['arms'][key]['observation_sha256']
                if fingerprints[:len(old)] != old:
                    errors.append(f'{key}: existing observations changed or removed')
            ids = [row['trial_id'] for row in observations]
            if len(ids) != len(set(ids)):
                errors.append(f'{key}: duplicate trial identities')
            thresholds = {}
            for row in observations:
                trial = arm / 'trials' / row['trial_id']
                label = f"{key}/{row['trial_id']}"
                candidate = read(trial / 'configuration.json')
                pair = [candidate['dbo_decode_token_threshold'], candidate['dbo_prefill_token_threshold']]
                thresholds[str(pair)] = thresholds.get(str(pair), 0) + 1
                if candidate['microbatches'] != 2 or pair not in plan['threshold_profiles']:
                    errors.append(f'{label}: DBO configuration mismatch')
                assigned = candidate['attention_gpus'] + candidate['expert_gpus']
                if len(assigned) != len(set(assigned)) or not set(assigned) <= set(cfg['gpus']):
                    errors.append(f'{label}: GPU allocation mismatch')
                if row.get('selection_split') != 'calibration' or row.get('trace_sha256') != cfg['trace']['sha256']:
                    errors.append(f'{label}: calibration provenance mismatch')
                receipt = read(trial / 'worker-result.json')
                if receipt['status'] != row['status'] or receipt.get('metrics') != row.get('metrics'):
                    errors.append(f'{label}: receipt and observation disagree')
                for artifact in row.get('artifacts', []):
                    checked_artifacts += 1
                    if file_digest(artifact['path']) != artifact['sha256']:
                        errors.append(f"{label}: artifact changed: {artifact['path']}")
                if row['status'] == 'ok':
                    launch = read(trial / 'launch.json')
                    for role in ('attention', 'ffn'):
                        command = launch[role]['command']
                        if '--enable-dbo' not in command:
                            errors.append(f'{label}/{role}: DBO not enabled')
                        for flag, value in zip(('--dbo-decode-token-threshold', '--dbo-prefill-token-threshold'), pair):
                            if flag not in command or command[command.index(flag)+1] != str(value):
                                errors.append(f'{label}/{role}: threshold CLI mismatch')
            arms[key] = dict(trials=len(observations), observation_sha256=fingerprints,
                             threshold_counts=thresholds)
    report = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Recorded search observations only; not a completion or superiority claim',
                  resume_only=True, previous_checkpoint=str(previous_files[-1]) if previous else None,
                  previous_observations_preserved=not any('observations changed' in e for e in errors) if previous else None,
                  checked_artifacts=checked_artifacts, arms=arms, errors=errors,
                  passed=not errors)
    destination.mkdir(exist_ok=True)
    path = destination / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), passed=report['passed'],
                         recorded_trials=sum(a['trials'] for a in arms.values()),
                         checked_artifacts=checked_artifacts, errors=errors)))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

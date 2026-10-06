#!/usr/bin/env python3
"""Verify the frozen artifacts immediately before launching a v10 evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('trace_root', type=Path)
    args = parser.parse_args()
    p = args.protocol
    freeze = json.loads((p / 'EVALUATION_FREEZE.json').read_text())
    if freeze['status'] != 'frozen_before_heldout' or freeze['heldout_requests_used_for_tuning'] != 0:
        raise ValueError('invalid freeze status')
    checks = dict(freeze['shared_code_sha256'])
    for name, key in {
        'combined-controller-v10.json': 'combined_controller_sha256',
        'reference-controller-v9.json': 'reference_controller_sha256',
        'heldout-max-deployment.json': 'baseline_deployment_sha256',
        'heldout-fbss-deployment.json': 'fbss_deployment_sha256',
        'heldout-schedule.json': 'schedule_sha256',
        'FREEZE.json': 'calibration_freeze_sha256',
        'fbss-decision-v6.json': 'decision_sha256',
        'expanded-profile-v6.json': 'expanded_profile_sha256',
    }.items():
        checks[str(p / name)] = freeze[key]
    for name, key in {'calibration-200.jsonl': 'calibration_trace_sha256',
                      'heldout-400.jsonl': 'heldout_trace_sha256',
                      'warmup-8.jsonl': 'warmup_trace_sha256'}.items():
        checks[str(args.trace_root / name)] = freeze[key]
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f'frozen artifact changed: {path}')
    audit = json.loads((p / 'freeze-trace-isolation-audit.json').read_text())
    if audit['status'] != 'PASS' or any(audit['overlap_counts'].values()):
        raise ValueError('identity audit failed')
    print(json.dumps({'status': 'PASS', 'verified_artifacts': len(checks), 'evaluation_freeze_sha256': digest(p / 'EVALUATION_FREEZE.json')}))


if __name__ == '__main__':
    main()

"""Resolve the recorded launch-only correction without replacing its freeze."""
import hashlib
import json
from pathlib import Path


def effective_calibration_hashes(protocol):
    protocol = Path(protocol)
    original = protocol / 'PRE_DYNAMIC_CALIBRATION_CODE_FREEZE.json'
    pre = json.loads(original.read_text())
    hashes = dict(pre['files_sha256'])
    amendment = protocol / 'CALIBRATION_LAUNCH_TECHNICAL_AMENDMENT_RETRY1.json'
    if amendment.exists():
        change = json.loads(amendment.read_text())
        if change['original_code_freeze_sha256'] != hashlib.sha256(original.read_bytes()).hexdigest():
            raise ValueError('technical amendment refers to a different code freeze')
        if change['controller_policy_changed'] or change['heldout_outcomes_inspected'] or change['dynamic_calibration_measurements_before_change'] != 0:
            raise ValueError('invalid launch-only amendment')
        runner = str((Path(__file__).parent / 'run_v026_causal_dvfs_v10_rate_suite.sh').resolve())
        if change['runner_path'] != runner or change['original_runner_sha256'] != hashes[runner]:
            raise ValueError('technical amendment original runner mismatch')
        hashes[runner] = change['corrected_runner_sha256']
    return pre, hashes

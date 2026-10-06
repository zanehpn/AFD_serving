#!/usr/bin/env python3
"""Freeze completed calibration development before a fresh three-arm evaluation."""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from dynamic_v10_provenance import effective_calibration_hashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('source_protocol', type=Path)
    parser.add_argument('trace_root', type=Path)
    args = parser.parse_args()
    protocol, source, traces = (p.resolve() for p in (args.protocol, args.source_protocol, args.trace_root))
    if (protocol / 'FREEZE.json').exists():
        raise FileExistsError('freeze is immutable')
    report_path = protocol / 'CALIBRATION_RESULTS_V10.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'complete_calibration_only' or report['heldout_outcomes_read'] != 0:
        raise ValueError('completed calibration-only development is required')
    if report['controller_sha256'] != digest(protocol / 'combined-controller-v10.json'):
        raise ValueError('controller changed after calibration')
    pre, expected_hashes = effective_calibration_hashes(protocol)
    for path, expected in expected_hashes.items():
        if digest(path) != expected:
            raise ValueError(f'policy implementation changed since calibration: {path}')
    # The development report and final freeze retain the measured calibration
    # outcomes, including failures; this script does not erase failed cells.
    old = json.loads((source / 'FREEZE.json').read_text())
    if digest(traces / 'calibration-200.jsonl') != old['calibration_trace_sha256']:
        raise ValueError('inherited calibration identities changed')
    audit_path = protocol / 'freeze-trace-isolation-audit.json'
    command = ['python3', 'scripts/audit_trace_isolation.py',
               '--calibration', str(traces / 'calibration-200.jsonl'), '--evaluation', str(traces / 'heldout-400.jsonl'),
               '--identity-field', 'source_index', '--identity-field', 'source_timestamp']
    audit = subprocess.check_output(command, text=True)
    audit_path.write_text(audit)
    if json.loads(audit)['status'] != 'PASS':
        raise ValueError('trace isolation failed')
    for name in ('fbss-decision-v6.json', 'expanded-profile-v6.json'):
        if (protocol / name).exists():
            if digest(protocol / name) != digest(source / name):
                raise ValueError(f'inherited artifact differs: {name}')
        else:
            shutil.copyfile(source / name, protocol / name)
    reference = protocol / 'reference-controller-v9.json'
    if reference.exists():
        if digest(reference) != digest(pre['source_v9_controller']):
            raise ValueError('reference controller differs')
    else:
        shutil.copyfile(pre['source_v9_controller'], reference)
    freeze = dict(old)
    freeze.update(frozen_at=datetime.now().astimezone().isoformat(),
                  max_calibration_mode=pre.get('migration', {}).get('max_calibration_mode', 'destination_max'),
                  calibration_comparison_scope=report.get('calibration_comparison_scope', 'same_server'),
                  inherited_v9_freeze=str(source / 'FREEZE.json'),
                  inherited_v9_freeze_sha256=digest(source / 'FREEZE.json'),
                  heldout_trace_sha256=digest(traces / 'heldout-400.jsonl'),
                  development_report_sha256=digest(report_path),
                  controller_sha256=digest(protocol / 'combined-controller-v10.json'),
                  additional_dynamic_development_cells=6,
                  additional_dynamic_development_requests=1200,
                  heldout_outcomes_opened_before_freeze=0,
                  reference_controller_sha256=digest(reference),
                  next_step='run precommitted MAX, v9 and v10 once on fresh heldout without feedback')
    (protocol / 'FREEZE.json').write_text(json.dumps(freeze, indent=2) + '\n')
    deployment = json.loads((protocol / 'calibration-max-deployment.json').read_text())
    subprocess.run(['python3', 'scripts/afd/make_dynamic_v10_evaluation.py', str(protocol), str(traces),
                    '--combined-controller', str(protocol / 'combined-controller-v10.json'),
                    '--reference-controller', str(reference), '--model', deployment['model'],
                    '--plugin-root', deployment['plugin']['root']], check=True)


if __name__ == '__main__':
    main()

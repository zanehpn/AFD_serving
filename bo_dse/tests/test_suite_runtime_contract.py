"""Execute the production shell/jq preflight without launching GPU services."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / 'scripts/afd/run_v026_ours_rate_suite.sh'


def preflight(tmp_path, m, dbo, runtime_m=None):
    deployment = json.loads((ROOT / 'inputs/protocols/qwen36/calibration-max-deployment.json').read_text())
    if m is not None:
        deployment['topology']['microbatches'] = m
        deployment['runtime_contract']['microbatches'] = m if runtime_m is None else runtime_m
    deployment['runtime_contract'].update(enable_dbo=dbo, stage_trace=True)
    path = tmp_path / 'deployment.json'
    path.write_text(json.dumps(deployment))
    # Run the actual assignments and complete jq gate, including its arguments.
    # Extracting avoids executing unrelated GPU/environment preflights.
    source = SUITE.read_text()
    assignments = '\n'.join(line for line in source.splitlines()
                            if line.startswith(('readonly MICROBATCHES=', 'readonly ENABLE_DBO=')))
    start = source.index('jq -e --argjson routing_sidecar ')
    end = source.index('\n', source.index('\' "${DEPLOYMENT}" >/dev/null', start))
    script = 'set -euo pipefail\n' + assignments + '\n' + source[start:end]
    env = dict(os.environ, DEPLOYMENT=str(path), routing_sidecar_expected='false',
               COMPUTE_GATE_ON_ATTENTION='false', MATH_PARAMETER_PROBE='true')
    return subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True)


@pytest.mark.parametrize('m', [1, 2, 4, 8])
def test_candidate_dbo_matches_real_shell_gate(tmp_path, m):
    result = preflight(tmp_path, m, m == 2)
    assert result.returncode == 0, result.stderr
    assert preflight(tmp_path, m, m != 2).returncode != 0


def test_legacy_m2_contract_and_mismatched_runtime(tmp_path):
    assert preflight(tmp_path, None, True).returncode == 0
    assert preflight(tmp_path, 4, False, runtime_m=1).returncode != 0


@pytest.mark.parametrize('m', [0, -1, 1.5, '4'])
def test_invalid_microbatch_count_rejected(tmp_path, m):
    # A JSON string is invalid even if jq -r would print it as a number.
    result = preflight(tmp_path, m, False)
    assert result.returncode != 0

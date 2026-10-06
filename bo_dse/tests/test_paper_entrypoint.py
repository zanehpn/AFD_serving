"""Paper entry points must use the measured FIFO prior without legacy fallbacks."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

BO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BO / 'scripts/afd'))
from static_dse.paper_model import MODEL, validate_feedback, validate_settings
from four_stage_dse_v6.model import STAGES


@pytest.mark.parametrize('entry', ['native.py', 'official.py'])
def test_default_cli_selects_paper_without_creating_outputs(tmp_path, entry):
    destination = tmp_path / 'absent'
    command = [sys.executable, str(BO / entry), 'start', '--directory', str(destination),
               '--rps', '8', '--dry-run']
    report = json.loads(subprocess.check_output(command, cwd=tmp_path, text=True))
    assert report['backend'] == 'paper'
    assert report['mechanism_model'] == MODEL
    assert report['require_four_stage'] is True
    assert report['pipeline_model'] == 'finite_microbatch_fifo_v1'
    assert report['energy_correction'] == 'standardized_log_energy_residual_gp'
    assert report['gpu_actions_performed'] is False
    assert not destination.exists()


@pytest.mark.parametrize('entry', ['native.py', 'official.py'])
def test_legacy_cli_is_explicit_and_does_not_claim_paper_model(tmp_path, entry):
    command = [sys.executable, str(BO / entry), '--backend', 'legacy-observable',
               'start', '--directory', str(tmp_path / 'absent'), '--dry-run']
    report = json.loads(subprocess.check_output(command, cwd=tmp_path, text=True))
    assert report['mechanism_model'] == 'external_power_duration_v1'
    assert report['require_four_stage'] is False
    assert report['paper_method'] is False


@pytest.mark.parametrize('change', [
    {'require_four_stage': False},
    {'workload': {'allow_model_feedback_fallback': True}},
    {'bo': {'exploration_policy': 'capacity_v2'}},
])
def test_paper_settings_reject_silent_legacy_behavior(change):
    settings = {'mechanism_model': MODEL, 'require_four_stage': True}
    settings.update(change)
    with pytest.raises(ValueError, match='Paper energy model'):
        validate_settings(settings)


@pytest.mark.parametrize('change', [
    {'model_feedback_supported': False},
    {'stage_models': {}},
    {'analytical_provisioning': {}},
])
def test_paper_feedback_rejects_unfitted_or_non_fifo_observations(change):
    feedback = {'stage_models': {s: {'intercept_ms': 1.} for s in STAGES}, 'analytical_provisioning': {
        'schedule_model': {'type': 'finite_microbatch_fifo_v1'}}}
    validate_feedback(feedback)
    damaged = copy.deepcopy(feedback)
    damaged.update(change)
    with pytest.raises(ValueError, match='Paper energy model'):
        validate_feedback(damaged)

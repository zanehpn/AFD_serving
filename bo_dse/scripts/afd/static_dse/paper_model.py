"""Fail-closed contract for the paper's measured four-stage FIFO energy prior."""
from four_stage_dse_v6.model import STAGES

MODEL = 'four_stage_fifo_v1'


def validate_settings(settings):
    if settings.get('mechanism_model') != MODEL:
        return
    if settings.get('require_four_stage') is not True:
        raise ValueError('Paper energy model requires four-stage feedback')
    if settings.get('workload', {}).get('allow_model_feedback_fallback'):
        raise ValueError('Paper energy model cannot fall back to end-to-end-only feedback')
    if settings.get('bo', {}).get('exploration_policy') == 'capacity_v2':
        raise ValueError('Paper energy model uses structure/control probes, not historical capacity_v2 search')


def validate_feedback(stage):
    if (stage.get('model_feedback_supported', True) is not True
            or set(stage.get('stage_models', {})) != set(STAGES)):
        raise ValueError('Paper energy model requires fitted four-stage models; fallback feedback is invalid')
    schedule = stage.get('analytical_provisioning', {}).get('schedule_model', {})
    if schedule.get('type') != 'finite_microbatch_fifo_v1':
        raise ValueError('Paper energy model requires the finite-microbatch FIFO schedule')

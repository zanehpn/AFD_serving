"""Leakage-free four-stage AFD design-space model (v6)."""

from .model import (  # noqa: F401
    STAGES,
    CandidateEvaluation,
    evaluate_candidate,
    pipeline_time_ms,
    select_candidate,
)

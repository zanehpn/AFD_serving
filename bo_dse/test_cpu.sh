#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
task_python="${BO_DSE_PYTHON:-python3}"
# A separate pytest process prevents collision with the legacy static_dse package.
"$task_python" -m pytest -q \
  scripts/afd/test_static_dse.py \
  scripts/afd/test_static_dse_extensions.py \
  scripts/afd/test_measure_command.py \
  scripts/afd/test_build_deepseek_v6_four_stage_profile.py \
  scripts/afd/four_stage_dse_v6 \
  tests

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then task_python=.venv/bin/python; else task_python=python3; fi
for test_file in \
  tests/test_native_migration.py \
  tests/test_static_dse_math.py \
  tests/test_general_dse.py \
  tests/test_full_math_workflow.py \
  scripts/afd/causal_dvfs_v10/test_controller.py \
  scripts/afd/test_make_dynamic_v10_controller.py \
  scripts/afd/test_v10_launch_cleanup.py; do
  "$task_python" -m pytest -q "$test_file"
done
"$task_python" scripts/afd/test_measure_command.py
"$task_python" scripts/afd/causal_dvfs_v5/test_controller.py
"$task_python" scripts/afd/causal_dvfs/test_replay_client.py
for script in migration/*.sh scripts/afd/run_dynamic_v10*.sh scripts/afd/start_server_native.sh scripts/afd/run_v026_ours_rate_suite.sh scripts/afd/run_v026_causal_dvfs{,_v10}_rate_suite.sh; do
  bash -n "$script"
done

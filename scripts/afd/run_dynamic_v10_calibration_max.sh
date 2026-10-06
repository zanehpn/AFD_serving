#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
task_root="$PWD"
trace_root="$task_root/results/afd_suites/dynamic-fbss-v10-inputs-20260906"
export ECODEP_CONTAINER_RUNTIME=native
export ECODEP_WARMUP_TRACE="$trace_root/warmup-8.jsonl"
export ECODEP_WARMUP_MANIFEST="$trace_root/warmup-manifest.json"
export ECODEP_CALIBRATION_REQUEST_LIMIT=200
export ECODEP_REPETITION_ID=1
export ECODEP_SCHEDULE_POSITION=1
export ECODEP_GPU_STABLE_FOR_S="${ECODEP_GPU_STABLE_FOR_S:-120}"
export ECODEP_MINIMUM_FREE_GIB="${ECODEP_MINIMUM_FREE_GIB:-79}"
export ECODEP_GPU_WAIT_TIMEOUT_S="${ECODEP_GPU_WAIT_TIMEOUT_S:-3600}"
for model in deepseek-v2-lite qwen36; do
  protocol="$task_root/results/afd_protocols/${model}-v026-dynamic-fbss-v10-20260906"
  export ECODEP_SCHEDULE_PATH="$protocol/calibration-schedule.json"
  model_tag="$(jq -r '.model' "$protocol/calibration-max-deployment.json")"
  suite_id="${model}-v026-dynamic-fbss-v10-calibration-max-20260906"
  if [[ ! -f "$task_root/results/afd_suites/$model_tag/$suite_id/COMPLETE" ]]; then
    bash scripts/afd/run_v026_ours_rate_suite.sh \
      "$protocol/calibration-max-deployment.json" "$trace_root/calibration-200.jsonl" \
      "$suite_id" 1,2,4
  fi
done

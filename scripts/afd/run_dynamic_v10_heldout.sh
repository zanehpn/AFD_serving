#!/usr/bin/env bash
# Execute the immutable three-arm protocol; never select retries by performance.
set -euo pipefail
cd "$(dirname "$0")/../.."
task_root="$PWD"
trace_root="$task_root/results/afd_suites/dynamic-fbss-v10-inputs-20260906"
export ECODEP_CONTAINER_RUNTIME=native
export ECODEP_WARMUP_TRACE="$trace_root/warmup-8.jsonl"
export ECODEP_WARMUP_MANIFEST="$trace_root/warmup-manifest.json"
export ECODEP_REPETITION_ID=1
export ECODEP_GPU_STABLE_FOR_S="${ECODEP_GPU_STABLE_FOR_S:-120}"
export ECODEP_MINIMUM_FREE_GIB="${ECODEP_MINIMUM_FREE_GIB:-79}"
export ECODEP_GPU_WAIT_TIMEOUT_S="${ECODEP_GPU_WAIT_TIMEOUT_S:-3600}"
for model in ${ECODEP_MODELS:-deepseek-v2-lite qwen36}; do
  protocol="$task_root/results/afd_protocols/${model}-v026-dynamic-fbss-v10-20260906"
  [[ -f "$protocol/EVALUATION_FREEZE.json" ]] || { echo "Missing evaluation freeze: $model" >&2; exit 1; }
  python3 scripts/afd/audit_dynamic_v10_freeze.py "$protocol" "$trace_root"
done
for model in ${ECODEP_MODELS:-deepseek-v2-lite qwen36}; do
  protocol="$task_root/results/afd_protocols/${model}-v026-dynamic-fbss-v10-20260906"
  model_tag="$(jq -r '.model' "$protocol/heldout-max-deployment.json")"
  export ECODEP_SCHEDULE_PATH="$protocol/heldout-schedule.json"
  base_id="${model}-v026-dynamic-fbss-v10-max-heldout-r1-20260906"
  old_id="${model}-v026-dynamic-fbss-v10-reference-v9-heldout-r1-20260906"
  new_id="${model}-v026-dynamic-fbss-v10-dynamic-heldout-r1-20260906"
  export ECODEP_SCHEDULE_POSITION=1
  if [[ ! -f "$task_root/results/afd_suites/$model_tag/$base_id/COMPLETE" ]]; then
    bash scripts/afd/run_v026_ours_rate_suite.sh "$protocol/heldout-max-deployment.json" \
      "$trace_root/heldout-400.jsonl" "$base_id" 1,2,4
  fi
  export ECODEP_SCHEDULE_POSITION=2
  python3 scripts/afd/audit_dynamic_v10_freeze.py "$protocol" "$trace_root"
  if [[ ! -f "$task_root/results/afd_suites/$model_tag/$old_id/COMPLETE" ]]; then
    bash scripts/afd/run_v026_causal_dvfs_rate_suite.sh "$protocol/heldout-fbss-deployment.json" \
      "$trace_root/heldout-400.jsonl" "$old_id" 1,2,4 dynamic-ae "$protocol/reference-controller-v9.json"
  fi
  export ECODEP_SCHEDULE_POSITION=3
  python3 scripts/afd/audit_dynamic_v10_freeze.py "$protocol" "$trace_root"
  if [[ ! -f "$task_root/results/afd_suites/$model_tag/$new_id/COMPLETE" ]]; then
    bash scripts/afd/run_v026_causal_dvfs_v10_rate_suite.sh "$protocol/heldout-fbss-deployment.json" \
      "$trace_root/heldout-400.jsonl" "$new_id" 1,2,4 dynamic-ae "$protocol/combined-controller-v10.json"
  fi
  for revision in 9 10; do
    dynamic_id="$old_id"
    [[ "$revision" == 9 ]] || dynamic_id="$new_id"
    python3 scripts/afd/summarize_dynamic_fbss_v10.py "$protocol" \
      "$task_root/results/afd_suites/$model_tag/$base_id" \
      "$task_root/results/afd_suites/$model_tag/$dynamic_id" --revision "$revision"
  done
done

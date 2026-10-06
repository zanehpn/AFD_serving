#!/usr/bin/env bash
# Isolated v0.26 deterministic causal Attention/Expert DVFS rate sweep.

set -euo pipefail

export ECODEP_AFD_IMAGE="${ECODEP_V026_AFD_IMAGE:-ecodep-vllm-afd:v0.26.0}"
source "$(dirname "$0")/common.sh"

readonly DEPLOYMENT="${1:?frozen deployment required}"
readonly TRACE_INPUT="${2:?trace required}"
readonly SUITE_ID="${3:?suite id required}"
readonly RATES_CSV="${4:-2,4,8,16}"
readonly CONTROLLER_ARM="${5:-dynamic-ae}"
readonly CONTROLLER_CONFIG_INPUT="${6:-${ECODEP_ROOT}/scripts/afd/causal_dvfs/safe-v1.json}"
[[ -f "${DEPLOYMENT}" ]] || { echo "missing deployment ${DEPLOYMENT}" >&2; exit 2; }
[[ -f "${TRACE_INPUT}" ]] || { echo "missing trace ${TRACE_INPUT}" >&2; exit 2; }
readonly TRACE="$(realpath "${TRACE_INPUT}")"
readonly CONTROLLER_CONFIG="$(realpath "${CONTROLLER_CONFIG_INPUT}")"
CONTROLLER_SCRIPT="${ECODEP_ROOT}/scripts/afd/causal_dvfs/controller.py"
readonly REPLAY_SCRIPT="${ECODEP_ROOT}/scripts/afd/causal_dvfs/replay_client.py"
readonly TRACE_AUDITOR="${ECODEP_TRACE_ISOLATION_AUDITOR:-${ECODEP_ROOT}/scripts/audit_trace_isolation.py}"
readonly MODEL_TAG="$(jq -r '.model' "${DEPLOYMENT}")"
readonly MODEL_PATH="${ECODEP_ROOT}/artifacts/models/${MODEL_TAG}"
readonly PLUGIN_ROOT="${ECODEP_AFD_PLUGIN_ROOT_OVERRIDE:-$(jq -r '.plugin.root' "${DEPLOYMENT}")}"
readonly PLUGIN_COMMIT="$(jq -r '.plugin.commit' "${DEPLOYMENT}")"
readonly PLACEMENT_ENABLED="$(jq -r 'if (.placement | has("enabled")) then .placement.enabled else true end' "${DEPLOYMENT}")"
readonly PLACEMENT_PATH="$(jq -r 'if ((.placement | has("enabled")) and (.placement.enabled == false)) then "" else .placement.path end' "${DEPLOYMENT}")"
readonly PLACEMENT_SHA256="$(jq -r 'if ((.placement | has("enabled")) and (.placement.enabled == false)) then "" else .placement.sha256 end' "${DEPLOYMENT}")"
readonly PERMUTATION_BY_LAYER="$(jq -r 'if ((.placement | has("enabled")) and (.placement.enabled == false)) then "" else .placement.permutation_by_layer end' "${DEPLOYMENT}")"
readonly SUITE_DIR="${ECODEP_ROOT}/results/afd_suites/${MODEL_TAG}/${SUITE_ID}"
readonly RUN_DIR="${ECODEP_ROOT}/results/afd_serving/${MODEL_TAG}/${SUITE_ID}"
readonly MONITOR_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${ECODEP_ROOT}/results/runtime}/ecodep-gpu-monitor/${SUITE_ID}"
readonly MONITOR_OUTPUT="${MONITOR_RUNTIME_DIR}/gpu-contamination.jsonl"
readonly MONITOR_FAILURE="${MONITOR_RUNTIME_DIR}/gpu-monitor-failure.json"
readonly CONTAINER_NAME="ecodep-afd-${SUITE_ID}"
readonly IMAGE="native-vllm-0.26.0-cu130"
readonly CONTAINER_RUNTIME="native"
readonly ATTENTION_GPUS="${ECODEP_ATTENTION_GPUS:-0,1}"
readonly EXPERT_GPUS="${ECODEP_EXPERT_GPUS:-2,3}"
readonly GPUS="${ATTENTION_GPUS},${EXPERT_GPUS}"
readonly MODEL_NAME="ecodep-v026"
readonly MAX_OUTPUT_TOKENS="${ECODEP_MAX_OUTPUT_TOKENS:-128}"
readonly CLOCK_URL="${ECODEP_CLOCK_URL:-http://127.0.0.1:9096}"
readonly ATTENTION_CLOCK_URL="${ECODEP_ATTENTION_CLOCK_URL:-${CLOCK_URL}}"
readonly EXPERT_CLOCK_URL="${ECODEP_EXPERT_CLOCK_URL:-${CLOCK_URL}}"
readonly GPU_CLOCK_URL_MAP="${ECODEP_GPU_CLOCK_URL_MAP:-}"
readonly MAX_ATTENTION_CLOCKS="${ECODEP_ATTENTION_CLOCKS_MHZ:-1410,1410}"
readonly MAX_EXPERT_CLOCKS="${ECODEP_EXPERT_CLOCKS_MHZ:-1410,1410}"
readonly MAX_ATTENTION_POWER_W="${ECODEP_ATTENTION_POWER_W:-400,400}"
readonly MAX_EXPERT_POWER_W="${ECODEP_EXPERT_POWER_W:-400,400}"
readonly SETTLE_S="${ECODEP_OPERATING_POINT_SETTLE_S:-2}"
readonly MAX_MODEL_LEN="${ECODEP_MAX_MODEL_LEN:-8192}"
readonly MAX_NUM_SEQS="${ECODEP_MAX_NUM_SEQS:-32}"
readonly MAX_NUM_BATCHED_TOKENS="${ECODEP_MAX_NUM_BATCHED_TOKENS:-3072}"
readonly CUDAGRAPH_CAPTURE_SIZE="${ECODEP_CUDAGRAPH_CAPTURE_SIZE:-32}"
readonly DBO_DECODE_TOKEN_THRESHOLD="${ECODEP_DBO_DECODE_TOKEN_THRESHOLD:-2}"
readonly DBO_PREFILL_TOKEN_THRESHOLD="${ECODEP_DBO_PREFILL_TOKEN_THRESHOLD:-12}"
readonly REPETITION="${ECODEP_REPETITION_ID:-1}"
readonly EVALUATION_SPLIT="${ECODEP_EVALUATION_SPLIT:-$(jq -r '.evaluation_split // "heldout"' "${DEPLOYMENT}")}"
readonly SCHEDULE_INPUT="${ECODEP_SCHEDULE_PATH:-}"
readonly ROUTING_SMOKE_LIMIT="${ECODEP_ROUTING_SMOKE_LIMIT:-}"
readonly CALIBRATION_REQUEST_LIMIT="${ECODEP_CALIBRATION_REQUEST_LIMIT:-}"
readonly GPU_STABLE_FOR_S="${ECODEP_GPU_STABLE_FOR_S:-10}"
readonly GPU_WAIT_TIMEOUT_S="${ECODEP_GPU_WAIT_TIMEOUT_S:-3600}"
readonly MINIMUM_FREE_GIB="${ECODEP_MINIMUM_FREE_GIB:-65}"
SCHEDULE_POSITION="${ECODEP_SCHEDULE_POSITION:-0}"
readonly WARMUP_TRACE_INPUT="${ECODEP_WARMUP_TRACE:?ECODEP_WARMUP_TRACE is required}"
readonly WARMUP_MANIFEST_INPUT="${ECODEP_WARMUP_MANIFEST:?ECODEP_WARMUP_MANIFEST is required}"

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model ${MODEL_PATH}" >&2; exit 2; }
python3 "${ECODEP_ROOT}/migration/check_gpus.py" --gpus "${GPUS}"
[[ -f "${CONTROLLER_CONFIG}" ]] || {
  echo "missing isolated causal DVFS config" >&2; exit 2;
}
jq -e --arg arm "${CONTROLLER_ARM}" '
  .selection_split == "calibration" and
  (.ablation_arms | index($arm) != null) and
  (if (.policy_version // 1) == 5 then
     (.method == "fbss_constrained_causal_stage_routing_joint_frequency_power_v8" or
      .method == "fbss_constrained_causal_stage_routing_joint_frequency_power_v9") and
     .events.submit_wakeup == "named_pipe" and
     .events.progress_mode == "batched" and
     .events.progress_event_interval_ms >= 200 and
     .events.progress_event_interval_ms <= 500 and
     .predictor.fixed_topology == "2A2E" and
     .predictor.static_dse_online == false and
     .predictor.future_trace_access == false and
     (.predictor.forbidden_inputs | index("evaluation_trace_file") != null) and
     (.predictor.fbss_binding.allowed_dynamic_candidates | index("c00-max") != null) and
     (.predictor.fbss_binding.allowed_dynamic_candidates as $allowed |
       .predictor.joint_operating_points | all(
         .id as $id | ($allowed | index($id)) != null
       )) and
     .compatibility.attention_dp == 2 and
     .compatibility.expert_ep == 2 and
     .compatibility.attention_tp == 1 and
     .compatibility.expert_tp == 1 and
     (.compatibility.compute_gate_on_attention | type) == "boolean" and
     (.compatibility.routing_source_role // "attention" | IN("attention", "ffn")) and
     .predictor.role_models.attention.guard_state == "f1410-p400" and
     .predictor.role_models.expert.guard_state == "f1410-p400"
   else
     ((.policy_version // 1) == 1 and
       .method == "deterministic_causal_role_dvfs_three_state" or
      (.policy_version // 1) == 2 and
       .method == "deterministic_causal_role_dvfs_backlog_v2" or
      (.policy_version // 1) == 3 and
       .method == "causal_afd_stage_bottleneck_predictor_v3" or
      (.policy_version // 1) == 4 and
       .method == "causal_afd_stage_bottleneck_predictor_joint_actuator_v4") and
     (.frequencies_mhz.attention.boost == 1410) and
     (.frequencies_mhz.expert.boost == 1410) and
     (if (.policy_version // 1) >= 2 then
        .events.submit_wakeup == "named_pipe" and
        .events.progress_mode == "batched" and
        .events.progress_event_interval_ms >= 200 and
        .events.progress_event_interval_ms <= 500 and
        .initial_states.expert == "normal" and
        (if (.policy_version // 1) >= 3 then
           .predictor.fixed_topology == "2A2E" and
           .predictor.static_dse_online == false and
           .predictor.future_trace_access == false and
           (.predictor.forbidden_inputs | index("evaluation_trace_file") != null) and
           .compatibility.attention_dp == 2 and
           .compatibility.expert_ep == 2 and
           .compatibility.attention_tp == 1 and
           .compatibility.expert_tp == 1
         else true end) and
        (if (.policy_version // 1) == 4 then
           .frequencies_mhz.attention.guard == 1410 and
           .frequencies_mhz.expert.guard == 1410 and
           .power_caps_w.attention.guard == 400 and
           .power_caps_w.expert.guard == 400 and
           (.power_caps_w.attention.eco >= 100) and
           (.power_caps_w.expert.eco >= 100)
         else true end)
      else true end)
   end)
' "${CONTROLLER_CONFIG}" >/dev/null || {
  echo "invalid causal DVFS config or ablation arm" >&2; exit 2;
}
readonly CONTROLLER_POLICY_VERSION="$(jq -r '.policy_version // 1' "${CONTROLLER_CONFIG}")"
CONTROLLER_ROUTING_SIDECAR=0
CONTROLLER_COMPUTE_GATE=0
CONTROLLER_ROUTING_SOURCE_ROLE=attention
CONTROLLER_ROUTING_FFN_SIDECAR=0
CONTROLLER_ROUTING_PREFILL_ONLY=0
CONTROLLER_ROUTING_LAYER_STRIDE=1
CONTROLLER_ROUTING_REQUEST_DETAIL=1
if (( CONTROLLER_POLICY_VERSION == 5 )); then
  CONTROLLER_SCRIPT="${ECODEP_ROOT}/scripts/afd/causal_dvfs_v5/controller.py"
  CONTROLLER_ROUTING_SIDECAR=1
  CONTROLLER_COMPUTE_GATE="$(jq -r '.compatibility.compute_gate_on_attention | if . then 1 else 0 end' "${CONTROLLER_CONFIG}")"
  CONTROLLER_ROUTING_SOURCE_ROLE="$(jq -r '.compatibility.routing_source_role // "attention"' "${CONTROLLER_CONFIG}")"
  [[ "${CONTROLLER_ROUTING_SOURCE_ROLE}" == "ffn" ]] && CONTROLLER_ROUTING_FFN_SIDECAR=1
  CONTROLLER_ROUTING_PREFILL_ONLY="$(jq -r '.predictor.routing_collection.prefill_only // false | if . then 1 else 0 end' "${CONTROLLER_CONFIG}")"
  CONTROLLER_ROUTING_LAYER_STRIDE="$(jq -r '.predictor.routing_collection.layer_stride // 1' "${CONTROLLER_CONFIG}")"
  CONTROLLER_ROUTING_REQUEST_DETAIL="$(jq -r '.predictor.routing_collection.request_detail | if . == null then true else . end | if . then 1 else 0 end' "${CONTROLLER_CONFIG}")"
fi
readonly CONTROLLER_SCRIPT CONTROLLER_ROUTING_SIDECAR CONTROLLER_COMPUTE_GATE
readonly CONTROLLER_ROUTING_SOURCE_ROLE CONTROLLER_ROUTING_FFN_SIDECAR
readonly CONTROLLER_ROUTING_PREFILL_ONLY
readonly CONTROLLER_ROUTING_LAYER_STRIDE CONTROLLER_ROUTING_REQUEST_DETAIL
[[ -f "${CONTROLLER_SCRIPT}" ]] || {
  echo "missing isolated causal DVFS controller" >&2; exit 2;
}
if (( CONTROLLER_POLICY_VERSION >= 3 )); then
  [[ "$(jq -r '.compatibility.model' "${CONTROLLER_CONFIG}")" == "${MODEL_TAG}" ]] || {
    echo "v3 predictor model compatibility mismatch" >&2; exit 2;
  }
  [[ "$(jq -r '.compatibility.gpu_model' "${CONTROLLER_CONFIG}")" == "${gpu_models[0]}" ]] || {
    echo "v3 predictor GPU compatibility mismatch" >&2; exit 2;
  }
  predictor_profile="$(jq -r '.predictor.calibration_profile.path' "${CONTROLLER_CONFIG}")"
  [[ -f "${predictor_profile}" ]] || {
    echo "v3 predictor calibration profile is missing" >&2; exit 2;
  }
  [[ "$(sha256sum "${predictor_profile}" | awk '{print $1}')" == \
     "$(jq -r '.predictor.calibration_profile.file_sha256' "${CONTROLLER_CONFIG}")" ]] || {
    echo "v3 predictor calibration profile hash mismatch" >&2; exit 2;
  }
fi
readonly PROGRESS_EVENT_MODE="$(jq -r '.events.progress_mode // "per-request"' "${CONTROLLER_CONFIG}")"
readonly PROGRESS_EVENT_INTERVAL_MS="$(jq -r '.events.progress_event_interval_ms // 50' "${CONTROLLER_CONFIG}")"
readonly SUBMIT_WAKEUP="$(jq -r '.events.submit_wakeup // "none"' "${CONTROLLER_CONFIG}")"
if (( CONTROLLER_POLICY_VERSION >= 2 )); then
  ARM="causal-dvfs-v${CONTROLLER_POLICY_VERSION}-${CONTROLLER_ARM}"
else
  ARM="causal-dvfs-${CONTROLLER_ARM}"
fi
readonly ARM

if [[ "${PLACEMENT_ENABLED}" == "true" ]]; then
  [[ -f "${PLACEMENT_PATH}" ]] || {
    echo "missing placement ${PLACEMENT_PATH}" >&2; exit 2;
  }
elif [[ "${PLACEMENT_ENABLED}" != "false" ]]; then
  echo "placement.enabled must be true or false" >&2
  exit 2
fi
[[ -f "${WARMUP_TRACE_INPUT}" ]] || {
  echo "missing warmup trace ${WARMUP_TRACE_INPUT}" >&2; exit 2;
}
[[ -f "${WARMUP_MANIFEST_INPUT}" ]] || {
  echo "missing warmup manifest ${WARMUP_MANIFEST_INPUT}" >&2; exit 2;
}
readonly WARMUP_MANIFEST="$(realpath "${WARMUP_MANIFEST_INPUT}")"
readonly WARMUP_TRACE="$(realpath "${WARMUP_TRACE_INPUT}")"
[[ "$(dirname "${WARMUP_TRACE}")" == "$(dirname "${TRACE}")" ]] || {
  echo "warmup and evaluated traces must share the mounted trace directory" >&2
  exit 2
}
[[ "${WARMUP_TRACE}" != "${TRACE}" ]] || {
  echo "warmup trace must differ from the evaluated trace" >&2; exit 2;
}
warmup_trace_sha256="$(python3 -c '
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
' "${WARMUP_TRACE}")"
warmup_manifest_sha256="$(python3 -c '
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
' "${WARMUP_MANIFEST}")"
jq -e --arg trace "${WARMUP_TRACE}" --arg sha "${warmup_trace_sha256}" '
  .selection_split == "calibration" and
  .output == $trace and
  .output_sha256 == $sha
' "${WARMUP_MANIFEST}" >/dev/null || {
  echo "warmup trace is not backed by a calibration-only manifest" >&2
  exit 2
}
[[ "${REPETITION}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ECODEP_REPETITION_ID must be a positive integer" >&2; exit 2;
}
[[ "${EVALUATION_SPLIT}" == "calibration" || \
   "${EVALUATION_SPLIT}" == "heldout" ]] || {
  echo "causal DVFS split must be calibration or heldout" >&2; exit 2;
}
if [[ -n "${CALIBRATION_REQUEST_LIMIT}" ]]; then
  [[ "${EVALUATION_SPLIT}" == "calibration" && \
     "${CALIBRATION_REQUEST_LIMIT}" =~ ^[0-9]+$ && \
     "${CALIBRATION_REQUEST_LIMIT}" -ge 200 ]] || {
    echo "calibration request limit requires calibration split and at least 200 requests" >&2
    exit 2
  }
fi
schedule_path=""
schedule_sha256=""
schedule_label="${ARM}"
[[ -z "${ROUTING_SMOKE_LIMIT}" ]] || {
  echo "routing calibration is intentionally outside causal DVFS" >&2
  exit 2
}
  [[ -f "${SCHEDULE_INPUT}" && "${SCHEDULE_POSITION}" =~ ^[1-9][0-9]*$ ]] || {
    echo "runs require ECODEP_SCHEDULE_PATH and a positive schedule position" >&2
    exit 2
  }
  schedule_path="$(realpath "${SCHEDULE_INPUT}")"
  schedule_sha256="$(python3 -c '
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
' "${schedule_path}")"
  python3 - "${schedule_path}" "${SCHEDULE_POSITION}" "${schedule_label}" \
    "${REPETITION}" "${RATES_CSV}" <<'PY'
import json, pathlib, sys
schedule = json.loads(pathlib.Path(sys.argv[1]).read_text())
position = int(sys.argv[2])
entries = [row for row in schedule["entries"] if int(row["position"]) == position]
if len(entries) != 1:
    raise ValueError("schedule position is missing or duplicated")
entry = entries[0]
rates = [float(value) for value in sys.argv[5].split(",")]
if entry["arm"] != sys.argv[3] or int(entry["repetition"]) != int(sys.argv[4]):
    raise ValueError("schedule arm/repetition does not match deployment")
if [float(value) for value in entry["rates_rps"]] != rates:
    raise ValueError("rate execution order does not match precommitted schedule")
PY
trace_actual_sha256="$(python3 -c '
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
' "${TRACE}")"
trace_base_arrival_rate_rps="$(python3 -c '
import json, pathlib, sys
rows=[json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if sys.argv[2]:
    limit=int(sys.argv[2])
    if len(rows) < limit:
        raise ValueError("trace is smaller than the calibration request limit")
    rows=rows[:limit]
arrivals=[float(row["arrival_s"]) for row in rows]
if len(arrivals) < 2 or max(arrivals) <= min(arrivals):
    raise ValueError("trace needs at least two distinct arrival timestamps")
print((len(arrivals)-1)/(max(arrivals)-min(arrivals)))
' "${TRACE}" "${CALIBRATION_REQUEST_LIMIT}")"
[[ "${warmup_trace_sha256}" != "${trace_actual_sha256}" ]] || {
  echo "warmup trace content must differ from the evaluated trace" >&2; exit 2;
}
if [[ "${EVALUATION_SPLIT}" == "heldout" ]]; then
  [[ "${trace_actual_sha256}" != \
     "$(jq -r '.calibration_trace_sha256' "${DEPLOYMENT}")" ]] || {
    echo "held-out trace is identical to the operating-point calibration trace" >&2
    exit 2
  }
else
  [[ "${trace_actual_sha256}" == \
     "$(jq -r '.calibration_trace_sha256' "${DEPLOYMENT}")" ]] || {
    echo "calibration trace differs from the frozen calibration deployment" >&2
    exit 2
  }
fi
placement_actual_sha256=""
permutation_actual_sha256=""
if [[ "${PLACEMENT_ENABLED}" == "true" ]]; then
  placement_actual_sha256="$(python3 -c '
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
' "${PLACEMENT_PATH}")"
  [[ "${placement_actual_sha256}" == "${PLACEMENT_SHA256}" ]] || {
    echo "placement hash mismatch" >&2; exit 2;
  }
  permutation_actual_sha256="$(python3 -c '
import hashlib, sys
print(hashlib.sha256(sys.argv[1].encode()).hexdigest())
' "${PERMUTATION_BY_LAYER}")"
  [[ "${permutation_actual_sha256}" == \
     "$(jq -r '.placement.permutation_env_sha256' "${DEPLOYMENT}")" ]] || {
    echo "layerwise permutation hash mismatch" >&2; exit 2;
  }
fi
[[ -z "$(git -C "${PLUGIN_ROOT}" status --short)" ]] || {
  echo "EcoDEP v0.26 plugin worktree is dirty" >&2; exit 2;
}
[[ "$(git -C "${PLUGIN_ROOT}" rev-parse HEAD)" == "${PLUGIN_COMMIT}" ]] || {
  echo "EcoDEP v0.26 plugin commit mismatch" >&2; exit 2;
}
jq -e '
  .schema_version >= 2 and
  .selection_split == "calibration" and
  .runtime_contract.cuda_graph_full_decode_only == false and
  .runtime_contract.enable_dbo == true and
  .runtime_contract.routing_sidecar == false and
  ((.runtime_contract.routing_ffn_sidecar // false) == false) and
  (.runtime_contract.routing_async_copy // true) == true and
  .runtime_contract.ffn_cudagraph == false and
  ((.runtime_contract.routing_source_role // "attention") == "attention") and
  ((.runtime_contract.compute_gate_on_attention // false) == false) and
  .runtime_contract.stage_trace == false and
  .runtime_contract.expert_boundary_action == false and
  .runtime_contract.request_path_clock_transitions == 0
' "${DEPLOYMENT}" >/dev/null
IFS=',' read -ra rates <<< "${RATES_CSV}"
for rate in "${rates[@]}"; do
  normalized_rate="$(python3 -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${rate}")"
  jq -e --arg rate "${normalized_rate}" \
    '.operating_points.by_rps[$rate].candidate_id | type == "string"' \
    "${DEPLOYMENT}" >/dev/null || {
      echo "base deployment lacks ${normalized_rate} rps point" >&2
      exit 2
    }
done
[[ ! -e "${SUITE_DIR}" && ! -e "${RUN_DIR}" ]] || {
  echo "suite/run directory already exists; use a new immutable suite id" >&2
  exit 2
}
mkdir -p "${SUITE_DIR}" "${RUN_DIR}" "${MONITOR_RUNTIME_DIR}"
controller_config_sha256="$(sha256sum "${CONTROLLER_CONFIG}" | awk '{print $1}')"
install -m 0444 "${CONTROLLER_CONFIG}" "${SUITE_DIR}/controller-config.json"
if [[ "${EVALUATION_SPLIT}" == "heldout" ]]; then
  [[ -x "${TRACE_AUDITOR}" ]] || {
    echo "heldout launch requires executable trace isolation auditor" >&2; exit 2;
  }
  python3 "${TRACE_AUDITOR}" \
    --calibration "$(jq -r '.calibration_trace' "${DEPLOYMENT}")" \
    --evaluation "${TRACE}" \
    --identity-field source_index --identity-field source_timestamp \
    > "${SUITE_DIR}/trace-isolation-audit.json"
  jq -e '.status == "PASS" and (.overlap_counts | all(. == 0))' \
    "${SUITE_DIR}/trace-isolation-audit.json" >/dev/null
else
  jq -n --arg trace "${TRACE}" --arg sha "${trace_actual_sha256}" \
    '{status:"CALIBRATION_ONLY",trace:$trace,trace_sha256:$sha,
      formal_evaluation_eligible:false}' \
    > "${SUITE_DIR}/trace-isolation-audit.json"
fi

watcher_pid=""
controller_pid=""
server_started=0
gpu_allocation_acquired=0
cleanup_done=0
cleanup() {
  local cleanup_status=0
  if [[ "${cleanup_done}" == "1" ]]; then
    return 0
  fi
  cleanup_done=1
  if [[ -n "${controller_pid}" ]] && kill -0 "${controller_pid}" 2>/dev/null; then
    kill "${controller_pid}" 2>/dev/null || true
    wait "${controller_pid}" 2>/dev/null || true
  fi
  controller_pid=""
  if [[ -n "${watcher_pid}" ]] && kill -0 "${watcher_pid}" 2>/dev/null; then
    kill "${watcher_pid}" 2>/dev/null || true
    wait "${watcher_pid}" 2>/dev/null || true
  fi
  if [[ -f "${MONITOR_OUTPUT}" ]]; then
    cp "${MONITOR_OUTPUT}" "${SUITE_DIR}/gpu-contamination.jsonl"
  fi
  if [[ -f "${MONITOR_FAILURE}" ]]; then
    cp "${MONITOR_FAILURE}" "${SUITE_DIR}/gpu-monitor-failure.json"
  fi
  if [[ "${server_started}" == "1" ]]; then
    if ! python3 "${ECODEP_ROOT}/migration/native_service.py" stop "${RUN_DIR}"; then
      cleanup_status=1
    fi
  fi
  if [[ "${gpu_allocation_acquired}" != "1" ]]; then
    return "${cleanup_status}"
  fi
  if ! python3 "${ECODEP_ROOT}/scripts/afd/reset_gpus.py" \
      --url "${CLOCK_URL}" --gpu-url-map "${GPU_CLOCK_URL_MAP}" \
      --gpus "${GPUS}" --output "${SUITE_DIR}/gpu-reset-ack.json" >/dev/null; then
    echo "failed to reset the complete GPU allocation" >&2
    cleanup_status=1
  elif ! python3 "${ECODEP_ROOT}/scripts/afd/check_gpu_reset.py" \
      --gpus "${GPUS}" --reset-ack "${SUITE_DIR}/gpu-reset-ack.json" \
      --output "${SUITE_DIR}/gpu-reset-check.json" >/dev/null; then
    echo "failed to verify the complete GPU reset" >&2
    cleanup_status=1
  fi
  return "${cleanup_status}"
}
on_exit() {
  local status=$?
  trap - EXIT
  if ! cleanup; then
    status=1
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

# GPU access remains shared and compute mode remains unchanged.  These checks
# only wait for a clean start and invalidate our measurement if a foreign CUDA
# context later overlaps it; they never deny or stop another user's process.
python3 "${ECODEP_ROOT}/scripts/afd/check_gpu_availability.py" \
  --gpus "${GPUS}" --minimum-free-gib "${MINIMUM_FREE_GIB}" --stable-for-s "${GPU_STABLE_FOR_S}" \
  --wait-timeout-s "${GPU_WAIT_TIMEOUT_S}" --poll-interval-s 1 \
  > "${SUITE_DIR}/gpu-availability.json"
gpu_allocation_acquired=1
python3 "${ECODEP_ROOT}/scripts/afd/reset_gpus.py" \
  --url "${CLOCK_URL}" --gpu-url-map "${GPU_CLOCK_URL_MAP}" \
  --gpus "${GPUS}" --output "${SUITE_DIR}/preflight-reset-ack.json" >/dev/null
python3 "${ECODEP_ROOT}/scripts/afd/check_gpu_reset.py" \
  --gpus "${GPUS}" --reset-ack "${SUITE_DIR}/preflight-reset-ack.json" \
  --output "${SUITE_DIR}/preflight-reset-check.json" >/dev/null

start_script="${ECODEP_ROOT}/scripts/afd/start_server_native.sh"
ECODEP_AFD_PLUGIN_ROOT="${PLUGIN_ROOT}" \
ECODEP_MODEL_PATH="${MODEL_PATH}" \
ECODEP_MODEL_TAG="${MODEL_TAG}" \
ECODEP_RUN_ID="${SUITE_ID}" \
ECODEP_TRACE_DIR="$(dirname "${TRACE}")" \
ECODEP_SERVED_MODEL_NAME="${MODEL_NAME}" \
ECODEP_ATTENTION_GPUS="${ATTENTION_GPUS}" \
ECODEP_EXPERT_GPUS="${EXPERT_GPUS}" \
ECODEP_ATTENTION_RANKS=2 \
ECODEP_EXPERT_RANKS=2 \
ECODEP_ATTENTION_TP=1 \
ECODEP_EXPERT_TP=1 \
ECODEP_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
ECODEP_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
ECODEP_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
ECODEP_ENABLE_PREFIX_CACHING=1 \
ECODEP_LANGUAGE_MODEL_ONLY=1 \
ECODEP_ENABLE_CUDA_COMPATIBILITY=1 \
ECODEP_CUDA_COMPATIBILITY_PATH="${ECODEP_CUDA_COMPATIBILITY_PATH:-}" \
ECODEP_VLLM_SERVER_DEV_MODE=1 \
ECODEP_CUDA_GRAPH_FULL_DECODE_ONLY=0 \
ECODEP_CUDAGRAPH_CAPTURE_SIZE="${CUDAGRAPH_CAPTURE_SIZE}" \
ECODEP_ENABLE_DBO=1 \
ECODEP_COMPUTE_GATE_ON_ATTENTION="${CONTROLLER_COMPUTE_GATE}" \
ECODEP_DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD}" \
ECODEP_DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD}" \
ECODEP_ENABLE_ROUTING_SIDECAR="${CONTROLLER_ROUTING_SIDECAR}" \
ECODEP_ROUTING_FFN_SIDECAR="${CONTROLLER_ROUTING_FFN_SIDECAR}" \
ECODEP_ROUTING_SOURCE_ROLE="${CONTROLLER_ROUTING_SOURCE_ROLE}" \
ECODEP_ROUTING_ASYNC_COPY=1 \
ECODEP_ROUTING_PREFILL_ONLY="${CONTROLLER_ROUTING_PREFILL_ONLY}" \
ECODEP_ROUTING_LAYER_STRIDE="${CONTROLLER_ROUTING_LAYER_STRIDE}" \
ECODEP_ROUTING_REQUEST_DETAIL="${CONTROLLER_ROUTING_REQUEST_DETAIL}" \
ECODEP_ROUTING_ACTIVE_MARKER="" \
ECODEP_FFN_CUDAGRAPH=0 \
ECODEP_ENABLE_STAGE_TRACE=0 \
ECODEP_ENABLE_EXPERT_EPOCH_TRACE=0 \
ECODEP_ENABLE_EXPERT_BOUNDARY_ACTION=0 \
ECODEP_EXPERT_PERMUTATION_BY_LAYER="${PERMUTATION_BY_LAYER}" \
  "${start_script}" >/dev/null
server_started=1

watcher_args=(
  --output "${MONITOR_OUTPUT}"
  --failure-marker "${MONITOR_FAILURE}"
  --gpus "${GPUS}"
  --container "${CONTAINER_NAME}"
  --interval-s 0.5
)
watcher_args+=(--owner-pid-file "${RUN_DIR}/native-service.pid")
python3 "${ECODEP_ROOT}/scripts/afd/monitor_shared_gpu.py" watch \
  "${watcher_args[@]}" > "${SUITE_DIR}/gpu-contamination-watch.log" 2>&1 &
watcher_pid=$!

python3 "${ECODEP_ROOT}/migration/native_service.py" wait "${RUN_DIR}" --timeout "${ECODEP_SERVER_TIMEOUT_S:-1800}"
kill -0 "${watcher_pid}" 2>/dev/null && \
  [[ ! -e "${MONITOR_FAILURE}" ]] || {
    echo "shared-GPU contamination monitor failed during startup" >&2; exit 1;
  }
runtime_exec=()
python3 "${ECODEP_ROOT}/migration/runtime_fingerprint.py" --check "${ECODEP_ROOT}/environment/native-runtime.json"
rg -q 'enable_prefix_caching.*True' "${RUN_DIR}/attention.log" || {
  echo "${ARM} v0.26 startup did not confirm prefix caching" >&2; exit 1;
}
rg -q "enforce_eager['\"]?: True" "${RUN_DIR}/attention.log" || {
  echo "${ARM} Attention startup did not confirm enforce_eager" >&2; exit 1;
}
rg -q "enforce_eager['\"]?: True" "${RUN_DIR}/expert.log" || {
  echo "${ARM} FFN startup did not confirm enforce_eager" >&2; exit 1;
}
rg -q 'enable_dbo.*True' "${RUN_DIR}/attention.log" || {
  echo "${ARM} v0.26 startup did not confirm DBO" >&2; exit 1;
}
if [[ "${PLACEMENT_ENABLED}" == "true" ]]; then
  rg -q "EcoDEP static expert placement installed: mode=layerwise.*sha256=${permutation_actual_sha256}" \
    "${RUN_DIR}/expert.log" || {
      echo "${ARM} FFN ranks did not confirm the frozen static placement" >&2
      exit 1
    }
fi
if (( CONTROLLER_POLICY_VERSION == 5 )); then
  mapfile -t routing_sidecars < <(
    find "${RUN_DIR}" -maxdepth 1 -type f \
      -name "routing-${CONTROLLER_ROUTING_SOURCE_ROLE}-*.jsonl" -print
  )
  [[ "${#routing_sidecars[@]}" -eq 2 ]] || {
    echo "v5 controller requires two ${CONTROLLER_ROUTING_SOURCE_ROLE} routing sidecars" >&2
    exit 1
  }
fi

python3 "${ECODEP_ROOT}/scripts/afd/set_clocks.py" \
  --attention-url "${ATTENTION_CLOCK_URL}" --expert-url "${EXPERT_CLOCK_URL}" \
  --gpu-url-map "${GPU_CLOCK_URL_MAP}" \
  --attention-gpus "${ATTENTION_GPUS}" --expert-gpus "${EXPERT_GPUS}" \
  --attention "${MAX_ATTENTION_CLOCKS}" --expert "${MAX_EXPERT_CLOCKS}" \
  --attention-power-w "${MAX_ATTENTION_POWER_W}" --expert-power-w "${MAX_EXPERT_POWER_W}" \
  --output "${SUITE_DIR}/warmup-operating-point-ack.json" >/dev/null
sleep "${SETTLE_S}"
warmup_request_log="${RUN_DIR}/warmup-${ARM}-r${REPETITION}.jsonl"
"${runtime_exec[@]}" python3 "${ECODEP_ROOT}/scripts/afd/replay_trace.py" \
  "${WARMUP_TRACE}" \
  --endpoint "http://127.0.0.1:${ECODEP_API_PORT:-18000}/v1/completions" \
  --model "${MODEL_NAME}" \
  --output "${RUN_DIR}/$(basename "${warmup_request_log}")" \
  --ttft-slo-ms 400 --tpot-slo-ms 120 \
  --time-scale 0.001 --max-output-tokens "${MAX_OUTPUT_TOKENS}"
python3 "${ECODEP_ROOT}/scripts/afd/summarize_replay.py" \
  "${warmup_request_log}" --output "${SUITE_DIR}/warmup-summary.json"
jq -e '.failed_requests == 0 and .completed_requests == .requests' \
  "${SUITE_DIR}/warmup-summary.json" >/dev/null
"${runtime_exec[@]}" python3 -c '
import json
import urllib.request
import os
call = urllib.request.Request("http://127.0.0.1:" + os.environ.get("ECODEP_API_PORT", "18000") + "/reset_prefix_cache", method="POST")
with urllib.request.urlopen(call, timeout=30) as response:
    assert response.status == 200
    assert json.load(response).get("success") is True
'

image_id="native:$(sha256sum "${ECODEP_ROOT}/environment/native-runtime.json" | awk '{print $1}')"
model_config_sha256="$(python3 -c '
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
' "${MODEL_PATH}/config.json")"
python3 - "${SUITE_DIR}/manifest.json" "${DEPLOYMENT}" "${TRACE}" \
  "${RATES_CSV}" "${IMAGE}" "${image_id}" "${model_config_sha256}" \
  "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}" \
  "${CUDAGRAPH_CAPTURE_SIZE}" "${DBO_DECODE_TOKEN_THRESHOLD}" \
  "${DBO_PREFILL_TOKEN_THRESHOLD}" "${MAX_OUTPUT_TOKENS}" \
  "${REPETITION}" "${EVALUATION_SPLIT}" "${ARM}" "${schedule_path}" \
  "${schedule_sha256}" "${SCHEDULE_POSITION}" "${warmup_trace_sha256}" \
  "${warmup_manifest_sha256}" "${trace_base_arrival_rate_rps}" \
  "${ROUTING_SMOKE_LIMIT}" "${CALIBRATION_REQUEST_LIMIT}" "${GPUS}" \
  "${CONTROLLER_ARM}" "${CONTROLLER_CONFIG}" "${controller_config_sha256}" \
  "${PLUGIN_ROOT}" "${PLUGIN_COMMIT}" <<'PY'
import hashlib, json, pathlib, sys
output, deployment_path, trace = map(pathlib.Path, sys.argv[1:4])
deployment = json.loads(deployment_path.read_text())
comparison_contract = {
    "schema_version": 1,
    "data_plane": "official_vllm_afd_v0.26_role_pruned_attention_ffn",
    "container": {"image": sys.argv[5], "image_id": sys.argv[6]},
    "model": {
        "name": deployment["model"],
        "config_sha256": sys.argv[7],
    },
    "topology": {
        "attention_dp": 2,
        "ffn_ep": 2,
        "attention_tp": 1,
        "expert_tp": 1,
    },
    "server": {
        "max_model_len": int(sys.argv[8]),
        "max_num_seqs": int(sys.argv[9]),
        "max_num_batched_tokens": int(sys.argv[10]),
        "prefix_caching": True,
        "cuda_graph_full_decode_only": False,
        "ffn_cudagraph": False,
        "enforce_eager": True,
        "cudagraph_capture_size": int(sys.argv[11]),
        "dbo": True,
        "dbo_decode_token_threshold": int(sys.argv[12]),
        "dbo_prefill_token_threshold": int(sys.argv[13]),
    },
    "generation": {
        "api": "v1/completions",
        "temperature": 0.0,
        "ignore_eos": True,
        "max_output_tokens": int(sys.argv[14]),
    },
    "measurement": {
        "gpu_ids": [int(value) for value in sys.argv[26].split(",")],
        "sample_interval_ms": 10,
        "energy_window": "request_first_submit_to_last_finish",
        "ttft_slo_ms": 400.0,
        "tpot_slo_ms": 120.0,
    },
    "arrival_rate_scaling": "(request_count-1)/(last_arrival-first_arrival)/target_rps",
    "warmup": {
        "trace_sha256": sys.argv[21],
        "manifest_sha256": sys.argv[22],
        "max_output_tokens": int(sys.argv[14]),
        "measured": False,
        "prefix_cache_reset_after": True,
    },
}
controller_config_path = pathlib.Path(sys.argv[28])
controller_config = json.loads(controller_config_path.read_text())
runtime_contract = dict(deployment["runtime_contract"])
runtime_contract.update({
    "causal_role_dvfs": True,
    "causal_event_source": "post_submit_request_lifecycle_only",
    "attention_frequency_control": sys.argv[27] in {"dynamic-a", "dynamic-ae"},
    "expert_frequency_control": sys.argv[27] in {"dynamic-e", "dynamic-ae"},
    "request_path_clock_transitions": "external_guarded_role_controller",
    "controller_policy_version": int(controller_config.get("policy_version", 1)),
    "submit_event_wakeup": controller_config.get("events", {}).get(
        "submit_wakeup", "polling"
    ),
    "progress_event_mode": controller_config.get("events", {}).get(
        "progress_mode", "per-request"
    ),
    "progress_event_interval_ms": controller_config.get("events", {}).get(
        "progress_event_interval_ms", 50
    ),
    "future_trace_access": False,
    "static_dse_online": False,
    "fixed_topology": "2A2E",
    "next_window_stage_predictor": int(controller_config.get("policy_version", 1)) >= 3,
    "joint_frequency_power_cap_control": int(
        controller_config.get("policy_version", 1)
    ) in {4, 5},
    "routing_sidecar": int(controller_config.get("policy_version", 1)) == 5,
    "routing_source_role": controller_config.get("compatibility", {}).get(
        "routing_source_role", "attention"
    ),
    "routing_ffn_sidecar": controller_config.get("compatibility", {}).get(
        "routing_source_role", "attention"
    ) == "ffn",
    "routing_prefill_only": controller_config.get("predictor", {}).get(
        "routing_collection", {}
    ).get("prefill_only", False),
    "routing_layer_stride": controller_config.get("predictor", {}).get(
        "routing_collection", {}
    ).get("layer_stride", 1),
    "routing_request_detail": controller_config.get("predictor", {}).get(
        "routing_collection", {}
    ).get("request_detail", True),
    "online_routing_access": int(controller_config.get("policy_version", 1)) == 5,
    "compute_gate_on_attention": controller_config.get("compatibility", {}).get(
        "compute_gate_on_attention", False
    ),
})
output.write_text(json.dumps({
    "schema_version": 2,
    "backend": "ecodep-vllm-afd-v026-causal-role-dvfs",
    "arm": sys.argv[17],
    "model": deployment["model"],
    "evaluation_split": sys.argv[16],
    "measurement_eligible": sys.argv[16] == "heldout",
    "repetition": int(sys.argv[15]),
    "schedule": {
        "path": str(pathlib.Path(sys.argv[18]).resolve()),
        "sha256": sys.argv[19],
        "position": int(sys.argv[20]),
    },
    "trace": str(trace.resolve()),
    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    "trace_base_arrival_rate_rps": float(sys.argv[23]),
    "measurement_request_limit": int(sys.argv[25]) if sys.argv[25] else None,
    "comparison_contract": comparison_contract,
    "comparison_contract_sha256": hashlib.sha256(json.dumps(
        comparison_contract, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest(),
    "deployment": str(deployment_path.resolve()),
    "deployment_sha256": hashlib.sha256(deployment_path.read_bytes()).hexdigest(),
    "plugin": {
        **deployment["plugin"],
        "root": str(pathlib.Path(sys.argv[30]).resolve()),
        "commit": sys.argv[31],
    },
    "placement": {k: v for k, v in deployment["placement"].items()
                  if k != "permutation_by_layer"},
    "runtime_contract": runtime_contract,
    "controller": {
        "method": controller_config["method"],
        "policy_version": int(controller_config.get("policy_version", 1)),
        "ablation_arm": sys.argv[27],
        "source_config": str(controller_config_path.resolve()),
        "config": str((output.parent / "controller-config.json").resolve()),
        "config_sha256": sys.argv[29],
        "script_sha256": hashlib.sha256(pathlib.Path(
            str(deployment_path.resolve().parents[3] / "scripts/afd/causal_dvfs_v5/controller.py")
            if int(controller_config.get("policy_version", 1)) == 5 else
            str(deployment_path.resolve().parents[3] / "scripts/afd/causal_dvfs/controller.py")
        ).read_bytes()).hexdigest(),
        "replay_script_sha256": hashlib.sha256(pathlib.Path(
            str(deployment_path.resolve().parents[3] / "scripts/afd/causal_dvfs/replay_client.py")
        ).read_bytes()).hexdigest(),
        "runner_script_sha256": hashlib.sha256(pathlib.Path(
            str(deployment_path.resolve().parents[3] / "scripts/afd/run_v026_causal_dvfs_rate_suite.sh")
        ).read_bytes()).hexdigest(),
        "selection_split": "calibration",
        "reads_trace": False,
        "predictor": controller_config.get("predictor"),
    },
    "operating_points": {k: v for k, v in deployment["operating_points"].items()
                         if k != "by_rps"},
    "rates_rps": [float(value) for value in sys.argv[4].split(",")],
    "routing_smoke": {
        "enabled": bool(sys.argv[24]),
        "request_limit": int(sys.argv[24]) if sys.argv[24] else None,
        "formal_provenance": not bool(sys.argv[24]),
    },
    "gpu_access_policy": {
        "device_acl": "unchanged",
        "compute_mode": "unchanged",
        "availability_gate": "passive",
        "foreign_cuda_context": "invalidate_measurement",
    },
    "paper_slo": {"quantile": "p90", "ttft_ms": 400.0, "tpot_ms": 120.0},
    "topology": {
        "attention_dp2": [int(value) for value in sys.argv[26].split(",")[:2]],
        "ffn_ep2": [int(value) for value in sys.argv[26].split(",")[2:]],
    },
    "native_memory_limit_bytes": None,
}, indent=2) + "\n")
PY
replay_limit_args=()
if [[ -n "${CALIBRATION_REQUEST_LIMIT}" ]]; then
  replay_limit_args=(--limit "${CALIBRATION_REQUEST_LIMIT}")
fi
for rate in "${rates[@]}"; do
  normalized_rate="$(python3 -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${rate}")"
  point="$(jq -c --arg arm "${CONTROLLER_ARM}" \
    --arg attention_power "${MAX_ATTENTION_POWER_W}" \
    --arg expert_power "${MAX_EXPERT_POWER_W}" '
    {candidate_id:(if (.policy_version // 1) >= 2 then
       "causal-dvfs-v" + ((.policy_version // 1) | tostring) + "-" + $arm
     else "causal-dvfs-" + $arm end),selection_split,method,
     policy_version:(.policy_version // 1),initial_states,events,signals,
     frequencies_mhz,power_caps_w,thresholds,safety,predictor,compatibility,
     attention_power_w:($attention_power | split(",") | map(tonumber)),
     expert_power_w:($expert_power | split(",") | map(tonumber)),
     transition_policy:"immediate_up_delayed_down",
     transition_energy_included:true}' "${CONTROLLER_CONFIG}")"
  case_dir="${RUN_DIR}/rps-${normalized_rate}"
  request_log="${RUN_DIR}/replay-ecodep-v026-rps-${normalized_rate}.jsonl"
  controller_events="${case_dir}/controller-events.jsonl"
  controller_log="${case_dir}/controller-actions.jsonl"
  controller_summary="${case_dir}/controller-summary.json"
  controller_stop="${case_dir}/controller.stop"
  controller_signal_fifo="${case_dir}/submit-signal.fifo"
  mkdir -p "${case_dir}"
  rm -f "${controller_events}" "${controller_log}" \
    "${controller_summary}" "${controller_stop}" "${controller_signal_fifo}"
  python3 "${ECODEP_ROOT}/scripts/afd/set_clocks.py" \
    --attention-url "${ATTENTION_CLOCK_URL}" --expert-url "${EXPERT_CLOCK_URL}" \
    --gpu-url-map "${GPU_CLOCK_URL_MAP}" \
    --attention-gpus "${ATTENTION_GPUS}" --expert-gpus "${EXPERT_GPUS}" \
    --attention "${MAX_ATTENTION_CLOCKS}" --expert "${MAX_EXPERT_CLOCKS}" \
    --attention-power-w "${MAX_ATTENTION_POWER_W}" --expert-power-w "${MAX_EXPERT_POWER_W}" \
    --output "${case_dir}/operating-point-ack.json" >/dev/null
  jq -e '.verified == true and (.verification_errors | length) == 0' \
    "${case_dir}/operating-point-ack.json" >/dev/null
  sleep "${SETTLE_S}"
  time_scale="$(python3 -c 'import sys; print(float(sys.argv[1]) / float(sys.argv[2]))' \
    "${trace_base_arrival_rate_rps}" "${rate}")"
  "${runtime_exec[@]}" python3 -c '
import json
import urllib.request
import os
call = urllib.request.Request("http://127.0.0.1:" + os.environ.get("ECODEP_API_PORT", "18000") + "/reset_prefix_cache", method="POST")
with urllib.request.urlopen(call, timeout=30) as response:
    assert response.status == 200
    assert json.load(response).get("success") is True
'
  if (( CONTROLLER_POLICY_VERSION == 5 )); then
    python3 "${ECODEP_ROOT}/scripts/afd/prepare_measurement_traces.py" \
      "${RUN_DIR}" --sidecar-mode sidecar-on --expert-epoch-mode epoch-off \
        --routing-role "${CONTROLLER_ROUTING_SOURCE_ROLE}" --routing-ranks 2 \
      --output "${case_dir}/measurement-trace-reset.json" >/dev/null
  fi
  controller_signal_args=()
  replay_signal_args=()
  if [[ "${SUBMIT_WAKEUP}" == "named_pipe" ]]; then
    controller_signal_args=(--submit-signal-fifo "${controller_signal_fifo}")
    replay_signal_args=(
      --submit-signal-fifo "${RUN_DIR}/rps-${normalized_rate}/submit-signal.fifo"
    )
  fi
  controller_routing_args=()
  if (( CONTROLLER_POLICY_VERSION == 5 )); then
    controller_routing_args=(
      --routing-dir "${RUN_DIR}" \
      --routing-pattern "routing-${CONTROLLER_ROUTING_SOURCE_ROLE}-*.jsonl"
    )
  fi
  python3 "${CONTROLLER_SCRIPT}" \
    --config "${CONTROLLER_CONFIG}" --arm "${CONTROLLER_ARM}" \
    --events "${controller_events}" --output "${controller_log}" \
    --summary "${controller_summary}" --stop-marker "${controller_stop}" \
    "${controller_signal_args[@]}" \
    "${controller_routing_args[@]}" \
    --attention-gpus "${ATTENTION_GPUS}" --expert-gpus "${EXPERT_GPUS}" \
    --attention-url "${ATTENTION_CLOCK_URL}" --expert-url "${EXPERT_CLOCK_URL}" \
    > "${case_dir}/controller.stdout.log" 2> "${case_dir}/controller.stderr.log" &
  controller_pid=$!
  sleep 1
  kill -0 "${controller_pid}" 2>/dev/null || {
    echo "causal DVFS controller failed before replay" >&2
    wait "${controller_pid}" || true
    exit 1
  }
  python3 "${ECODEP_ROOT}/scripts/afd/measure_command.py" \
    --gpus "${GPUS}" --output "${case_dir}/telemetry.json" \
    --samples-output "${case_dir}/power-samples.jsonl" \
    --request-log "${request_log}" -- \
    "${runtime_exec[@]}" python3 "${REPLAY_SCRIPT}" \
      "${TRACE}" \
      --endpoint "http://127.0.0.1:${ECODEP_API_PORT:-18000}/v1/completions" \
      --model "${MODEL_NAME}" \
      --output "${RUN_DIR}/$(basename "${request_log}")" \
      --events "${RUN_DIR}/rps-${normalized_rate}/controller-events.jsonl" \
      --progress-event-mode "${PROGRESS_EVENT_MODE}" \
      --progress-event-interval-ms "${PROGRESS_EVENT_INTERVAL_MS}" \
      "${replay_signal_args[@]}" \
      --ttft-slo-ms 400 --tpot-slo-ms 120 \
      --time-scale "${time_scale}" \
      "${replay_limit_args[@]}" \
      --max-output-tokens "${MAX_OUTPUT_TOKENS}"
  controller_deadline=$((SECONDS + 60))
  while kill -0 "${controller_pid}" 2>/dev/null && \
      (( SECONDS < controller_deadline )); do
    sleep 0.2
  done
  if kill -0 "${controller_pid}" 2>/dev/null; then
    touch "${controller_stop}"
    echo "causal DVFS controller did not finish after replay" >&2
    wait "${controller_pid}" || true
    controller_pid=""
    exit 1
  fi
  if ! wait "${controller_pid}"; then
    controller_pid=""
    echo "causal DVFS controller failed" >&2
    exit 1
  fi
  controller_pid=""
  if (( CONTROLLER_POLICY_VERSION == 5 )); then
    jq -e '
      .status == "complete" and .replay_started == true and
      .replay_ended == true and .outstanding_at_exit == 0 and
      .restored_to_guard == true and .routing_updates > 0
    ' "${controller_summary}" >/dev/null
  else
    jq -e '
      .status == "complete" and .replay_started == true and
      .replay_ended == true and .outstanding_at_exit == 0 and
      .restored_to_max == true
    ' "${controller_summary}" >/dev/null
  fi
  energy="$(jq -r '.energy_j' "${case_dir}/telemetry.json")"
  python3 "${ECODEP_ROOT}/scripts/afd/summarize_replay.py" \
    "${request_log}" --energy-j "${energy}" --output "${case_dir}/summary.json"
  jq -e '.failed_requests == 0 and .completed_requests == .requests' \
    "${case_dir}/summary.json" >/dev/null
  jq -e '
    .returncode == 0 and
    .measurement_window_source == "request_first_submit_to_last_finish" and
    .sample_time_coverage >= 0.99 and
    .sample_error_count == 0
  ' "${case_dir}/telemetry.json" >/dev/null
  python3 "${ECODEP_ROOT}/scripts/afd/monitor_shared_gpu.py" validate \
    --output "${MONITOR_OUTPUT}" \
    --failure-marker "${MONITOR_FAILURE}" \
    --telemetry "${case_dir}/telemetry.json" --container "${CONTAINER_NAME}" \
    --report "${case_dir}/gpu-contamination-validation.json" \
    --max-gap-s 1.5 --wait-timeout-s 3
  jq -n --arg rate "${rate}" --arg scale "${time_scale}" \
    --arg split "${EVALUATION_SPLIT}" --argjson repetition "${REPETITION}" \
    --argjson settle_s "${SETTLE_S}" --argjson point "${point}" \
    --argjson base_rps "${trace_base_arrival_rate_rps}" \
    --argjson request_limit "${CALIBRATION_REQUEST_LIMIT:-null}" \
    '{offered_rps:($rate|tonumber),time_scale:($scale|tonumber),
      evaluation_split:$split,repetition:$repetition,
      trace_base_arrival_rate_rps:$base_rps,
      measurement_request_limit:$request_limit,
      arrival_rate_scaling:"base_arrival_rate_rps / offered_rps",
      operating_point:$point,operating_point_settle_s:$settle_s,
      controller_summary:"controller-summary.json",
      controller_actions:"controller-actions.jsonl",
      causal_events:"controller-events.jsonl",
      controller_policy_version:$point.policy_version,
      submit_event_wakeup:$point.events.submit_wakeup,
      progress_event_mode:$point.events.progress_mode,
      progress_event_interval_ms:$point.events.progress_event_interval_ms,
      transition_energy_included:true}' \
    > "${case_dir}/case.json"
done

if ! cleanup; then
  echo "suite measurements finished, but mandatory cleanup failed" >&2
  exit 1
fi
touch "${SUITE_DIR}/COMPLETE"

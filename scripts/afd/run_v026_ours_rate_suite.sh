#!/usr/bin/env bash
# Matched v0.26 static-placement rate sweep for B2, B3, or EcoDEP calibration/heldout.

set -euo pipefail

export ECODEP_AFD_IMAGE="${ECODEP_V026_AFD_IMAGE:-ecodep-vllm-afd:v0.26.0}"
source "$(dirname "$0")/common.sh"

readonly DEPLOYMENT="${1:?frozen deployment required}"
readonly TRACE_INPUT="${2:?trace required}"
readonly SUITE_ID="${3:?suite id required}"
readonly RATES_CSV="${4:-2,4,8,16}"
[[ -f "${DEPLOYMENT}" ]] || { echo "missing deployment ${DEPLOYMENT}" >&2; exit 2; }
[[ -f "${TRACE_INPUT}" ]] || { echo "missing trace ${TRACE_INPUT}" >&2; exit 2; }
readonly TRACE="$(realpath "${TRACE_INPUT}")"
readonly MODEL_TAG="$(jq -r '.model' "${DEPLOYMENT}")"
readonly MODEL_PATH="${ECODEP_ROOT}/artifacts/models/${MODEL_TAG}"
readonly PLUGIN_ROOT="$(jq -r '.plugin.root' "${DEPLOYMENT}")"
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
readonly GPUS="${ECODEP_MEASUREMENT_GPUS:-${ATTENTION_GPUS},${EXPERT_GPUS}}"
readonly ATTENTION_RANKS="$(jq -r '.topology.attention_ranks // .topology.attention_dp' "${DEPLOYMENT}")"
readonly EXPERT_RANKS="$(jq -r '.topology.expert_ranks // .topology.ffn_ep' "${DEPLOYMENT}")"
readonly ATTENTION_TP="$(jq -r '.topology.attention_tp // 1' "${DEPLOYMENT}")"
readonly EXPERT_TP="$(jq -r '.topology.expert_tp // 1' "${DEPLOYMENT}")"
readonly MICROBATCHES="$(jq -r '.topology.microbatches // 2' "${DEPLOYMENT}")"
readonly JOINT_LAYOUT="$(jq -r '.topology.layout_contract // ""' "${DEPLOYMENT}")"
readonly ENABLE_DBO="$([[ "${MICROBATCHES}" == "2" ]] && printf 1 || printf 0)"
# BO supports larger P2P topologies; legacy suites retain their four-card scope.
python3 - "${ATTENTION_GPUS}" "${EXPERT_GPUS}" "${GPUS}" "${ATTENTION_RANKS}" "${EXPERT_RANKS}" "${ECODEP_ROOT}" "${ECODEP_BO_FEEDBACK:-0}" <<'PYGPU'
import sys
sys.path.insert(0, sys.argv[6] + '/migration')
from bo_topology import validate_groups
attention, expert, allocation = ([int(x) for x in v.split(',')] for v in sys.argv[1:4])
validate_groups(attention, expert, allocation)
if (len(attention) != int(sys.argv[4]) or len(expert) != int(sys.argv[5])
    or (sys.argv[7] != '1' and (len(attention) != 2 or len(expert) not in (1, 2)))):
    raise SystemExit('Invalid static topology or measurement allocation')
PYGPU
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
readonly EVALUATION_SPLIT="$(jq -r '.evaluation_split // "heldout"' "${DEPLOYMENT}")"
readonly ARM="$(jq -r '.arm // "ours"' "${DEPLOYMENT}")"
readonly COMPUTE_GATE_ON_ATTENTION="$(jq -r '.runtime_contract.compute_gate_on_attention // false' "${DEPLOYMENT}")"
readonly SCHEDULE_INPUT="${ECODEP_SCHEDULE_PATH:-}"
readonly ROUTING_SMOKE_LIMIT="${ECODEP_ROUTING_SMOKE_LIMIT:-}"
readonly CALIBRATION_REQUEST_LIMIT="${ECODEP_CALIBRATION_REQUEST_LIMIT:-}"
readonly MATH_CONTROLLER="$(jq -r '.math_dynamic_controller.path // ""' "${DEPLOYMENT}")"
readonly MATH_PARAMETER_PROBE="$(jq -r '.math_dse_parameter_probe // false' "${DEPLOYMENT}")"
readonly BO_FEEDBACK="${ECODEP_BO_FEEDBACK:-0}"
[[ "${BO_FEEDBACK}" == "0" || "${BO_FEEDBACK}" == "1" ]] || exit 2
if [[ "${BO_FEEDBACK}" == "1" ]]; then
  [[ "${MATH_PARAMETER_PROBE}" == "true" && "${EVALUATION_SPLIT}" == "calibration" && "${RATES_CSV}" != *,* ]] || exit 2
fi
readonly ORACLE_PROFILE_MODE="${ECODEP_ORACLE_PROFILE_MODE:-0}"
readonly GPU_STABLE_FOR_S="${ECODEP_GPU_STABLE_FOR_S:-10}"
readonly GPU_WAIT_TIMEOUT_S="${ECODEP_GPU_WAIT_TIMEOUT_S:-3600}"
readonly MINIMUM_FREE_GIB="${ECODEP_MINIMUM_FREE_GIB:-65}"
SCHEDULE_POSITION="${ECODEP_SCHEDULE_POSITION:-0}"
readonly WARMUP_TRACE_INPUT="${ECODEP_WARMUP_TRACE:?ECODEP_WARMUP_TRACE is required}"
readonly WARMUP_MANIFEST_INPUT="${ECODEP_WARMUP_MANIFEST:?ECODEP_WARMUP_MANIFEST is required}"
if [[ -n "${MATH_CONTROLLER}" ]]; then
  [[ "$(sha256sum "${MATH_CONTROLLER}" | awk '{print $1}')" == "$(jq -r '.math_dynamic_controller.sha256' "${DEPLOYMENT}")" ]] || exit 2
  python3 - "${MATH_CONTROLLER}" "${ECODEP_ROOT}" "${ATTENTION_GPUS}" "${EXPERT_GPUS}" <<'PYMATH'
import sys
sys.path.insert(0, sys.argv[2] + '/scripts/afd')
from math_dynamic.controller import load_config
c = load_config(sys.argv[1])
if c['attention_gpus'] != list(map(int, sys.argv[3].split(','))) or c['expert_gpus'] != list(map(int, sys.argv[4].split(','))):
    raise SystemExit('Static frozen GPU mapping differs from dynamic launch')
PYMATH
fi

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model ${MODEL_PATH}" >&2; exit 2; }

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
[[ "${ORACLE_PROFILE_MODE}" == "0" || "${ORACLE_PROFILE_MODE}" == "1" ]] || {
  echo "ECODEP_ORACLE_PROFILE_MODE must be 0 or 1" >&2; exit 2
}
if [[ "${ORACLE_PROFILE_MODE}" == "1" ]]; then
  [[ "${EVALUATION_SPLIT}" == "routing_calibration" ]] || {
    echo "Oracle profiling is calibration-only routing instrumentation" >&2
    exit 2
  }
  [[ "${RATES_CSV}" != *,* ]] || {
    echo "Oracle profiling requires one offered rate per immutable run" >&2
    exit 2
  }
fi
[[ "${EVALUATION_SPLIT}" == "calibration" || \
   "${EVALUATION_SPLIT}" == "heldout" || \
   "${EVALUATION_SPLIT}" == "routing_calibration" ]] || {
  echo "deployment evaluation_split must be calibration, routing_calibration, or heldout" >&2; exit 2;
}
if [[ -n "${CALIBRATION_REQUEST_LIMIT}" ]]; then
  [[ "${EVALUATION_SPLIT}" == "calibration" && \
     "${CALIBRATION_REQUEST_LIMIT}" =~ ^[0-9]+$ && \
     "${CALIBRATION_REQUEST_LIMIT}" -ge 200 ]] || {
    echo "calibration request limit requires calibration split and at least 200 requests" >&2
    exit 2
  }
fi
[[ "${ARM}" == "ours" || "${ARM}" == "B2" || "${ARM}" == "B3" ]] || {
  echo "deployment arm must be ours, B2, or B3" >&2; exit 2;
}
[[ "${EVALUATION_SPLIT}" == "calibration" || \
   "${EVALUATION_SPLIT}" == "routing_calibration" && "${ARM}" == "B2" || \
   "${EVALUATION_SPLIT}" == "heldout" ]] || {
  echo "invalid arm for the deployment split" >&2; exit 2;
}
schedule_path=""
schedule_sha256=""
schedule_label="${ARM}"
if [[ "${EVALUATION_SPLIT}" == "calibration" && "${ARM}" == "ours" ]]; then
  schedule_label="$(jq -r '.candidate_id' "${DEPLOYMENT}")"
fi
if [[ "${EVALUATION_SPLIT}" == "routing_calibration" ]]; then
  [[ "${RATES_CSV}" != *,* ]] || {
    echo "routing calibration requires exactly one offered rate" >&2; exit 2;
  }
  schedule_path="$(realpath "${DEPLOYMENT}")"
  schedule_sha256="$(python3 -c '
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
' "${schedule_path}")"
  SCHEDULE_POSITION=1
  if [[ -n "${ROUTING_SMOKE_LIMIT}" && \
        ! "${ROUTING_SMOKE_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ECODEP_ROUTING_SMOKE_LIMIT must be a positive integer" >&2
    exit 2
  fi
else
  [[ -z "${ROUTING_SMOKE_LIMIT}" ]] || {
    echo "ECODEP_ROUTING_SMOKE_LIMIT is valid only for routing calibration" >&2
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
fi
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
     "$(jq -r '.operating_points.calibration_trace_sha256' "${DEPLOYMENT}")" ]] || {
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
routing_sidecar_expected=false
[[ "${EVALUATION_SPLIT}" == "routing_calibration" ]] && routing_sidecar_expected=true
jq -e --argjson routing_sidecar "${routing_sidecar_expected}" \
  --argjson microbatches "${MICROBATCHES}" \
  --argjson compute_gate_on_attention "${COMPUTE_GATE_ON_ATTENTION}" --argjson math_probe "${MATH_PARAMETER_PROBE}" '
  .schema_version >= 2 and
  .selection_split == "calibration" and
  .runtime_contract.cuda_graph_full_decode_only == false and
  ($microbatches | type == "number" and . >= 1 and floor == .) and
  ((.runtime_contract.microbatches // 2) == $microbatches) and
  (.runtime_contract.enable_dbo == ($microbatches == 2)) and
  .runtime_contract.routing_sidecar == $routing_sidecar and
  ((.runtime_contract.routing_ffn_sidecar // false) == $routing_sidecar) and
  (.runtime_contract.routing_async_copy // true) == true and
  (if $routing_sidecar
    then .runtime_contract.routing_window_gate == "replay_client_marker"
    else true
   end) and
  .runtime_contract.ffn_cudagraph == false and
  ((.runtime_contract.routing_source_role // "attention") ==
    (if $routing_sidecar then "ffn" else "attention" end)) and
  ((.runtime_contract.compute_gate_on_attention // false) == $compute_gate_on_attention) and
  .runtime_contract.stage_trace == $math_probe and
  (if $math_probe then .evaluation_split == "calibration" else true end) and
  .runtime_contract.expert_boundary_action == false and
  .runtime_contract.request_path_clock_transitions == 0
' "${DEPLOYMENT}" >/dev/null
if [[ "${EVALUATION_SPLIT}" == "heldout" && "${ARM}" == "ours" ]]; then
  jq -e '
    (.operating_points.calibration_trace_sha256 | type == "string") and
    (.operating_points.comparison_contract_sha256 | type == "string")
  ' "${DEPLOYMENT}" >/dev/null
fi
IFS=',' read -ra rates <<< "${RATES_CSV}"
for rate in "${rates[@]}"; do
  normalized_rate="$(python3 -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${rate}")"
  if [[ "${EVALUATION_SPLIT}" == "heldout" && "${ARM}" == "ours" ]]; then
    jq -e --arg rate "${normalized_rate}" \
      '.operating_points.by_rps[$rate].calibration_feasible == true' \
      "${DEPLOYMENT}" >/dev/null || {
        echo "deployment lacks a calibration-feasible ${normalized_rate} rps point" >&2
        exit 2
      }
  else
    jq -e --arg rate "${normalized_rate}" \
      '.operating_points.by_rps[$rate].candidate_id | type == "string"' \
      "${DEPLOYMENT}" >/dev/null || {
        echo "deployment lacks ${normalized_rate} rps point" >&2
        exit 2
      }
  fi
done
[[ ! -e "${SUITE_DIR}" && ! -e "${RUN_DIR}" ]] || {
  echo "suite/run directory already exists; use a new immutable suite id" >&2
  exit 2
}
mkdir -p "${SUITE_DIR}" "${RUN_DIR}" "${MONITOR_RUNTIME_DIR}"

watcher_pid=""
server_started=0
gpu_allocation_acquired=0
cleanup_done=0
cleanup() {
  local cleanup_status=0
  if [[ "${cleanup_done}" == "1" ]]; then
    return 0
  fi
  cleanup_done=1
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
      --gpus "${GPUS}" --output "${SUITE_DIR}/clock-reset.json" >/dev/null; then
    echo "failed to verify allocation clock/power reset" >&2
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
  --gpus "${GPUS}" --output "${SUITE_DIR}/allocation-reset-before.json" >/dev/null

start_script="${ECODEP_ROOT}/scripts/afd/start_server_native.sh"
joint_layout_path=""
if [[ "${JOINT_LAYOUT}" == "independent_afd_replicas_v1" ]]; then
  joint_layout_path="${SUITE_DIR}/joint-layout.json"
  python3 - "${DEPLOYMENT}" "${ATTENTION_GPUS}" "${EXPERT_GPUS}" "${joint_layout_path}" "${DBO_DECODE_TOKEN_THRESHOLD}" "${DBO_PREFILL_TOKEN_THRESHOLD}" <<'PYLAYOUT'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
c = {k: d['topology'][k] for k in ('attention_dp','attention_tp','expert_dp','expert_ep','expert_tp','microbatches')}
c.update(attention_gpus=list(map(int,sys.argv[2].split(','))), expert_gpus=list(map(int,sys.argv[3].split(','))), execution_mode='eager')
Path(sys.argv[4]).write_text(json.dumps({'configuration':c, 'microbatch_thresholds': {
    'dbo_decode_token_threshold':int(sys.argv[5]), 'dbo_prefill_token_threshold':int(sys.argv[6])}})+'\n')
PYLAYOUT
fi
ECODEP_AFD_PLUGIN_ROOT="${PLUGIN_ROOT}" \
ECODEP_MODEL_PATH="${MODEL_PATH}" \
ECODEP_MODEL_TAG="${MODEL_TAG}" \
ECODEP_RUN_ID="${SUITE_ID}" \
ECODEP_TRACE_DIR="$(dirname "${TRACE}")" \
ECODEP_SERVED_MODEL_NAME="${MODEL_NAME}" \
ECODEP_ATTENTION_GPUS="${ATTENTION_GPUS}" \
ECODEP_EXPERT_GPUS="${EXPERT_GPUS}" \
ECODEP_ATTENTION_RANKS="${ATTENTION_RANKS}" \
ECODEP_EXPERT_RANKS="${EXPERT_RANKS}" \
ECODEP_ATTENTION_TP="${ATTENTION_TP}" \
ECODEP_EXPERT_TP="${EXPERT_TP}" \
ECODEP_LAYOUT_JSON="${joint_layout_path}" \
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
ECODEP_ENABLE_DBO="${ENABLE_DBO}" \
ECODEP_COMPUTE_GATE_ON_ATTENTION="$([[ "${COMPUTE_GATE_ON_ATTENTION}" == "true" ]] && printf 1 || printf 0)" \
ECODEP_DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD}" \
ECODEP_DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD}" \
ECODEP_ENABLE_ROUTING_SIDECAR="$([[ "${EVALUATION_SPLIT}" == "routing_calibration" ]] && printf 1 || printf 0)" \
ECODEP_ROUTING_FFN_SIDECAR="$([[ "${EVALUATION_SPLIT}" == "routing_calibration" ]] && printf 1 || printf 0)" \
ECODEP_ROUTING_SOURCE_ROLE="$([[ "${EVALUATION_SPLIT}" == "routing_calibration" ]] && printf ffn || printf attention)" \
ECODEP_ROUTING_ASYNC_COPY=1 \
ECODEP_ROUTING_ACTIVE_MARKER="$([[ "${EVALUATION_SPLIT}" == "routing_calibration" ]] && printf '%s' "${RUN_DIR}/.routing-active" || true)" \
ECODEP_FFN_CUDAGRAPH=0 \
ECODEP_ENABLE_STAGE_TRACE="$([[ "${MATH_PARAMETER_PROBE}" == "true" ]] && printf 1 || printf 0)" \
ECODEP_ENABLE_EXPERT_EPOCH_TRACE="${ORACLE_PROFILE_MODE}" \
ECODEP_ENABLE_EXPERT_BOUNDARY_ACTION=0 \
ECODEP_TRACE_SYNC_CUDA="${ORACLE_PROFILE_MODE}" \
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
if [[ "${JOINT_LAYOUT}" == "independent_afd_replicas_v1" ]]; then
  python3 "${ECODEP_ROOT}/migration/verify_joint_launch.py" --layout "${joint_layout_path}" --run "${RUN_DIR}"
elif [[ "${ENABLE_DBO}" == "1" ]]; then
  rg -q 'enable_dbo.*True' "${RUN_DIR}/attention.log" || exit 1
fi
if [[ "${PLACEMENT_ENABLED}" == "true" ]]; then
  rg -q "EcoDEP static expert placement installed: mode=layerwise.*sha256=${permutation_actual_sha256}" \
    "${RUN_DIR}/expert.log" || {
      echo "${ARM} FFN ranks did not confirm the frozen static placement" >&2
      exit 1
    }
fi

if [[ "${EVALUATION_SPLIT}" == "routing_calibration" ]]; then
  mapfile -t routing_sidecars < <(
    find "${RUN_DIR}" -maxdepth 1 -type f -name 'routing-ffn-*.jsonl' -print
  )
  [[ "${#routing_sidecars[@]}" -eq 2 ]] || {
    echo "routing startup did not materialize exactly two FFN sidecars" >&2
    exit 1
  }
  [[ -z "$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'routing-attention-*.jsonl' -print -quit)" ]] || {
    echo "routing startup emitted a forbidden Attention sidecar" >&2
    exit 1
  }
  truncate -s 0 "${routing_sidecars[@]}"
  if [[ "${ORACLE_PROFILE_MODE}" == "1" ]]; then
    mapfile -t expert_epochs < <(
      find "${RUN_DIR}" -maxdepth 1 -type f -name 'expert-epoch-*.jsonl' -print
    )
    [[ "${#expert_epochs[@]}" -eq 2 ]] || {
      echo "Oracle profile startup did not materialize two Expert epoch traces" >&2
      exit 1
    }
    truncate -s 0 "${expert_epochs[@]}"
  fi
fi

python3 "${ECODEP_ROOT}/scripts/afd/set_clocks.py" \
  --attention-url "${ATTENTION_CLOCK_URL}" --expert-url "${EXPERT_CLOCK_URL}" \
  --gpu-url-map "${GPU_CLOCK_URL_MAP}" \
  --attention-gpus "${ATTENTION_GPUS}" --expert-gpus "${EXPERT_GPUS}" \
  --attention "${MAX_ATTENTION_CLOCKS}" --expert "${MAX_EXPERT_CLOCKS}" \
  --attention-power-w "${MAX_ATTENTION_POWER_W}" --expert-power-w "${MAX_EXPERT_POWER_W}" \
  --output "${SUITE_DIR}/warmup-operating-point-ack.json" >/dev/null
sleep "${SETTLE_S}"
if [[ "${EVALUATION_SPLIT}" == "routing_calibration" ]]; then
  jq -n '{schema_version:5,requests:0,completed_requests:0,failed_requests:0,
    measured:false,reason:"routing profile starts with empty sidecars"}' \
    > "${SUITE_DIR}/warmup-summary.json"
else
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
fi
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
  "${ROUTING_SMOKE_LIMIT}" "${CALIBRATION_REQUEST_LIMIT}" "${GPUS}" "${ATTENTION_GPUS}" "${EXPERT_GPUS}" <<'PY'
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
    "topology": deployment["topology"],
    "server": {
        "max_model_len": int(sys.argv[8]),
        "max_num_seqs": int(sys.argv[9]),
        "max_num_batched_tokens": int(sys.argv[10]),
        "prefix_caching": True,
        "cuda_graph_full_decode_only": False,
        "ffn_cudagraph": False,
        "enforce_eager": True,
        "cudagraph_capture_size": int(sys.argv[11]),
        "dbo": deployment["topology"].get("microbatches", 2) == 2,
        "microbatches": deployment["topology"].get("microbatches", 2),
        "dbo_decode_token_threshold": int(sys.argv[12]),
        "dbo_prefill_token_threshold": int(sys.argv[13]),
        "compute_gate_on_attention": deployment["runtime_contract"][
            "compute_gate_on_attention"
        ],
    },
    "generation": {
        "api": "v1/completions",
        "temperature": 0.0,
        "ignore_eos": True,
        "max_output_tokens": int(sys.argv[14]),
        **({'generation_config': 'vllm', 'seed': 0, 'record_output_token_ids': True}
           if deployment['topology'].get('layout_contract') == 'independent_afd_replicas_v1' else {}),
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
output.write_text(json.dumps({
    "schema_version": 2,
    "backend": (
        "ecodep-vllm-afd-v026-calibration"
        if sys.argv[16] == "calibration"
        else (
            "ecodep-vllm-afd-v026-routing-calibration"
            if sys.argv[16] == "routing_calibration"
            else f"ecodep-vllm-afd-v026-{sys.argv[17].lower()}"
        )
    ),
    "arm": sys.argv[17],
    "math_method": deployment.get("math_method"),
    "math_dynamic_controller": deployment.get("math_dynamic_controller"),
    "model": deployment["model"],
    "evaluation_split": sys.argv[16],
    "measurement_eligible": sys.argv[16] != "routing_calibration",
    "formal_evaluation_eligible": sys.argv[16] == "heldout",
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
    "plugin": deployment["plugin"],
    "placement": {k: v for k, v in deployment["placement"].items()
                  if k != "permutation_by_layer"},
    "runtime_contract": deployment["runtime_contract"],
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
        "attention_gpus": [int(value) for value in sys.argv[27].split(",")],
        "expert_gpus": [int(value) for value in sys.argv[28].split(",")],
        "attention_dp": deployment["topology"]["attention_dp"],
        "ffn_ep": deployment["topology"]["ffn_ep"],
    },
    "native_memory_limit_bytes": None,
}, indent=2) + "\n")
PY
if [[ "${EVALUATION_SPLIT}" == "heldout" && "${ARM}" == "ours" ]]; then
  jq -e --arg expected "$(jq -r '.operating_points.comparison_contract_sha256' \
      "${DEPLOYMENT}")" \
    '.comparison_contract_sha256 == $expected' "${SUITE_DIR}/manifest.json" >/dev/null || {
    echo "held-out comparison contract differs from calibration" >&2
    exit 2
  }
fi

replay_limit_args=()
replay_routing_window_args=()
if [[ -n "${ROUTING_SMOKE_LIMIT}" ]]; then
  replay_limit_args=(--limit "${ROUTING_SMOKE_LIMIT}")
elif [[ -n "${CALIBRATION_REQUEST_LIMIT}" ]]; then
  replay_limit_args=(--limit "${CALIBRATION_REQUEST_LIMIT}")
fi
if [[ "${EVALUATION_SPLIT}" == "routing_calibration" ]]; then
  replay_routing_window_args=(--routing-active-marker ${RUN_DIR}/.routing-active)
fi
for rate in "${rates[@]}"; do
  normalized_rate="$(python3 -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${rate}")"
  point="$(jq -c --arg rate "${normalized_rate}" \
    '.operating_points.by_rps[$rate]' "${DEPLOYMENT}")"
  attention_mhz="$(jq -r '.attention_mhz | join(",")' <<<"${point}")"
  expert_mhz="$(jq -r '.expert_mhz | join(",")' <<<"${point}")"
  attention_power="$(jq -r '.attention_power_w | join(",")' <<<"${point}")"
  expert_power="$(jq -r '.expert_power_w | join(",")' <<<"${point}")"
  case_dir="${RUN_DIR}/rps-${normalized_rate}"
  request_log="${RUN_DIR}/replay-ecodep-v026-rps-${normalized_rate}.jsonl"
  mkdir -p "${case_dir}"
  python3 "${ECODEP_ROOT}/scripts/afd/set_clocks.py" \
    --attention-url "${ATTENTION_CLOCK_URL}" --expert-url "${EXPERT_CLOCK_URL}" \
    --gpu-url-map "${GPU_CLOCK_URL_MAP}" \
    --attention-gpus "${ATTENTION_GPUS}" --expert-gpus "${EXPERT_GPUS}" \
    --attention "${attention_mhz}" --expert "${expert_mhz}" \
    --attention-power-w "${attention_power}" --expert-power-w "${expert_power}" \
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
  replay_command=(python3 "${ECODEP_ROOT}/scripts/afd/replay_trace.py")
  if [[ -n "${MATH_CONTROLLER}" ]]; then
    replay_command=(python3 "${ECODEP_ROOT}/scripts/afd/math_dynamic/run_replay.py"
      --config "${MATH_CONTROLLER}" --case-dir "${case_dir}" --clock-url "${CLOCK_URL}" --)
  fi
  measure_script="${ECODEP_ROOT}/scripts/afd/measure_command.py"
  measure_args=()
  if [[ "${BO_FEEDBACK}" == "1" ]]; then
    python3 "${ECODEP_ROOT}/scripts/afd/prepare_measurement_traces.py" "${RUN_DIR}" \
      --sidecar-mode sidecar-off --expert-epoch-mode epoch-off \
      --attention-ranks "${ATTENTION_RANKS}" --output "${case_dir}/measurement-trace-reset.json"
    measure_script="${ECODEP_ROOT}/bo_dse/scripts/afd/measure_command.py"
    measure_args=(--record-operating-state)
    if [[ -n "${joint_layout_path:-}" ]]; then
      replay_command+=(--record-output-token-ids)
    fi
  fi
  python3 "${measure_script}" "${measure_args[@]}" \
    --gpus "${GPUS}" --output "${case_dir}/telemetry.json" \
    --samples-output "${case_dir}/power-samples.jsonl" \
    --request-log "${request_log}" -- \
    "${runtime_exec[@]}" "${replay_command[@]}" \
      "${TRACE}" \
      --endpoint "http://127.0.0.1:${ECODEP_API_PORT:-18000}/v1/completions" \
      --model "${MODEL_NAME}" \
      --output "${RUN_DIR}/$(basename "${request_log}")" \
      --ttft-slo-ms 400 --tpot-slo-ms 120 \
      --time-scale "${time_scale}" \
      "${replay_limit_args[@]}" \
      "${replay_routing_window_args[@]}" \
      --max-output-tokens "${MAX_OUTPUT_TOKENS}"
  if [[ "${BO_FEEDBACK}" == "1" ]]; then
    python3 "${ECODEP_ROOT}/bo_dse/native_backend.py" snapshot --run "${RUN_DIR}" --cell "${case_dir}"
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
      transition_energy_included:false}' \
    > "${case_dir}/case.json"
done

if ! cleanup; then
  echo "suite measurements finished, but mandatory cleanup failed" >&2
  exit 1
fi
if [[ "${EVALUATION_SPLIT}" == "routing_calibration" && \
      "${ORACLE_PROFILE_MODE}" == "0" ]]; then
  routing_output="${SUITE_DIR}/routing-provenance.json"
  routing_finalize_args=()
  if [[ -n "${ROUTING_SMOKE_LIMIT}" ]]; then
    routing_output="${SUITE_DIR}/routing-smoke-validation.json"
    routing_finalize_args=(--smoke-limit "${ROUTING_SMOKE_LIMIT}")
  fi
  python3 "${ECODEP_ROOT}/scripts/afd/finalize_v026_routing_provenance.py" \
    --run-dir "${RUN_DIR}" \
    --calibration-trace "${TRACE}" \
    --replay-summary "${case_dir}/summary.json" \
    --operating-point-ack "${case_dir}/operating-point-ack.json" \
    --plugin-root "${PLUGIN_ROOT}" \
    --image-id "${image_id}" \
    --collection-rate-rps "${normalized_rate}" \
    "${routing_finalize_args[@]}" \
    --output "${routing_output}"
fi
if [[ "${ORACLE_PROFILE_MODE}" == "1" ]]; then
  plan_path="$(jq -r '.oracle_profile.deployed_plan' "${DEPLOYMENT}")"
  [[ -f "${plan_path}" ]] || {
    echo "Oracle profile deployment lacks a frozen deployed plan" >&2; exit 2
  }
  python3 - "${case_dir}/run-config.json" "${DEPLOYMENT}" "${plan_path}" \
    "${TRACE}" "${RUN_DIR}" "${normalized_rate}" <<'PY'
import hashlib
import json
import pathlib
import sys

output, deployment_path, plan_path, trace_path, result_root = map(
    pathlib.Path, sys.argv[1:6]
)
plan = json.loads(plan_path.read_text())
payload = {
    "schema_version": 3,
    "method": "clairvoyant_expert_oracle_profile_calibration",
    "formal_evaluation_eligible": False,
    "oracle_scope": "same-trace calibration upper bound",
    "leakage_semantics": "intentional clairvoyance; never held-out evidence",
    "deployed_plan": str(plan_path.resolve()),
    "deployed_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "deployment": str(deployment_path.resolve()),
    "trace": str(trace_path.resolve()),
    "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    "split": "calibration_oracle",
    "offered_rps": float(sys.argv[6]),
    "result_root": str(result_root.resolve()),
    "data_plane": "official_vllm_afd_v0.26_role_pruned_attention_ffn",
    "routing_sidecar_role": "ffn",
    "routing_availability": "observed",
    "expert_frequency_mhz": plan["expert_frequencies_mhz"][0],
}
output.write_text(json.dumps(payload, indent=2) + "\n")
PY
  python3 - "${SUITE_DIR}/oracle-profile-provenance.json" "${case_dir}" \
    "${RUN_DIR}" "${DEPLOYMENT}" "${TRACE}" <<'PY'
import hashlib
import json
import pathlib
import sys

output, case_dir, run_dir, deployment, trace = map(pathlib.Path, sys.argv[1:])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sidecars = sorted(run_dir.glob("routing-ffn-*.jsonl"))
epochs = sorted(run_dir.glob("expert-epoch-*.jsonl"))
if len(sidecars) != 2 or len(epochs) != 2:
    raise SystemExit("Oracle profile run lacks two routing and two epoch traces")
payload = {
    "schema_version": 1,
    "method": "clairvoyant_expert_oracle_profile_calibration",
    "formal_evaluation_eligible": False,
    "selection_split": "calibration",
    "evaluation_split": "calibration_same_trace_oracle",
    "leakage_semantics": "intentional clairvoyance upper bound",
    "deployment": {"path": str(deployment.resolve()), "sha256": sha(deployment)},
    "trace": {"path": str(trace.resolve()), "sha256": sha(trace)},
    "summary": {
        "path": str((case_dir / "summary.json").resolve()),
        "sha256": sha(case_dir / "summary.json"),
    },
    "telemetry": {
        "path": str((case_dir / "telemetry.json").resolve()),
        "sha256": sha(case_dir / "telemetry.json"),
    },
    "routing_sidecars": [
        {"path": str(path.resolve()), "sha256": sha(path)} for path in sidecars
    ],
    "expert_epoch_traces": [
        {"path": str(path.resolve()), "sha256": sha(path)} for path in epochs
    ],
}
output.write_text(json.dumps(payload, indent=2) + "\n")
PY
fi
touch "${SUITE_DIR}/COMPLETE"

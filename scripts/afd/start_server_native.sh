#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
readonly MODEL_PATH="${ECODEP_MODEL_PATH:?}"
readonly MODEL_TAG="${ECODEP_MODEL_TAG:?}"
readonly RUN_ID="${ECODEP_RUN_ID:?}"
readonly AFD_PLUGIN_ROOT="${ECODEP_AFD_PLUGIN_ROOT:?}"
readonly RESULT_DIR="${ECODEP_ROOT}/results/afd_serving/${MODEL_TAG}/${RUN_ID}"
readonly PID_FILE="${RESULT_DIR}/native-service.pid"
readonly PLUGIN_SOURCE_SHA256="$(find "${AFD_PLUGIN_ROOT}/afd_plugin" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
mkdir -p "${RESULT_DIR}"
trap 'python3 "${ECODEP_ROOT}/migration/native_service.py" stop "${RESULT_DIR}"' ERR
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 VLLM_USE_V2_MODEL_RUNNER=0
unset CUDA_VISIBLE_DEVICES
export HF_HOME="${RESULT_DIR}/cache/huggingface"
export XDG_CACHE_HOME="${RESULT_DIR}/cache/xdg"
export PYTHONPATH=${AFD_PLUGIN_ROOT}
export VLLM_ENGINE_READY_TIMEOUT_S="${ECODEP_VLLM_ENGINE_READY_TIMEOUT_S:-1200}"
export VLLM_SERVER_DEV_MODE="${ECODEP_VLLM_SERVER_DEV_MODE:-1}"
export NCCL_DEBUG="${ECODEP_NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${ECODEP_NCCL_DEBUG_SUBSYS:-INIT,NET,P2P}"
export "NCCL_DEBUG_FILE=${RESULT_DIR}/nccl-%h-%p.log"
export NCCL_CUMEM_ENABLE="${ECODEP_NCCL_CUMEM_ENABLE:-0}"
export NCCL_MAX_P2P_NCHANNELS="${ECODEP_NCCL_MAX_P2P_NCHANNELS:-8}"
export NCCL_SET_STACK_SIZE="${ECODEP_NCCL_SET_STACK_SIZE:-0}"
export NCCL_RUNTIME_CONNECT="${ECODEP_NCCL_RUNTIME_CONNECT:-1}"
export ECODEP_ROUTING_SIDECAR="${ECODEP_ENABLE_ROUTING_SIDECAR:-0}"
export ECODEP_ROUTING_TRACE="$([[ "${ECODEP_ENABLE_ROUTING_SIDECAR:-0}" == "1" ]] && printf "${RESULT_DIR}/routing-{role}-{rank}-{pid}.jsonl" || true)"
export ECODEP_ROUTING_FFN_SIDECAR="${ECODEP_ROUTING_FFN_SIDECAR:-0}"
export ECODEP_ROUTING_SOURCE_ROLE="${ECODEP_ROUTING_SOURCE_ROLE:-attention}"
export ECODEP_ROUTING_ASYNC_COPY="${ECODEP_ROUTING_ASYNC_COPY:-1}"
export ECODEP_ROUTING_PREFILL_ONLY="${ECODEP_ROUTING_PREFILL_ONLY:-0}"
export ECODEP_ROUTING_LAYER_STRIDE="${ECODEP_ROUTING_LAYER_STRIDE:-1}"
export ECODEP_ROUTING_REQUEST_DETAIL="${ECODEP_ROUTING_REQUEST_DETAIL:-1}"
export ECODEP_ROUTING_ACTIVE_MARKER="${ECODEP_ROUTING_ACTIVE_MARKER:-}"
export ECODEP_STAGE_TRACE="$([[ "${ECODEP_ENABLE_STAGE_TRACE:-0}" == "1" ]] && printf "${RESULT_DIR}/stage-{role}-{rank}-{pid}.jsonl" || true)"
export ECODEP_EXPERT_EPOCH_TRACE="$([[ "${ECODEP_ENABLE_EXPERT_EPOCH_TRACE:-0}" == "1" ]] && printf "${RESULT_DIR}/expert-epoch-{rank}-{pid}.jsonl" || true)"
export ECODEP_FFN_CUDAGRAPH="${ECODEP_FFN_CUDAGRAPH:-0}"
export ECODEP_EXPERT_PERMUTATION="${ECODEP_EXPERT_PERMUTATION:-}"
export ECODEP_EXPERT_PERMUTATION_BY_LAYER="${ECODEP_EXPERT_PERMUTATION_BY_LAYER:-}"
export ECODEP_TRACE_CUDA_EVENTS="${ECODEP_TRACE_CUDA_EVENTS:-1}"
export ECODEP_TRACE_SYNC_CUDA="${ECODEP_TRACE_SYNC_CUDA:-0}"
export ECODEP_HYBRID_COMPACTION_TRACE="${ECODEP_HYBRID_COMPACTION_TRACE:-}"
native_command=(
  python3 "${ECODEP_ROOT}/scripts/afd/launch_pair.py"
  --model "${MODEL_PATH}"
  --attention-gpus "${ECODEP_ATTENTION_GPUS:-0,1}"
  --expert-gpus "${ECODEP_EXPERT_GPUS:-2,3}"
  --attention-ranks "${ECODEP_ATTENTION_RANKS:-2}"
  --expert-ranks "${ECODEP_EXPERT_RANKS:-2}"
  --attention-tp "${ECODEP_ATTENTION_TP:-1}"
  --expert-tp "${ECODEP_EXPERT_TP:-1}"
  --api-port "${ECODEP_API_PORT:-18000}"
  --expert-api-port "${ECODEP_EXPERT_API_PORT:-18001}"
  --dp-rpc-base-port "${ECODEP_DP_RPC_BASE_PORT:-29550}"
  --afd-port "${ECODEP_AFD_PORT:-16239}"
  --served-model-name "${ECODEP_SERVED_MODEL_NAME:-official-afd}"
  --gpu-memory-utilization "${ECODEP_GPU_MEMORY_UTILIZATION:-0.85}"
  --cpu-offload-gb "${ECODEP_CPU_OFFLOAD_GB:-0}"
  --max-model-len "${ECODEP_MAX_MODEL_LEN:-8192}"
  --max-num-seqs "${ECODEP_MAX_NUM_SEQS:-32}"
  --max-num-batched-tokens "${ECODEP_MAX_NUM_BATCHED_TOKENS:-3072}"
  --safetensors-load-strategy "${ECODEP_SAFETENSORS_LOAD_STRATEGY:-lazy}"
  --enable-prefix-caching "${ECODEP_ENABLE_PREFIX_CACHING:-1}"
  --cuda-graph-full-decode-only "${ECODEP_CUDA_GRAPH_FULL_DECODE_ONLY:-0}"
  --cudagraph-capture-size "${ECODEP_CUDAGRAPH_CAPTURE_SIZE:-32}"
  --enable-dbo "${ECODEP_ENABLE_DBO:-1}"
  --compute-gate-on-attention "${ECODEP_COMPUTE_GATE_ON_ATTENTION:-0}"
  --p2p-wire-dtype "${ECODEP_P2P_WIRE_DTYPE:-native_bf16}"
  --dbo-decode-token-threshold "${ECODEP_DBO_DECODE_TOKEN_THRESHOLD:-2}"
  --dbo-prefill-token-threshold "${ECODEP_DBO_PREFILL_TOKEN_THRESHOLD:-12}"
  --results "${RESULT_DIR}"
  --text-only
)
if [[ -n "${ECODEP_LAYOUT_JSON:-}" ]]; then
  native_command+=(--layout-json "${ECODEP_LAYOUT_JSON}")
fi
if [[ "${ECODEP_LANGUAGE_MODEL_ONLY:-1}" == "1" ]]; then
  native_command+=(--language-model-only)
fi

python3 "${ECODEP_ROOT}/migration/native_service.py" start "${RESULT_DIR}" -- "${native_command[@]}"
python3 - "${RESULT_DIR}/launch_config.json" "${PID_FILE}" "${ECODEP_ROOT}/environment/native-runtime.json" <<PY
import json, pathlib, sys
pid = int(pathlib.Path(sys.argv[2]).read_text())
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "runtime": "native",
    "launch_mode": "process_group",
    "runtime_version": "$(python3 --version)",

    "api_port": ${ECODEP_API_PORT:-18000},
    "expert_api_port": ${ECODEP_EXPERT_API_PORT:-18001},
    "afd_port": ${ECODEP_AFD_PORT:-16239},
    "dp_rpc_base_port": ${ECODEP_DP_RPC_BASE_PORT:-29550},
    "service_pid": pid,
    "environment_manifest": str(pathlib.Path(sys.argv[3]).resolve()),
    "model": "${MODEL_TAG}",
    "run_id": "${RUN_ID}",
    "attention_gpus": "${ECODEP_ATTENTION_GPUS:-0,1}",
    "expert_gpus": "${ECODEP_EXPERT_GPUS:-2,3}",
    "attention_ranks": ${ECODEP_ATTENTION_RANKS:-2},
    "expert_ranks": ${ECODEP_EXPERT_RANKS:-2},
    "attention_tp": ${ECODEP_ATTENTION_TP:-1},
    "expert_tp": ${ECODEP_EXPERT_TP:-1},
    "max_model_len": ${ECODEP_MAX_MODEL_LEN:-8192},
    "max_num_seqs": ${ECODEP_MAX_NUM_SEQS:-32},
    "max_num_batched_tokens": ${ECODEP_MAX_NUM_BATCHED_TOKENS:-3072},
    "prefix_caching_enabled": True,
    "cuda_graph_full_decode_only": False,
    "cudagraph_capture_size": ${ECODEP_CUDAGRAPH_CAPTURE_SIZE:-32},
    "dbo_enabled": bool(${ECODEP_ENABLE_DBO:-1}),
    "compute_gate_on_attention": $([[ "${ECODEP_COMPUTE_GATE_ON_ATTENTION:-0}" == "1" ]] && printf True || printf False),
    "hybrid_oracle_enabled": $([[ -n "${ECODEP_HYBRID_ORACLE_PLACEMENT:-}" ]] && printf True || printf False),
    "hybrid_oracle_placement": "${ECODEP_HYBRID_ORACLE_PLACEMENT:-}",
    "hybrid_cache_size": ${ECODEP_HYBRID_CACHE_SIZE:-0},
    "dbo_decode_token_threshold": ${ECODEP_DBO_DECODE_TOKEN_THRESHOLD:-2},
    "dbo_prefill_token_threshold": ${ECODEP_DBO_PREFILL_TOKEN_THRESHOLD:-12},
    "expert_permutation": "${ECODEP_EXPERT_PERMUTATION:-}",
    "expert_permutation_by_layer": "${ECODEP_EXPERT_PERMUTATION_BY_LAYER:-}",
    "expert_placement_mode": "$([[ -n "${ECODEP_EXPERT_PERMUTATION_BY_LAYER:-}" ]] && printf layerwise || printf global)",
    "routing_sidecar_enabled": ${ECODEP_ENABLE_ROUTING_SIDECAR:-0},
    "routing_async_copy_enabled": ${ECODEP_ROUTING_ASYNC_COPY:-1},
    "routing_prefill_only": ${ECODEP_ROUTING_PREFILL_ONLY:-0},
    "routing_layer_stride": ${ECODEP_ROUTING_LAYER_STRIDE:-1},
    "routing_request_detail": ${ECODEP_ROUTING_REQUEST_DETAIL:-1},
    "routing_ffn_sidecar_enabled": ${ECODEP_ROUTING_FFN_SIDECAR:-0},
    "routing_source_role": "${ECODEP_ROUTING_SOURCE_ROLE:-attention}",
    "routing_active_marker": "${ECODEP_ROUTING_ACTIVE_MARKER:-}",
    "ffn_cudagraph_enabled": ${ECODEP_FFN_CUDAGRAPH:-0},
    "trace_sync_cuda": ${ECODEP_TRACE_SYNC_CUDA:-0},
    "stage_trace_enabled": ${ECODEP_ENABLE_STAGE_TRACE:-0},
    "expert_epoch_trace_enabled": ${ECODEP_ENABLE_EXPERT_EPOCH_TRACE:-0},
    "expert_epoch_gate_enabled": 0,
    "expert_boundary_action_enabled": 0,
    "image": "native-vllm-0.26.0-cu130",
    "plugin_import_root": "${AFD_PLUGIN_ROOT}",
    "plugin_host_root": "${AFD_PLUGIN_ROOT}",
    "plugin_source_sha256": "${PLUGIN_SOURCE_SHA256}",
    "memory_limit_bytes": None,
    "memory_swap_limit_bytes": None,
}, indent=2) + "\n")
PY

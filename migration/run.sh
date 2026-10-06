#!/usr/bin/env bash
# Complete mathematical static DSE followed by bound dynamic calibration/heldout.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source migration/env.sh
hardware_args=(--legacy-topology-probes)
if [[ "${ECODEP_LEGACY_TOPOLOGY_PROBES:-0}" != 1 ]]; then
  export ECODEP_HARDWARE_PROFILE="${ECODEP_HARDWARE_PROFILE:-${PWD}/results/primitives/HARDWARE.json}"
  python3 migration/ensure_hardware.py --profile "$ECODEP_HARDWARE_PROFILE" \
    --gpus "${ECODEP_ATTENTION_GPUS},${ECODEP_EXPERT_GPUS}"
  hardware_args=(--hardware-profile "$ECODEP_HARDWARE_PROFILE")
fi
for model in ${ECODEP_MODELS:-deepseek-v2-lite qwen36}; do
  [[ "$model" == deepseek-v2-lite || "$model" == qwen36 ]] || { echo 'Invalid model selector' >&2; exit 2; }
  python3 migration/workflow.py --model "$model" \
    --campaign "results/math-full/${ECODEP_STATIC_RUN_ID:-math-full-v1}-${model}" \
    --gpus "${ECODEP_ATTENTION_GPUS},${ECODEP_EXPERT_GPUS}" --target "${ECODEP_STATIC_TARGET:-fixed}" "${hardware_args[@]}"
done

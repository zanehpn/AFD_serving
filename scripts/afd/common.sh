#!/usr/bin/env bash
set -euo pipefail

readonly ECODEP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ECODEP_AFD_IMAGE="${ECODEP_AFD_IMAGE:-ecodep-vllm-afd:v0.19.1}"
readonly ECODEP_CONTAINER_MEMORY="50g"
readonly ECODEP_CONTAINER_MEMORY_BYTES="53687091200"
readonly ECODEP_SHM_SIZE="8g"
readonly ECODEP_SHM_SIZE_BYTES="8589934592"

ecodep_docker_memory_args() {
  printf '%s\n' \
    "--memory=${ECODEP_CONTAINER_MEMORY}" \
    "--memory-swap=${ECODEP_CONTAINER_MEMORY}" \
    "--shm-size=${ECODEP_SHM_SIZE}"
}

ecodep_assert_container_limit() {
  local container_name="$1"
  local memory_limit
  local memory_swap_limit
  local shm_size
  memory_limit="$(docker inspect --format '{{.HostConfig.Memory}}' "${container_name}")"
  memory_swap_limit="$(docker inspect --format '{{.HostConfig.MemorySwap}}' "${container_name}")"
  shm_size="$(docker inspect --format '{{.HostConfig.ShmSize}}' "${container_name}")"
  if [[ "${memory_limit}" != "${ECODEP_CONTAINER_MEMORY_BYTES}" ]]; then
    echo "container ${container_name} memory limit is ${memory_limit}, expected ${ECODEP_CONTAINER_MEMORY_BYTES}" >&2
    return 1
  fi
  if [[ "${memory_swap_limit}" != "${ECODEP_CONTAINER_MEMORY_BYTES}" ]]; then
    echo "container ${container_name} memory-swap limit is ${memory_swap_limit}, expected ${ECODEP_CONTAINER_MEMORY_BYTES}" >&2
    return 1
  fi
  if [[ "${shm_size}" != "${ECODEP_SHM_SIZE_BYTES}" ]]; then
    echo "container ${container_name} shm size is ${shm_size}, expected ${ECODEP_SHM_SIZE_BYTES}" >&2
    return 1
  fi
}

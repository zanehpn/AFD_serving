#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source migration/env.sh
[[ "$(id -u)" == 0 ]] || { echo 'GPU clock control needs root.' >&2; exit 2; }
export PYTHIA_ALLOWED_GPUS="${ECODEP_ATTENTION_GPUS},${ECODEP_EXPERT_GPUS}"
export PYTHIA_NVCTL_PORT="${PYTHIA_NVCTL_PORT:-9096}"
exec .venv/bin/python services/nvcontrold.py

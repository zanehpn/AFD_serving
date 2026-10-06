#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" == 0 || "${1:-}" == '--backend' && "${2:-}" == 'official' ]]; then
  exec bash migration/setup_official.sh
fi
[[ "$#" == 2 && "$1" == '--backend' && "$2" == 'customized' ]] || {
  echo 'Usage: setup_native.sh [--backend official|customized]' >&2; exit 2;
}
[[ "$(id -u)" == 0 ]] || { echo 'Run setup as root on the destination server.' >&2; exit 2; }
[[ ! -f environment/PREPARED.json ]] || { echo 'Already prepared; do not mutate an active experiment environment.' >&2; exit 2; }
if command -v apt-get >/dev/null; then
  apt-get update
  apt-get install -y python3 python3-venv git jq ripgrep build-essential g++-12 libnuma1 libgomp1 curl
fi
command -v nvidia-smi >/dev/null || { echo 'Install the NVIDIA driver on the destination first.' >&2; exit 2; }
# uv and managed Python stay inside this checkout; no system Python replacement.
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install 'uv==0.11.32'
.bootstrap/bin/uv venv --python 3.12 --seed .venv
.bootstrap/bin/uv pip install --python .venv/bin/python --torch-backend=cu130 -r environment/native-requirements.lock.txt
.venv/bin/python migration/restore_plugins.py
# Build in a disposable copy so the source checkouts remain clean for commit checks.
build_dir="$(mktemp -d)"
trap 'rm -rf -- "$build_dir"' EXIT
cp -a third_party/afd-plugin-ecodep-v026-dvfs-v6-four-stage/. "$build_dir/"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_AFD_PLUGIN=0.26.0.dev0 \
  .bootstrap/bin/uv pip install --python .venv/bin/python --no-deps --no-build-isolation "$build_dir"
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > environment/native-pip-freeze.txt
bash migration/test_cpu.sh
printf '%s\n' 'Installed. Next: source .venv/bin/activate; source migration/env.sh; python migration/download_models.py'

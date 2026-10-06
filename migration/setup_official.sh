#!/usr/bin/env bash
# Separate, unmodified upstream runtime. Never load patches/afd-custom.bundle.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ ! -e .venv-official && ! -e third_party/afd-plugin-official ]] || {
  echo 'Official environment already exists; use a fresh checkout instead of overwriting it.' >&2; exit 2;
}
python3 -m venv .bootstrap-official
.bootstrap-official/bin/python -m pip install 'uv==0.11.32'
.bootstrap-official/bin/uv venv --python 3.12 --seed .venv-official
.bootstrap-official/bin/uv pip install --python .venv-official/bin/python --torch-backend=cu130 -r environment/native-requirements.lock.txt
.venv-official/bin/python migration/official_install.py
.venv-official/bin/python -m pip check
echo 'Official runtime installed. Download models with migration/download_models.py; see bo_dse/OFFICIAL_RUNTIME.md.'

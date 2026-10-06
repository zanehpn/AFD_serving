#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source migration/env.sh
mkdir -p results
# Prevent two native campaigns in this checkout; GPU monitors guard other jobs.
exec 9>results/native-campaign.lock
flock -n 9 || { echo 'A campaign is already active in this checkout.' >&2; exit 2; }
python3 migration/preflight.py
[[ -f environment/PREPARED.json ]] || python3 migration/prepare.py
python3 migration/reuse_max.py
# Source A100 MAX guards are already materialized by prepare.py.
# Each phase propagates failures. No detached continuation silently waits after a failed phase.
bash scripts/afd/run_dynamic_v10_calibration_controller.sh
python3 scripts/afd/continue_dynamic_v10_after_calibration.py
python3 - <<'PY'
import json
from pathlib import Path
p = Path('results/afd_suites/dynamic-fbss-v10-inputs-20260906/CONTINUATION_STATUS.json')
d = json.loads(p.read_text())
print(json.dumps(d, indent=2))
if d['status'] != 'complete':
    raise SystemExit('Calibration gate did not pass; heldout remains unused. Inspect calibration results.')
PY

"""Read-only inventory for a restored checkpoint; does not authorize or launch resume."""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    base = Path(__file__).resolve().parent
    external = json.loads((base / 'external-dependencies.json').read_text())
    commands = {name: shutil.which(name) for name in ('nvidia-smi', 'apptainer', 'dcgmi', 'nvcc', 'python3', 'git')}
    gpu = dict(available=bool(commands['nvidia-smi']))
    if commands['nvidia-smi']:
        result = subprocess.run([commands['nvidia-smi'], '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        gpu.update(returncode=result.returncode, inventory=result.stdout.strip(), error=result.stderr.strip())
    states, count, receipts = {}, 0, 0
    for owner in ('dbo_on_1256_v3', 'dbo_on_0347_v3'):
        for path in sorted((root / 'results' / owner / 'search').glob('*/comparison/*/state.json')):
            state = json.loads(path.read_text())
            total = len(state['observations'])
            n_receipts = len(list((path.parent / 'trials').glob('*/worker-result.json')))
            states[str(path.relative_to(root))] = dict(observations=total, search_receipts=n_receipts, frozen=state['frozen'], pending=bool(state['pending']))
            count += total
            receipts += n_receipts
    original_image = Path(external['container']['path'])
    image_path = root / 'artifacts/containers' / original_image.name
    weights = {}
    for model, metadata in external['models'].items():
        missing = []
        mismatched = []
        for item in metadata['files']:
            if item['archived']:
                continue
            path = root / 'artifacts/models' / model / item['name']
            if not path.exists():
                missing.append(item['name'])
            elif path.stat().st_size != item['size']:
                mismatched.append(item['name'])
        weights[model] = dict(missing_files=missing, size_mismatches=mismatched, weights_sha256_verified=False)
    print(json.dumps(dict(uid=os.geteuid(), python=platform.python_version(), workspace=str(root), commands=commands, gpu=gpu,
        checkpoint=dict(method_states=len(states), observations=count, search_receipts=receipts, states=states),
        container=dict(expected_sha256=external['container']['sha256'], expected_size=external['container']['size_bytes'], restored_path=str(image_path), exists=image_path.exists(), sha256_verified=False),
        models=weights, gpu_controls_applied=False, launched=False, direct_cross_host_resume_supported=False,
        next_action='Read ROOT_AGENT_HANDOFF.md; provision missing runtime/weights and create an explicitly new host campaign. Never run cleanup against archived PIDs.'), indent=2))


if __name__ == '__main__':
    main()

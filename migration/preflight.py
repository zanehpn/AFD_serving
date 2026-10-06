#!/usr/bin/env python3
"""Validate native runtime, plugin ABI, models and read-only clock-service access."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import urllib.request
from bo_topology import topologies

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    if os.geteuid() != 0:
        raise SystemExit('Run from the destination root environment.')
    selected = os.environ.get('ECODEP_MODELS', 'deepseek-v2-lite qwen36').split()
    if not selected or len(selected) != len(set(selected)) or set(selected) - {'deepseek-v2-lite', 'qwen36'}:
        raise RuntimeError('Invalid ECODEP_MODELS selection')
    gpu_groups = [os.environ.get('ECODEP_ATTENTION_GPUS', '0,1'), os.environ.get('ECODEP_EXPERT_GPUS', '2,3')]
    bo_size = os.environ.get('ECODEP_BO_ALLOCATION_SIZE')
    if bo_size:
        ids = [int(x) for group in gpu_groups for x in group.split(',')]
        first = topologies(ids)[0]
        if (int(bo_size) != len(ids) or
                gpu_groups != [','.join(map(str, first[r + '_gpus'])) for r in ('attention', 'expert')]):
            raise SystemExit('BO preflight mapping must match the allocation reference topology')
    elif any(len(x.split(',')) != 2 for x in gpu_groups):
        raise SystemExit('The legacy topology requires exactly two GPUs per role; use native.py for BO.')
    gpus = ','.join(gpu_groups)
    gpu_check = [sys.executable, 'migration/check_gpus.py', '--gpus', gpus]
    if bo_size:
        gpu_check += ['--bo-allocation', bo_size]
    subprocess.run(gpu_check, check=True)
    expected = {'vllm': '0.26.0', 'torch': '2.11.0', 'transformers': '5.14.1'}
    for package, version in expected.items():
        if md.version(package).split('+')[0] != version:
            raise RuntimeError(f'{package} must be {version}')
    if not shutil.which('nvcc'):
        raise RuntimeError('CUDA Toolkit nvcc is missing. Install CUDA 13.0 Toolkit and add its bin directory to PATH.')
    import torch
    if torch.version.cuda != '13.0' or not torch.cuda.is_available():
        raise RuntimeError('Native CUDA 13.0 initialization failed; check the driver and installed cu130 wheels.')
    for name, commit in json.loads((ROOT / 'environment/plugins.lock.json').read_text())['targets'].items():
        plugin = ROOT / 'third_party' / name
        actual = subprocess.check_output(['git', '-C', str(plugin), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit or subprocess.check_output(['git', '-C', str(plugin), 'status', '--short'], text=True).strip():
            raise RuntimeError('Plugin commit/worktree differs')
        env = dict(os.environ, PYTHONPATH=str(plugin), PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable, '-c', 'import afd_plugin; afd_plugin.register_afd(); import vllm; print(afd_plugin.__file__, vllm.__version__)'], env=env, check=True)
    for model in json.loads((ROOT / 'environment/models.lock.json').read_text()):
        path = ROOT / 'artifacts/models' / model['name']
        if hashlib.sha256((path / 'config.json').read_bytes()).hexdigest() != model['config_sha256']:
            raise RuntimeError(f'Model config differs: {path}')
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        for shard in set(index['weight_map'].values()):
            if not (path / shard).is_file():
                raise FileNotFoundError(path / shard)
    with urllib.request.urlopen(os.environ.get('ECODEP_CLOCK_URL', 'http://127.0.0.1:9096') + '/health', timeout=5) as response:
        health = json.load(response)
    if health['protocol'] != 'nvcontrold.applied_ack.v2' or not set(map(int, gpus.split(','))) <= set(health['allowed']):
        raise RuntimeError('Clock service does not cover the requested GPUs or ACK protocol')
    platform_path = ROOT / 'environment/native-platform.json'
    platform_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv,noheader,nounits'], text=True)
    marker = ROOT / 'environment/PREPARED.json'
    if marker.exists():
        if platform_path.exists() and json.loads(platform_path.read_text())['nvidia_smi'] != platform_info:
            raise RuntimeError('GPU identity or driver changed since preparation')
        prepared = json.loads(marker.read_text())
        if prepared.get('selected_models', ['deepseek-v2-lite', 'qwen36']) != selected:
            raise RuntimeError('Model selection changed after preparation')
        for key, expected in prepared.get('service_ports', {}).items():
            if int(os.environ.get(key, expected)) != expected:
                raise RuntimeError('Service ports changed after preparation')
        if prepared['gpu_groups'] != gpu_groups:
            raise RuntimeError('GPU mapping changed after preparation; use a fresh campaign clone.')
        subprocess.run([sys.executable, 'migration/runtime_fingerprint.py', '--check', 'environment/native-runtime.json'], check=True)
    if not marker.exists():
        platform_path.write_text(json.dumps({'nvidia_smi': platform_info}, indent=2) + '\n')
    print('PASS: native runtime, models, plugin registration, GPU mapping and clock service. No clocks changed.')


if __name__ == '__main__':
    main()

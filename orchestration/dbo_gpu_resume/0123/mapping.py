"""Explicit physical execution mapping; logical search identities stay immutable."""
import copy
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
GPU_MAP = {0: 0, 3: 1, 4: 2, 7: 3}
PHYSICAL_GPUS = [0, 1, 2, 3]

def load_manifest():
    path = BASE/'MIGRATION.json'
    manifest = json.loads(path.read_text())
    assert manifest['logical_to_physical'] == {str(k):v for k,v in GPU_MAP.items()}
    for filename, digest in manifest['files_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
    return manifest

def transform(config, candidate=None):
    manifest = load_manifest()
    result = copy.deepcopy(config)
    assert result['gpus'] == list(GPU_MAP), result['gpus']
    result['gpus'] = PHYSICAL_GPUS
    result['hardware'] = manifest['hardware']
    result['clock_url'] = 'http://127.0.0.1:19099'
    result['_physical_migration'] = str(BASE/'MIGRATION.json')
    if result.get('correctness'):
        result['correctness']['reference_gpu_uuids'] = [manifest['hardware']['devices'][str(g)]['uuid'] for g in PHYSICAL_GPUS[:2]]
    if candidate is None:
        return result
    mapped = copy.deepcopy(candidate)
    for role in ('attention', 'expert'):
        mapped[role+'_gpus'] = [GPU_MAP[g] for g in candidate[role+'_gpus']]
    return result, mapped

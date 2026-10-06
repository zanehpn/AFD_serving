"""Explicit physical execution mapping; logical search identities stay immutable."""
import copy
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
GPU_MAP = {0: 0, 3: 1, 4: 2, 7: 4}
PRIMARY_MAP = {1: 0, 2: 1, 5: 2, 6: 4}
PHYSICAL_GPUS = [0, 1, 2, 4]

def load_manifest():
    path = BASE/'MIGRATION.json'
    manifest = json.loads(path.read_text())
    assert manifest['logical_to_physical'] == {str(k):v for k,v in GPU_MAP.items()}
    assert manifest['primary_logical_to_physical'] == {str(k):v for k,v in PRIMARY_MAP.items()}
    for filename, digest in manifest['files_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
    return manifest

def transform(config, candidate=None):
    manifest = load_manifest()
    result = copy.deepcopy(config)
    physical_map = mapping_for(result['gpus'])
    primary = result['gpus'] == list(PRIMARY_MAP)
    result['gpus'] = PHYSICAL_GPUS
    result['hardware'] = manifest['hardware']
    result['clock_url'] = 'http://127.0.0.1:19101' if primary else 'http://127.0.0.1:19100'
    result['_physical_migration'] = str(BASE/'MIGRATION.json')
    if result.get('correctness'):
        result['correctness']['reference_gpu_uuids'] = [manifest['hardware']['devices'][str(g)]['uuid'] for g in PHYSICAL_GPUS[:2]]
    if candidate is None:
        return result
    return result, map_candidate(config['gpus'], candidate)

def mapping_for(gpus):
    if gpus == list(GPU_MAP):return GPU_MAP
    if gpus == list(PRIMARY_MAP):return PRIMARY_MAP
    raise ValueError(f'Unexpected logical GPU allocation: {gpus}')

def map_candidate(gpus, candidate):
    physical_map = mapping_for(gpus)
    mapped = copy.deepcopy(candidate)
    for role in ('attention','expert'):
        mapped[role+'_gpus'] = [physical_map[g] for g in candidate[role+'_gpus']]
    return mapped

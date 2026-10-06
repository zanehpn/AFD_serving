"""CPU-only checkpoint inventory and conservative AFD capacity estimates.

Only unavoidable, unquantized weight bytes can prove a structure impossible.
Cache, workspace and replication estimates guide ordering, not hard rejection.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

from .space import digest, structure

MIB = 1024 ** 2
SUPPORTED = {'deepseek_v2', 'qwen3_5_moe', 'qwen3_5_moe_text'}


def inventory(model_path):
    root = Path(model_path)
    raw = (root / 'config.json').read_bytes()
    outer = json.loads(raw)
    cfg = outer.get('text_config', outer)
    model_type = cfg.get('model_type', outer.get('model_type'))
    result = dict(schema_version=1, model_type=model_type, config_sha256=hashlib.sha256(raw).hexdigest(),
                  status='unknown', tensor_headers=[], weight_bytes={}, lower_bound_bytes={}, assumptions=[])
    if model_type not in SUPPORTED or outer.get('quantization_config') or cfg.get('quantization_config'):
        result['reason'] = 'Unsupported or quantized checkpoint: capacity remains unproven'
        return result
    files = sorted(root.glob('*.safetensors'))
    index = root / 'model.safetensors.index.json'
    expected = json.loads(index.read_text())['weight_map'] if index.exists() else None
    if expected:
        if any(Path(name).name != name or not name.endswith('.safetensors') for name in expected.values()):
            raise ValueError('Checkpoint index must reference local safetensors shards')
        files = [root / name for name in sorted(set(expected.values()))]
    if not files:
        result['reason'] = 'No safetensors headers available'
        return result
    totals = dict(attention=0, expert_routed=0, expert_other=0)
    lower = dict(attention=0, expert=0)
    seen = set()
    for path in files:
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('Truncated safetensors header')
            length = struct.unpack('<Q', prefix)[0]
            if not 2 <= length <= min(100 * MIB, path.stat().st_size - 8):
                raise ValueError('Invalid safetensors header length')
            header = stream.read(length)
        result['tensor_headers'].append(dict(file=path.name, sha256=hashlib.sha256(prefix + header).hexdigest()))
        for name, tensor in json.loads(header).items():
            if name == '__metadata__':
                continue
            if name in seen or (expected is not None and expected.get(name) != path.name):
                raise ValueError('Duplicate or incorrectly indexed tensor: ' + name)
            seen.add(name)
            offsets = tensor['data_offsets']
            if len(offsets) != 2 or any(type(n) is not int for n in offsets) or not 0 <= offsets[0] <= offsets[1] <= path.stat().st_size - 8 - length:
                raise ValueError('Invalid tensor data offsets: ' + name)
            shape = tensor['shape']
            if any(type(n) is not int or n < 0 for n in shape):
                raise ValueError('Invalid tensor shape: ' + name)
            width = {'BF16': 2, 'F16': 2, 'F32': 4}.get(tensor['dtype'])
            if width is None:
                result['reason'] = 'Unsupported tensor dtype: capacity remains unproven'
                return result
            if math.prod(shape) * width != offsets[1] - offsets[0]:
                raise ValueError('Tensor shape/dtype disagrees with data size: ' + name)
            # Official BF16/FP16 serving: retain 32-bit scalar/state tensors in
            # the estimate, while two bytes/element is a conservative bound.
            size = math.prod(shape) * width
            minimum = math.prod(shape) * 2
            parts = name.split('.')
            if 'visual' in parts or 'mtp' in parts:
                continue  # Official Qwen language-only / no speculative MTP.
            if '.mlp.' in name:
                routed = '.mlp.experts.' in name
                totals['expert_routed' if routed else 'expert_other'] += size
                if routed:
                    lower['expert'] += minimum
            else:
                totals['attention'] += size
                if '.self_attn.' in name or '.linear_attn.' in name:
                    lower['attention'] += minimum
    if expected is not None and seen != set(expected):
        raise ValueError('Checkpoint index is incomplete')
    if not totals['attention'] or not totals['expert_routed']:
        result['reason'] = 'Missing recognizable attention/expert tensors'
        return result
    layers = cfg['num_hidden_layers']
    if cfg.get('kv_lora_rank'):
        cache_per_token = layers * (cfg['kv_lora_rank'] + cfg.get('qk_rope_head_dim', 0)) * 2
        cache_kind = 'MLA latent cache (replicated across TP in estimate)'
    else:
        types = cfg.get('layer_types', ['full_attention'] * layers)
        full = sum(t == 'full_attention' for t in types)
        cache_per_token = full * 2 * cfg.get('num_key_value_heads', cfg['num_attention_heads']) * cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads']) * 2
        cache_kind = 'Full-attention KV; hybrid recurrent state covered by workspace reserve'
    result.update(status='estimated', weight_bytes=totals, lower_bound_bytes=lower,
                  kv_bytes_per_token=cache_per_token, cache_kind=cache_kind,
                  assumptions=['Pinned official AFD, compute_gate_on_attention=false, BF16/FP16',
                               'Attention DP replicates weights; TP estimate divides weights',
                               'Routed experts divide across EP world once, not DP*EP*TP',
                               'Unknown buffers/state and cache occupancy require physical preflight'])
    return result


def build_profile(model_path, candidates, hardware, *, max_model_len=8192,
                  max_num_seqs=32, workspace_mib=2048, memory_fraction=.85):
    if not 0 < memory_fraction <= 1 or workspace_mib < 0 or max_model_len <= 0 or max_num_seqs <= 0:
        raise ValueError('Invalid capacity sizing settings')
    model = inventory(model_path)
    profile = dict(schema_version=1, model=model, structures={}, memory_fraction=memory_fraction,
                   workspace_mib=workspace_mib, max_model_len=max_model_len, max_num_seqs=max_num_seqs)
    for candidate in candidates:
        key = digest(structure(candidate))
        if key in profile['structures']:
            continue
        t = candidate['topology']
        entry = dict(status='unknown', per_gpu={}, reason=model.get('reason'))
        if model['status'] == 'estimated':
            weights = model['weight_bytes']
            na, ne = len(t['attention_gpus']), len(t['expert_gpus'])
            cache = model['kv_bytes_per_token'] * max_model_len * math.ceil(max_num_seqs / t['attention_dp'])
            estimates = dict(attention=weights['attention'] / t['attention_tp'] + cache,
                expert=weights['expert_routed'] / t['expert_ep'] + weights['expert_other'] / t['expert_tp'])
            for role, count in [('attention', na), ('expert', ne)]:
                for gpu in t[role + '_gpus']:
                    device = hardware['devices'].get(str(gpu))
                    if device is None:
                        continue
                    entry['per_gpu'][str(gpu)] = dict(role=role, capacity_mib=device['memory_mib'],
                        usable_mib=device['memory_mib'] * memory_fraction,
                        weight_lower_bound_mib=model['lower_bound_bytes'][role] / count / MIB,
                        estimated_mib=estimates[role] / MIB + workspace_mib)
            rows = list(entry['per_gpu'].values())
            if len(rows) != na + ne:
                entry['status'] = 'unknown'
            elif any(x['weight_lower_bound_mib'] > x['usable_mib'] for x in rows):
                entry['status'] = 'proven_impossible'
            elif all(x['estimated_mib'] <= x['usable_mib'] for x in rows):
                entry['status'] = 'estimated_fit'
            else:
                entry['status'] = 'uncertain'
        profile['structures'][key] = entry
    return profile


def assessment(candidate, profile):
    return profile.get('structures', {}).get(digest(structure(candidate)), {'status': 'unknown'})

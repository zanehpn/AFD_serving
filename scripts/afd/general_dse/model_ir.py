"""Explicit per-layer operator ledger from model structure, without loading tensors.

Adapters describe architecture, not measured performance. Unknown architectures
must supply an audited ledger instead of silently becoming a standard Transformer.
"""
import hashlib
import json
from pathlib import Path


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def validate(model):
    if model.get('schema') != 'llm_operator_ledger_v1':
        raise ValueError('Expected llm_operator_ledger_v1')
    positive_int(model['hidden_size'], 'hidden_size')
    positive_int(model['dtype_bytes'], 'dtype_bytes')
    if not model['blocks']:
        raise ValueError('Empty operator ledger')
    import math
    for block in model['blocks']:
        a, f = block['attention'], block['ffn']
        for obj, keys in ((a, ('linear_flops', 'weight_bytes', 'context_flops', 'kv_bytes', 'state_bytes', 'activation_bytes')),
                          (f, ('expert_flops', 'expert_weight_bytes', 'shared_flops', 'shared_weight_bytes', 'router_flops', 'router_weight_bytes'))):
            if any(not math.isfinite(obj[k]) or obj[k] < 0 for k in keys):
                raise ValueError('Invalid operator arithmetic')
        e, k = positive_int(f['experts'], 'experts'), positive_int(f['top_k'], 'top_k')
        if k > e:
            raise ValueError('top_k exceeds experts')
        probs = f.get('routing_probabilities')
        if probs is not None and (len(probs) != e or any(not math.isfinite(x) or not 0 <= x <= 1 for x in probs)
                                  or not math.isclose(sum(probs), k)):
            raise ValueError('Routing marginals must sum to top_k')
    for key in ('embedding_weight_bytes', 'output_weight_bytes', 'output_flops_per_token'):
        if not math.isfinite(model[key]) or model[key] < 0:
            raise ValueError('Invalid embedding/output ledger')
    return model


def from_config(path, dtype_bytes=2):
    path = Path(path)
    raw = json.loads(path.read_text())
    c = raw.get('text_config', raw)
    kind = c['model_type']
    dense = {'llama', 'mistral', 'qwen2', 'qwen3'}
    moe = {'mixtral', 'qwen2_moe', 'qwen3_moe', 'qwen3_5_moe_text'}
    mla = {'deepseek_v2', 'deepseek_v3'}
    if kind not in dense | moe | mla:
        raise ValueError(f'No audited architecture adapter for {kind}; supply llm_operator_ledger_v1 JSON')
    if c.get('quantization_config') or raw.get('quantization_config'):
        raise ValueError('Quantized operator arithmetic needs an explicit ledger/kernel profile')
    h, layers, vocab = (positive_int(c[k], k) for k in ('hidden_size', 'num_hidden_layers', 'vocab_size'))
    heads = positive_int(c['num_attention_heads'], 'num_attention_heads')
    d = positive_int(c.get('head_dim', h // heads), 'head_dim')
    kvheads = positive_int(c.get('num_key_value_heads', heads), 'num_key_value_heads')
    experts = c.get('num_experts', c.get('n_routed_experts', c.get('num_local_experts', 1))) if kind not in dense else 1
    top_k = c.get('num_experts_per_tok', 1) if experts > 1 else 1
    blocks = []
    for i in range(layers):
        attn_kind = c.get('layer_types', ['full_attention'] * layers)[i]
        if kind in mla:
            qrank = c.get('q_lora_rank')
            qdim = heads * (c['qk_nope_head_dim'] + c['qk_rope_head_dim'])
            vdim = heads * c['v_head_dim']
            weights = ((h * qrank + qrank * qdim) if qrank else h * qdim)
            weights += h * (c['kv_lora_rank'] + c['qk_rope_head_dim'])
            weights += c['kv_lora_rank'] * heads * (c['qk_nope_head_dim'] + c['v_head_dim']) + vdim * h
            context = 2 * heads * (c['qk_nope_head_dim'] + c['qk_rope_head_dim'] + c['v_head_dim'])
            kv, state, recurrent = dtype_bytes * (c['kv_lora_rank'] + c['qk_rope_head_dim']), 0, 0
            attn_kind = 'mla'
        elif attn_kind == 'linear_attention' and kind == 'qwen3_5_moe_text':
            nk, nv, dk, dv = (c[k] for k in ('linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim', 'linear_value_head_dim'))
            channels = 2 * nk * dk + nv * dv
            weights = h * (channels + nv * dv + 2 * nv) + nv * dv * h
            weights += channels * c.get('linear_conv_kernel_dim', 4)
            recurrent = 12 * nv * dk * dv
            context, kv = 0, 0
            state = 4 * nv * dk * dv + dtype_bytes * channels * c.get('linear_conv_kernel_dim', 4)
        elif attn_kind == 'full_attention':
            weights = h * (heads + 2 * kvheads) * d + heads * d * h
            # Qwen3.5 full attention emits an additional output gate projection.
            if kind == 'qwen3_5_moe_text':
                weights += h * heads * d
            context, kv, state, recurrent = 4 * heads * d, 2 * dtype_bytes * kvheads * d, 0, 0
        else:
            raise ValueError(f'Unsupported layer operator {attn_kind}')
        attention = dict(kind=attn_kind, linear_flops=2 * weights + recurrent,
            weight_bytes=dtype_bytes * weights, context_flops=context, kv_bytes=kv,
            state_bytes=state, activation_bytes=4 * dtype_bytes * h,
            projection_shape=[h, h], heads=heads, kv_heads=kvheads)
        is_moe = experts > 1
        if kind in mla:
            is_moe = is_moe and i >= c.get('first_k_dense_replace', 0) and i % c.get('moe_layer_freq', 1) == 0
        if kind in {'qwen2_moe', 'qwen3_moe'}:
            is_moe = is_moe and i not in c.get('mlp_only_layers', []) and (i + 1) % c.get('decoder_sparse_step', 1) == 0
        e, k = (experts, top_k) if is_moe else (1, 1)
        inter = c.get('moe_intermediate_size', c.get('intermediate_size')) if is_moe else c.get('intermediate_size')
        inter = positive_int(inter, 'FFN intermediate size')
        shared = c.get('shared_expert_intermediate_size', 0) if is_moe else 0
        if is_moe and kind in mla:
            shared = c.get('n_shared_experts', 0) * inter
        ffn = dict(kind='moe' if is_moe else 'dense', experts=e, top_k=k,
            expert_flops=6 * h * inter, expert_weight_bytes=3 * h * inter * dtype_bytes,
            shared_flops=6 * h * shared, shared_weight_bytes=3 * h * shared * dtype_bytes,
            router_flops=2 * h * e if is_moe else 0, router_weight_bytes=h * e * dtype_bytes if is_moe else 0,
            projection_shape=[h, inter])
        blocks.append(dict(attention=attention, ffn=ffn))
    tied = c.get('tie_word_embeddings', raw.get('tie_word_embeddings', False))
    return validate(dict(schema='llm_operator_ledger_v1', name=raw.get('_name_or_path') or path.parent.name,
        architecture_type=kind, hidden_size=h, dtype_bytes=dtype_bytes, blocks=blocks,
        embedding_weight_bytes=h * vocab * dtype_bytes,
        output_weight_bytes=0 if tied else h * vocab * dtype_bytes,
        output_flops_per_token=2 * h * vocab,
        sources={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()},
        assumptions=['structural matrix arithmetic, not measured performance',
            'norm/activation/softmax/cache-management overhead represented only by declared traffic and kernel proxy',
            'full causal context; sliding-window reuse and specialized kernel fusion are not reproduced',
            'MLA uses compressed KV storage; linear attention uses a separate recurrent state ledger'] ))

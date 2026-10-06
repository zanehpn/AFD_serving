#!/usr/bin/env python3
"""Generate calibration token references in a fresh stock-vLLM process."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'migration'))
sys.path.insert(0, str(ROOT / 'scripts/afd'))
from output_correctness import trace_rows, validate_reference
from replay_trace import request_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    rows = trace_rows(contract)
    if os.environ.get('VLLM_PLUGINS') != '' or any(k.startswith('ECODEP_') for k in os.environ):
        raise ValueError('Stock reference requires plugins disabled and no AFD environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(contract['reference_gpu_uuids']):
        raise ValueError('Stock reference GPU UUID mapping differs from the frozen allocation')
    # Import only after checking the isolated environment, before any vLLM worker.
    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != '0.26.0':
        raise ValueError('Reference requires the locked vLLM 0.26.0 runtime')
    llm = LLM(model=str(Path(contract['model_config']['path']).parent),
              tensor_parallel_size=contract['reference_tp'], data_parallel_size=1,
              enable_expert_parallel=False, enable_dbo=False, ubatch_size=0,
              enforce_eager=True, additional_config={}, language_model_only=True,
              enable_prefix_caching=False, max_model_len=8192,
              max_num_seqs=32, max_num_batched_tokens=3072,
              gpu_memory_utilization=.85, trust_remote_code=True,
              generation_config='vllm', seed=0)
    try:
        config = llm.llm_engine.vllm_config
        pc = config.parallel_config
        if config.additional_config or pc.use_ubatching or any(n.startswith('afd_plugin') for n in sys.modules):
            raise ValueError('Reference runtime unexpectedly loaded AFD/microbatching')
        prompts = [request_prompt(row) for row in rows]
        prompts = [{'prompt_token_ids': p} if isinstance(p, list) else p for p in prompts]
        params = [SamplingParams(temperature=0, seed=0, ignore_eos=True,
                                 max_tokens=min(r['output_tokens'], contract['max_output_tokens'])) for r in rows]
        results = llm.generate(prompts, params, use_tqdm=False)
        if len(results) != len(rows):
            raise ValueError('Reference generation did not return the whole cohort')
        reference = {'contract': contract, 'runtime': {
            'backend': 'stock_vllm_colocated', 'plugins': [], 'vllm_version': vllm.__version__,
            'tensor_parallel_size': pc.tensor_parallel_size, 'microbatches': 1, 'afd_enabled': False},
            'outputs': [{**row, 'output_token_ids': list(result.outputs[0].token_ids)}
                        for row, result in zip(rows, results)]}
        validate_reference(reference, contract)
        with args.output.open('x') as stream:
            json.dump(reference, stream, indent=2, allow_nan=False)
            stream.write('\n')
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()

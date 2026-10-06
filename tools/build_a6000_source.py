"""Build a new-host source snapshot from archived V2 plus the GA addition.

Never modify the historical source or import its optimizer observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def replace(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError(f'Expected exactly one replacement in {path}: {before[:80]}')
    path.write_text(text.replace(before, after))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--eager-cuda-loading', action='store_true')
    parser.add_argument('--disable-torch-compile', action='store_true')
    parser.add_argument('--async-cuda-allocator', action='store_true')
    parser.add_argument('--runtime-patch-manifest', type=Path)
    parser.add_argument('--a6000-memory-pstates', action='store_true')
    args = parser.parse_args()
    source = args.destination.resolve()
    shutil.copytree(args.archive, source)
    # These additions do not replace the archived capacity search implementation.
    additions = ['scripts/afd/static_dse/genetic.py', 'scripts/afd/static_dse/comparison.py',
                 'scripts/afd/static_dse_cli.py', 'tests/test_genetic.py']
    for rel in additions:
        shutil.copy2(ROOT/'bo_dse'/rel, source/'bo_dse'/rel)
    campaign = source/'bo_dse/scripts/afd/static_dse/campaign.py'
    replace(campaign, 'from .optimizer import expected_cost',
            'from .optimizer import expected_cost\nfrom .genetic import DEFAULT_GA, validate_options as validate_ga_options')
    replace(campaign,
            '    if bo["method"] not in ("bo", "random"):\n        raise ValueError("method must be bo or random")',
            '    if bo["method"] not in ("bo", "random", "ga"):\n'
            '        raise ValueError("method must be bo, random or ga")\n'
            '    if bo["method"] == "ga":\n'
            '        bo = {**DEFAULT_GA, **bo}\n        validate_ga_options(bo)')
    optimizer = source/'bo_dse/scripts/afd/static_dse/optimizer.py'
    signature = 'def propose(candidates, observations, settings, eligible_ids, remaining_gpu_hours, *, _local=False, _current=None):\n'
    replace(optimizer, signature, signature +
            '    if settings["bo"]["method"] == "ga":\n'
            '        from .genetic import propose as genetic_propose\n'
            '        return genetic_propose(candidates, observations, settings, eligible_ids, remaining_gpu_hours)\n')
    official = source/'bo_dse/official.py'
    replace(official, 'from static_dse.comparison import create_comparison, comparison_report',
            'from static_dse.comparison import VARIANTS, create_comparison, comparison_report')
    replace(official, "config['comparison_methods'] = getattr(args, 'comparison_methods', ['v2', 'generic_bo', 'random'])",
            "config['comparison_methods'] = getattr(args, 'comparison_methods', None) or list(VARIANTS)\n"
            "    config['expected_gpu_model'] = getattr(args, 'expected_gpu_model', 'NVIDIA A100-SXM4-80GB')\n"
            "    config['cuda_architecture'] = 'sm_86' if config['expected_gpu_model'] == 'NVIDIA RTX A6000' else 'sm_80'")
    replace(official, "for method in ('v2', 'generic_bo', 'random') for seed in config['seeds']",
            "for method in (config.get('comparison_methods') or ['v2', 'generic_bo', 'random']) for seed in config['seeds']")
    replace(official, "choices=('v2', 'generic_bo', 'random')", "choices=('v2', 'generic_bo', 'random', 'ga')")
    # Preserve the original absolute TBT limit on the new host instead of
    # deriving a different limit from the new reference measurement.
    replace(official, "    if config['tbt_slo'] and config['slo_mode'] != 'relative_max':\n        raise ValueError('TBT SLO requires relative Max mode')",
            "    absolute_tbt = getattr(args, 'tbt_ms', None)\n"
            "    if absolute_tbt is not None:\n"
            "        if config['slo_mode'] != 'absolute' or not math.isfinite(absolute_tbt) or absolute_tbt <= 0:\n"
            "            raise ValueError('Explicit TBT limit requires absolute mode and a positive finite value')\n"
            "        config['tbt_slo'] = True\n        config['limits']['tbt_ms'] = absolute_tbt\n"
            "        config['tbt_protocol'] = 'p90 pooled within-request client SSE token-ID arrival intervals; coalesced tokens share arrival time'\n"
            "    if config['tbt_slo'] and config['slo_mode'] != 'relative_max' and absolute_tbt is None:\n"
            "        raise ValueError('Absolute TBT SLO requires --tbt-ms')")
    replace(official, "    p.add_argument('--frequencies',",
            "    p.add_argument('--expected-gpu-model', choices=['NVIDIA A100-SXM4-80GB', 'NVIDIA RTX A6000'], default='NVIDIA A100-SXM4-80GB')\n"
            "    p.add_argument('--tbt-ms', type=float, help='Predeclared absolute TBT p90 limit in milliseconds')\n"
            "    p.add_argument('--reference-dbo-thresholds', default='2:12', help='Fixed reference decode:prefill pair, selected before measurement')\n"
            "    p.add_argument('--prior-preparation-cost-directory', type=Path, help='Explicit preparation-only protocol revision; carry costs without observations')\n"
            "    p.add_argument('--frequencies',")
    replace(official, "    reference = next(c for c in groups.values()",
            "    reference_pair = tuple(map(int, getattr(args, 'reference_dbo_thresholds', '2:12').split(':')))\n"
            "    if len(reference_pair) != 2 or min(reference_pair) <= 0:\n"
            "        raise ValueError('Reference thresholds must be a positive decode:prefill pair')\n"
            "    reference = next(c for c in groups.values()")
    replace(official, "c.get('dbo_decode_token_threshold', 2) == 2\n"
            "                     and c.get('dbo_prefill_token_threshold', 12) == 12)",
            "c.get('dbo_decode_token_threshold', 2) == reference_pair[0]\n"
            "                     and c.get('dbo_prefill_token_threshold', 12) == reference_pair[1])")
    replace(official, "    if getattr(args, 'reuse_max_directory', None):",
            "    prior_cost_directory = getattr(args, 'prior_preparation_cost_directory', None)\n"
            "    if prior_cost_directory is not None:\n"
            "        if prior is not None or prior_run is not None or getattr(args, 'reuse_max_directory', None):\n"
            "            raise ValueError('Select only one preparation cost source')\n"
            "        from preparation_revision import carry\n"
            "        carry(config, prior_cost_directory, inputs)\n"
            "    if getattr(args, 'reuse_max_directory', None):")
    shutil.copy2(ROOT/'tools/preparation_revision.py', source/'bo_dse/preparation_revision.py')
    worker = source/'bo_dse/official_worker.py'
    text = worker.read_text()
    upstream = (ROOT/'bo_dse/official_worker.py').read_text()
    # The new native installation has its own wheel identity, not the old
    # container's direct_url metadata. Preserve archived warmup validation.
    start, end = text.index('def runtime_identity():'), text.index('def verify_runtime():')
    new_start, new_end = upstream.index('def runtime_identity():'), upstream.index('def verify_runtime():')
    worker.write_text(text[:start] + upstream[new_start:new_end] + text[end:])
    if args.a6000_memory_pstates:
        replace(official, "    config['memory_clock_mhz'] = max(memory_clocks)",
                "    config['memory_clock_mhz'] = max(memory_clocks)\n"
                "    config['memory_clock_policy'] = ('driver_managed_pstates' if config['expected_gpu_model'] == 'NVIDIA RTX A6000' else 'fixed')")
        replace(official, "connector='p2p_nccl', execution_modes=['eager'], memory_clock_mhz=config['memory_clock_mhz'],",
                "connector='p2p_nccl', execution_modes=['eager'], memory_clock_mhz=config['memory_clock_mhz'], memory_clock_policy=config['memory_clock_policy'],")
        replace(worker, 'from official_space import commands',
                'from official_space import commands\nfrom a6000_operating_policy import check_operating_point, clock_summary')
        replace(worker,
                "            if abs(state['power_limit_w']-c[role+'_power_w']) > 1 or state['memory_clock_mhz'] != config['memory_clock_mhz']:\n"
                "                raise ValueError('Measured cap/memory clock differs from frozen operating point')",
                "            check_operating_point(config, c, gpu, state)")
        replace(worker, 'role_mean_power_w=roles, role_peak_rank_p95_power_w=power_p95,',
                "memory_clock_policy=config.get('memory_clock_policy', 'fixed'), measured_clocks=clock_summary(window),\n"
                "                                          role_mean_power_w=roles, role_peak_rank_p95_power_w=power_p95,")
        shutil.copy2(ROOT/'tools/a6000_operating_policy.py', source/'bo_dse/a6000_operating_policy.py')
    if args.runtime_patch_manifest:
        replace(worker, "            if f.hash.mode != 'sha256' or expected != f.hash.value:",
                "            overrides = read(ROOT / 'environment/runtime-patches.json')['files']\n"
                "            override = overrides.get(str(path.resolve()))\n"
                "            allowed_patch = (override is not None and digest == override['after_sha256']\n"
                "                and f.hash.value == base64.urlsafe_b64encode(bytes.fromhex(override['before_sha256'])).decode().rstrip('='))\n"
                "            if f.hash.mode != 'sha256' or (expected != f.hash.value and not allowed_patch):")
        replace(worker, "                if sha(installed) != sha(path):",
                "                patch = read(ROOT / 'environment/runtime-patches.json')['files'].get(str(installed.resolve()))\n"
                "                allowed = patch is not None and sha(path) == patch['before_sha256'] and sha(installed) == patch['after_sha256']\n"
                "                if sha(installed) != sha(path) and not allowed:")
    replace(worker, 'def compiler_preflight(directory):', "def compiler_preflight(directory, architecture='sm_80'):")
    replace(worker, "'-arch=sm_80'", "'-arch=' + architecture")
    replace(worker, 'compiler = compiler_preflight(args.directory)',
            "compiler = compiler_preflight(args.directory, config.get('cuda_architecture', 'sm_80'))")
    replace(worker,
            "        if any('A100-SXM4-80GB' not in d['name'] for d in hardware['devices'].values()):\n"
            "            raise ValueError('Initial official protocol requires A100-SXM4-80GB')",
            "        expected = config.get('expected_gpu_model', 'NVIDIA A100-SXM4-80GB')\n"
            "        if any(d['name'] != expected for d in hardware['devices'].values()):\n"
            "            raise ValueError(f'GPU model differs from the declared new-host protocol: {expected}')")
    if args.eager_cuda_loading:
        replace(worker, "    env.update(VLLM_PLUGINS='afd',",
                "    env.update(CUDA_MODULE_LOADING='EAGER', CUDA_MODULE_DATA_LOADING='EAGER', VLLM_PLUGINS='afd',")
    if args.disable_torch_compile:
        replace(worker, "VLLM_USE_V2_MODEL_RUNNER='0', PYTHONDONTWRITEBYTECODE='1',",
                "VLLM_USE_V2_MODEL_RUNNER='0', TORCH_COMPILE_DISABLE='1', TORCHDYNAMO_DISABLE='1', PYTHONDONTWRITEBYTECODE='1',")
    if args.async_cuda_allocator:
        replace(worker, "PYTHONUNBUFFERED='1', PATH=",
                "PYTORCH_ALLOC_CONF='backend:cudaMallocAsync', VLLM_USE_FLASHINFER_SAMPLER='0', PYTHONUNBUFFERED='1', PATH=")
    # Historical three-arm tests explicitly retain their historical arm list.
    replace(source/'bo_dse/tests/test_official.py', 'dp_rpc_port=28000, comparison=True)',
            "dp_rpc_port=28000, comparison=True, comparison_methods=['v2', 'generic_bo', 'random'])")
    shutil.copy2(ROOT/'tools/tests/test_a6000_protocol.py', source/'bo_dse/tests/test_a6000_protocol.py')
    # Link the verified native plugin checkout; it remains unmodified.
    plugin = source/'third_party/afd-plugin-official'
    if plugin.exists():
        plugin.rename(source/'third_party/archived-plugin-source')
    plugin.symlink_to(ROOT/'third_party/afd-plugin-official', target_is_directory=True)
    for name in ('official-runtime.lock.json', 'OFFICIAL_INSTALLED.json', 'native-requirements.lock.txt'):
        shutil.copy2(ROOT/'environment'/name, source/'environment'/name)
    if args.runtime_patch_manifest:
        shutil.copy2(args.runtime_patch_manifest, source/'environment/runtime-patches.json')
        lock_path = source/'environment/official-runtime.lock.json'
        lock = json.loads(lock_path.read_text())
        lock.update(source_modifications_allowed=True, runtime_scope='A6000 startup repair; explicit hash-verified patches only')
        lock_path.write_text(json.dumps(lock, indent=2)+'\n')
    record = dict(algorithm_origin=str(args.archive.resolve()), ga_origin_commit='c3c12f92424a24bdcbd4fe05a614ba6197443373',
                  cuda_module_loading='EAGER' if args.eager_cuda_loading else 'inherited_runtime_default',
                  torch_compile_disabled=args.disable_torch_compile,
                  async_cuda_allocator=args.async_cuda_allocator,
                  historical_observations_imported=False,
                  changes=['GA integration', 'explicit GPU model and SM architecture', 'absolute TBT CLI', 'new native runtime fingerprint'],
                  files_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ('bo_dse', 'migration', 'scripts', 'services', 'environment')
                                for p in (source/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    (source/'BUILD_MANIFEST.json').write_text(json.dumps(record, indent=2)+'\n')
    print(source)


if __name__ == '__main__':
    main()

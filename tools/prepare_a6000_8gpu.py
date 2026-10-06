"""Prepare a fresh 8-GPU RPS8/16 plan from the archived A6000 protocol."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiment-records/2026-09-16-a6000-progress'
RUN = ROOT / 'results/a6000-8gpu-rps8-16-20260916'


def replace_option(command, option, values):
    start = command.index(option) + 1
    end = start
    while end < len(command) and not command[end].startswith('--'):
        end += 1
    command[start:end] = values


def main():
    original = json.loads((ARCHIVE / 'PLAN.json').read_text())
    plan = {key: copy.deepcopy(original[key]) for key in (
        'algorithm', 'methods', 'frequencies_mhz', 'power_caps_w',
        'search_attempts_per_arm', 'dbo_threshold_profiles',
        'reference_dbo_thresholds', 'slo_mode', 'frequency_semantics',
        'cuda_module_loading', 'cuda_module_data_loading',
        'torch_compile_disabled', 'cuda_allocator', 'flashinfer_sampler',
        'runtime_scope', 'memory_clock_policy')}
    plan.update(status='prepared', source=str(RUN/'source'),
        gpus=list(range(8)), rps=[8, 16], baseline_first=True,
        search_attempts_total=256, historical_observations_imported=False,
        heldout_evaluated=False, common_setup_evaluations={}, jobs=[],
        default_topology=dict(attention_gpus=[0, 1, 2, 3], expert_gpus=[4, 5, 6, 7],
                              attention_tp=1, expert_tp=1, microbatches=2),
        origin_commit='2bbd51140bd725b2787a5c1d85cf668667a811e2',
        runtime_patches=str(RUN/'source/environment/runtime-patches.json'),
        trace_policy='Archived RPS8 request identities for both rates; official.py rescales arrivals to requested RPS. Heldout is not replayed.',
        prior_measurements_policy='Historical four-card observations and costs remain in the archive; fresh baselines and optimizer states on this host.')
    for rate in (8, 16):
        for model in ('deepseek', 'qwen'):
            job = copy.deepcopy(next(j for j in original['jobs'] if j['id'] == f'{model}-rps8'))
            job['id'] = f'{model}-rps{rate}'
            command = job['command']
            command[1] = str(RUN/'source/bo_dse/official.py')
            for option, values in {
                '--directory': [str(RUN/'search'/job['id'])],
                '--gpus': list(map(str, range(8))), '--rps': [str(rate)],
                '--evaluations': ['17'],
                '--calibration': [str(RUN/'inputs'/f'{model}-rps8'/'calibration.jsonl')],
                '--heldout': [str(RUN/'inputs'/f'{model}-rps8'/'heldout.jsonl')],
            }.items():
                replace_option(command, option, values)
            for name in ('calibration.jsonl', 'heldout.jsonl'):
                path = RUN/'inputs'/f'{model}-rps8'/name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == job['trace_hashes'][name]
            job.pop('archived_settings', None)
            job['limits_note'] = 'Historical A100 limits are reported only; fresh 4A4E MAX defines local relative limits.'
            plan['jobs'].append(job)
            plan['common_setup_evaluations'][job['id']] = 1
    with (RUN/'PLAN.json').open('x') as f:
        json.dump(plan, f, indent=2)
        f.write('\n')
    print(RUN/'PLAN.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Prepare independent GA arms from existing settings without launching GPUs."""
import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', nargs='+', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    configs = [p.resolve(strict=True) for p in args.settings]
    if len({p.parent.name for p in configs}) != len(configs):
        raise ValueError('Each scenario must have a distinct parent directory name')
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]/'bo_dse'
    snapshot = root/'source/bo_dse'
    shutil.copytree(source/'scripts/afd', snapshot/'scripts/afd',
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for path in source.glob('*.py'):
        shutil.copy2(path, snapshot/path.name)
    # Import only the private snapshot; future repository edits cannot change
    # these new campaigns' code hashes or the older three-arm campaigns.
    sys.path.insert(0, str(snapshot/'scripts/afd'))
    from static_dse.campaign import file_hash, read_json, write_json
    from static_dse.comparison import create_comparison
    from static_dse.space import configuration, digest

    rows = []
    for config_path in configs:
        baseline_path = config_path.parent/'comparison/generic_bo-seed0/bundle.json'
        old = read_json(baseline_path)
        manifest = create_comparison(config_path, root/config_path.parent.name,
                                     seeds=args.seeds, methods=['ga'])
        for arm in manifest['campaigns']:
            path = Path(arm['directory'])
            new = read_json(path/'bundle.json')
            state = read_json(path/'state.json')
            if state['observations'] or state['pending']:
                raise ValueError('A new GA arm must start without search labels')
            for key in ('hardware', 'runtime', 'calibration_plan', 'isolation', 'model_workload'):
                if old[key] != new[key]:
                    raise ValueError(f'GA/baseline context differs: {key}')
            for key in ('limits', 'budget', 'setup_cost', 'reference_candidate_id'):
                if old['settings'][key] != new['settings'][key]:
                    raise ValueError(f'GA/baseline protocol differs: {key}')
            for key, value in old['sources'].items():
                if key != 'campaign_config' and new['sources'].get(key) != value:
                    raise ValueError(f'GA/baseline input differs: {key}')
            if {c['id']: configuration(c) for c in old['candidates']} != {
                    c['id']: configuration(c) for c in new['candidates']}:
                raise ValueError('GA candidate configurations differ from the baseline')
            if digest(old['audit']) != digest(new['audit']):
                raise ValueError('GA eligibility differs from the baseline')
            rows.append(dict(scenario=config_path.parent.name, **arm,
                             candidate_count=len(new['candidates']),
                             search_trials_available=new['settings']['budget']['evaluations']-
                                 new['settings']['setup_cost']['evaluations'],
                             baseline_bundle=dict(path=str(baseline_path), sha256=file_hash(baseline_path)),
                             ga_bundle_sha256=file_hash(path/'bundle.json'),
                             matching_space_inputs_limits_budget=True))
    report = dict(status='prepared', selection_split='calibration', gpu_execution_started=False,
                  heldout_evaluation_completed=False, campaigns=rows,
                  source_files_sha256={str(p.relative_to(root)): file_hash(p)
                      for p in sorted(snapshot.rglob('*.py'))})
    write_json(root/'PREPARED.json', report)
    print(json.dumps({k: v for k,v in report.items() if k != 'source_files_sha256'}, indent=2))


if __name__ == '__main__':
    main()

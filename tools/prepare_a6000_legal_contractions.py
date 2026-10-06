"""Freeze legal_contractions_v2 with the verified eight-GPU runtime adapter."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'results/a6000-8gpu-rps8-16-20260916-nccl'
POLICY_FILES = (
    'bo_dse/scripts/afd/static_dse/campaign.py',
    'bo_dse/scripts/afd/static_dse/capacity_search.py',
)
TEST_FILES = (
    'test_capacity_v2.py', 'test_capacity_dbo_pairs.py',
    'test_capacity_joint_knobs.py', 'test_threshold_contraction.py',
    'test_capacity_legal_contractions.py', 'test_threshold_search.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def replace_option(command, option, values):
    start = command.index(option) + 1
    end = start
    while end < len(command) and not command[end].startswith('--'):
        end += 1
    command[start:end] = values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    run = parser.parse_args().directory.resolve()
    run.mkdir()  # Never replace an existing experiment.
    shutil.copytree(PREVIOUS / 'source', run / 'source',
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    shutil.copytree(PREVIOUS / 'inputs', run / 'inputs')
    changes = {}
    for name in POLICY_FILES:
        target = run / 'source' / name
        before = sha(target)
        shutil.copy2(ROOT / name, target)
        changes[name] = {'before_sha256': before, 'after_sha256': sha(target),
                         'origin': str(ROOT / name)}
    # The frozen adapter stamps the old revision during preparation. Remove
    # that override so the latest DEFAULT_BO records legal_contractions_v2.
    adapter = run / 'source/bo_dse/official.py'
    before = sha(adapter)
    old = "            settings['bo']['capacity_policy_revision'] = (\n                'budgeted_joint_knobs_thresholds_v3' if config.get('dbo_threshold_profiles')\n                else 'budgeted_joint_knobs_v1')\n"
    content = adapter.read_text()
    assert content.count(old) == 1
    adapter.write_text(content.replace(old, ''))
    changes['bo_dse/official.py'] = {
        'before_sha256': before, 'after_sha256': sha(adapter),
        'reason': 'Use latest policy revision from DEFAULT_BO; retain verified runtime adapter',
    }
    for name in TEST_FILES:
        shutil.copy2(ROOT / 'bo_dse/tests' / name, run / 'source/bo_dse/tests' / name)
    shutil.copy2(ROOT / 'bo_dse/CAPACITY_V2.md', run / 'source/bo_dse/CAPACITY_V2.md')
    assert "\"capacity_policy_revision\": \"legal_contractions_v2\"" in (
        run / 'source' / POLICY_FILES[0]).read_text()
    plan = copy.deepcopy(json.loads((PREVIOUS / 'PLAN.json').read_text()))
    plan.update(algorithm='legal_contractions_v2_plus_ga',
                capacity_policy_revision='legal_contractions_v2',
                source=str(run / 'source'),
                origin_commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                runtime_patches=str(run / 'source/environment/runtime-patches.json'),
                common_setup_evaluations={j['id']: 1 for j in plan['jobs']},
                historical_observations_imported=False,
                prior_measurements_policy='Fresh baselines and optimizer states; previous results retained in their own directory.',
                policy_source_changes=changes,
                capacity_explore_every=3, capacity_structure_fraction=.375,
                created_at=datetime.now(timezone.utc).isoformat())
    plan.pop('prior_preparation', None)
    for job in plan['jobs']:
        command = [arg.replace(str(PREVIOUS), str(run)) for arg in job['command']]
        if '--prior-preparation-cost-directory' in command:
            at = command.index('--prior-preparation-cost-directory')
            del command[at:at + 2]
        replace_option(command, '--evaluations', ['17'])
        job['command'] = command
        for name, expected in job['trace_hashes'].items():
            model = job['id'].split('-rps')[0]
            assert sha(run / 'inputs' / f'{model}-rps8' / name) == expected
    assert not (run / 'search').exists()
    write(run / 'PLAN.json', plan)
    overlay = json.loads((PREVIOUS / 'RESIDENT_EXECUTION.json').read_text())
    overlay.update(authorized_at=datetime.now(timezone.utc).isoformat(),
                   user_request='确认换到 legal_contractions_v2 最新版开始跑',
                   plan_sha256=sha(run / 'PLAN.json'),
                   scope='Fresh legal_contractions_v2, all four methods, both models at RPS8 and RPS16',
                   driver_sha256=sha(Path(overlay['driver'])))
    write(run / 'RESIDENT_EXECUTION.json', overlay)
    write(run / 'SOURCE_MANIFEST.json', {
        'base_runtime_source': str(PREVIOUS / 'source'),
        'policy_commit': plan['origin_commit'], 'changes': changes,
        'files_sha256': {str(p.relative_to(run)): sha(p)
                         for p in sorted((run / 'source').rglob('*'))
                         if p.is_file() and '.git' not in p.parts},
    })
    (run / 'README.md').write_text(
        '# A6000 legal_contractions_v2 RPS8/16 comparison\n\n'
        'Fresh MAX baselines and optimizer states for DeepSeek and Qwen on eight A6000 GPUs. '
        'Four methods: v2, generic_bo, random, ga; seed 0; 16 search attempts per arm. '
        'The verified NCCL/AFD runtime adapter is retained, with the latest capacity policy '
        'and its legal_contractions_v2 defaults. See SOURCE_MANIFEST.json for exact changes.\n')
    print(run)


if __name__ == '__main__':
    main()

"""Freeze one fresh four-A6000 Qwen RPS8 run with the current V2 policy."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'results/a6000-4gpu-qwen-rps8-20260917-latest-v2'
LATEST = ROOT / 'results/a6000-8gpu-rps8-16-20260917-latest-v2'
REV = '523b7a4d555015478728648db3af9898fb04d1cb'
PREFIX = 'experiment-records/2026-09-16-a6000-search-complete/'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def archived(name):
    return subprocess.check_output(['git', 'show', REV + ':' + PREFIX + name], cwd=ROOT)

def option(command, name, values):
    start = command.index(name) + 1
    end = start
    while end < len(command) and not command[end].startswith('--'):
        end += 1
    command[start:end] = values

def main():
    RUN.mkdir()
    history = RUN / 'historical-four-gpu'
    history.mkdir()
    manifest = json.loads(archived('manifest.json'))
    write(history / 'archive-manifest.json', manifest)
    payload = archived('qwen-rps8.tar.gz')
    (history / 'qwen-rps8.tar.gz').write_bytes(payload)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        for member in archive:
            dest = history / member.name
            if member.isdir():
                continue
            if not member.isfile() or not dest.resolve().is_relative_to(history.resolve()):
                raise ValueError('Unexpected archive member: ' + member.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.extractfile(member).read())
    for name in ('PLAN.json', 'COMPARISON.json', 'COMPARISON.md', 'BASELINES.json', 'README.md'):
        (history / name).write_bytes(archived(name))
    old_scene = history / 'search/qwen-rps8'
    old_config = json.loads((old_scene / 'official-config.json').read_text())
    assert old_config['gpus'] == [0, 1, 2, 3] and not old_config['resident_search']
    shutil.copytree(LATEST / 'source', RUN / 'source',
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    inputs = RUN / 'inputs/qwen-rps8'
    inputs.mkdir(parents=True)
    for name in ('calibration.jsonl', 'heldout.jsonl'):
        shutil.copy2(LATEST / 'inputs/qwen-rps8' / name, inputs / name)
    old_plan = json.loads((history / 'PLAN.json').read_text())
    job = next(j for j in old_plan['jobs'] if j['id'] == 'qwen-rps8')
    for name, expected in job['trace_hashes'].items():
        assert sha(inputs / name) == expected
    cmd = job['command']
    cmd[1] = str(RUN / 'source/bo_dse/official.py')
    for name, values in {
        '--directory': [str(RUN / 'search/qwen-rps8')],
        '--calibration': [str(inputs / 'calibration.jsonl')],
        '--heldout': [str(inputs / 'heldout.jsonl')],
        '--comparison-methods': ['v2'],
        '--clock-url': ['http://127.0.0.1:19097'],
        '--ttft-ms': [str(old_config['limits']['ttft_ms'])],
        '--tpot-ms': [str(old_config['limits']['tpot_ms'])],
        '--min-output-tps': [str(old_config['limits']['min_output_tps'])],
    }.items():
        option(cmd, name, values)
    cmd.extend(['--reuse-max-directory', str(old_scene)])
    job.update(command=cmd, limits=old_config['limits'], expected_limits=old_config['limits'],
               baseline_source=str(old_scene), slo_policy='Frozen historical four-A6000 MAX-derived SLOs')
    job.pop('archived_settings', None)
    policy = {}
    for name in ('campaign.py', 'capacity_search.py'):
        rel = Path('bo_dse/scripts/afd/static_dse') / name
        assert sha(RUN / 'source' / rel) == sha(ROOT / rel)
        policy[str(rel)] = sha(RUN / 'source' / rel)
    plan = dict(status='frozen', created_at=datetime.now(timezone.utc).isoformat(),
        authorized_user_request='Qwen RPS8 四卡 A6000 用最新版 V2 重新跑一遍',
        source=str(RUN / 'source'), methods=['v2'], jobs=[job], gpus=[0, 1, 2, 3],
        algorithm='legal_contractions_v2', capacity_policy_revision='legal_contractions_v2',
        policy_sha256=policy, policy_origin_commit='1dc910f6ad5e3e0e35c9d5c631e481344f271a28',
        search_attempts_per_arm=16, search_attempts_total=16, seed=0,
        historical_observations_imported=False, common_setup_evaluations=1,
        historical_max_reused=True, historical_archive_commit=REV,
        resident_search=False, execution_policy='Cold service startup per trial, matching historical four-GPU batch',
        comparison_scope='Historical four-A6000 results, same requests and SLOs; different physical host and current verified runtime adapter',
        frequencies_mhz=old_plan['frequencies_mhz'], power_caps_w=old_plan['power_caps_w'],
        dbo_threshold_profiles=old_plan['dbo_threshold_profiles'],
        reference_dbo_thresholds=old_plan['reference_dbo_thresholds'],
        runtime_environment={'NCCL_CUMEM_ENABLE': '0', 'NCCL_DEBUG': 'WARN'},
        heldout_evaluated=False, confirmation_evaluated=False)
    write(RUN / 'PLAN.json', plan)
    write(RUN / 'SOURCE_MANIFEST.json', dict(origin=str(LATEST / 'source'),
        files_sha256={str(p.relative_to(RUN)): sha(p) for p in sorted((RUN / 'source').rglob('*'))
                      if p.is_file() and '.git' not in p.parts}))
    (RUN / 'README.md').write_text(
        '# Qwen RPS8: latest V2, four A6000 GPUs\n\n'
        'Fresh legal_contractions_v2 search, seed 0, 16 attempts including failures. '
        'Each measurement replays 200 original calibration requests after 8 warmups. '
        'GPU pool 0-3; original frequency, power, DBO threshold candidates and exact four-GPU SLOs. '
        'Cold service startup per trial matches the historical batch. Historical MAX is inherited once; '
        'no previous optimizer observations are imported. Raw historical results remain separate. '
        'Physical GPU host and runtime adapter differ from the historical batch; this is a calibration comparison. '
        'No heldout or additional confirmation is scheduled.\n')
    print(RUN)

if __name__ == '__main__':
    main()

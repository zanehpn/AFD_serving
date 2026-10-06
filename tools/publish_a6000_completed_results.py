"""Capture completed observations while a later trial continues running."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from publish_dbo_completed_scene import checked, sha, write


def feasible(observation, limits):
    metrics = observation.get('metrics', {})
    return observation.get('status') == 'ok' and all(
        isinstance(metrics.get('output_tps' if key == 'min_output_tps' else key), (int, float))
        and (metrics['output_tps'] >= limit if key == 'min_output_tps' else metrics[key] <= limit)
        for key, limit in limits.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--include-four-gpu-rerun', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(root / 'tools/restore_a6000_completed_results.py', dest / 'restore.py')
    base = root / 'experiment-records/2026-09-16-a6000-checkpoint-1705'
    base_manifest = json.loads(checked(base / 'manifest.json'))
    base_files = {row['member']: row for row in base_manifest['files']}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        code_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        base_checkpoint=str(base.relative_to(root)), base_manifest_sha256=sha(checked(base / 'manifest.json')),
        scope='Completed observations only; in-flight trial files and live shared service logs excluded',
        files=[], inherited_files=[], parts=[])
    captures, files, states, rows, arms = {}, set(), {}, [], []
    runs = ['a6000-8gpu-rps8-16-20260916-nccl', 'a6000-8gpu-rps8-16-20260917-latest-v2']
    if args.include_four_gpu_rerun:
        runs.append('a6000-4gpu-qwen-rps8-20260917-latest-v2')

    def include_tree(folder):
        for p in folder.rglob('*'):
            if p.is_file() and not p.is_symlink() and not any(
                    part in ('.git', '__pycache__', '.pytest_cache') for part in p.parts
                    ) and p.suffix not in ('.tmp', '.lock', '.pyc', '.o'):
                files.add(p)

    for run_name in runs:
        run = root / 'results' / run_name
        include_tree(run / 'source')
        include_tree(run / 'inputs')
        if run_name == 'a6000-4gpu-qwen-rps8-20260917-latest-v2':
            history = run / 'historical-four-gpu'
            include_tree(history / 'search')
            for p in history.glob('*.json'):
                files.add(p)
            for p in history.glob('*.md'):
                files.add(p)
        if (run / 'README.md').exists():
            files.add(run / 'README.md')
        for p in run.glob('*.json'):
            captures[p] = checked(p)
        for scene in sorted((run / 'search').iterdir()):
            config = json.loads(checked(scene / 'official-config.json'))
            settings = json.loads(checked(scene / 'campaign-settings.json'))
            reference = json.loads(checked(scene / 'inputs/slo-reference.json'))
            limits = settings['limits']
            assert limits == config['limits'] == reference['limits']
            include_tree(scene / 'inputs')
            include_tree(scene / 'validation')
            for p in scene.glob('*.json'):
                captures[p] = checked(p)
            comparison = scene / 'comparison'
            if (comparison / 'comparison.json').exists():
                files.add(comparison / 'comparison.json')
            for state_path in sorted(comparison.glob('*/state.json')):
                data = checked(state_path)
                state = json.loads(data)
                captures[state_path] = data
                states[state_path] = state
                method = state_path.parent.name
                for p in state_path.parent.iterdir():
                    if p.is_file() and p.suffix == '.json' and p != state_path:
                        # Deployment is only stable after freeze; bundle/settings are immutable.
                        if state.get('frozen') or p.name in ('bundle.json', 'settings.json'):
                            captures[p] = checked(p)
                group = []
                for observation in state['observations']:
                    trial = state_path.parent / 'trials' / observation['trial_id']
                    receipt = json.loads(checked(trial / 'worker-result.json'))
                    assert receipt['status'] == observation['status']
                    assert receipt.get('metrics') == observation.get('metrics')
                    assert observation['selection_split'] == 'calibration'
                    assert observation['trace_sha256'] == config['trace']['sha256']
                    include_tree(trial)
                    metrics = observation.get('metrics', {})
                    row = dict(run=run_name, scene=scene.name, method=method,
                        trial_id=observation['trial_id'], status=observation['status'],
                        slo_feasible=feasible(observation, limits),
                        measurement_host=observation.get('measurement_host',
                            'current_host' if run_name.endswith('latest-v2') else 'historical_host'),
                        **{key: metrics.get(key) for key in ('energy_j', 'ttft_ms', 'tpot_ms', 'tbt_ms', 'output_tps')},
                        failure_reason=observation.get('failure_reason', ''))
                    rows.append(row)
                    group.append(row)
                passing = [o for o in state['observations'] if feasible(o, limits)]
                best = min(passing, key=lambda o: o['metrics']['energy_j']) if passing else None
                best_config = json.loads(checked(state_path.parent / 'trials' / best['trial_id'] / 'configuration.json')) if best else None
                arms.append(dict(run=run_name, scene=scene.name, method=method,
                    attempts=len(group), successful=sum(o['status'] == 'ok' for o in group),
                    failures=sum(o['status'] != 'ok' for o in group), feasible=len(passing),
                    complete=len(group) == 16 and not state.get('pending'), frozen=state.get('frozen', False),
                    in_flight_excluded=state.get('pending', {}).get('trial_id') if state.get('pending') else None,
                    limits=limits, max_energy_j=settings['default_energy_j'],
                    best_metrics=best['metrics'] if best else None, best_configuration=best_config,
                    saving_vs_historical_max_pct=100 * (1 - best['metrics']['energy_j'] / settings['default_energy_j']) if best else None))
            scene_states = [snapshot for path, snapshot in states.items() if path.parent.parent == comparison]
            if scene_states and all(s.get('frozen') and not s.get('pending') for s in scene_states):
                include_tree(scene / 'resident-session')
    folders = ['resume-qwen-rps16', 'queue-latest-v2']
    if args.include_four_gpu_rerun:
        folders.append('rerun-qwen-rps8-4gpu')
    for folder in folders:
        for p in (root / folder).iterdir():
            if p.is_file() and (p.suffix in ('.py', '.json', '.md') or p.name.startswith('native-python')):
                captures[p] = checked(p)
    files.update(root / 'tools' / name for name in ('publish_a6000_completed_results.py',
        'restore_a6000_completed_results.py', 'publish_dbo_completed_scene.py'))
    # Capture metadata first, then only trial directories named by the captured observations.
    files.update(captures)
    archive_name = 'completed-results-delta.tar.gz'
    with tempfile.TemporaryFile() as compressed:
        with tarfile.open(fileobj=compressed, mode='w:gz', compresslevel=6) as tar:
            for i, p in enumerate(sorted(files)):
                data = captures[p] if p in captures else checked(p)
                member = str(p.relative_to(root))
                entry = dict(member=member, size=len(data), sha256=sha(data))
                if member in base_files and entry['sha256'] == base_files[member]['sha256']:
                    manifest['inherited_files'].append(entry)
                else:
                    info = tarfile.TarInfo(member)
                    info.size, info.mode = len(data), p.stat().st_mode & 0o777
                    tar.addfile(info, io.BytesIO(data))
                    manifest['files'].append(entry)
                if i % 500 == 0:
                    print(f'Captured {i}/{len(files)} files', flush=True)
        compressed.seek(0)
        expected = {row['member']: row for row in manifest['files']}
        verified = 0
        with tarfile.open(fileobj=compressed, mode='r:gz') as tar:
            for member in tar:
                data = tar.extractfile(member).read()
                assert sha(data) == expected[member.name]['sha256']
                verified += 1
        assert verified == len(expected)
        compressed.seek(0)
        digest = hashlib.sha256()
        while chunk := compressed.read(24 * 1024 * 1024):
            name = archive_name + f'.part{len(manifest["parts"]):03d}'
            (dest / name).write_bytes(chunk)
            manifest['parts'].append(dict(file=name, size=len(chunk), sha256=sha(chunk)))
            digest.update(chunk)
        manifest['archive_sha256'] = digest.hexdigest()
    # Later observations are allowed; captured completed history must remain unchanged.
    for path, snapshot in states.items():
        now = json.loads(checked(path))
        assert now['observations'][:len(snapshot['observations'])] == snapshot['observations'], path
    baseline = [a for a in arms if a['run'] == runs[0] and a['method'] != 'v2-seed0']
    latest = [a for a in arms if a['run'] == runs[1]]
    rerun = [a for a in arms if a['run'] == 'a6000-4gpu-qwen-rps8-20260917-latest-v2']
    assert sum(a['attempts'] for a in baseline) == 192 and all(a['complete'] for a in baseline)
    summary = dict(snapshot_utc=manifest['created_at'], baseline_attempts=192,
        latest_v2_attempts=sum(a['attempts'] for a in latest),
        four_gpu_latest_v2_attempts=sum(a['attempts'] for a in rerun),
        four_gpu_rerun_included=args.include_four_gpu_rerun,
        historical_v2_attempts=sum(a['attempts'] for a in arms if a['run'] == runs[0] and a['method'] == 'v2-seed0'),
        heldout_evaluated=False, confirmation_evaluated=False, arms=arms,
        limitations=['Calibration only; no same-host confirmation or heldout evaluation.',
                     'Historical baselines and new-host V2 measurements are identified separately.',
                     'Energy comparisons are relative to archived MAX, not evidence of stable causal method gains.',
                     'In-flight attempts are excluded; partial scenarios are explicitly labelled.',
                     'Only request completion was verified, not semantic or exact-token correctness.'])
    write(dest / 'summary.json', summary)
    write(dest / 'manifest.json', manifest)
    write(dest / 'validation.json', dict(status='passed', archived_files_verified=verified,
        unchanged_files_referenced_from_base=len(manifest['inherited_files']),
        completed_history_prefix_unchanged=True, observations_receipts_match=True,
        calibration_only=True, credential_pattern_scan='passed', measurements=len(rows)))
    with (dest / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# A6000 已完成结果快照', '', f'快照时间：{manifest["created_at"]}', '',
        f'八卡 BO / Random / GA 基线 192/192 全部完成。八卡最新 legal_contractions_v2 已完成 {summary["latest_v2_attempts"]}/64 次；旧 V2 的 18 次历史观测单独列出。进行中的试验不计入结果。', '',
        '| 批次 | 场景 | 方法 | 已完成尝试 | 成功 | 满足 SLO | 最优能耗 kJ | 相对历史 MAX 节能 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        energy = f'{arm["best_metrics"]["energy_j"]/1000:.3f}' if arm['best_metrics'] else '—'
        saving = f'{arm["saving_vs_historical_max_pct"]:.2f}%' if arm['best_metrics'] else '—'
        label = ('四卡新版 V2 重跑' if arm in rerun else '八卡最新 V2' if arm['run'] == runs[1] else '八卡原批次')
        lines.append(f'| {label} | {arm["scene"]} | {arm["method"]} | {arm["attempts"]}/16 | {arm["successful"]} | {arm["feasible"]} | {energy} | {saving} |')
    if rerun:
        lines += ['', f'四卡 Qwen RPS8 最新版 V2 重跑已完成 {summary["four_gpu_latest_v2_attempts"]}/16 次。该批使用独立四卡 SLO，不能与八卡结果混合排名。沿用原始 200 条请求、8 条预热、seed 0、频率/功率/DBO 候选及每次冷启动协议；物理主机与历史批次不同。', '',
            '历史四卡 Qwen RPS8：旧 V2 39.439 kJ、BO 43.121 kJ、Random 36.380 kJ、GA 49.062 kJ；完整 256 次历史搜索已在 `../2026-09-16-a6000-search-complete` 归档。此次快照另外保留该场景历史原始记录和 MAX 继承证据。']
    lines += ['', '比较使用各场景及卡池对应的固定 SLO。旧主机基线和本机新 V2 均保留真实来源；节能幅度是校准测量相对历史 MAX 的差异，尚无同机对照复测或 heldout 验证，不能据此宣称稳定胜出。', '',
        '`summary.json` 保存配置、指标、SLO 和完成状态；`measurements.csv` 保留每次已完成试验（包括失败）。', '',
        '原始配置、回执、逐请求记录、NVML 遥测、冻结源码和迁移元数据采用增量归档。未变化文件引用仓库内 `../2026-09-16-a6000-checkpoint-1705`，新增文件在本目录的分块归档中；两者的逐文件和分块 SHA-256 均有记录。模型权重、虚拟环境和正在写入的共享服务日志不打包。', '',
        '恢复到新目录：`python3 restore.py results-restored`。恢复只校验和写文件，不启动 GPU 实验。原始状态快照可能含 pending 字段，它只记录当时队列状态；相应未完成 trial 不在快照结果中。', '']
    (dest / 'README.md').write_text('\n'.join(lines))
    for p in dest.iterdir():
        if '.tar.gz.part' not in p.name:
            checked(p)
    print(json.dumps(dict(destination=str(dest), baseline_attempts=192, latest_v2_attempts=summary['latest_v2_attempts'],
        files=verified, inherited_files=len(manifest['inherited_files']),
        bytes=sum(p['size'] for p in manifest['parts'])), indent=2), flush=True)


if __name__ == '__main__':
    main()

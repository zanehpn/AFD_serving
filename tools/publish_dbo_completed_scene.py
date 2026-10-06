#!/usr/bin/env python3
"""Archive a completed calibration scene without modifying experimental inputs."""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked(path):
    if path.is_symlink():
        raise ValueError(f'Symlink refused: {path}')
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed while reading: {path}')
    if SECRET.search(data):
        raise ValueError(f'Possible credential in {path}; contents suppressed')
    return data


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    campaign = root / 'results/dbo_on_1256_v3'
    search = campaign / 'search/deepseek-rps4'
    confirmation = campaign / 'confirmation/deepseek-rps4'
    read = lambda path: json.loads(checked(path))
    plan = read(campaign / 'PLAN.json')
    config = read(search / 'official-config.json')
    limits = config['limits']
    reference = read(search / 'inputs/slo-reference.json')
    confirmations = read(confirmation / 'RESULTS.json')
    assert confirmations['split'] == 'calibration_confirmation'
    assert len(confirmations['rows']) == 4
    assert confirmations['configurations_and_thresholds_frozen_before_confirmation']
    assert confirmations['repeats_per_configuration'] == 1

    def passes(metrics):
        return bool(metrics) and all(k in metrics for k in ('ttft_ms', 'tpot_ms', 'output_tps')) and metrics['ttft_ms'] <= limits['ttft_ms'] and metrics['tpot_ms'] <= limits['tpot_ms'] and metrics['output_tps'] >= limits['min_output_tps']

    artifact_count = 0
    def audit_artifacts(receipt):
        nonlocal artifact_count
        for item in receipt.get('artifacts', []):
            assert sha(checked(Path(item['path']))) == item['sha256'], item['path']
            artifact_count += 1

    rows, methods = [], {}
    for method in ('v2', 'generic_bo', 'random'):
        folder = search / f'comparison/{method}-seed0'
        state = read(folder / 'state.json')
        observations = state['observations']
        assert len(observations) == 16 and state['frozen'] and state['pending'] is None
        assert len(list((folder / 'trials').glob('*/worker-result.json'))) == 16
        for obs in observations:
            receipt = read(folder / 'trials' / obs['trial_id'] / 'worker-result.json')
            assert receipt['status'] == obs['status']
            assert receipt.get('metrics') == obs.get('metrics')
            assert not receipt.get('cleanup_required', False)
            assert obs['selection_split'] == 'calibration'
            assert obs['trace_sha256'] == config['trace']['sha256']
            audit_artifacts(receipt)
            metrics = obs.get('metrics', {})
            rows.append(dict(phase='calibration_search', method=method, trial_id=obs['trial_id'], status=obs['status'], passes_original_slo=obs['status'] == 'ok' and passes(metrics), **{k: metrics.get(k) for k in ('energy_j','ttft_ms','tpot_ms','output_tps')}, failure_reason=receipt.get('failure_reason', '')))
        group = [row for row in rows if row['method'] == method]
        best = state['deployment']['best']
        assert best['feasible'] and passes(best['metrics'])
        assert best['metrics']['energy_j'] == min(row['energy_j'] for row in group if row['passes_original_slo'])
        methods[method] = dict(attempts=16, successful=sum(row['status']=='ok' for row in group), slo_feasible=sum(row['passes_original_slo'] for row in group), deployment=state['deployment'])
    for row in confirmations['rows']:
        receipt = read(confirmation / row['threshold_profile'] / row['method'] / 'worker-result.json')
        assert receipt['metrics'] == row['metrics'] and receipt['status'] == row['status']
        assert passes(row['metrics']) == row['passes_original_slo']
        audit_artifacts(receipt)
        rows.append(dict(phase='calibration_confirmation', method=row['method'], trial_id='confirmation-0', status=row['status'], passes_original_slo=row['passes_original_slo'], **row['metrics'], failure_reason=''))
    assert len(rows) == 52

    isolation = read(search / 'inputs/isolation.json')
    traces = {}
    for split in ('calibration', 'heldout'):
        data = checked(search / 'inputs' / f'{split}.jsonl')
        assert sha(data) == isolation[split]['sha256']
        traces[split] = [json.loads(line) for line in data.splitlines() if line.strip()]
        assert len(traces[split]) == isolation[split]['requests']
    overlap = {}
    for field in isolation['identity_fields']:
        sets = {split: {json.dumps(row[field], sort_keys=True) for row in records} for split, records in traces.items()}
        assert all(len(sets[split]) == len(traces[split]) for split in sets)
        overlap[field] = len(sets['calibration'] & sets['heldout'])
        assert overlap[field] == 0
    warmup = [json.loads(line) for line in checked(search / 'inputs/warmup.jsonl').splitlines() if line.strip()]
    for field in isolation['identity_fields']:
        assert {json.dumps(row[field], sort_keys=True) for row in warmup} <= {json.dumps(row[field], sort_keys=True) for row in traces['calibration']}

    files = set()
    for folder in (search, confirmation):
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix not in ('.tmp', '.lock', '.pyc', '.o'):
                files.add(path)
    for name in ('PLAN.json', 'EXPERIMENT.md', 'container.json', 'REVISION_CHECKS.json', 'FIRST_LAUNCH_CHECK.json', 'FIRST_COMPLETED_TRIAL.json'):
        files.add(campaign / name)
    for path, expected in plan['files_sha256'].items():
        path = Path(path)
        assert sha(checked(path)) == expected, path
        files.add(path)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='completed DeepSeek RPS4 calibration search and confirmation only', workspace_original=str(root), archives=[], files=[])
    archive_name = 'deepseek-rps4-dbo-on-completed.tar.gz'
    with tempfile.TemporaryFile() as compressed:
        with tarfile.open(fileobj=compressed, mode='w:gz', compresslevel=6) as tar:
            for path in sorted(files):
                data = checked(path)
                member = str(path.relative_to(root))
                info = tarfile.TarInfo(member)
                info.size = len(data)
                info.mode = path.stat().st_mode & 0o777
                tar.addfile(info, io.BytesIO(data))
                manifest['files'].append(dict(archive=archive_name, member=member, source=str(path), size=len(data), sha256=sha(data)))
        compressed.seek(0)
        parts, digest, total = [], hashlib.sha256(), 0
        while chunk := compressed.read(24 * 1024 * 1024):
            name = archive_name + f'.part{len(parts):03d}'
            (dest / name).write_bytes(chunk)
            digest.update(chunk)
            total += len(chunk)
            parts.append(dict(file=name, size=len(chunk), sha256=sha(chunk)))
        manifest['archives'].append(dict(name=archive_name, size=total, sha256=digest.hexdigest(), parts=parts))
    write(dest / 'manifest.json', manifest)
    repo = Path(__file__).resolve().parents[1]
    shutil.copyfile(repo / 'experiment-records/2026-09-11-143809-rps8-200-8gpu-complete/restore.py', dest / 'restore.py')
    limitations = [
        'Single seed, 16 attempts per method; one confirmation per frozen configuration on the same 200 calibration requests.',
        'No unseen heldout evaluation. Included legacy heldout input is provenance only; it is not the separate planned formal evaluation split.',
        'All four calibration confirmations, including MAX, fail the original fixed SLO. Search-best V2 is not a confirmed SLO winner.',
        'Historical RPS4 MAX used a modified plugin and prefix cache; this campaign uses the upstream plugin with prefix cache disabled. Historical MAX energy ratios cannot isolate optimizer effects.',
        'Request completion was checked; semantic answer correctness and exact token equality were not verified.',
        'Raw provenance retains original absolute paths. Model weights, container image, dependency caches, and unfinished scenes are not bundled.',
    ]
    summary = dict(scene='deepseek-rps4', completion='48 search attempts and 4 calibration confirmations complete', heldout_evaluated=False, methods=methods, confirmations=confirmations, frozen_slo_reference=reference, limitations=limitations)
    write(dest / 'summary.json', summary)
    with (dest / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    labels = {'v2':'V2', 'generic_bo':'BO', 'random':'Random', 'MAX':'MAX'}
    lines = ['# DeepSeek RPS 4：DBO 开启、阈值参与搜索（已完成）', '',
        '本快照包含 48 次搜索尝试（V2、BO、Random 各 16 次，seed 0）及 4 次冻结配置复测。每次测量使用相同的 200 条 calibration 请求。其他未完成场景不在本快照中。', '',
        '**V2 在搜索阶段找到最低能耗的 SLO 可行配置；但四个配置在之后的 calibration 复测中均未通过原始 SLO，包括 MAX，因此目前不能认定存在经复测确认的 SLO 最优方法。没有 heldout 结果。**', '',
        '## 搜索结果', '', '| 方法 | 尝试 | 成功 | SLO 可行 | 最佳可行能耗 kJ | TTFT ms | TPOT ms | 吞吐 token/s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, value in methods.items():
        m = value['deployment']['best']['metrics']
        lines.append(f"| {labels[method]} | 16 | {value['successful']} | {value['slo_feasible']} | {m['energy_j']/1000:.3f} | {m['ttft_ms']:.3f} | {m['tpot_ms']:.3f} | {m['output_tps']:.3f} |")
    lines += ['', 'Random 的一次 warmup 失败保留在 16 次预算和原始记录中。', '',
        '**V2 选出的配置：3 张活跃 GPU，2 A（GPU 1、2）+ 1 E（GPU 5）；A/E 均 1050 MHz、每卡 200 W；DBO decode/prefill 阈值 32/512。** GPU 6 未参与该配置。A 为 DP=2、TP=1；E 为 DP=1、EP=1、TP=1。', '',
        '可用池为 GPU 1、2、5、6，采用 NVML locked clocks。全程 DBO 开启，microbatches=2；阈值候选为 2/12、8/128、32/512。使用 upstream vLLM 0.26.0 AFD、eager、禁用 prefix cache。', '',
        '## 冻结配置复测（同一 calibration，各 1 次）', '', '| 方法 | 能耗 kJ | TTFT ms | TPOT ms | 吞吐 token/s | 未通过的原始 SLO |', '|---|---:|---:|---:|---:|---|']
    for row in confirmations['rows']:
        m = row['metrics']
        lines.append(f"| {labels[row['method']]} | {m['energy_j']/1000:.3f} | {m['ttft_ms']:.3f} | {m['tpot_ms']:.3f} | {m['output_tps']:.3f} | {', '.join(row['failed_slos'])} |")
    lines += ['', f"固定 SLO：TTFT ≤ {limits['ttft_ms']:.6f} ms；TPOT ≤ {limits['tpot_ms']:.6f} ms；吞吐 ≥ {limits['min_output_tps']:.6f} token/s。延迟上限为历史 MAX 的 105%，吞吐下限为其 95%，所有条件须同时满足。未用新 MAX 复测重设门槛。", '',
        '历史 RPS 4 MAX 使用修改过的 plugin 且启用 prefix cache，与本次运行时不同；相对历史 MAX 的能耗差异不能单独归因于搜索方法。复测明显波动，单次 calibration 结果不能证明泛化或稳定优越性。', '',
        '## 文件与校验', '',
        '- `summary.json`：完整最佳配置、指标、复测结果及限制。',
        '- `measurements.csv`：全部 52 条记录，区分搜索与复测，保留失败。',
        '- `manifest.json` / `*.tar.gz.part*`：原始测量、日志、遥测、输入、冻结源码和运行时元数据，逐文件及压缩分块 SHA-256。',
        '- `validation.json`：预算、冻结状态、测量证据、357 个冻结文件、请求身份隔离及恢复校验。',
        '- `restore.py`：恢复并核验全部原始文件；不会执行归档内的实验代码。', '',
        '恢复命令：`python restore.py restored-output`。归档保留相对工作区路径；原始 JSON 内的绝对路径仍用于历史溯源。模型权重、容器镜像和依赖缓存未打包，因此恢复文件不等于可直接重新启动实验。', '',
        '归档中的 legacy heldout 输入仅保留隔离溯源，不代表已经测量，也不是另外计划的正式 heldout split。calibration 与该 legacy heldout 在 source_index、source_timestamp 上均无交集；warmup 来自 calibration。只核验了请求完成，未核验语义正确性或输出 token 完全相等。', '']
    (dest / 'README.md').write_text('\n'.join(lines))
    with tempfile.TemporaryDirectory(prefix='dbo-archive-restore-') as restored:
        result = subprocess.run([sys.executable, str(dest / 'restore.py'), restored], check=True, text=True, capture_output=True)
    for entry in manifest['files']:
        assert sha(checked(Path(entry['source']))) == entry['sha256'], entry['source']
    write(dest / 'validation.json', dict(status='passed', search_attempts=48, confirmation_runs=4, frozen_method_states=3, pending_trials=0, frozen_source_files_verified=len(plan['files_sha256']), measurement_artifact_references_verified=artifact_count, archived_files=len(files), identity_overlap=overlap, warmup_from_calibration=True, heldout_evaluated=False, restored_all_file_hashes=True, source_unchanged_after_archive=True, credential_pattern_scan='passed', restoration_output=result.stdout))
    for path in dest.iterdir():
        if '.tar.gz.part' not in path.name:
            checked(path)
    print(json.dumps(dict(destination=str(dest), files=len(files), archive_bytes=total, parts=len(parts), artifact_references_verified=artifact_count)))


if __name__ == '__main__':
    main()

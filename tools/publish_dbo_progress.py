#!/usr/bin/env python3
"""Export current four-scene results, frozen sources, and GPU migration changes."""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from publish_dbo_completed_scene import checked, sha, write

CAMPAIGNS = ('dbo_on_1256_v3', 'dbo_on_0347_v3', 'dbo_on_parallel_v3')
SCENES = {'deepseek-rps4': CAMPAIGNS[0], 'qwen-rps4': CAMPAIGNS[0], 'deepseek-rps8': CAMPAIGNS[1], 'qwen-rps8': CAMPAIGNS[1]}


def search_boundary(root):
    paths = []
    for scene, owner in SCENES.items():
        folder = root / 'results' / owner / 'search' / scene / 'comparison'
        paths.extend(folder.glob('*/state.json'))
        paths.extend(folder.glob('*/trials/*/worker-result.json'))
    return {str(p.relative_to(root)): sha(checked(p)) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    root, dest = args.workspace.resolve(), args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    boundary = search_boundary(root)
    assert len([p for p in boundary if p.endswith('/state.json')]) == 12
    frozen = {}
    for campaign in CAMPAIGNS:
        for path, digest in json.loads(checked(root / 'results' / campaign / 'PLAN.json'))['files_sha256'].items():
            if path in frozen:
                assert frozen[path] == digest
            frozen[path] = digest
    for path, digest in frozen.items():
        assert sha(checked(Path(path))) == digest, path
    files, links, exclusions = set(), [], []
    bases = [root / 'results' / name for name in CAMPAIGNS]
    bases.append(root / 'results/dbo_on_1256/native-packages')
    migrations = ('dbo_on_0123_resume', 'dbo_on_0124_resume')
    bases.extend(root / 'results' / name for name in migrations)
    for folder in bases:
        for directory, dirs, names in os.walk(folder, followlinks=False):
            dirs[:] = [name for name in dirs if name not in ('__pycache__', '.git')]
            for name in list(dirs) + names:
                path = Path(directory) / name
                if path.is_symlink():
                    target = path.resolve()
                    links.append(dict(member=str(path.relative_to(root)), target_member=str(target.relative_to(root))))
                elif path.is_file():
                    if path.suffix in ('.pyc', '.lock', '.o', '.tmp') or '.tmp-' in path.name or path.name == 'host-resource-observations.jsonl':
                        exclusions.append(str(path.relative_to(root)))
                    else:
                        files.add(path)
    files.update(Path(p) for p in frozen)
    models = {}
    for model in ('DeepSeek-V2-Lite-Chat', 'Qwen3.6-35B-A3B'):
        folder = root / 'artifacts/models' / model
        models[model] = dict(original_directory=str(folder), files=[])
        for path in sorted(folder.iterdir()):
            if path.is_file():
                stat = path.stat()
                weight = path.suffix in ('.safetensors', '.bin', '.pt', '.pth')
                models[model]['files'].append(dict(name=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns, archived=not weight))
                if not weight:
                    files.add(path)
        metadata = folder / '.cache/huggingface/download'
        if metadata.exists():
            files.update(metadata.rglob('*.metadata'))
    # Frozen references can traverse shared directory aliases. Store canonical
    # files once and recreate aliases after extraction.
    files = {path.resolve() for path in files}
    external = dict(container=json.loads(checked(root / 'results/dbo_on_1256_v3/container.json')), models=models,
        container_included=False, model_weights_included=False, bo_virtualenv_included=False, compiled_caches_included=False,
        installed_plugin_source_included=True, runtime_manifest='results/dbo_on_1256_v3/source/environment/OFFICIAL_INSTALLED.json',
        note='No public download URL for the exact original container is established. Transport it separately or document a rebuilt runtime in a new campaign.')
    summary = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), observations=0, search_receipts=0, confirmation_receipts=0,
        scenes={}, pending_failures=[], source_machine_stopped=False, direct_cross_host_resume_supported=False, heldout_evaluated=False)
    artifact_references = 0
    for scene, owner in SCENES.items():
        folder = root / 'results' / owner / 'search' / scene
        methods = {}
        for state_path in sorted(folder.glob('comparison/*/state.json')):
            state = json.loads(checked(state_path))
            receipts = list((state_path.parent / 'trials').glob('*/worker-result.json'))
            methods[state_path.parent.name] = dict(observations=len(state['observations']), receipts=len(receipts), frozen=state['frozen'], pending=bool(state['pending']))
            summary['observations'] += len(state['observations'])
            summary['search_receipts'] += len(receipts)
            for path in receipts:
                receipt = json.loads(checked(path))
                if receipt.get('cleanup_required'):
                    summary['pending_failures'].append(dict(path=str(path.relative_to(root)), sha256=sha(checked(path)), status=receipt['status'], failure_reason=receipt.get('failure_reason'), old_host_cleanup_required=not (path.parent/'CLEANUP_RESOLVED.json').exists()))
                for artifact in receipt.get('artifacts', []):
                    assert sha(checked(Path(artifact['path']))) == artifact['sha256'], artifact['path']
                    artifact_references += 1
        summary['scenes'][scene] = methods
    for owner in CAMPAIGNS[:2]:
        summary['confirmation_receipts'] += len(list((root / 'results' / owner / 'confirmation').glob('**/worker-result.json')))
    isolation = []
    for scene, owner in SCENES.items():
        inputs = root / 'results' / owner / 'search' / scene / 'inputs'
        calibration = [json.loads(line) for line in checked(inputs / 'calibration.jsonl').splitlines() if line.strip()]
        for label, path in [('legacy_heldout', inputs / 'heldout.jsonl'), ('planned_formal_heldout', root / 'results/dbo_on_1256_v3/validation/heldout-12000-12399.jsonl')]:
            evaluation = [json.loads(line) for line in checked(path).splitlines() if line.strip()]
            overlap = {}
            for field in ('source_index', 'source_timestamp'):
                left, right = ({json.dumps(row[field], sort_keys=True) for row in records} for records in (calibration, evaluation))
                assert len(left) == len(calibration) and len(right) == len(evaluation)
                overlap[field] = len(left & right)
                assert not overlap[field]
            isolation.append(dict(scene=scene, evaluation=label, calibration_requests=len(calibration), evaluation_requests=len(evaluation), overlap=overlap,
                calibration_sha256=sha(checked(inputs / 'calibration.jsonl')), evaluation_sha256=sha(checked(path))))
    # The exporter snapshots local files, never stops controllers or touches GPUs.
    # Search states and receipts must remain byte-identical for the entire export.
    manifest = dict(created_utc=summary['snapshot_utc'], workspace_original=str(root), archives=[], files=[], links=links, exclusions=exclusions)
    name = 'dbo-progress-0124.tar.gz'
    with tempfile.TemporaryFile() as compressed:
        with tarfile.open(fileobj=compressed, mode='w:gz', compresslevel=6) as tar:
            for path in sorted(files):
                data = checked(path)
                member = str(path.relative_to(root))
                info = tarfile.TarInfo(member)
                info.size, info.mode = len(data), path.stat().st_mode & 0o777
                tar.addfile(info, io.BytesIO(data))
                manifest['files'].append(dict(archive=name, member=member, source=str(path), size=len(data), sha256=sha(data)))
        assert search_boundary(root) == boundary, 'Search advanced during export; reject this snapshot'
        for path, digest in frozen.items():
            assert sha(checked(Path(path))) == digest
        compressed.seek(0)
        import hashlib
        digest, parts, size = hashlib.sha256(), [], 0
        while chunk := compressed.read(24 * 1024 * 1024):
            part = name + f'.part{len(parts):03d}'
            (dest / part).write_bytes(chunk)
            parts.append(dict(file=part, size=len(chunk), sha256=sha(chunk)))
            size += len(chunk)
            digest.update(chunk)
        manifest['archives'].append(dict(name=name, sha256=digest.hexdigest(), size=size, parts=parts))
    write(dest / 'manifest.json', manifest)
    write(dest / 'summary.json', summary)
    write(dest / 'external-dependencies.json', external)
    write(dest / 'search-boundary.json', boundary)
    write(dest / 'trace-isolation.json', isolation)
    shutil.copyfile(repo / 'tools/restore_dbo_checkpoint.py', dest / 'restore.py')
    shutil.copyfile(repo / 'tools/preflight_dbo_checkpoint.py', dest / 'preflight.py')
    (dest / 'README.md').write_text('# 四卡 DBO 实验当前结果与代码快照\n\n'
        f"快照时间：{summary['snapshot_utc']}。包含 {summary['observations']} 条优化器观测、{summary['search_receipts']} 份搜索回执、{summary['confirmation_receipts']} 份配置复测回执。失败和待记账结果均保留。\n\n"
        '包含全部四个场景的搜索状态、请求回放、能耗采样、启动记录、原有冻结源码，以及 0123 / 0124 GPU 迁移代码、映射和验证记录。\n\n'
        '当前用户指定剩余实验使用 GPU 0、1、2、4 串行续跑；旧结果不重跑。物理 GPU 在校准阶段发生过变化，比较应披露阶段差异。正式 heldout 尚未评测，不能据此宣称方法最终胜出。\n\n'
        '恢复文件：`python3 restore.py restored-output`。这只恢复快照；旧 PID、端口、GPU UUID 和启动脚本不能直接用于另一台机器。\n\n'
        '模型权重、容器、虚拟环境和编译缓存未打包，清单见 external-dependencies.json。所有归档分块和文件均有 SHA-256，恢复校验见 validation.json。\n\n'
        '当前进度说明：[DBO_PROGRESS_20260915.md](../../docs/DBO_PROGRESS_20260915.md)。\n')
    with tempfile.TemporaryDirectory(prefix='dbo-checkpoint-verify-') as restored:
        output = subprocess.run([sys.executable, str(dest / 'restore.py'), restored], check=True, text=True, capture_output=True)
        # Verify the byte boundary inside the restored checkpoint, including aliases.
        assert search_boundary(Path(restored)) == boundary
        preflight = subprocess.run([sys.executable, str(dest / 'preflight.py'), restored], check=True, text=True, capture_output=True)
        report = json.loads(preflight.stdout)
        assert report['checkpoint']['observations'] == summary['observations']
        assert report['checkpoint']['search_receipts'] == summary['search_receipts']
    write(dest / 'validation.json', dict(status='passed', archived_files=len(files), internal_links=len(links), frozen_files_verified=len(frozen), search_boundary_unchanged_during_export=True,
        artifact_references_verified=artifact_references, restore_all_file_hashes=True, restored_search_boundary_verified=True, preflight_executed_read_only=True,
        cross_host_resume_tested=False, target_machine_tested=False, credential_scan='passed', restoration_output=output.stdout))
    for path in dest.iterdir():
        if '.tar.gz.part' not in path.name:
            checked(path)
    print(json.dumps(dict(destination=str(dest), files=len(files), archive_bytes=size, parts=len(parts), observations=summary['observations'], search_receipts=summary['search_receipts'])))


if __name__ == '__main__':
    main()

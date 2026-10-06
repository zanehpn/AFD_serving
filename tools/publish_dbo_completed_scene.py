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
    lines = ['# DeepSeek RPS 4: DBO enabled with threshold search (complete)', '',
        'This snapshot includes 48 search attempts (16 each for V2, BO, and Random, seed 0) and four frozen-configuration confirmations. Every measurement uses the same 200 calibration requests. Other unfinished scenarios are excluded.', '',
        '**V2 found the lowest-energy SLO-feasible configuration during search. However, all four configurations, including MAX, failed the original SLOs during later calibration confirmations. No method has a confirmed SLO-feasible optimum from those repeats. No held-out results are available.**', '',
        '## Search results', '', '| Method | Attempts | Successful | SLO feasible | Best feasible energy kJ | TTFT ms | TPOT ms | Throughput token/s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, value in methods.items():
        m = value['deployment']['best']['metrics']
        lines.append(f"| {labels[method]} | 16 | {value['successful']} | {value['slo_feasible']} | {m['energy_j']/1000:.3f} | {m['ttft_ms']:.3f} | {m['tpot_ms']:.3f} | {m['output_tps']:.3f} |")
    lines += ['', 'The Random warmup failure remains charged against its 16-attempt budget and retained in the raw records.', '',
        '**V2 selected three active GPUs: 2 A (GPUs 1, 2) + 1 E (GPU 5), both roles at 1050 MHz and 200 W per GPU, with DBO decode/prefill thresholds of 32/512.** GPU 6 was unused. A: DP=2, TP=1; E: DP=1, EP=1, TP=1.', '',
        'The available pool was GPUs 1, 2, 5, 6 with NVML locked clocks. DBO remained enabled with microbatches=2; threshold candidates were 2/12, 8/128, and 32/512. Runtime: upstream vLLM 0.26.0 AFD, eager execution, prefix caching disabled.', '',
        '## Frozen-configuration confirmations (same calibration, one each)', '', '| Method | Energy kJ | TTFT ms | TPOT ms | Throughput token/s | Failed original SLOs |', '|---|---:|---:|---:|---:|---|']
    for row in confirmations['rows']:
        m = row['metrics']
        lines.append(f"| {labels[row['method']]} | {m['energy_j']/1000:.3f} | {m['ttft_ms']:.3f} | {m['tpot_ms']:.3f} | {m['output_tps']:.3f} | {', '.join(row['failed_slos'])} |")
    lines += ['', f"Fixed SLOs: TTFT <= {limits['ttft_ms']:.6f} ms; TPOT <= {limits['tpot_ms']:.6f} ms; throughput >= {limits['min_output_tps']:.6f} token/s. Latency ceilings are 105% of historical MAX and the throughput floor is 95%; all constraints must hold together. The new MAX confirmation did not reset these thresholds.", '',
        'Historical RPS 4 MAX used a modified plugin and prefix caching, unlike this runtime; differences from its energy cannot be attributed solely to search. Confirmation measurements varied substantially. A single calibration run does not establish generalization or consistent superiority.', '',
        '## Files and verification', '',
        '- `summary.json`: complete selected configurations, metrics, confirmations, and limitations.',
        '- `measurements.csv`: all 52 records, separating search and confirmations and retaining failures.',
        '- `manifest.json` / `*.tar.gz.part*`: raw measurements, logs, telemetry, inputs, frozen sources, and runtime metadata with file and archive-part SHA-256 hashes.',
        '- `validation.json`: budgets, frozen state, measurement evidence, 357 frozen files, request-identity isolation, and restoration checks.',
        '- `restore.py`: restore and verify every original file without executing archived experiment code.', '',
        'Restore with `python restore.py restored-output`. Archive members use relative workspace paths; absolute paths in original JSON records preserve historical provenance. Model weights, container images, and dependency caches are excluded, so restoring files alone does not make the experiment runnable.', '',
        'Legacy held-out inputs preserve isolation provenance only; they do not establish completed measurements and are not the separately planned formal held-out split. Calibration and legacy held-out have no overlap in source_index or source_timestamp; warmup comes from calibration. Checks establish request completion, not semantic correctness or exact output-token equality.', '']
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

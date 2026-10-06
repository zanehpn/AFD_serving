"""Archive the completed 8/16 RPS suite and push an isolated results branch.

The watcher never changes a campaign. Authentication is inherited through an
environment variable and is never written to the repository or configuration.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bo_dse/scripts/afd'))
from static_dse.optimizer import best_measured

SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path):
    if path.is_symlink():
        raise ValueError('Refusing symlink: ' + str(path))
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed during capture: ' + str(path))
    if SECRET.search(data):
        raise ValueError('Credential-like content in ' + str(path))
    return data


def git(repo, *args, env=None):
    result = subprocess.run(['git', '-C', str(repo), *args], env=env,
                            text=True, capture_output=True, timeout=600)
    if result.returncode:
        # Never echo authentication environment or credential-bearing URLs.
        raise RuntimeError('git ' + args[0] + ' failed: ' + result.stderr[-3000:])
    return result.stdout.strip()


def ready(config):
    progress = []
    for rate, name in config['runs'].items():
        directory = Path(name)
        state = read(directory / 'status.json')
        progress.append({'rps': int(rate), 'phase': state['phase'],
                         'active_model': state.get('active_model')})
        if state['phase'] != 'complete':
            return False, progress
        if state.get('completed_models') != ['qwen', 'deepseek']:
            raise ValueError('Incomplete model list: ' + name)
        if state.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            if Path('/proc/' + str(state['pid'])).exists():
                return False, progress
        for model in ('qwen', 'deepseek'):
            campaign = read(directory / model / 'campaign/state.json')
            if len(campaign['observations']) != 16 or campaign.get('pending'):
                raise ValueError('Campaign must complete exactly 16 attempts: ' + model)
            session = read(directory / model / 'resident-session/session-cost.json')
            if session.get('cleanup_error'):
                raise ValueError('Campaign cleanup failed: ' + name)
    return True, progress


def summary(config):
    result = {'captured_utc': now(), 'base_commit': config['base_commit'],
              'interpretation': 'Best measured SLO-feasible participating-GPU energy; historical baselines; seed 0; no heldout evaluation.',
              'models': [], 'baseline_archives': config['baseline_archives'],
              'interrupted_run': config['interrupted_run']}
    measurements = []
    for rate, name in config['runs'].items():
        baseline = Path(config['baselines'][rate])
        for model in ('qwen', 'deepseek'):
            reference = read(baseline / model / 'inputs/slo-reference.json')
            record = {'rps': int(rate), 'model': model, 'reference': reference['metrics'],
                      'limits': reference['limits'], 'methods': {}}
            result['models'].append(record)
            for method, campaign in (
                    ('random', baseline / model / 'comparison/random-seed0'),
                    ('generic_bo', baseline / model / 'comparison/generic_bo-seed0'),
                    ('capacity_v2', Path(name) / model / 'campaign')):
                if not (campaign / 'state.json').exists():
                    record['methods'][method] = {'started': False}
                    continue
                state = read(campaign / 'state.json')
                bundle = read(campaign / 'bundle.json')
                observations = state['observations']
                limits = bundle['settings']['limits']
                if limits != record['limits']:
                    raise ValueError('SLO mismatch: ' + str(campaign))
                best = best_measured(observations, limits)
                item = {'started': True, 'directory': str(campaign),
                        'attempts': len(observations), 'frozen': state.get('frozen'),
                        'pending': (state.get('pending') or {}).get('trial_id'),
                        'successful': sum(o['status'] == 'ok' for o in observations),
                        'feasible_attempts': sum(bool(best_measured([o], limits)) for o in observations),
                        'best': best, 'budget': bundle['settings']['budget'],
                        'setup_cost': bundle['settings']['setup_cost']}
                record['methods'][method] = item
                if best:
                    first = next(o for o in observations if o['candidate_id'] == best['candidate_id'])
                    item['configuration'] = read(campaign / 'trials' / first['trial_id'] / 'configuration.json')
                    item['saving_vs_max_pct'] = 100 * (1 - best['metrics']['energy_j'] / reference['metrics']['energy_j'])
                for step, observation in enumerate(observations, 1):
                    c = read(campaign / 'trials' / observation['trial_id'] / 'configuration.json')
                    best_so_far = best_measured(observations[:step], limits)
                    row = {'rps': int(rate), 'model': model, 'method': method, 'seed': 0,
                           'step': step, 'trial_id': observation['trial_id'],
                           'candidate_id': observation['candidate_id'], 'status': observation['status'],
                           'slo_feasible': bool(best_measured([observation], limits)),
                           'failure_reason': observation.get('failure_reason', ''),
                           'configuration': json.dumps(c, sort_keys=True),
                           'best_so_far_energy_j': best_so_far['metrics']['energy_j'] if best_so_far else None}
                    row.update({key: observation.get('metrics', {}).get(key) for key in
                                ('energy_j', 'ttft_ms', 'tpot_ms', 'tbt_ms', 'output_tps')})
                    row.update({key: observation.get('cost', {}).get(key) for key in
                                ('wall_seconds', 'gpu_hours', 'tuning_energy_j')})
                    measurements.append(row)
    return result, measurements


def archive(destination, name, files, manifest):
    files = sorted(set(files))
    with tempfile.TemporaryFile() as compressed:
        with tarfile.open(fileobj=compressed, mode='w:gz', compresslevel=6) as tf:
            for path in files:
                data = checked_bytes(path)
                member = str(path.relative_to(ROOT.parent))
                info = tarfile.TarInfo(member)
                info.size, info.mtime, info.mode = len(data), int(path.stat().st_mtime), path.stat().st_mode & 0o777
                tf.addfile(info, io.BytesIO(data))
                manifest['files'].append({'source': str(path), 'archive': name, 'member': member,
                                          'bytes': len(data), 'sha256': sha(data)})
        compressed.seek(0)
        digest = hashlib.sha256()
        parts = []
        while data := compressed.read(24 * 1024 * 1024):
            part = name + '.tar.gz.part' + str(len(parts)).zfill(3)
            (destination / part).write_bytes(data)
            digest.update(data)
            parts.append({'file': part, 'bytes': len(data), 'sha256': sha(data)})
        manifest['archives'].append({'name': name, 'sha256': digest.hexdigest(), 'parts': parts})


def result_files(directory):
    return [p for p in directory.rglob('*') if p.is_file() and not p.is_symlink()
            and p.suffix not in ('.tmp', '.o', '.pyc') and not p.name.endswith('.lock')]


def build(config):
    sys.path.insert(0, str(ROOT / 'bo_dse'))
    from official import validate_context
    for name in config['runs'].values():
        for model in ('qwen', 'deepseek'):
            validate_context(read(Path(name) / model / 'official-config.json'))
    repo = Path(config['export_checkout'])
    branch = config['branch']
    if not repo.exists():
        git(ROOT, 'worktree', 'add', '-b', branch, str(repo), config['base_commit'])
    if git(repo, 'branch', '--show-current') != branch:
        raise ValueError('Unexpected export branch')
    destination = repo / 'experiment-records' / config['snapshot']
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'snapshot_utc': now(), 'base_commit': config['base_commit'],
                'files': [], 'archives': [], 'excluded': ['weights', 'virtual environments', 'caches', 'credentials']}
    tracked = git(ROOT, 'ls-files', '-z').split('\0')
    source = [ROOT / name for name in tracked if name and not name.startswith('experiment-records/')]
    source += [ROOT / name for name in config['new_source_files']]
    source += [ROOT / 'environment/OFFICIAL_INSTALLED.json']
    source += [ROOT / 'bo-dse-cpu-tests.log']
    archive(destination, 'capacity-v2-source', source, manifest)
    for rate, name in config['runs'].items():
        archive(destination, 'capacity-v2-rps' + rate + '-results', result_files(Path(name)), manifest)
    archive(destination, 'interrupted-rps16-preserved', result_files(Path(config['interrupted_run'])), manifest)
    evidence = []
    for name in config['baselines'].values():
        base = Path(name)
        for model in ('qwen', 'deepseek'):
            evidence += [base / model / 'official-config.json', base / model / 'inputs/slo-reference.json']
            for method in ('random-seed0', 'generic_bo-seed0'):
                arm = base / model / 'comparison' / method
                evidence += [arm / 'state.json', arm / 'bundle.json']
                evidence += list(arm.glob('trials/*/configuration.json'))
    archive(destination, 'historical-baseline-evidence', evidence, manifest)
    write(destination / 'manifest.json', manifest)
    shutil.copy2(ROOT / 'experiment-records/2026-09-11-143809-rps8-200-8gpu-complete/restore.py', destination / 'restore.py')
    report, rows = summary(config)
    write(destination / 'summary.json', report)
    with (destination / 'measurements.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Completed capacity v2: 8/16 RPS, 200 requests, eight-GPU pool', '',
             'Qwen and DeepSeek each completed 16 search attempts at each rate, seed 0. '
             'Each measurement uses the same 200 calibration requests and eight warmup requests. '
             'The historical eight-GPU MAX and SLO limits are reused separately for every model/rate; '
             'one historical MAX cost is charged per campaign. Candidates may use fewer than eight GPUs.', '',
             'Best means lowest measured participating-GPU energy satisfying all frozen SLOs, using the '
             'optimizer repeat-aggregation rule. Search energy and reserved GPU-hours remain separate. '
             'These are historical-baseline comparisons, with one seed and no heldout validation.', '',
             '| RPS | Model | Method | Attempts | Successful | Feasible attempts | Best kJ | Saving vs MAX |',
             '|---:|---|---|---:|---:|---:|---:|---:|']
    for record in report['models']:
        for method, item in record['methods'].items():
            best = item.get('best')
            energy = f"{best['metrics']['energy_j']/1000:.2f}" if best else 'none'
            saving = f"{item['saving_vs_max_pct']:.2f}%" if best else '—'
            lines.append(f"| {record['rps']} | {record['model']} | {method} | {item['attempts']} | {item['successful']} | {item['feasible_attempts']} | {energy} | {saving} |")
    lines += ['', 'All failed attempts and raw replay/NVML data are retained. The earlier user-interrupted '
              'RPS16 attempt is archived separately and is not a new campaign observation. Its failure '
              'receipt and shutdown costs must not be discarded or merged into successful observations.', '',
              'Historical baseline raw archives: ' + ', '.join('../' + x for x in config['baseline_archives'].values()) + '.', '',
              'Run `python3 restore.py restored-records` to verify and restore every archived byte. '
              'Models and dependencies must be recreated from the pinned lock files. Restored PID/status '
              'files are historical records, not live machine state.', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    for name in config['new_source_files']:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(checked_bytes(ROOT / name))
    with tempfile.TemporaryDirectory(prefix='capacity-suite-verify-') as td:
        subprocess.run([sys.executable, str(destination / 'restore.py'), td], check=True, timeout=600)
    write(destination / 'validation.json', {'restored_files': len(manifest['files']),
                                           'all_archive_and_file_hashes_verified': True,
                                           'credential_scan': 'passed', 'verified_utc': now(),
                                           'pre_run_cpu_suite': '300 passed, 7 failed, 2 skipped; full log archived',
                                           'capacity_v2_tests': '28 passed before the RPS16 run'})
    git(repo, 'add', '--', 'experiment-records/' + config['snapshot'], *config['new_source_files'])
    if git(repo, 'diff', '--cached', '--name-only'):
        git(repo,
            'commit', '-m', 'Archive completed capacity v2 eight-GPU experiments at 8 and 16 RPS')
    return git(repo, 'rev-parse', 'HEAD')


def push(config, commit):
    if not os.environ.get('MOE_GITHUB_TOKEN'):
        raise RuntimeError('GitHub token unavailable to publisher')
    with tempfile.TemporaryDirectory(prefix='capacity-publish-auth-') as td:
        askpass = Path(td) / 'askpass'
        askpass.write_text('#!/bin/sh\ncase "$1" in\n*Username*) printf "%s\\n" "x-access-token" ;;\n*) printf "%s\\n" "$MOE_GITHUB_TOKEN" ;;\nesac\n')
        askpass.chmod(0o700)
        env = dict(os.environ, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0')
        repo, branch = config['export_checkout'], config['branch']
        git(repo, '-c', 'credential.helper=', 'push', 'origin', 'HEAD:refs/heads/' + branch, env=env)
        remote = git(repo, '-c', 'credential.helper=', 'ls-remote', 'origin', 'refs/heads/' + branch, env=env)
        if not remote or remote.split()[0] != commit:
            raise RuntimeError('Remote commit does not match the verified archive commit')
    return 'experiment-records/' + config['snapshot']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    if args.check:
        for name in config['new_source_files']:
            checked_bytes(ROOT / name)
        report, rows = summary(config)
        print(json.dumps({'ready': ready(config), 'comparison_groups': len(report['models']),
                          'measurement_rows': len(rows), 'branch': config['branch']}))
        return
    runtime = args.config.parent
    with (runtime / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = runtime / 'status.json'
        state = read(state_path) if state_path.exists() else {}
        if state.get('phase') == 'pushed':
            return
        commit = state.get('commit')
        while True:
            try:
                complete, progress = ready(config)
                if not complete:
                    write(state_path, {'phase': 'waiting_for_completion', 'pid': os.getpid(),
                                       'updated_utc': now(), 'progress': progress, 'branch': config['branch']})
                    time.sleep(30)
                    continue
                if not commit:
                    write(state_path, {'phase': 'building_and_verifying_archive', 'pid': os.getpid(), 'updated_utc': now()})
                    commit = build(config)
                write(state_path, {'phase': 'pushing', 'pid': os.getpid(), 'commit': commit, 'updated_utc': now()})
                url = push(config, commit)
                write(state_path, {'phase': 'pushed', 'pid': os.getpid(), 'commit': commit,
                                   'url': url, 'branch': config['branch'], 'updated_utc': now()})
                print('Published ' + url, flush=True)
                return
            except Exception as error:
                write(state_path, {'phase': 'retry_pending', 'pid': os.getpid(), 'commit': commit,
                                   'error': str(error), 'updated_utc': now()})
                time.sleep(60)


if __name__ == '__main__':
    main()

"""Budgeted, resumable stock reference preparation before AFD calibration."""
import os
from pathlib import Path
import signal
import subprocess
import time

from output_correctness import validate_reference


def measurement(config, directory, native):
    env = native.environment(config)
    env = {k: v for k, v in env.items() if not k.startswith('ECODEP_')}
    env.pop('PYTHONPATH', None)
    env.update(VLLM_PLUGINS='', CUDA_VISIBLE_DEVICES=','.join(config['correctness']['reference_gpu_uuids']))
    command = [config['native_python'], str(native.BO / 'scripts/afd/measure_command.py'),
               '--gpus', ','.join(map(str, config['correctness']['reference_gpus'])),
               '--interval-ms', '100', '--output', str(directory / 'tuning-telemetry.json'),
               '--samples-output', str(directory / 'tuning-power.jsonl'), '--',
               config['native_python'], str(native.BO / 'stock_reference.py'),
               '--contract', str(directory / 'contract.json'), '--output', str(directory / 'reference.json')]
    with (directory / 'native.log').open('a') as log:
        process = subprocess.Popen(command, env=env, cwd=native.ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        native.write_json(directory / 'meter-process.json', {
            'pid': process.pid, 'start_ticks': native.process_identity(process.pid)})
        try:
            code = process.wait(timeout=config['trial_timeout_seconds'])
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    native.check(code == 0, f'Stock reference exited {code}; inspect {directory / "native.log"}')


def prepare(config, retry_failed, native):
    """Charge every attempt, then bind a successful reference into the context."""
    root = Path(config['config_path']).parent
    contract = config['correctness']
    results = []

    def check_available():
        native.invoke(config, [native.ROOT / 'scripts/afd/check_gpu_availability.py',
                               '--gpus', ','.join(map(str, config['gpus'])),
                               '--minimum-free-gib', '65', '--stable-for-s', '2', '--wait-timeout-s', '30'],
                      stdout=subprocess.DEVNULL)

    with native.gpu_locks(config['gpus']):
        attempts = sorted(root.glob('correctness-attempt-*'))
        for attempt in attempts:
            path = attempt / 'result.json'
            if not path.exists():
                native.check(retry_failed, f'Interrupted reference at {attempt}; use prepare --retry-failed')
                native.stop_meter(attempt)
                cost, source = native.receipt_cost(config, attempt)
                native.write_json(path, {'status': 'failed', 'failure_reason': 'Interrupted stock reference',
                                        'cost': cost, 'cost_energy_source': source, 'artifacts': []})
            result = native.read_json(path)
            for item in result['artifacts']:
                native.check(native.file_hash(item['path']) == item['sha256'], 'Reference attempt artifact changed')
            results.append(result)
        if not results or results[-1]['status'] != 'ok':
            native.check(not results or retry_failed, 'Stock reference failed; inspect logs and use prepare --retry-failed')
            native.check(len(results) < config['budget']['evaluations'] - 3 and
                         sum(r['cost']['gpu_hours'] for r in results) < config['budget']['gpu_hours'],
                         'Stock reference exhausted budget reserved for bootstrap and two BO trials')
            attempt = root / f'correctness-attempt-{len(results):03d}'
            attempt.mkdir()
            native.write_json(attempt / 'PLAN.json', {'measurement_gpus': contract['reference_gpus']})
            native.write_json(attempt / 'contract.json', contract)
            native.write_json(attempt / 'STARTED.json', {'started_wall_seconds': time.time()})
            try:
                native.check(os.geteuid() == 0, 'Stock reference requires the target root environment')
                native.invoke(config, [native.ROOT / 'migration/runtime_fingerprint.py', '--check',
                                       native.ROOT / 'environment/native-runtime.json'])
                native.invoke(config, [native.BO / 'native_backend.py', 'verify-platform'])
                check_available()
                measurement(config, attempt, native)
                validate_reference(native.read_json(attempt / 'reference.json'), contract)
                cost, source = native.receipt_cost(config, attempt)
                native.check(source == 'nvml_full_suite_window', 'Stock reference tuning telemetry invalid')
                result = {'status': 'ok', 'artifacts': [native.artifact(attempt / name) for name in
                          ('reference.json', 'contract.json', 'tuning-telemetry.json')]}
            except Exception as error:
                native.stop_meter(attempt)
                result = {'status': 'failed', 'failure_reason': f'{type(error).__name__}: {error}',
                          'artifacts': [native.artifact(attempt / 'contract.json')]}
            result['cost'], result['cost_energy_source'] = native.receipt_cost(config, attempt)
            native.write_json(attempt / 'result.json', result)
            results.append(result)
            native.check(result['status'] == 'ok', f'Stock reference failed at {attempt}; inspect logs and use prepare --retry-failed')
        else:
            attempt = attempts[-1]
        # Require full GPU release even after a normal subprocess exit. A leaked
        # reference worker must never overlap the subsequent AFD bootstrap.
        check_available()
        source = attempt / 'reference.json'
        validate_reference(native.read_json(source), contract)
        destination = root / 'inputs/correctness-reference.json'
        if destination.exists():
            native.check(destination.read_bytes() == source.read_bytes(), 'Frozen reference differs from successful attempt')
        else:
            with destination.open('xb') as stream:
                stream.write(source.read_bytes())
        manifest = native.read_json(config['context_manifest'])
        files = manifest['files_sha256']
        for path in (destination, attempt / 'result.json'):
            key, value = str(path), native.file_hash(path)
            native.check(key not in files or files[key] == value, 'Frozen correctness evidence changed')
            files[key] = value
        native.write_json(config['context_manifest'], manifest)
    return results

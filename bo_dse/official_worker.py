"""External GPU control, unmodified CLI services, public replay, and NVML.

Runs in the official vLLM Python environment; no BO/scikit-learn dependency.
"""
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'migration'))
sys.path.insert(0, str(ROOT / 'scripts/afd'))
from native_service import boot_id, identity, stop_group, group_members
from output_correctness import compare
from official_space import commands


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def runtime_identity():
    lock = read(ROOT / 'environment/official-runtime.lock.json')
    plugin = ROOT / 'third_party/afd-plugin-official'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(plugin), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != lock['plugin_commit'] or git('remote', 'get-url', 'origin') != lock['plugin_url']:
        raise ValueError('Official plugin origin/commit mismatch')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Official plugin checkout is modified')
    packages, files = {}, {}
    for name in ('vllm', 'vllm-afd-plugin'):
        dist = metadata.distribution(name)
        packages[name] = dist.version
        if name == 'vllm' and (dist.version != lock['vllm_version'] or dist.read_text('direct_url.json')):
            raise ValueError('vLLM must be the locked official wheel, not a local/editable checkout')
        direct = json.loads(dist.read_text('direct_url.json') or '{}')
        if direct.get('dir_info', {}).get('editable'):
            raise ValueError('Editable runtime packages are prohibited')
        for f in dist.files or []:
            if not f.hash:
                continue
            path = Path(dist.locate_file(f))
            digest = sha(path)
            expected = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
            if f.hash.mode != 'sha256' or expected != f.hash.value:
                raise ValueError(f'Installed runtime differs from wheel RECORD: {path}')
            files[str(path.resolve())] = digest
        if name == 'vllm-afd-plugin':
            for path in (plugin / 'afd_plugin').rglob('*.py'):
                installed = Path(dist.locate_file(str(path.relative_to(plugin))))
                if sha(installed) != sha(path):
                    raise ValueError(f'Installed plugin differs from official source: {path}')
            if not any(e.group == 'vllm.general_plugins' and e.name == 'afd' and e.value == 'afd_plugin:register_afd'
                       for e in dist.entry_points):
                raise ValueError('Official AFD entrypoint missing')
    return dict(packages=packages, files_sha256=files, plugin_commit=lock['plugin_commit'], python=platform.python_version(),
                dependency_versions={d.metadata['Name'].lower(): d.version for d in metadata.distributions()})


def verify_runtime():
    expected = read(ROOT / 'environment/OFFICIAL_INSTALLED.json')
    current = runtime_identity()
    if any(current[k] != expected[k] for k in current):
        raise ValueError('Official installed runtime changed since setup')
    return current


def clean_environment():
    prefixes = ('ECODEP_', 'VLLM_', 'CUDA_', 'NCCL_', 'TORCH_', 'PYTORCH_', 'PYTHON')
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes) and k != 'LD_LIBRARY_PATH'}
    env.update(VLLM_PLUGINS='afd', VLLM_USE_V2_MODEL_RUNNER='0', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', PATH=str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', ''))
    return env


def http(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        if not 200 <= response.status < 300:
            raise ValueError('HTTP operation failed')
        payload = response.read()
    return json.loads(payload) if payload else {}


def clocks(config, c, directory):
    receipts = []
    for role in ('attention', 'expert'):
        for gpu in c[role + '_gpus']:
            p = http(config['clock_url'] + '/set_power_limit', dict(gpu=gpu, watts=c[role + '_power_w']))
            f = http(config['clock_url'] + '/set_clock', dict(gpu=gpu, sm_mhz=c[role + '_mhz']))
            receipts.append(dict(gpu=gpu, power=p, clock=f))
            write(directory / 'clock-receipts.json', receipts)
            if (p.get('gpu') != gpu or f.get('gpu') != gpu or
                    p.get('applied_w') != c[role + '_power_w'] or p.get('power_control') != 'limited'
                    or f.get('applied_mhz') != c[role + '_mhz']
                    or f.get('clock_control') != 'locked'):
                raise ValueError('Clock/power applied acknowledgement mismatch')


def cleanup(config, directory):
    errors = []
    for path in sorted(directory.glob('service-*.json')):
        try:
            service = read(path)
            stop_group(service)
            service['stopped'] = True
            write(path, service)
        except Exception as error:
            errors.append(f'{path.name}: {error}')
    # Try every owned GPU even when a service or earlier reset fails.
    owned = read(directory / 'control-ownership.json')['gpus'] if (directory / 'control-ownership.json').exists() else []
    if not set(owned) <= set(config['gpus']):
        raise ValueError('Cleanup allocation mismatch')
    for gpu in owned:
        try:
            reply = http(config['clock_url'] + '/reset', {'gpu': gpu})
            power = http(config['clock_url'] + '/set_power_limit', {'gpu': gpu, 'reset': True})
            if reply.get('gpu') != gpu or reply.get('clock_reset') is not True or power.get('power_control') != 'default':
                raise ValueError('Reset not acknowledged')
        except Exception as error:
            errors.append(f'GPU {gpu}: {error}')
    if owned:
        try:
            import pynvml as nv
            nv.nvmlInit()
            try:
                for gpu in owned:
                    if nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByIndex(gpu)):
                        errors.append(f'GPU {gpu}: compute processes remain after cleanup')
            finally:
                nv.nvmlShutdown()
        except Exception as error:
            errors.append(f'GPU process cleanup verification: {error}')
    write(directory / 'cleanup.json', dict(verified=not errors, errors=errors))
    if errors:
        raise RuntimeError('; '.join(errors))


class Monitor:
    def __init__(self, gpus):
        import pynvml as nv
        self.nv, self.gpus = nv, gpus
        nv.nvmlInit()
        self.handles = [nv.nvmlDeviceGetHandleByIndex(g) for g in gpus]
        self.rows, self.errors = [], []
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        try:
            states = [dict(sm_clock_mhz=self.nv.nvmlDeviceGetClockInfo(h, self.nv.NVML_CLOCK_SM),
                           memory_clock_mhz=self.nv.nvmlDeviceGetClockInfo(h, self.nv.NVML_CLOCK_MEM),
                           power_limit_w=self.nv.nvmlDeviceGetEnforcedPowerLimit(h)/1000,
                           gpu_utilization=self.nv.nvmlDeviceGetUtilizationRates(h).gpu,
                           throttle_reasons=self.nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)) for h in self.handles]
            powers = [self.nv.nvmlDeviceGetPowerUsage(h)/1000 for h in self.handles]
            self.rows.append(dict(timestamp_ns=time.time_ns(), gpu_ids=self.gpus, power_w=powers, operating_state=states))
        except Exception as error:
            self.errors.append(str(error))

    def loop(self):
        while not self.event.wait(.1):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def stop(self, directory):
        self.event.set()
        self.thread.join()
        self.sample()
        self.nv.nvmlShutdown()
        with (directory / 'nvml.jsonl').open('w') as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + '\n')


def metrics(config, c, directory, monitor):
    sys.path.insert(0, str(ROOT / 'bo_dse/scripts/afd'))
    from measure_command import integrate_window_energy
    from summarize_replay import percentile
    records = [json.loads(s) for s in (directory / 'replay.jsonl').read_text().splitlines()]
    if config.get('output_validation', 'exact_tokens') == 'exact_tokens':
        gate = compare(config['correctness'], config['correctness_reference'], records)
    else:
        expected = [json.loads(s) for s in Path(config['trace']['path']).read_text().splitlines() if s.strip()]
        expected_ids = {r['source_index'] for r in expected}
        returned_ids = [r.get('source_index') for r in records]
        lengths = {r['source_index']: min(r['output_tokens'], config['max_output_tokens']) for r in expected}
        complete = (len(records) == len(expected) and len(set(returned_ids)) == len(expected)
                    and set(returned_ids) == expected_ids
                    and all(r.get('http_status') == 200 and not r.get('error')
                            and r.get('actual_output_tokens') == lengths[r['source_index']] for r in records))
        gate = dict(protocol='request_completion_only_v1', verified=False,
                    request_completion_verified=complete, exact_tokens_checked=False,
                    semantic_correctness_checked=False, requests=len(records), selection_split='calibration')
    write(directory / 'correctness.json', gate)
    if not (gate.get('verified') or gate.get('request_completion_verified')) or any(r.get('error') for r in records):
        raise ValueError('Calibration output/request correctness failed; inspect correctness.json and replay.jsonl')
    tbt_metrics = {}
    if config.get('tbt_slo'):
        intervals = []
        for r in records:
            values = r.get('tbt_ms', [])
            if (len(r.get('token_arrival_s', [])) != r['actual_output_tokens']
                    or len(values) != max(0, r['actual_output_tokens']-1)
                    or any(not math.isfinite(v) or v < 0 for v in values)):
                raise ValueError('Missing or invalid client token-arrival evidence for TBT')
            intervals.extend(values)
        if not intervals:
            raise ValueError('TBT requires at least one observed token interval')
        tbt_metrics['tbt_ms'] = percentile(intervals, .9)
    start = min(r['submit_wall_ns'] for r in records)
    end = max(r['finish_wall_ns'] for r in records)
    samples = [(0, r['timestamp_ns'], r['power_w']) for r in monitor.rows]
    energy, coverage = integrate_window_energy(samples, start_ns=start, end_ns=end, gpu_count=len(monitor.gpus))
    window = [r for r in monitor.rows if start <= r['timestamp_ns'] <= end]
    if monitor.errors or len(window) < 2 or any(b['timestamp_ns']-a['timestamp_ns'] > 1e9 for a, b in zip(window, window[1:])):
        raise ValueError('Incomplete NVML measurement coverage')
    for row in window:
        for i, gpu in enumerate(monitor.gpus):
            role = 'attention' if gpu in c['attention_gpus'] else 'expert'
            state = row['operating_state'][i]
            if abs(state['power_limit_w']-c[role+'_power_w']) > 1 or state['memory_clock_mhz'] != config['memory_clock_mhz']:
                raise ValueError('Measured cap/memory clock differs from frozen operating point')
    duration = (end-start)/1e9
    roles = {role: sum(energy[i] for i, gpu in enumerate(monitor.gpus) if gpu in c[role+'_gpus'])/duration
             for role in ('attention', 'expert')}
    power_p95 = {}
    window = [r for r in monitor.rows if start <= r['timestamp_ns'] <= end]
    for role in ('attention', 'expert'):
        per_rank = []
        for gpu in c[role + '_gpus']:
            values = [r['power_w'][r['gpu_ids'].index(gpu)] for r in window
                      if gpu in r.get('gpu_ids', [])]
            if values:
                per_rank.append(percentile(values, .95))
        if len(per_rank) == len(c[role + '_gpus']):
            power_p95[role] = max(per_rank)
    role_util = {}
    for role in ('attention', 'expert'):
        values = [row['operating_state'][i].get('gpu_utilization') for row in window
                  for i, gpu in enumerate(row.get('gpu_ids', monitor.gpus)) if gpu in c[role + '_gpus']]
        values = [v for v in values if v is not None and math.isfinite(v)]
        if values:
            role_util[role] = sum(values) / len(values)
    client_lags = [r['queue_lag_ms'] for r in records if r.get('queue_lag_ms') is not None]
    return dict(status='ok', requests=len(records), completed_requests=len(records), failed_requests=0,
                output_correctness=gate, execution_verified=True, telemetry_valid=True,
                metrics=dict(energy_j=sum(energy), ttft_ms=percentile([r['ttft_ms'] for r in records], .9),
                             tpot_ms=percentile([r['tpot_ms'] for r in records], .9),
                             output_tps=sum(r['actual_output_tokens'] for r in records)/duration, **tbt_metrics),
                external_observables=dict(provenance='public_replay_and_nvml', duration_s=duration,
                                          role_mean_power_w=roles, role_peak_rank_p95_power_w=power_p95, sample_time_coverage=coverage,
                                          role_mean_gpu_utilization_pct=role_util,
                                          client_queue_lag_ms_p90=percentile(client_lags, .9) if client_lags else None,
                                          dbo_requested=c['microbatches'] == 2, actual_microbatch_splits=None,
                                          stage_times=None, execution_evidence='frozen CLI, complete public API requests and NVML; output policy: '
                                          + config.get('output_validation', 'exact_tokens')))



def check_ports(config):
    ports = [config['api_port'], config['api_port']+1, config['afd_port'],
             config['dp_rpc_port'], config['dp_rpc_port']+100]
    if len(set(ports)) != len(ports):
        raise ValueError('Official service port ranges overlap')
    for port in ports:
        with socket.socket() as sock:
            # A closed reusable server can leave TIME_WAIT without a live owner.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as error:
                raise OSError(error.errno, f'Official preflight cannot bind 127.0.0.1:{port}: {error.strerror}') from error


def compiler_preflight(directory):
    env = clean_environment()
    nvcc = shutil.which('nvcc', path=env['PATH'])
    if nvcc is None:
        raise RuntimeError('CUDA preflight: nvcc is missing from PATH')
    source = directory/'cuda-preflight.cu'
    source.write_text('#include <cuda_runtime.h>\n#include <curand.h>\n#include <curand_kernel.h>\n'
                      '#include <cublas_v2.h>\n#include <cusparse.h>\n#include <cuda/std/utility>\n'
                      '__global__ void smoke(float* p) { p[threadIdx.x] = 0.0f; }\n')
    command = [nvcc, '-std=c++17', '-arch=sm_80', '-c', str(source), '-o', str(directory/'cuda-preflight.o')]
    with (directory/'cuda-preflight.log').open('wb') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    return dict(nvcc=nvcc, compile_passed=True, source=artifact(source))


def start_services(config, c, directory):
    launch = commands(config, c)
    write(directory / 'launch.json', launch)
    for role, item in launch.items():
        env = clean_environment()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(config['hardware']['devices'][str(g)]['uuid'] for g in item['gpus'])
        with (directory / f'{role}.log').open('ab') as log:
            process = subprocess.Popen(item['command'], env=env, cwd=directory, stdout=log, stderr=log, start_new_session=True)
        ticks = identity(process.pid)
        if ticks is None:
            process.wait()
            raise RuntimeError(f'{role} exited before service registration')
        write(directory / f'service-{role}.json', dict(pid=process.pid, start_ticks=ticks, boot_id=boot_id(), command=item['command']))
    deadline = time.monotonic() + config['trial_timeout_seconds']
    while time.monotonic() < deadline:
        if any(not group_members(read(p)) for p in directory.glob('service-*.json')):
            raise RuntimeError('Official AFD service exited before readiness')
        try:
            http(f"http://127.0.0.1:{config['api_port']}/health")
            break
        except (OSError, ValueError):
            time.sleep(2)
    else:
        raise TimeoutError('Official service startup timed out')


def measure_requests(config, c, directory, monitor):
    base = [sys.executable, str(ROOT / 'scripts/afd/replay_trace.py')]
    common = ['--endpoint', f"http://127.0.0.1:{config['api_port']}/v1/completions", '--model', 'official-afd',
              '--max-output-tokens', str(config['max_output_tokens']), '--record-output-token-ids',
              '--timeout-s', str(config['trial_timeout_seconds'])]
    subprocess.run(base + [config['warmup_trace'], '--output', str(directory/'warmup.jsonl'), '--time-scale', '0'] + common,
                   check=True, env=clean_environment(), timeout=config['trial_timeout_seconds'])
    warm = [json.loads(s) for s in (directory/'warmup.jsonl').read_text().splitlines()]
    if len(warm) != config['warmup_requests'] or any(r.get('error') for r in warm):
        raise ValueError('Official warmup failed')
    # Prefix caching is disabled at launch; no private reset endpoint needed.
    subprocess.run(base + [config['trace']['path'], '--output', str(directory/'replay.jsonl'),
                           '--time-scale', str(config['time_scale'])] + common,
                   check=True, env=clean_environment(), timeout=config['trial_timeout_seconds'])
    monitor.sample()
    return metrics(config, c, directory, monitor)

def trial(config, c, directory, reference=False):
    if config.get('resident_session_directory') and not reference:
        from resident_worker import resident_trial
        return resident_trial(config, c, directory)
    started = time.monotonic()
    result, monitor = {}, None
    try:
        verify_runtime()
        check_ports(config)
        monitor = Monitor(c['attention_gpus'] + c['expert_gpus'])
        for i, gpu in enumerate(monitor.gpus):
            handle = monitor.handles[i]
            if monitor.nv.nvmlDeviceGetUUID(handle) != config['hardware']['devices'][str(gpu)]['uuid']:
                raise ValueError('Physical GPU identity changed')
            if monitor.nv.nvmlDeviceGetComputeRunningProcesses(handle):
                raise ValueError('GPU has existing compute processes')
        monitor.start()
        write(directory / 'control-ownership.json', {'gpus': monitor.gpus})
        clocks(config, c, directory)
        if reference:
            env = clean_environment()
            env['VLLM_PLUGINS'] = ''
            env['CUDA_VISIBLE_DEVICES'] = ','.join(config['correctness']['reference_gpu_uuids'])
            command = [sys.executable, str(ROOT/'bo_dse/stock_reference.py'),
                       '--contract', config['correctness_contract'], '--output', config['correctness_reference_path']]
            with (directory/'reference.log').open('ab') as log:
                process = subprocess.Popen(command, env=env, cwd=directory, stdout=log, stderr=log, start_new_session=True)
            ticks = identity(process.pid)
            if ticks is None:
                process.wait()
                raise RuntimeError('Stock reference exited at startup')
            write(directory/'service-reference.json', dict(pid=process.pid, start_ticks=ticks, boot_id=boot_id(), command=command))
            process.wait(timeout=config['trial_timeout_seconds'])
            if process.returncode:
                raise RuntimeError('Stock reference failed; inspect reference.log')
            from output_correctness import validate_reference
            validate_reference(read(config['correctness_reference_path']), config['correctness'])
            result = dict(status='ok', reference=artifact(config['correctness_reference_path']))
            return result
        start_services(config, c, directory)
        result = measure_requests(config, c, directory, monitor)
    except Exception as error:
        result = dict(status='failed', failure_reason=f'{type(error).__name__}: {error}')
    finally:
        try:
            cleanup(config, directory)
        except Exception as error:
            result.update(status='failed', failure_reason=result.get('failure_reason', '') + '; cleanup: ' + str(error), cleanup_required=True)
        if monitor is not None:
            if monitor.thread.ident is not None:
                monitor.stop(directory)
            else:
                monitor.nv.nvmlShutdown()
        elapsed = time.monotonic()-started
        energy = 0.
        if monitor is not None:
            energy = sum((b['timestamp_ns']-a['timestamp_ns'])/1e9*(sum(a['power_w'])+sum(b['power_w']))/2
                         for a, b in zip(monitor.rows, monitor.rows[1:]))
        result['cost'] = dict(wall_seconds=elapsed, gpu_hours=elapsed*len(config['gpus'])/3600,
                              tuning_energy_j=max(energy, result.get('metrics', {}).get('energy_j', 0)))
        result['cost_energy_complete'] = monitor is not None and bool(monitor.rows) and not monitor.errors
        immutable = ('STARTED.json', 'launch.json', 'clock-receipts.json', 'nvml.jsonl',
                     'replay.jsonl', 'warmup.jsonl', 'correctness.json', 'configuration.json')
        result['artifacts'] = [artifact(directory/name) for name in immutable if (directory/name).exists()]
        write(directory / 'worker-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'trial', 'reference', 'cleanup'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--configuration', type=Path)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.action == 'preflight':
        from capture_hardware import capture
        import pynvml
        check_ports(config)
        compiler = compiler_preflight(args.directory)
        identity_record = verify_runtime()
        hardware = capture(config['gpus'], pynvml)
        if any('A100-SXM4-80GB' not in d['name'] for d in hardware['devices'].values()):
            raise ValueError('Initial official protocol requires A100-SXM4-80GB')
        health = http(config['clock_url'] + '/health')
        if health.get('protocol') != 'nvcontrold.applied_ack.v2' or not set(config['gpus']) <= set(health.get('allowed', [])):
            raise ValueError('Clock daemon protocol/allocation mismatch')
        write(args.directory/'preflight.json', dict(runtime=identity_record, hardware=hardware, clock=health, compiler=compiler))
    elif args.action == 'cleanup':
        cleanup(config, Path(config.get('resident_session_directory', args.directory)))
    else:
        trial(config, read(args.configuration), args.directory, reference=args.action == 'reference')


if __name__ == '__main__':
    main()

"""Isolated calibration experiment; never writes to the search campaigns."""
import copy
import fcntl
import json
import os
import tempfile
from pathlib import Path
import signal
import sys
import time

ROOT = (Path(__file__).resolve().parents[1] / 'results/a6000-p2p-pstates-20260915')
OUT = (Path(__file__).resolve().parents[1] / 'results/resident-audit-20260915')
sys.path.insert(0, str(ROOT/'source/bo_dse'))
import official_worker as w
import resident_worker as rw
from native_service import process_info


def save(name, data):
    w.write(OUT/name, data)


def config_and_candidates():
    config = w.read(ROOT/'search/deepseek-rps4/official-config.json')
    a = w.read(ROOT/'BASELINES.json')['deepseek-rps4']['configuration']
    b = dict(a, attention_mhz=1920, expert_mhz=1920)
    return config, a, b


def audit():
    config, a, b = config_and_candidates()
    checks = {}
    key = rw.service_key(config, a)
    checks['frequency_only_reuses'] = key == rw.service_key(config, b)
    checks['power_only_reuses'] = key == rw.service_key(config, dict(a, attention_power_w=250))
    checks['threshold_change_restarts'] = key != rw.service_key(config, dict(a, dbo_decode_token_threshold=64, dbo_prefill_token_threshold=1024))
    changed = copy.deepcopy(a)
    changed['attention_gpus'], changed['expert_gpus'] = [2, 3], [0, 1]
    checks['allocation_change_restarts'] = key != rw.service_key(config, changed)
    checks['port_change_restarts'] = key != rw.service_key(dict(config, api_port=config['api_port']+1000), a)
    checks['dbo_change_restarts'] = key != rw.service_key(config, dict(a, microbatches=1))
    checks['calibration_only'] = '/calibration.jsonl' in config['trace']['path']
    checks['full_replay'] = config['trace']['requests'] == 200
    checks['warmup_eight'] = config['warmup_requests'] == 8
    save('offline-checks.json', checks)
    if not all(checks.values()):
        raise RuntimeError(f'Offline audit failed: {checks}')
    return checks


def snapshot():
    paths = [ROOT/'PLAN.json'] + list(ROOT.glob('search/*/comparison/*/state.json'))
    return {str(p): w.sha(p) for p in sorted(paths)}


def measure(config, candidate, name, resident=False):
    directory = OUT/name
    directory.mkdir()
    cfg = dict(config)
    if resident:
        cfg['resident_session_directory'] = str(OUT/'session')
    else:
        cfg.pop('resident_session_directory', None)
    w.write(directory/'worker-config.json', cfg)
    w.write(directory/'configuration.json', candidate)
    w.write(directory/'STARTED.json', {'started_wall_ns': time.time_ns(), 'diagnostic_only': True})
    phases = {}
    originals = {}
    for method in ('verify_runtime', 'start_services', 'measure_requests', 'cleanup', 'clocks'):
        original = getattr(w, method)
        originals[method] = original
        def timed(*args, _name=method, _fn=original, **kwargs):
            start = time.monotonic()
            try:
                return _fn(*args, **kwargs)
            finally:
                phases[_name] = phases.get(_name, 0) + time.monotonic()-start
        setattr(w, method, timed)
    try:
        result = w.trial(cfg, candidate, directory)
    finally:
        for name_, fn in originals.items():
            setattr(w, name_, fn)
    w.write(directory/'diagnostic-phases.json', phases)
    row = {'name': name, 'status': result['status'], 'cost': result['cost'],
           'metrics': result.get('metrics'), 'phases': phases,
           'resident_service': result.get('resident_service'),
           'failure_reason': result.get('failure_reason'),
           'telemetry_valid': result.get('telemetry_valid'),
           'completed_requests': result.get('completed_requests')}
    print(json.dumps(row), flush=True)
    if result['status'] != 'ok' or result.get('completed_requests') != 200 or not result.get('telemetry_valid'):
        raise RuntimeError(f'Diagnostic failed: {row}')
    if resident and name.startswith('03') and not result['resident_service']['reused']:
        raise RuntimeError('Expected actual service reuse')
    return row


def run():
    audit()
    cfg, a, b = config_and_candidates()
    status = w.read(ROOT/'STATUS.json')
    pid = status['pid']
    info = process_info(pid)
    if not info or info['state'] in ('T', 't', 'Z', 'X'):
        raise RuntimeError('Queue is not in an active resumable state')
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if str(ROOT/'run_queue.py') not in cmd or str(ROOT) not in cmd:
        raise RuntimeError(f'Unexpected queue command: {cmd}')
    parent_fd = os.pidfd_open(pid)
    if process_info(pid)['start_ticks'] != info['start_ticks']:
        raise RuntimeError('Queue identity changed')
    locks = []
    paused = False
    meter = None
    rows = []
    before = None
    record = {'status': 'waiting_for_round_boundary', 'queue_pid': pid,
              'queue_start_ticks': info['start_ticks'], 'started_at': time.time(),
              'diagnostic_only': True, 'rows': rows}
    save('STATUS.json', record)
    try:
        # Stop only the scheduler. Its current child and all GPU services run on.
        signal.pidfd_send_signal(parent_fd, signal.SIGSTOP)
        paused = True
        deadline = time.monotonic()+5400
        while True:
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            active = [int(x) for x in children if (process_info(int(x)) or {}).get('state') not in (None, 'Z', 'X')]
            if not active:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Current round did not finish within 90 minutes')
            time.sleep(5)
        for gpu in sorted(cfg['gpus']):
            fd = os.open(str(Path(tempfile.gettempdir()) / f'moe-bo-gpu-{gpu}.lock'), os.O_RDONLY | os.O_NOFOLLOW)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p in ROOT.glob('search/*/comparison/*/state.json'):
            if w.read(p).get('pending'):
                raise RuntimeError(f'Unfinished search request at boundary: {p}')
        meter = w.Monitor(cfg['gpus'])
        rw.check_identity_and_owners(cfg, meter, OUT/'session', False)
        w.check_ports(cfg)
        before = snapshot()
        save('search-boundary-before.json', before)
        meter.start()
        record.update(status='validating', boundary_at=time.time())
        save('STATUS.json', record)
        # Two matched candidates: cold A vs reused A, resident cold B vs cold B.
        for name, candidate, resident in [('01-cold-a', a, False), ('02-resident-cold-b', b, True),
                                           ('03-resident-warm-a', a, True), ('04-cold-b', b, False)]:
            if name == '04-cold-b':
                w.cleanup(cfg, OUT/'session')
            record['current'] = name
            save('STATUS.json', record)
            rows.append(measure(cfg, candidate, name, resident))
            save('STATUS.json', record)
        record['status'] = 'validation_complete'
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cleanup_errors = []
        for directory in [OUT/'session'] + sorted(OUT.glob('0*-*')):
            if (directory/'control-ownership.json').exists():
                try:
                    w.cleanup(cfg, directory)
                except Exception as error:
                    cleanup_errors.append(f'{directory}: {error}')
        if meter:
            try:
                if meter.thread.ident is not None:
                    meter.stop(OUT)
                    samples = meter.rows
                    elapsed = (samples[-1]['timestamp_ns']-samples[0]['timestamp_ns'])/1e9 if len(samples)>1 else 0
                    energy = sum((y['timestamp_ns']-x['timestamp_ns'])/1e9*(sum(x['power_w'])+sum(y['power_w']))/2 for x,y in zip(samples,samples[1:]))
                    record['reservation_cost'] = dict(wall_seconds=elapsed, gpu_hours=elapsed*len(cfg['gpus'])/3600, tuning_energy_j=energy)
                    record['meter_errors'] = meter.errors
                else:
                    meter.nv.nvmlShutdown()
            except Exception as error:
                record['meter_finalization_error'] = repr(error)
        if before is not None:
            try:
                after = snapshot()
                save('search-boundary-after.json', after)
                record['search_unchanged'] = before == after
            except Exception as error:
                record['snapshot_error'] = repr(error)
        for fd in reversed(locks):
            os.close(fd)
        record['cleanup_errors'] = cleanup_errors
        if paused and not cleanup_errors:
            signal.pidfd_send_signal(parent_fd, signal.SIGCONT)
            record['queue_resumed'] = True
        else:
            record['queue_resumed'] = False
        os.close(parent_fd)
        record['finished_at'] = time.time()
        save('STATUS.json', record)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted: {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if '--audit-only' in sys.argv:
        print(json.dumps(audit(), indent=2))
    else:
        run()

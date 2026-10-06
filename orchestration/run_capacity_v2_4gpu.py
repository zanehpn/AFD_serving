"""Run frozen capacity_v2 at 4/8 RPS, each model on its own four-GPU pool."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'bo_dse/.venv/bin/python'
NATIVE = ROOT / '.venv-official/bin/python'
sys.path.insert(0, str(ROOT / 'bo_dse'))
import official as o


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def setup(cfg, model, rate):
    directory = Path(cfg['runs'][str(rate)]) / model
    old = Path(cfg['rps8_baseline']) / model
    gpus = list(range(4)) if model == 'qwen' else list(range(4, 8))
    offset = 0 if model == 'qwen' else 1000
    source = ROOT / 'inputs/traces' if rate == 4 else old / 'inputs'
    args = ['start', '--directory', str(directory), '--model', 'qwen36' if model == 'qwen' else 'deepseek-v2-lite',
            '--gpus', *map(str, gpus), '--rps', str(rate),
            '--calibration', str(source / ('calibration-200.jsonl' if rate == 4 else 'calibration-source.jsonl')),
            '--heldout', str(source / ('heldout-400.jsonl' if rate == 4 else 'heldout.jsonl')),
            '--evaluation-requests', '200', '--evaluations', '17', '--gpu-hours', '32',
            '--microbatches', '2', '--seeds', '0', '--direct-search',
            '--output-validation', 'request_completion', '--slo-mode', 'relative_max',
            '--resident-search', '--exploration-policy', 'capacity_v2',
            '--api-port', str(24000+offset), '--afd-port', str(26000+offset),
            '--dp-rpc-port', str(34000+offset), '--clock-url', 'http://127.0.0.1:' + str(9098 if model == 'qwen' else 9099),
            '--reuse-max-directory', str(ROOT/'inputs/calibration-max'/('qwen36' if model == 'qwen' else 'deepseek-v2-lite') if rate == 4 else old),
            '--prepare-only']
    if rate == 8:
        args += ['--tbt-slo']
    return directory, gpus, args


def prepare(cfg, model, rate):
    if rate == 4:
        import inherited_reference
        import legacy_max_reference
        inherited_reference.inherit = legacy_max_reference.inherit
    original = o.initialize

    def initialize(args):
        config = original(args)
        manifest = o.read(config['context_manifest'])
        paths = [Path(__file__), ROOT/'orchestration/legacy_max_reference.py']
        paths += [p for p in (Path(config['directory'])/'inputs/legacy-max-source').rglob('*') if p.is_file()]
        manifest['files_sha256'].update({str(p): o.sha(p) for p in paths})
        o.write(config['context_manifest'], manifest)
        return config

    o.initialize = initialize
    sys.argv = [str(ROOT/'bo_dse/official.py'), *setup(cfg, model, rate)[2]]
    o.main()


def search(cfg, model, rate):
    directory = setup(cfg, model, rate)[0]
    config = o.read(directory/'official-config.json')
    o.validate_context(config)
    original = o.tell

    def tell(path, result):
        value = original(path, result)
        state = o.read(Path(path)/'state.json')
        o.write(directory.parent/(model+'-progress.json'), dict(model=model, rps=rate,
                attempts=len(state['observations']), successful=sum(x['status']=='ok' for x in state['observations']),
                last_status=result['status'], last_failure=result.get('failure_reason'), updated_utc=now()))
        reason = result.get('failure_reason', '')
        if result['status'] != 'ok' and any(x in reason for x in ('Official plugin', 'Official installed runtime',
                'Installed runtime differs', 'Installed plugin differs', 'No such file or directory',
                'ModuleNotFoundError', 'Clock daemon protocol/allocation mismatch')):
            raise RuntimeError('Infrastructure failure recorded; stopping: '+reason)
        return value

    o.tell = tell
    report = o.run(config)
    state = o.read(directory/'campaign/state.json')
    if len(state['observations']) != 16 or state['pending']:
        raise RuntimeError('Search did not finish all 16 attempts')
    o.write(directory.parent/(model+'-summary.json'), report)


def model_run(cfg, model):
    runtime = Path(cfg['runtime'])
    status_path = runtime/(model+'-status.json')
    record = dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  model=model, completed_rates=[], method='capacity_v2', revision='neighbor_reduction')
    def status(phase, **fields):
        record.update(phase=phase, updated_utc=now(), **fields)
        o.write(status_path, record)
        print(json.dumps(record), flush=True)
    gpus = setup(cfg, model, 8)[1]
    port = 9098 if model == 'qwen' else 9099
    env = dict(os.environ, PATH=os.environ.get('PATH', ''),
               PYTHIA_ALLOWED_GPUS=','.join(map(str,gpus)), PYTHIA_NVCTL_PORT=str(port), PYTHONUNBUFFERED='1')
    clock = child = None
    with (runtime/(model+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            raise RuntimeError('Existing status; inspect before resuming')
        try:
            status('starting_clock', gpus=gpus)
            with (runtime/(model+'-clock.log')).open('ab') as log:
                clock = subprocess.Popen([str(NATIVE), str(ROOT/'services/nvcontrold.py')], cwd=ROOT, env=env, stdout=log, stderr=log)
            for _ in range(60):
                if clock.poll() is not None:
                    raise RuntimeError('Clock daemon exited')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                        health = json.load(response)
                    if health['allowed'] != gpus:
                        raise RuntimeError('Clock daemon allocation mismatch')
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('Clock daemon not ready')
            for rate in cfg['rates']:
                out = Path(cfg['runs'][str(rate)])
                out.mkdir(parents=True, exist_ok=True)
                for phase in ('prepare', 'search'):
                    status(phase, rps=rate)
                    with (out/(model+'.log')).open('ab') as log:
                        child = subprocess.Popen([str(PYTHON), __file__, phase, cfg['config_path'], model, str(rate)],
                                                 cwd=ROOT, env=env, stdout=log, stderr=log)
                    if child.wait() != 0:
                        raise RuntimeError(f'{model} RPS{rate} {phase} failed; inspect model log')
                    child = None
                record['completed_rates'].append(rate)
            status('complete')
        except BaseException as error:
            status('needs_attention', error=repr(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=180)
            if clock is not None and clock.poll() is None:
                clock.terminate()
                clock.wait(timeout=30)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    action, config_path, model, *rest = sys.argv[1:]
    cfg = o.read(config_path)
    cfg['config_path'] = str(Path(config_path).resolve())
    if action == 'run':
        model_run(cfg, model)
    elif action == 'prepare':
        prepare(cfg, model, int(rest[0]))
    elif action == 'search':
        search(cfg, model, int(rest[0]))
    else:
        raise SystemExit('Expected run, prepare or search')

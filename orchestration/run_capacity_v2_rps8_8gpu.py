"""Run latest capacity_v2 sequentially on both models with inherited RPS8 Max."""
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
OUT = ROOT / 'results' / os.environ.get('MOE_RPS8_EXPERIMENT', 'rps8-200-capacity-v2-latest-8gpu')
REVISION = os.environ.get('MOE_RPS8_REVISION', 'neighbor_reduction')
BASELINES = (Path(__file__).resolve().parents[2] / 'baselines-rps8-restored/MOE_DVFS-rps8-200-8gpu/results/rps8-200')
PYTHON = ROOT / 'bo_dse/.venv/bin/python'
NATIVE = ROOT / '.venv-official/bin/python'
sys.path.insert(0, str(ROOT / 'bo_dse'))
import official as o


def command(model):
    return [str(PYTHON), str(ROOT/'bo_dse/official.py'), 'start',
            '--directory', str(OUT/model),
            '--model', 'qwen36' if model == 'qwen' else 'deepseek-v2-lite',
            '--gpus', *map(str, range(8)), '--rps', '8',
            '--calibration', str(BASELINES/model/'inputs/calibration-source.jsonl'),
            '--heldout', str(BASELINES/model/'inputs/heldout.jsonl'),
            '--evaluation-requests', '200', '--evaluations', '17', '--gpu-hours', '64',
            '--microbatches', '2', '--seeds', '0', '--direct-search',
            '--output-validation', 'request_completion', '--slo-mode', 'relative_max',
            '--tbt-slo', '--resident-search', '--exploration-policy', 'capacity_v2',
            '--api-port', '24000', '--afd-port', '25000', '--dp-rpc-port', '34000',
            '--clock-url', 'http://127.0.0.1:9098', '--reuse-max-directory', str(BASELINES/model), '--prepare-only']



def model_run(model):
    directory = OUT/model
    config = o.read(directory/'official-config.json')
    o.validate_context(config)
    original_tell = o.tell

    def tell(path, result):
        value = original_tell(path, result)
        state = o.read(Path(path)/'state.json')
        o.write(OUT/(model+'-progress.json'), dict(
            model=model, attempts=len(state['observations']),
            successful=sum(x['status']=='ok' for x in state['observations']),
            last_status=result['status'], last_failure=result.get('failure_reason'),
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        reason = result.get('failure_reason', '')
        if result['status'] != 'ok' and any(x in reason for x in (
            'Official plugin', 'Official installed runtime', 'Installed runtime differs',
            'Installed plugin differs', 'No such file or directory', 'ModuleNotFoundError',
            'Clock daemon protocol/allocation mismatch')):
            raise RuntimeError('Infrastructure failure recorded; stopping: '+reason)
        return value

    o.tell = tell
    try:
        report = o.run(config)
    finally:
        o.tell = original_tell
    state = o.read(directory/'campaign/state.json')
    if len(state['observations']) != 16 or state['pending']:
        raise RuntimeError('Search did not finish all 16 attempts')
    o.write(OUT/(model+'-summary.json'), report)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  method='capacity_v2', revision=REVISION, gpus=list(range(8)), rps=8, evaluation_requests=200,
                  search_attempts_per_model=16, completed_models=[])
    clock = child = None

    def status(phase, **fields):
        record.update(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
        o.write(OUT/'status.json', record)
        print(json.dumps(record), flush=True)

    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT/'status.json').exists():
            raise RuntimeError('Existing queue status; inspect before resuming')
        try:
            predecessor = os.environ.get('MOE_RPS8_PREDECESSOR')
            if predecessor:
                previous = Path(predecessor)
                status('waiting_for_previous_experiment', predecessor=str(previous))
                while True:
                    old = o.read(previous)
                    if old['phase'] != 'complete' and (
                            old.get('boot_id') != record['boot_id'] or
                            not Path('/proc/' + str(old.get('pid'))).exists()):
                        raise RuntimeError('Previous queue stopped before completion; inspect its status')
                    if old['phase'] == 'needs_attention':
                        raise RuntimeError('Previous experiment needs attention; refusing concurrent GPU use')
                    if old['phase'] == 'complete':
                        # Its finalizer must also release the clock daemon port.
                        try:
                            with urllib.request.urlopen('http://127.0.0.1:9098/health', timeout=1):
                                pass
                        except OSError:
                            break
                    time.sleep(10)
            status('starting_clock')
            env = dict(os.environ, PATH=os.environ.get('PATH', ''), PYTHIA_ALLOWED_GPUS=','.join(map(str, range(8))),
                       PYTHIA_NVCTL_PORT='9098', PYTHONUNBUFFERED='1')
            with (OUT/'clock.log').open('ab') as log:
                clock = subprocess.Popen([str(NATIVE), str(ROOT/'services/nvcontrold.py')],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            for _ in range(60):
                if clock.poll() is not None:
                    raise RuntimeError('Clock daemon exited; inspect clock.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:9098/health', timeout=1) as response:
                        health = json.load(response)
                    if health['allowed'] != list(range(8)):
                        raise RuntimeError('Clock daemon GPU allocation mismatch')
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('Clock daemon did not become ready')
            for model in ('qwen', 'deepseek'):
                for phase, cmd in [('preparing_with_inherited_reference', command(model)),
                                   ('running_search', [str(PYTHON), __file__, model])]:
                    status(phase, active_model=model)
                    with (OUT/(model+'.log')).open('ab') as log:
                        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if child.wait() != 0:
                        raise RuntimeError(model+' '+phase+' failed; inspect '+model+'.log')
                    child = None
                record['completed_models'].append(model)
            status('complete', active_model=None)
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
    if sys.argv[1:] == ['run']:
        run()
    elif len(sys.argv) == 2 and sys.argv[1] in ('qwen', 'deepseek'):
        model_run(sys.argv[1])
    else:
        raise SystemExit('usage: run_capacity_v2_rps8_8gpu.py run|qwen|deepseek')

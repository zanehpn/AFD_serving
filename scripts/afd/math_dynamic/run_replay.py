"""Run the existing causal replay client with an isolated mathematical controller."""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

AFD = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--clock-url', required=True)
    args, replay_args = p.parse_known_args()
    if replay_args and replay_args[0] == '--':
        replay_args.pop(0)
    directory = args.case_dir; directory.mkdir(parents=True, exist_ok=True)
    events, stop, ready = (directory / name for name in ('events.jsonl', 'controller.stop', 'controller.ready'))
    if any(f.exists() for f in (events, stop, ready)):
        raise FileExistsError('Dynamic replay state already exists')
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    fifo = directory / 'submit.fifo'
    controller = replay = None
    result = 1
    try:
        controller = subprocess.Popen([sys.executable, str(AFD / 'math_dynamic/controller.py'),
            '--config', str(args.config), '--events', str(events), '--actions', str(directory / 'controller-actions.jsonl'),
            '--summary', str(directory / 'controller-summary.json'), '--stop', str(stop), '--ready', str(ready),
            '--clock-url', args.clock_url, '--submit-fifo', str(fifo)])
        deadline = time.monotonic() + 60
        while not ready.exists():
            if controller.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Controller did not become ready')
            time.sleep(.05)
        replay = subprocess.Popen([sys.executable, str(AFD / 'causal_dvfs/replay_client.py'), *replay_args,
            '--events', str(events), '--submit-signal-fifo', str(fifo), '--progress-event-mode', 'batched', '--progress-event-interval-ms', '250'])
        while replay.poll() is None:
            if controller.poll() is not None:
                summary_path = directory / 'controller-summary.json'
                if controller.returncode == 0 and summary_path.exists() and json.loads(summary_path.read_text()).get('replay_ended'):
                    replay.wait(timeout=10)
                    break
                raise RuntimeError('Controller exited before replay')
            time.sleep(.05)
        result = replay.returncode
        if result:
            raise RuntimeError(f'Replay failed: {result}')
        if controller.wait(timeout=60):
            raise RuntimeError('Controller failed')
        summary = json.loads((directory / 'controller-summary.json').read_text())
        if not (summary['status'] == 'complete' and summary['replay_ended'] and summary['restored_to_guard'] and summary['outstanding'] == 0):
            raise RuntimeError('Incomplete controller lifecycle')
    finally:
        stop.touch()
        for process in (replay, controller):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
    return result

if __name__ == '__main__':
    raise SystemExit(main())

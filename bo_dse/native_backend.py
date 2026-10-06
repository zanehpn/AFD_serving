#!/usr/bin/env python3
"""Native-environment helpers; no BO imports or scikit-learn dependency."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot(run, cell):
    paths = sorted(run.glob('stage-*.jsonl'))
    if not paths:
        raise ValueError('No stage traces were emitted by the backend')
    previous = None
    deadline = time.monotonic() + 30
    while True:
        current = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        if current == previous:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Stage writers did not settle after replay')
        previous = current
        time.sleep(.5)
    target = cell / 'traces'
    target.mkdir(exist_ok=False)
    for path in paths:
        shutil.copyfile(path, target / path.name)
    if current != [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]:
        raise ValueError('Stage files changed during snapshot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--cell', type=Path, required=True)
    p = sub.add_parser('validate')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    sub.add_parser('verify-platform')
    args = parser.parse_args()
    if args.action == 'snapshot':
        snapshot(args.run, args.cell)
    elif args.action == 'verify-platform':
        expected = json.loads((ROOT / 'environment/native-platform.json').read_text())['nvidia_smi']
        actual = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total',
                                          '--format=csv,noheader,nounits'], text=True)
        if actual != expected:
            raise ValueError('Live GPU identity/driver differs from the frozen platform')
    else:
        # Runs in its own process: legacy static_dse cannot shadow the BO package.
        sys.path.insert(0, str(ROOT / 'migration'))
        spec = importlib.util.spec_from_file_location('native_static_adapter', ROOT / 'migration/static_dse.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        plan = json.loads(args.plan.read_text())
        result = native.collect_entry(plan, plan['schedule'][0])
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()

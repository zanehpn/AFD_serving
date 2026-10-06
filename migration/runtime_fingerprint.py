#!/usr/bin/env python3
"""Record/check native software versions without importing CUDA or changing GPUs."""
import argparse
import importlib.metadata as md
import json
import platform
from pathlib import Path


def snapshot():
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in md.distributions()}
    # The plugin is supplied through a separately frozen Git tree.
    packages.pop('vllm-afd-plugin', None)
    return {'runtime': 'native', 'python': platform.python_version(),
            'machine': platform.machine(), 'packages': dict(sorted(packages.items()))}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--check', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    data = snapshot()
    if args.check:
        if data != json.loads(args.check.read_text()):
            raise SystemExit('Native environment changed since preparation; refuse to mix runs.')
    elif args.output:
        if args.output.exists() and json.loads(args.output.read_text()) != data:
            raise SystemExit('Refusing to replace a different environment fingerprint.')
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

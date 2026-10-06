#!/usr/bin/env python3
"""Download upstream code and apply the local-only Git bundle, keeping exact commits."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(list(map(str, args)), text=True).strip()


def main():
    lock = json.loads((ROOT / 'environment/plugins.lock.json').read_text())
    bundle = ROOT / lock['bundle']
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == lock['bundle_sha256']
    upstream = ROOT / 'third_party/afd-upstream'
    upstream.parent.mkdir(exist_ok=True)
    if not upstream.exists():
        run('git', 'clone', '--filter=blob:none', '--no-checkout', lock['upstream_url'], upstream)
    run('git', '-C', upstream, 'fetch', 'origin', lock['base_commit'])
    # Materialize base blobs before cloning this partial repository locally.
    run('git', '-C', upstream, 'checkout', '--detach', lock['base_commit'])
    run('git', '-C', upstream, 'bundle', 'verify', bundle)
    run('git', '-C', upstream, 'fetch', bundle, 'HEAD:refs/heads/ecodep-migration')
    for name, ref in lock.get('target_refs', {}).items():
        run('git', '-C', upstream, 'fetch', bundle, f'{ref}:refs/heads/{name}')
    for name, commit in lock['targets'].items():
        target = ROOT / 'third_party' / name
        if not target.exists():
            run('git', 'clone', '--no-hardlinks', '--no-checkout', upstream, target)
            run('git', '-C', target, 'checkout', '--detach', commit)
        assert run('git', '-C', target, 'rev-parse', 'HEAD') == commit
        assert not run('git', '-C', target, 'status', '--short')
        print(name, commit)


if __name__ == '__main__':
    main()

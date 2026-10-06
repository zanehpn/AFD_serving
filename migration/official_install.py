"""Install exactly the official AFD commit, building outside the source tree."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / 'environment/official-runtime.lock.json').read_text())
    destination = ROOT / 'third_party/afd-plugin-official'
    if destination.exists():
        raise FileExistsError('Refusing to replace an existing plugin checkout')
    destination.parent.mkdir(exist_ok=True)
    subprocess.run(['git', 'clone', lock['plugin_url'], str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', lock['plugin_commit']], check=True)
    with tempfile.TemporaryDirectory(prefix='official-afd-build-') as temporary:
        build = Path(temporary) / 'source'
        shutil.copytree(destination, build)
        env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_AFD_PLUGIN='0.26.0.dev0')
        subprocess.run([str(ROOT / '.bootstrap-official/bin/uv'), 'pip', 'install', '--python', sys.executable,
                        '--no-deps', '--no-build-isolation', str(build)], env=env, check=True)
    sys.path.insert(0, str(ROOT / 'bo_dse'))
    from official_worker import runtime_identity
    record = runtime_identity()
    record['install_lock'] = lock
    with (ROOT / 'environment/OFFICIAL_INSTALLED.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()

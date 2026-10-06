"""Archive current source and experiment bytes, including unfinished runs."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone

repo = Path(__file__).resolve().parents[1]
stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M%S') + '-capacity-feedback-progress'
out = repo / 'experiment-records' / stamp
out.mkdir(parents=True)
manifest = {'snapshot_utc': datetime.now(timezone.utc).isoformat(),
            'complete': False, 'files': [], 'archives': [],
            'note': 'Live file-by-file snapshot; unfinished trials included. Not an atomic checkpoint.'}
secrets = re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
skip = {'.git', '.venv', '.venv-official', '.bootstrap', '.bootstrap-official', '__pycache__', '.pytest_cache', 'third_party', 'artifacts', 'experiment-records'}
extensions = {'.py', '.sh', '.md', '.json', '.jsonl', '.toml', '.txt', '.patch', '.yaml', '.yml', '.csv', '.log'}
def digest(data):
    return hashlib.sha256(data).hexdigest()

for root in [repo, (Path(__file__).resolve().parents[2] / 'MOE_DVFS-capacity-v2-8gpu'), (Path(__file__).resolve().parents[2] / 'MOE_DVFS-capacity-v2-feedback'),
             (Path(__file__).resolve().parents[2] / 'MOE_DVFS-capacity-v2-neighbor-reduction')]:
    if not root.exists():
        continue
    name = root.name + '-current'
    with tempfile.TemporaryFile() as compressed:
        with tarfile.open(fileobj=compressed, mode='w:gz') as archive:
            for p in sorted(root.rglob('*')):
                rel = p.relative_to(root)
                if any(x in skip or x.startswith('.') for x in rel.parts) or p.is_symlink() or not p.is_file():
                    continue
                if 'results' not in rel.parts and p.suffix not in extensions:
                    continue
                if p.suffix in {'.lock', '.tmp', '.pyc'}:
                    continue
                data = p.read_bytes()
                if secrets.search(data):
                    raise ValueError('Credential-like content: ' + str(p))
                member = root.name + '/' + str(rel)
                info = tarfile.TarInfo(member)
                info.size = len(data)
                info.mode = p.stat().st_mode & 0o777
                archive.addfile(info, io.BytesIO(data))
                manifest['files'].append({'archive': name, 'source': str(p), 'member': member, 'bytes': len(data), 'sha256': digest(data)})
        compressed.seek(0)
        parts = []
        combined = hashlib.sha256()
        while data := compressed.read(24 * 1024 * 1024):
            filename = name + f'.tar.gz.part{len(parts):03d}'
            (out / filename).write_bytes(data)
            combined.update(data)
            parts.append({'file': filename, 'bytes': len(data), 'sha256': digest(data)})
        manifest['archives'].append({'name': name, 'sha256': combined.hexdigest(), 'parts': parts})
        print(name, sum(p['bytes'] for p in parts), flush=True)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
shutil.copy2(repo / 'experiment-records/2026-09-11-163050-capacity-v2-complete/restore.py', out / 'restore.py')
(out / 'README.md').write_text('# Current capacity feedback experiment snapshot\n\nContains current repository source, original eight-GPU capacity experiment source/results, and Qwen joint-feedback source/results including raw logs and measurements. The Qwen feedback experiment was still running at snapshot time; this is partial progress, not a completed experiment. Files are captured individually while the experiment runs. Historical and current campaigns overlap; do not concatenate them. Dependencies, model weights, caches and credentials are excluded.\n\nRestore and verify every archived file with `python3 restore.py restored-records`. See manifest.json for capture time and SHA-256 checksums.\n')
with tempfile.TemporaryDirectory(prefix='feedback-verify-') as target:
    subprocess.run(['python3', str(out / 'restore.py'), target], check=True)
print(out, flush=True)

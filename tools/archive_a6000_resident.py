"""Export a search-boundary snapshot and isolated resident diagnostic evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
RUN = REPO/'results/a6000-p2p-pstates-20260915'
DIAG = REPO/'results/resident-audit-20260915'
OUT = REPO/'experiment-records/2026-09-15-a6000-resident-audit'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pack(paths, destination, base):
    records = {}
    with tarfile.open(destination, 'w:gz', compresslevel=6, dereference=False) as tar:
        for p in sorted(paths):
            if any(x in p.parts for x in ('.git', '__pycache__', '.pytest_cache')):
                continue
            if not p.is_file() or p.is_symlink():
                continue
            name = str(p.relative_to(base))
            before = digest(p)
            tar.add(p, arcname=name, recursive=False)
            if digest(p) != before:
                raise RuntimeError(f'File changed during snapshot: {p}')
            records[name] = before
    # Verify the actual compressed export, not only its input files.
    with tarfile.open(destination, 'r:gz') as tar:
        for name, expected in records.items():
            member = tar.extractfile(name)
            if hashlib.sha256(member.read()).hexdigest() != expected:
                raise RuntimeError(f'Archive verification failed: {name}')
    return {'archive': destination.name, 'sha256': digest(destination),
            'size_bytes': destination.stat().st_size, 'files_sha256': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['boundary', 'diagnostic'])
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    if stage == 'boundary':
        status = json.loads((DIAG/'STATUS.json').read_text())
        if status['status'] != 'validating':
            raise RuntimeError('Search must be paused at the acquired diagnostic boundary')
        frozen = json.loads((DIAG/'search-boundary-before.json').read_text())
        if any(digest(Path(p)) != h for p,h in frozen.items()):
            raise RuntimeError('Search boundary differs from diagnostic snapshot')
        paths = list((RUN/'source').rglob('*')) + list((RUN/'search').rglob('*'))
        paths += [p for p in RUN.iterdir() if p.is_file() and p.suffix in ('.json', '.py', '.md')
                  and p.name not in ('COMPARISON.json', 'COMPARISON.md')]
        record = pack(paths, OUT/'search-boundary.tar.gz', RUN)
        if any(digest(Path(p)) != h for p,h in frozen.items()):
            raise RuntimeError('Search advanced during snapshot')
        record['search_boundary'] = frozen
        record['scope'] = 'Current A6000 source and search files; models, installed environments, and upstream git metadata remain external dependencies. Absolute paths are preserved; this is evidence, not an automatic cross-host resume package.'
    else:
        status = json.loads((DIAG/'STATUS.json').read_text())
        if status['status'] not in ('validation_complete', 'failed') or not status.get('finished_at'):
            raise RuntimeError('Diagnostic must be finalized')
        record = pack(DIAG.rglob('*'), OUT/'resident-diagnostic.tar.gz', DIAG)
        record['scope'] = 'Independent calibration pilot; no search observations added.'
        shutil.copyfile(DIAG/'STATUS.json', OUT/'STATUS.json')
    record['created_at'] = datetime.now(timezone.utc).isoformat()
    (OUT/f'{stage}-manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('files_sha256', 'search_boundary')}, indent=2))


if __name__ == '__main__':
    main()

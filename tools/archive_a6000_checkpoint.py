"""Export current A6000 runs and verify every archived byte before publication."""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from publish_dbo_completed_scene import checked, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    results = root / 'results'

    def boundary():
        return {str(p.relative_to(root)): sha(checked(p))
                for p in sorted(results.glob('*/search/*/comparison/*/state.json'))}

    before = boundary()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    code_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    files=[], links=[], exclusions=[], parts=[])
    files = set()
    bases = [results, root / 'third_party/afd-plugin-official', root / 'tools']
    for base in bases:
        for p in base.rglob('*'):
            rel = p.relative_to(root)
            if any(part in ('.git', '__pycache__', '.pytest_cache') for part in rel.parts):
                continue
            if p.is_symlink():
                manifest['links'].append(dict(member=str(rel), target=str(p.readlink())))
            elif p.is_file():
                if p.suffix in ('.pyc', '.o', '.lock', '.tmp'):
                    manifest['exclusions'].append(str(rel))
                else:
                    files.add(p)
    files.update(p for p in (root / 'environment').glob('*') if p.is_file() and not p.is_symlink())
    operational = sorted(Path(tempfile.gettempdir()).glob('*a6000*.py'))
    entries = [(p, str(p.relative_to(root))) for p in sorted(files)]
    entries += [(p, 'operational-scripts/' + p.name) for p in operational]
    name = 'a6000-checkpoint.tar.gz'
    active = results / 'a6000-8gpu-rps8-16-20260916-nccl'
    visible = {str(p.relative_to(root)): p.name for p in
               [active / 'STATUS.json', active / 'SUPERVISOR.json',
                results / 'A6000_AFTER_BASELINES.json', results / 'CHAIN_STATUS.json']}
    captured = {}
    with tempfile.TemporaryFile() as compressed:
        with tarfile.open(fileobj=compressed, mode='w:gz', compresslevel=6) as tar:
            for index, (p, member) in enumerate(entries):
                data = checked(p)
                if member in visible:
                    captured[visible[member]] = data
                info = tarfile.TarInfo(member)
                info.size, info.mode = len(data), p.stat().st_mode & 0o777
                tar.addfile(info, io.BytesIO(data))
                manifest['files'].append(dict(member=member, size=len(data), sha256=sha(data)))
                if index % 500 == 0:
                    print(f'Archived {index}/{len(entries)} files', flush=True)
        assert boundary() == before, 'Search states changed during snapshot'
        compressed.seek(0)
        expected = {f['member']: f for f in manifest['files']}
        verified = 0
        with tarfile.open(fileobj=compressed, mode='r:gz') as tar:
            for member in tar:
                data = tar.extractfile(member).read()
                assert sha(data) == expected[member.name]['sha256'], member.name
                verified += 1
        assert verified == len(expected)
        compressed.seek(0)
        digest = hashlib.sha256()
        while chunk := compressed.read(24 * 1024 * 1024):
            part = f'{name}.part{len(manifest["parts"]):03d}'
            (dest / part).write_bytes(chunk)
            manifest['parts'].append(dict(file=part, size=len(chunk), sha256=sha(chunk)))
            digest.update(chunk)
        manifest['archive_sha256'] = digest.hexdigest()
    assert boundary() == before, 'Search states changed during verification'
    write(dest / 'manifest.json', manifest)
    write(dest / 'search-boundary.json', before)
    summary = dict(snapshot_utc=manifest['created_at'], scenes={},
                   baseline_observations=0, baseline_failures=0,
                   heldout_evaluated=False, confirmation_evaluated=False)
    for scene in sorted((active / 'search').iterdir()):
        methods = {}
        for p in sorted(scene.glob('comparison/*/state.json')):
            state = json.loads(checked(p))
            obs = state['observations']
            failures = sum(o.get('status') != 'ok' for o in obs)
            methods[p.parent.name] = dict(attempts=len(obs), failures=failures,
                pending=bool(state.get('pending')), frozen=state.get('frozen', False))
            if not p.parent.name.startswith('v2'):
                summary['baseline_observations'] += len(obs)
                summary['baseline_failures'] += failures
        summary['scenes'][scene.name] = methods
    for filename, data in captured.items():
        (dest / filename).write_bytes(data)
    write(dest / 'summary.json', summary)
    write(dest / 'validation.json', dict(status='passed', files_verified=verified,
        archive_readback_sha256_verified=True, search_boundary_unchanged=True,
        credential_pattern_scan='passed', cross_host_resume_tested=False))
    print(json.dumps(dict(destination=str(dest), files=verified,
        compressed_bytes=sum(p['size'] for p in manifest['parts']),
        parts=len(manifest['parts']), summary=summary), indent=2), flush=True)


if __name__ == '__main__':
    main()

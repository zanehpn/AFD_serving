"""Restore the verified base plus completed-result delta; never launch experiments."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restore_archive(folder, manifest, wanted, destination):
    restored = set()
    with tempfile.TemporaryFile() as archive:
        digest = hashlib.sha256()
        for part in manifest['parts']:
            data = (folder / part['file']).read_bytes()
            assert len(data) == part['size'] and sha(data) == part['sha256'], part['file']
            digest.update(data)
            archive.write(data)
        assert digest.hexdigest() == manifest['archive_sha256']
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode='r:gz') as tar:
            for member in tar:
                rel = PurePosixPath(member.name)
                assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
                if member.name not in wanted:
                    continue
                assert member.name not in restored
                data = tar.extractfile(member).read()
                expected = wanted[member.name]
                assert len(data) == expected['size'] and sha(data) == expected['sha256'], member.name
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists(), member.name
                target.write_bytes(data)
                target.chmod(member.mode & 0o777)
                restored.add(member.name)
    assert restored == set(wanted)
    return len(restored)


def main():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'manifest.json').read_text())
    base = here.parents[1] / manifest['base_checkpoint']
    raw = (base / 'manifest.json').read_bytes()
    assert sha(raw) == manifest['base_manifest_sha256']
    destination = Path(sys.argv[1]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    inherited = {row['member']: row for row in manifest['inherited_files']}
    added = {row['member']: row for row in manifest['files']}
    assert not set(inherited) & set(added)
    count = restore_archive(base, json.loads(raw), inherited, destination)
    count += restore_archive(here, manifest, added, destination)
    print(json.dumps(dict(restored_files=count, sha256_verified=True,
                         destination=str(destination), gpu_actions_performed=False)))


if __name__ == '__main__':
    main()

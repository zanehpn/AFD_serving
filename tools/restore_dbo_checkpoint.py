"""Verify files and restore relative internal links; never execute old controllers."""
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def main():
    base = Path(__file__).resolve().parent
    manifest = json.loads((base / 'manifest.json').read_text())
    destination = Path(sys.argv[1]).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def safe(name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unsafe member path: ' + name)
        path = destination / name
        current = path
        while current != destination:
            if current.is_symlink():
                raise ValueError('Refusing path through existing symlink: ' + str(current))
            current = current.parent
        return path

    count = 0
    for archive in manifest['archives']:
        expected = {item['member']: item for item in manifest['files'] if item['archive'] == archive['name']}
        seen = set()
        with tempfile.TemporaryFile() as joined:
            digest = hashlib.sha256()
            for part in archive['parts']:
                data = (base / part['file']).read_bytes()
                if hashlib.sha256(data).hexdigest() != part['sha256']:
                    raise ValueError('Part checksum mismatch: ' + part['file'])
                digest.update(data)
                joined.write(data)
            if digest.hexdigest() != archive['sha256']:
                raise ValueError('Archive checksum mismatch')
            joined.seek(0)
            with tarfile.open(fileobj=joined, mode='r|gz') as tar:
                for member in tar:
                    if not member.isfile() or member.name not in expected or member.name in seen:
                        raise ValueError('Unexpected or repeated member: ' + member.name)
                    data = tar.extractfile(member).read()
                    if hashlib.sha256(data).hexdigest() != expected[member.name]['sha256']:
                        raise ValueError('File checksum mismatch: ' + member.name)
                    path = safe(member.name)
                    if path.exists() and (not path.is_file() or path.read_bytes() != data):
                        raise ValueError('Refusing overwrite: ' + str(path))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    path.chmod(member.mode & 0o777)
                    seen.add(member.name)
                    count += 1
            if seen != set(expected):
                raise ValueError('Missing indexed files')
    for link in manifest.get('links', []):
        # Target paths are checked inside the destination before creating any link.
        safe(link['target_member'])
        safe(link['member'])
    for link in manifest.get('links', []):
        path = destination / link['member']
        target = destination / link['target_member']
        if path.exists() or path.is_symlink():
            raise ValueError('Refusing existing link location: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)
    print('Verified and restored', count, 'files and', len(manifest.get('links', [])), 'internal relative links')
    print('Archive only: do not execute old process-control or resume scripts on another host.')


if __name__ == '__main__':
    main()

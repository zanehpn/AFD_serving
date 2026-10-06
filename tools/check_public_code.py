#!/usr/bin/env python3
"""Scan publishable files without printing credential values or personal data."""
import argparse
import hashlib
import json
from pathlib import Path
import re


RULES = {
    'credential': re.compile(
        r'(?:sk-[A-Za-z0-9_-]{16,}|(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{16,}'
        r'|AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{20,}'
        r'|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
    'authenticated_url': re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
    'credential_assignment': re.compile(
        r'''(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret[_-]?key)'''
        r'''\b["']?\s*[:=]\s*["'][^"'\s]{4,}["']'''),
    'jwt': re.compile(r'eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}'),
    'machine_path': re.compile(r'/(?:home|root|shared|mnt|workspace|tmp)/|/usr/' r'local/'),
    'email': re.compile(r'(?<![\w.+-])[A-Za-z0-9][\w.+-]*@[\w.-]+\.[a-zA-Z]{2,}'),
    'private_address': re.compile(
        r'\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))'
        r'\.\d{1,3}\.\d{1,3}\b'),
}
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv',
            '.venv-official', '.bootstrap', '.bootstrap-official', 'third_party',
            'artifacts', 'results', 'parallel_runs'}


def scan(root):
    findings, binaries = [], []
    count = 0
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in EXCLUDED or part.startswith('.venv-') for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if path.is_symlink():
            findings.append({'path': relative.as_posix(), 'rule': 'symlink_needs_review'})
            continue
        count += 1
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            binaries.append({'path': relative.as_posix(),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for rule, pattern in RULES.items():
                if rule == 'private_address' and re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+', line):
                    continue  # A four-component package version is not an address.
                matches = pattern.findall(line)
                if rule == 'email':
                    matches = [value for value in matches if not value.endswith('@example.invalid')]
                if matches:
                    findings.append({'path': relative.as_posix(), 'line': number, 'rule': rule})
    return {'files_scanned': count, 'findings': findings, 'binary_files_for_separate_review': binaries,
            'scope': 'Working files; excludes Git metadata, caches, environments and generated results. '
                     'Pattern matching is not proof that every possible secret has been detected.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = scan(args.root)
    print(json.dumps(report, indent=2))
    return int(bool(report['findings']))


if __name__ == '__main__':
    raise SystemExit(main())

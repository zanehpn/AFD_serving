#!/usr/bin/env python3
"""Fetch original pinned model snapshots; no weights are stored in Git."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
for model in json.loads((ROOT / 'environment/models.lock.json').read_text()):
    snapshot_download(repo_id=model['repo_id'], revision=model['revision'],
                      local_dir=str(ROOT / 'artifacts/models' / model['name']))

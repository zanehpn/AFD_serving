#!/usr/bin/env python3
"""Create or verify one shared primitive profile before any case model launches."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/afd'))
from general_dse import engine,primitives


def ensure(profile,gpus):
    profile=Path(profile).resolve()
    if not profile.exists():
        plan=profile.parent/'PLAN.json';raw=profile.parent/'MEASUREMENTS.json'
        tool=[sys.executable,str(ROOT/'migration/general_dse.py')]
        if not plan.exists():
            subprocess.run([*tool,'probe-plan','--gpus',','.join(map(str,gpus)),
                '--frequencies','810,1050,1290,1410','--output',str(plan)],check=True)
        planned=json.loads(plan.read_text());primitives.verify_sources(planned)
        if planned['physical_gpus']!=gpus:
            raise ValueError('Existing primitive plan has another allocation; use a separate profile path')
        if not raw.exists():
            subprocess.run([*tool,'probe-run',str(plan),'--output',str(raw)],check=True)
        subprocess.run([*tool,'build-hardware',str(raw),'--output',str(profile)],check=True)
    h=json.loads(profile.read_text());engine.validate_hardware(h);primitives.verify_sources(h)
    if not h.get('sources') or not set(gpus)<={g['id'] for g in h['gpus']}:
        raise ValueError('Shared profile lacks measured sources or requested GPUs')
    actual = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    uuids = {int(row[0]):row[1].strip() for row in csv.reader(io.StringIO(actual))}
    expected = {g['id']:g.get('uuid') for g in h['gpus']}
    if any(expected[i] != uuids.get(i) for i in gpus):
        raise ValueError('Portable profile GPU UUIDs differ from this allocation; measure this server before launching')
    return profile


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--gpus',required=True)
    a=p.parse_args();print(ensure(a.profile,list(map(int,a.gpus.split(',')))))
if __name__=='__main__':main()

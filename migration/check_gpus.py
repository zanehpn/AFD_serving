#!/usr/bin/env python3
"""Check the hardware family supported by the inherited A100 policy."""
import argparse
import csv
import io
import subprocess


def validate(ids, rows, allowed_counts=(4,)):
    if len(ids) not in allowed_counts or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
        raise ValueError(f'Require distinct physical GPU indices, count in {allowed_counts}')
    found = {int(r[0]): (r[1].strip(), float(r[2])) for r in rows}
    chosen = [found[i] for i in ids]
    if len({name for name, _ in chosen}) != 1:
        raise ValueError('All selected GPUs must have the same model')
    if any('A100-SXM4-80GB' not in name or memory < 80000 for name, memory in chosen):
        raise ValueError('Inherited policy supports A100-SXM4-80GB only; another GPU family needs a new calibration/profile protocol')
    return chosen


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--gpus', required=True)
    p.add_argument('--bo-allocation', type=int, choices=(4, 6, 8))
    args = p.parse_args()
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'], text=True)
    counts = (args.bo_allocation,) if args.bo_allocation else (4,)
    print(validate([int(x) for x in args.gpus.split(',')], csv.reader(io.StringIO(output)), counts))


if __name__ == '__main__':
    main()

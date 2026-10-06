"""Read-only measurement summary; never call ask/tell or change experiment state."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def feasible(observation, limits):
    metrics = observation.get('metrics', {})
    if observation.get('status') != 'ok':
        return False
    for key, limit in limits.items():
        value = metrics.get('output_tps' if key == 'min_output_tps' else key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        if (value < limit) if key == 'min_output_tps' else (value > limit):
            return False
    return isinstance(metrics.get('energy_j'), (int, float)) and math.isfinite(metrics['energy_j'])


def report(root):
    plan = read(root/'PLAN.json')
    rows, trials = [], []
    for job in plan['jobs']:
        settings_path = root/'search'/job['id']/'campaign-settings.json'
        limits = read(settings_path)['limits'] if settings_path.exists() else job['limits']
        for method in plan['methods']:
            directory = root/'search'/job['id']/'comparison'/f'{method}-seed0'
            state = read(directory/'state.json') if (directory/'state.json').exists() else {}
            observations = state.get('observations', [])
            passing = [o for o in observations if feasible(o, limits)]
            best = min(passing, key=lambda o: o['metrics']['energy_j']) if passing else None
            rows.append(dict(scenario=job['id'], method=method, attempts=len(observations),
                             failures=sum(o.get('status') != 'ok' for o in observations),
                             feasible=len(passing), best=best, pending=bool(state.get('pending'))))
            for iteration, observation in enumerate(observations, 1):
                metrics = observation.get('metrics', {})
                trials.append(dict(scenario=job['id'], method=method, iteration=iteration,
                    candidate_id=observation['candidate_id'], status=observation['status'],
                    feasible=feasible(observation, limits),
                    **{key: metrics.get(key) for key in ('energy_j', 'ttft_ms', 'tpot_ms', 'tbt_ms', 'output_tps')}))
    summary = dict(updated_at=datetime.now(timezone.utc).isoformat(),
                   scope='New A6000 calibration measurements only; historical A100 measurements remain separate',
                   search_attempts=sum(row['attempts'] for row in rows), heldout_evaluated=False, rows=rows)
    tmp = root/'COMPARISON.tmp'
    tmp.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    tmp.replace(root/'COMPARISON.json')
    fields = ['scenario', 'method', 'iteration', 'candidate_id', 'status', 'feasible',
              'energy_j', 'ttft_ms', 'tpot_ms', 'tbt_ms', 'output_tps']
    with (root/'trials.csv.tmp').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trials)
    (root/'trials.csv.tmp').replace(root/'trials.csv')
    lines = ['# A6000 四方法搜索进度', '', f"更新时间：{summary['updated_at']}", '',
             '仅汇总本机 calibration；旧 A100 结果单独保留。尚未执行 heldout。', '',
             '| 场景 | 方法 | 搜索尝试 | 失败 | 满足当前 SLO | 最低可行能耗 J |',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        energy = f"{row['best']['metrics']['energy_j']:.2f}" if row['best'] else '—'
        lines.append(f"| {row['scenario']} | {row['method']} | {row['attempts']}/16 | {row['failures']} | {row['feasible']} | {energy} |")
    (root/'COMPARISON.md.tmp').write_text('\n'.join(lines)+'\n')
    (root/'COMPARISON.md.tmp').replace(root/'COMPARISON.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        report(args.directory)
        status = read(args.directory/'STATUS.json')
        if not args.watch or status['status'] != 'running':
            break
        time.sleep(30)


if __name__ == '__main__':
    main()

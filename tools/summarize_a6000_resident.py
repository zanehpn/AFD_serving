"""Summarize the bounded resident pilot without changing any search result."""
import json
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[1] / 'results/resident-audit-20260915')


def main():
    status = json.loads((ROOT/'STATUS.json').read_text())
    if status['status'] != 'validation_complete' or not status.get('queue_resumed'):
        raise RuntimeError('Four successful measurements and queue resumption are required')
    rows = {r['name']: r for r in status['rows']}
    cold = rows['01-cold-a']
    warm = rows['03-resident-warm-a']
    cold_s, warm_s = cold['cost']['wall_seconds'], warm['cost']['wall_seconds']
    reused = []
    for name in ['02-resident-cold-b', '03-resident-warm-a']:
        reused.append(json.loads((ROOT/name/'resident-services.json').read_text())['services'])
    identity_fields = ('pid', 'start_ticks', 'boot_id')
    same_services = all(all(reused[0][role][f] == reused[1][role][f] for f in identity_fields)
                        for role in ('service-attention.json', 'service-ffn.json'))
    summary = {
        'scope': 'DeepSeek RPS4, one matched cold/reused pair; calibration only, no significance claim',
        'cold_a_seconds': cold_s, 'reused_a_seconds': warm_s,
        'saved_seconds_per_compatible_trial': cold_s-warm_s,
        'elapsed_reduction_pct': 100*(cold_s-warm_s)/cold_s,
        'speedup': cold_s/warm_s,
        'same_service_processes': same_services,
        'all_requests_completed': all(r['completed_requests'] == 200 for r in rows.values()),
        'all_telemetry_valid': all(r['telemetry_valid'] for r in rows.values()),
        'search_unchanged': status['search_unchanged'],
        'queue_resumed': status['queue_resumed'],
        'reservation_cost': status.get('reservation_cost'),
        'metrics_change_pct': {k: 100*(warm['metrics'][k]/cold['metrics'][k]-1)
                               for k in ('ttft_ms', 'tpot_ms', 'output_tps', 'energy_j')},
        'cold_b_seconds': rows['04-cold-b']['cost']['wall_seconds'],
        'resident_cold_b_seconds': rows['02-resident-cold-b']['cost']['wall_seconds'],
        'rollout': 'Current search remains cold-start. Reuse requires a separately frozen, consistent cost protocol.',
    }
    (ROOT/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# A6000 常驻服务小规模验证结果', '',
             'DeepSeek RPS4；每次 8 条预热、200 条 calibration；结果未进入搜索优化器。', '',
             '| 测量 | 总耗时（秒） | 服务启动（秒） | 预热及回放（秒） |',
             '|---|---:|---:|---:|']
    for r in status['rows']:
        lines.append(f"| {r['name']} | {r['cost']['wall_seconds']:.2f} | {r['phases'].get('start_services',0):.2f} | {r['phases'].get('measure_requests',0):.2f} |")
    lines += ['', f"相同 A 配置冷启动对比常驻复用：节省 {cold_s-warm_s:.2f} 秒（{summary['elapsed_reduction_pct']:.1f}%），单次加速 {summary['speedup']:.2f} 倍。", '',
              f"进程身份确实复用：{same_services}。全部请求完成且遥测有效：{summary['all_requests_completed'] and summary['all_telemetry_valid']}。搜索状态前后相同：{status['search_unchanged']}。队列已恢复：{status['queue_resumed']}。", '',
              '每条件仅一次；该结果衡量兼容配置的单次工程收益，不能外推为整个搜索或 Qwen 的加速倍数。冷启动 B 的两种启动方式仅作额外参考，未进行随机交叉和重复测量。', '',
              '当前四方法搜索仍使用原冷启动协议；未改候选顺序、尝试预算、trace、SLO 或 heldout 状态。整个验证预约期的成本单独保存在 SUMMARY.json 和原始 NVML 记录。', '',
              '遥测及延迟变化见 SUMMARY.json；请求完成不代表输出语义等价。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

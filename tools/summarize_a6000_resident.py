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
    lines = ['# A6000 resident-service pilot results', '',
             'DeepSeek RPS4; eight warmup and 200 calibration requests per measurement. Results were not fed into the search optimizer.', '',
             '| Measurement | Total seconds | Service startup seconds | Warmup and replay seconds |',
             '|---|---:|---:|---:|']
    for r in status['rows']:
        lines.append(f"| {r['name']} | {r['cost']['wall_seconds']:.2f} | {r['phases'].get('start_services',0):.2f} | {r['phases'].get('measure_requests',0):.2f} |")
    lines += ['', f"Cold start versus resident reuse for the same A configuration: saved {cold_s-warm_s:.2f} seconds ({summary['elapsed_reduction_pct']:.1f}%), a {summary['speedup']:.2f}x speedup for this measurement.", '',
              f"Process identities reused: {same_services}. All requests completed and telemetry valid: {summary['all_requests_completed'] and summary['all_telemetry_valid']}. Search state unchanged: {status['search_unchanged']}. Queue resumed: {status['queue_resumed']}.", '',
              'Each condition was measured once. This measures an engineering benefit for compatible configurations, not a speedup for the full search or Qwen. The two cold-start B launch modes are additional references; randomized crossovers and repeated measurements were not performed.', '',
              'The four-method search retains its original cold-start protocol. Candidate order, attempt budgets, traces, SLOs, and held-out status are unchanged. Costs over the complete reserved validation period are recorded separately in SUMMARY.json and raw NVML records.', '',
              'See SUMMARY.json for telemetry and latency changes. Request completion does not establish semantic output equivalence.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

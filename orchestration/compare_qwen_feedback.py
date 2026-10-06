"""Compare observed feasibility under identical Qwen RPS16 SLOs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bo_dse/scripts/afd'))
from static_dse.optimizer import aggregate, best_measured

OLD = (Path(__file__).resolve().parents[2] / 'MOE_DVFS-capacity-v2-8gpu/results/rps16-200-capacity-v2-8gpu-reuse-max/qwen')
NEW = ROOT/'results/qwen-rps16-200-capacity-v2-feedback/qwen'


def read(path):
    return json.loads(path.read_text())


def compare():
    configs = {label: read(base/'official-config.json') for label, base in [('old', OLD), ('new', NEW)]}
    for key in ('model', 'rps', 'limits', 'budget', 'max_model_len', 'max_output_tokens',
                'evaluation_requests', 'microbatches', 'time_scale', 'output_validation', 'slo_ratios'):
        if configs['old'].get(key) != configs['new'].get(key):
            raise ValueError('Comparison protocol mismatch: '+key)
    if configs['old']['trace']['sha256'] != configs['new']['trace']['sha256']:
        raise ValueError('Comparison trace differs')
    limits = configs['new']['limits']
    reference = read(NEW/'inputs/slo-reference.json')['metrics']['energy_j']
    report = dict(rps=16, model='qwen', limits=limits, shared_reference_energy_j=reference,
                  interpretation='Calibration search comparison, one seed; no heldout validation', runs={})
    for label, base in [('old', OLD), ('new', NEW)]:
        state = read(base/'campaign/state.json')
        obs = state['observations']
        def feasible(o):
            return o['status'] == 'ok' and all(
                o['metrics']['output_tps'] >= value if key == 'min_output_tps' else o['metrics'][key] <= value
                for key, value in limits.items())
        best = best_measured(obs, limits)
        if best:
            best['energy_saving_vs_reference_pct'] = 100*(1-best['metrics']['energy_j']/reference)
        report['runs'][label] = dict(directory=str(base), attempts=len(obs),
            complete=len(obs)==16 and state['pending'] is None,
            successful=sum(o['status']=='ok' for o in obs),
            feasible_trials=sum(feasible(o) for o in obs),
            feasible_unique_candidates=sum(x['feasible'] for x in aggregate(obs, limits).values()),
            best=best, pending_trial=(state['pending'] or {}).get('trial_id'),
            minimum_ttft_ms=min((o['metrics']['ttft_ms'] for o in obs if o['status']=='ok'), default=None))
    output = NEW.parent/'comparison-vs-original.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    compare()

"""Read an existing campaign and preview v3; never alter it or execute a trial."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bo_dse/scripts/afd'))
from static_dse.campaign import updated_candidates, costs
from static_dse.optimizer import propose
from static_dse.space import digest, structure

source = Path(sys.argv[1])
bundle = json.loads((source / 'bundle.json').read_text())
state = json.loads((source / 'state.json').read_text())
settings = copy.deepcopy(bundle['settings'])
settings['bo'].update(json.loads((ROOT / 'hierarchical-policy.json').read_text()))
candidates = updated_candidates(bundle, state)
checks = {r['id']: r for r in bundle['audit']['candidates']}
verified = {digest(structure(c)) for c in candidates for o in state['observations']
            if c['id'] == o['candidate_id'] and o['status'] == 'ok'
            and o.get('execution_verified') and o.get('telemetry_valid')}
eligible = {r['id'] for r in checks.values() if r['status'] == 'eligible'}
for c in candidates:
    row = checks[c['id']]
    if (row['status'] != 'hard_rejected'
        and (settings.get('allow_structure_probes') or digest(structure(c)) in verified)
        and not set(row['pending_reasons']) - {'structure_execution_unverified'}):
        eligible.add(c['id'])
remaining = settings['budget']['gpu_hours'] - costs(bundle, state)['gpu_hours']
if costs(bundle, state)['evaluations'] >= settings['budget']['evaluations']:
    candidate, decision = None, {'reason': 'budget_exhausted'}
else:
    candidate, decision = propose(candidates, state['observations'], settings, eligible, remaining)
print(json.dumps(dict(source=str(source), observations=len(state['observations']),
    preview_only=True, source_pending_ignored_for_preview=state['pending'] is not None,
    candidate=candidate, decision=decision), indent=2))

import hashlib
import json
from pathlib import Path
import sys
import urllib.request

BASE = Path(__file__).resolve().parent
OLD = BASE.with_name('dbo_on_0347_v3')
sys.path.insert(0, str(OLD/'source/bo_dse'))
import pynvml as nv
from capture_hardware import capture

hardware = capture([0,1,2,4], nv)
nv.nvmlInit()
try:
    assert all(not nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByIndex(g)) for g in [0,1,2,4])
finally: nv.nvmlShutdown()
with urllib.request.urlopen('http://127.0.0.1:9095/health', timeout=5) as response:
    health = json.load(response)
assert {0,1,2} <= set(health['allowed']), health
boundary = json.loads((BASE/'BOUNDARY.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in boundary['receipts_sha256'].items())
# Retire the old primary GPU allocation without touching other users' jobs.
# Original failures and metrics remain unchanged; only ownership release is audited.
for receipt_path in OLD.with_name('dbo_on_1256_v3').glob('search/*/comparison/*/trials/*/worker-result.json'):
    receipt=json.loads(receipt_path.read_text());resolved=receipt_path.parent/'CLEANUP_RESOLVED.json'
    if not receipt.get('cleanup_required') or resolved.exists():continue
    cleanup=json.loads((receipt_path.parent/'cleanup.json').read_text())
    assert cleanup['errors'] and all(e in ('GPU 5: compute processes remain after cleanup','GPU 6: compute processes remain after cleanup') for e in cleanup['errors']), cleanup
    evidence=[]
    for service_path in receipt_path.parent.glob('service-*.json'):
        service=json.loads(service_path.read_text());assert service.get('stopped'),service
        process=Path('/proc')/str(service['pid'])
        try:
            fields=(process/'stat').read_text().rsplit(')',1)[1].split()
            assert fields[0]=='Z' or int(fields[19])!=int(service['start_ticks']),service
        except FileNotFoundError:pass
        evidence.append({'path':str(service_path),'sha256':hashlib.sha256(service_path.read_bytes()).hexdigest()})
    assert evidence
    report=dict(verified=True,original_receipt={'path':str(receipt_path),'sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest()},
                ownership_scope='Our registered services exited. Prior cleanup reported only outside processes on retired GPUs 5,6; clock/power reset acknowledgements passed.',
                prior_cleanup=cleanup,service_evidence=evidence,old_gpu_controls_changed=False,
                failure_status_preserved=True,physical_allocation_retired=True)
    resolved.write_text(json.dumps(report,indent=2)+'\n')
manifest = dict(reason='User requested 0,3,4,7 -> 0,1,2,4',
                logical_to_physical={'0':0,'3':1,'4':2,'7':4},
                primary_logical_to_physical={'1':0,'2':1,'5':2,'6':4},
                physical_gpus=[0,1,2,4], hardware=hardware,
                clock_control='DCGM application, same backend as previous RPS8 phase',
                measurements_rerun=False, previous_boundary=str(BASE/'BOUNDARY.json'),
                reporting='GPU UUIDs changed at this boundary. Retain all calibration outcomes; report phase differences. Final heldout must freeze physical mapping before evaluation.',
                files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.rglob('*.py')})
path = BASE/'MIGRATION.json'
assert not any(json.loads(f.read_text())['manifest']['path'] == str(path) for f in OLD.glob('search/*/comparison/*/trials/*/physical-execution.json'))
path.write_text(json.dumps(manifest,indent=2)+'\n')
from mapping import transform, load_manifest, map_candidate
from official_space import commands
tested = 0
config_paths=list(OLD.glob('search/*/official-config.json'))+list(OLD.with_name('dbo_on_1256_v3').glob('search/*/official-config.json'))
for config_path in config_paths:
    config = json.loads(config_path.read_text())
    before = json.dumps(config, sort_keys=True)
    # The execution mapping touches these fields only; avoid copying the whole
    # 15,000-candidate search space for each candidate in this preflight.
    execution = {key:config[key] for key in ('gpus','hardware','correctness') if key in config}
    command_settings = {key:value for key,value in config.items() if key not in ('configurations','validation_candidates')}
    physical = transform(execution)
    for candidate in config['configurations']:
        mapped = map_candidate(config['gpus'], candidate)
        assert all(mapped[k] == v for k,v in candidate.items() if k not in ('attention_gpus','expert_gpus'))
        assert sorted(mapped['attention_gpus'] + mapped['expert_gpus']) == sorted(set(mapped['attention_gpus'] + mapped['expert_gpus']))
        assert set(mapped['attention_gpus'] + mapped['expert_gpus']) <= {0,1,2,4}
        launch = commands(dict(command_settings, **physical), mapped)
        for role in ('attention','ffn'):
            assert set(launch[role]['gpus']) <= {0,1,2,4}
            cmd = launch[role]['command']
            assert '--enable-dbo' in cmd
            for flag,key in [('--dbo-decode-token-threshold','dbo_decode_token_threshold'),('--dbo-prefill-token-threshold','dbo_prefill_token_threshold')]:
                assert int(cmd[cmd.index(flag)+1]) == candidate[key]
        tested += 1
    assert json.dumps(config, sort_keys=True) == before
report = dict(passed=True,candidates_checked=tested,prior_receipts_preserved=len(boundary['receipts_sha256']),
              no_gpu_controls_applied=True,physical_gpus=[0,1,2,4])
(BASE/'PREPARE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

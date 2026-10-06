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

hardware = capture([0,1,2,3], nv)
nv.nvmlInit()
try:
    assert all(not nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByIndex(g)) for g in [0,1,2,3])
finally: nv.nvmlShutdown()
with urllib.request.urlopen('http://127.0.0.1:9095/health', timeout=5) as response:
    health = json.load(response)
assert {0,1,2} <= set(health['allowed']), health
boundary = json.loads((BASE/'BOUNDARY.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in boundary['receipts_sha256'].items())
manifest = dict(reason='User requested 0,3,4,7 -> 0,1,2,3',
                logical_to_physical={'0':0,'3':1,'4':2,'7':3},
                physical_gpus=[0,1,2,3], hardware=hardware,
                clock_control='DCGM application, same backend as previous RPS8 phase',
                measurements_rerun=False, previous_boundary=str(BASE/'BOUNDARY.json'),
                reporting='GPU UUIDs changed at this boundary. Retain all calibration outcomes; report phase differences. Final heldout must freeze physical mapping before evaluation.',
                files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.rglob('*.py')})
path = BASE/'MIGRATION.json'
assert not list(OLD.glob('search/*/comparison/*/trials/*/physical-execution.json'))
path.write_text(json.dumps(manifest,indent=2)+'\n')
from mapping import transform, load_manifest
from official_space import commands
tested = 0
for config_path in OLD.glob('search/*/official-config.json'):
    config = json.loads(config_path.read_text())
    before = json.dumps(config, sort_keys=True)
    # The execution mapping touches these fields only; avoid copying the whole
    # 15,000-candidate search space for each candidate in this preflight.
    execution = {key:config[key] for key in ('gpus','hardware','correctness') if key in config}
    command_settings = {key:value for key,value in config.items() if key not in ('configurations','validation_candidates')}
    for candidate in config['configurations']:
        physical, mapped = transform(execution, candidate)
        assert all(mapped[k] == v for k,v in candidate.items() if k not in ('attention_gpus','expert_gpus'))
        assert sorted(mapped['attention_gpus'] + mapped['expert_gpus']) == sorted(set(mapped['attention_gpus'] + mapped['expert_gpus']))
        assert set(mapped['attention_gpus'] + mapped['expert_gpus']) <= {0,1,2,3}
        launch = commands(dict(command_settings, **physical), mapped)
        for role in ('attention','ffn'):
            assert set(launch[role]['gpus']) <= {0,1,2,3}
            cmd = launch[role]['command']
            assert '--enable-dbo' in cmd
            for flag,key in [('--dbo-decode-token-threshold','dbo_decode_token_threshold'),('--dbo-prefill-token-threshold','dbo_prefill_token_threshold')]:
                assert int(cmd[cmd.index(flag)+1]) == candidate[key]
        tested += 1
    assert json.dumps(config, sort_keys=True) == before
report = dict(passed=True,candidates_checked=tested,prior_receipts_preserved=len(boundary['receipts_sha256']),
              no_gpu_controls_applied=True,physical_gpus=[0,1,2,3])
(BASE/'PREPARE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

import hashlib
import json
import os
import signal
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD = BASE.with_name('dbo_on_0347_v3')
COMBINED = BASE.with_name('dbo_on_parallel_v3')

def live(pid, needle):
    try:
        p = Path('/proc') / str(pid)
        return needle.encode() in (p/'cmdline').read_bytes() and (p/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False

state = json.loads((OLD/'STATUS.json').read_text())
assert state['phase'] == 'waiting_for_gpus', state
launch = json.loads((COMBINED/'LAUNCH.json').read_text())
pid = launch['extra']['pid']
assert live(pid, 'dbo_on_0347_v3/run_experiment.py')
os.kill(pid, signal.SIGSTOP)
state = json.loads((OLD/'STATUS.json').read_text())
assert state['phase'] == 'waiting_for_gpus', state
receipts = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in OLD.glob('search/*/comparison/*/trials/*/worker-result.json')}
snapshot = {'previous_status': state, 'previous_launch': launch, 'receipts_sha256': receipts,
            'reason': 'User requested physical pool 0,3,4,7 -> 0,1,2,3; preserve all existing trials'}
(BASE/'BOUNDARY.json').write_text(json.dumps(snapshot, indent=2)+'\n')
for target, needle in [(launch['coordinator_pid'], 'resume_after_contention.py'),
                       (launch['validation']['pid'], 'validation/run_validation.py')]:
    if live(target, needle): os.kill(target, signal.SIGTERM)
os.kill(pid, signal.SIGTERM)
os.kill(pid, signal.SIGCONT)
for _ in range(40):
    if not live(pid, 'dbo_on_0347_v3/run_experiment.py'): break
    time.sleep(.5)
else: raise RuntimeError('Old controller still live; migration must not launch')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in receipts.items())
print(json.dumps({'stopped_at_wait_boundary': True, 'preserved_receipts': len(receipts)}))

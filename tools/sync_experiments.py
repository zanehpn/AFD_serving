"""Upload completed experimental progress; retry failures without altering experiments."""
import fcntl,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from datetime import datetime,timezone
REPO=(Path(__file__).resolve().parents[2] / 'MOE_DVFS-github-export');RUNTIME=(Path(__file__).resolve().parents[2] / 'MOE_DVFS-github-sync')
PYTHON=str(Path(__file__).resolve().parents[1] / 'bo_dse/.venv/bin/python')
BRANCH='experiments/20260910-results-and-policies'
ROOTS=['MOE_DVFS-policy-v3-replay','MOE_DVFS-rps-sweep','MOE_DVFS-8gpu-sweep','MOE_DVFS-resident-deploy','MOE_DVFS-8gpu-resident']
def read(p):return json.loads(p.read_text())
def write(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)
def completed():
 result={}
 for name in ROOTS:
  for p in (Path(__file__).resolve().parents[2]/name/'results').rglob('state.json'):
   try:
    data=p.read_bytes();s=json.loads(data)
    if s.get('frozen') and s.get('observations') and not s.get('pending'):result[str(p)]=hashlib.sha256(data).hexdigest()
   except (OSError,ValueError):continue
 for p in list((Path(__file__).resolve().parents[2] / 'MOE_DVFS-heldout-validation/results').glob('*/RESULT.json')) + list((Path(__file__).resolve().parents[2] / 'MOE_DVFS-heldout-attn1170/results').glob('*/RESULT.json')):
  result[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 p=(Path(__file__).resolve().parents[2] / 'MOE_DVFS-resident-deploy/orchestration/transition-status.json')
 if p.exists() and read(p).get('phase')=='deployed':result[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return result
def git(*args,env=None):return subprocess.check_output(['git','-C',str(REPO),*args],env=env,text=True).strip()
def upload():
 env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0')
 stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M%S');env['MOE_SNAPSHOT_NAME']=stamp
 out=REPO/'experiment-records'/stamp
 assert git('branch','--show-current')==BRANCH
 subprocess.run([PYTHON,str(REPO/'tools/export_experiments.py')],env=env,check=True)
 previous=REPO/'experiment-records/2026-09-10-1235'
 shutil.copy2(previous/'restore.py',out/'restore.py')
 text=(previous/'README.md').read_text()
 text+='\n## Current queue revision\n\nFuture four-GPU 2/4 RPS cohorts and eight-GPU 1/2/4 RPS cohorts include v2, generic BO and random, each with 16 search attempts plus a matched Max reference. Internal method v2 uses broad_v2; baselines explicitly use legacy policy. Qwen 1-RPS baseline additions are cancelled; their 21 existing attempts remain archived. Eight-GPU cohorts wait for both four-GPU queues. The eight-GPU pool permits candidate allocations using fewer than eight GPUs. GPU execution for new cohorts is pending unless completed receipts are present.\n\nThe uploader watches completed campaigns every 60 seconds; uploads are serialized and retried after errors. This live snapshot contains overlapping and superseded histories. Select the appropriate campaign before plotting.\n'
 text+='\nService residence deployment status and physical validation receipts are archived under MOE_DVFS-resident-deploy. When its transition-status is deployed, current four-GPU searches continue there, and future eight-GPU searches use MOE_DVFS-8gpu-resident. Original sweep histories overlap and are superseded; do not concatenate them. Only same-service consecutive configurations reuse processes. Each trial still applies controls and repeats warmup/replay.\n'
 (out/'README.md').write_text(text)
 if env.get('MOE_FULL_CHECKPOINT')=='1':
  note='\n## User-requested paused checkpoint\n\nBoth four-GPU search schedulers and the waiting eight-GPU queue are paused by user request. Current RPS=4 worker measurements drained successfully; their receipts are retained, including pending observations not yet told to the optimizer. RPS=8 diagnostic scripts are archived as prepared code only: no RPS=8 measurement has started. This checkpoint also includes unfinished historical trial directories for recovery; they are explicitly listed in manifest.json and must not be treated as completed measurements. The MOE_DVFS root includes its full results tree and preflight records in this checkpoint.\n'
  with (out/'README.md').open('a') as f:f.write(note)
 with tempfile.TemporaryDirectory(prefix='moe-sync-verify-') as td:
  subprocess.run(['python3',str(out/'restore.py'),td],check=True)
 manifest=read(out/'manifest.json');write(out/'validation.json',{'restored_and_hash_verified_files':len(manifest['files']),'credentials_scan':'passed during export'})
 git('add','tools','experiment-records/'+stamp)
 git('commit','-m','Archive completed experiment progress and current queue '+stamp)
 git('push','origin',BRANCH,env=env)
 commit=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/'+BRANCH,env=env).split()[0]
 assert remote==commit
 return {'commit':commit,'snapshot':stamp,'url':'experiment-records/'+stamp}
def main():
 RUNTIME.mkdir(exist_ok=True)
 with (RUNTIME/'watcher.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ledger=RUNTIME/'ledger.json';last=read(ledger) if ledger.exists() else {}
  while True:
   now=completed()
   if not last.get('commit') or now!=last.get('completed'):
    write(RUNTIME/'status.json',{'phase':'uploading','pid':os.getpid(),'updated_unix':time.time()})
    try:
     result=upload();last={**result,'completed':now,'uploaded_unix':time.time()};write(ledger,last)
     write(RUNTIME/'status.json',{'phase':'watching','pid':os.getpid(),**last})
    except Exception as exc:
     write(RUNTIME/'status.json',{'phase':'retry_pending','pid':os.getpid(),'error':repr(exc),'updated_unix':time.time()})
     time.sleep(60);continue
   time.sleep(60)
if __name__=='__main__':main()

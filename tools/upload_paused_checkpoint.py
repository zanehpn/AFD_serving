"""Serialize a user-requested full checkpoint with the existing uploader."""
import json,os,signal,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import sync_experiments as s

def main():
 status=s.RUNTIME/'status.json'
 state=s.read(status)
 assert state['phase']=='watching', 'Wait until the existing upload has completed'
 pid=state['pid'];proc=Path(f'/proc/{pid}')
 assert str(s.REPO/'tools/sync_experiments.py').encode() in (proc/'cmdline').read_bytes().split(b'\0')
 ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
 os.kill(pid,signal.SIGSTOP)
 try:
  assert s.read(status)['phase']=='watching', 'Uploader changed phase; retry later'
  os.environ['MOE_FULL_CHECKPOINT']='1'
  s.write(status,{'phase':'uploading_user_checkpoint','pid':pid,'uploader_pid':os.getpid(),'updated_unix':time.time()})
  completed=s.completed()
  result=s.upload()
  ledger={**result,'completed':completed,'uploaded_unix':time.time(),'full_paused_checkpoint':True}
  s.write(s.RUNTIME/'ledger.json',ledger)
  s.write(status,{'phase':'watching','pid':pid,**ledger})
  print(json.dumps(ledger,indent=2),flush=True)
 except BaseException as exc:
  s.write(status,{'phase':'checkpoint_failed','pid':pid,'error':repr(exc),'updated_unix':time.time()})
  raise
 finally:
  if proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==ticks:os.kill(pid,signal.SIGCONT)

if __name__=='__main__':main()

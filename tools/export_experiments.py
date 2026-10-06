import csv,hashlib,io,json,os,re,shutil,subprocess,sys,tarfile,time
import tempfile
from pathlib import Path
from datetime import datetime,timezone
DEST=(Path(__file__).resolve().parents[2] / 'MOE_DVFS-github-export')
OUT=DEST/'experiment-records'/os.environ['MOE_SNAPSHOT_NAME']
OUT.mkdir(parents=True,exist_ok=True)
roots=[Path(__file__).resolve().parents[1],(Path(__file__).resolve().parents[2] / 'MOE_DVFS-deepseek-rerun'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-policy-v2'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-policy-v3'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-resident-v1'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-qwen-extension'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-rps-sweep'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-policy-v3-fix1'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-policy-v3-replay'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-8gpu-sweep'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-heldout-validation'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-heldout-attn1170'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-resident-deploy'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-8gpu-resident'),(Path(__file__).resolve().parents[2] / 'MOE_DVFS-heldout-attn1410')]
full_checkpoint=os.environ.get('MOE_FULL_CHECKPOINT')=='1'
sys.path.insert(0,str(roots[2]/'bo_dse/scripts/afd'))
from static_dse.optimizer import best_measured
pattern=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
def read(p):
 data=p.read_bytes()
 if pattern.search(data):raise RuntimeError('Credential-like content requires review: '+str(p))
 return data
def sha(data):return hashlib.sha256(data).hexdigest()
manifest={'snapshot_utc':datetime.now(timezone.utc).isoformat(),'base_commit':subprocess.check_output(['git','-C',str(roots[0]),'rev-parse','HEAD'],text=True).strip(),'files':[],'archives':[],'omitted_unfinished_trial_directories':[],'campaigns':[]}
folders=('bo_dse','migration','scripts','services','tests','inputs','environment','patches')
def source_files(root):
 result=[]
 for name in folders:
  for p in (root/name).rglob('*'):
   rel=p.relative_to(root)
   excluded=('__pycache__','results') if full_checkpoint else ('__pycache__','preflight-records','results')
   if not p.is_file() or p.is_symlink() or any(x.startswith('.') or x in excluded for x in rel.parts):continue
   if p.suffix in ('.py','.sh','.md','.json','.jsonl','.toml','.txt','.patch','.yaml','.yml'):
    result.append(p)
 for p in root.iterdir():
  if p.is_file() and p.suffix in ('.md','.json','.patch'):result.append(p)
 for name in ('orchestration',):
  if (root/name).exists():result += [p for p in (root/name).glob('*') if p.is_file() and p.suffix in ('.py','.json','.log')]
 return sorted(set(result))

def archive(root,kind,files):
 tmp=Path(tempfile.gettempdir())/(root.name+'-'+kind+'.tar.gz')
 with tarfile.open(tmp,'w:gz',compresslevel=6) as tf:
  for p in files:
   rel=str(p.relative_to(root));data=read(p)
   info=tarfile.TarInfo(root.name+'/'+rel);info.size=len(data);info.mtime=int(p.stat().st_mtime);info.mode=p.stat().st_mode & 0o777
   tf.addfile(info,io.BytesIO(data))
   manifest['files'].append({'source':str(p),'archive':root.name+'-'+kind,'member':info.name,'bytes':len(data),'sha256':sha(data)})
 parts=[];digest=hashlib.sha256()
 with tmp.open('rb') as f:
  index=0
  while data:=f.read(24*1024*1024):
   name=f'{root.name}-{kind}.tar.gz.part{index:03d}';(OUT/name).write_bytes(data);digest.update(data)
   parts.append({'file':name,'bytes':len(data),'sha256':sha(data)});index+=1
 manifest['archives'].append({'name':root.name+'-'+kind,'sha256':digest.hexdigest(),'parts':parts})
 print(root.name,kind,'files',len(files),'compressed_bytes',tmp.stat().st_size,flush=True)

for root in roots:
 archive(root,'source',source_files(root))
 if not (root/'results').exists():continue
 resultroot=(root/'results/official-qwen-rps1/revisions/relative-max-8-tbt' if root.name=='MOE_DVFS' and not full_checkpoint else root/'results');unfinished=set()
 for p in resultroot.rglob('STARTED.json'):
  if not (p.parent/'worker-result.json').exists():unfinished.add(p.parent)
 for p in resultroot.rglob('worker-config.json'):
  if not (p.parent/'worker-result.json').exists():unfinished.add(p.parent)
 files=[]
 for p in resultroot.rglob('*'):
  if not p.is_file() or p.is_symlink() or p.suffix in ('.tmp','.o','.pyc') or p.name.endswith('.lock'):continue
  if not full_checkpoint and any(parent in unfinished for parent in p.parents):continue
  files.append(p)
 manifest.setdefault('included_unfinished_trial_directories',[])
 manifest['included_unfinished_trial_directories' if full_checkpoint else 'omitted_unfinished_trial_directories']+=sorted(map(str,unfinished))
 archive(root,'results',sorted(files))

# CSV is derived from the archived bytes, so the index and raw snapshot agree.
rows=[]
for archive_info in manifest['archives']:
 if not archive_info['name'].endswith('-results'):continue
 tmp=Path(tempfile.gettempdir())/(archive_info['name']+'.tar.gz')
 with tarfile.open(tmp,'r|gz') as tf:
  payload={m.name:json.load(tf.extractfile(m)) for m in tf if m.name.endswith(('/state.json','/bundle.json'))}
  names=set(payload)
  for name in sorted(names):
   if not name.endswith('/state.json'):continue
   base=name[:-len('state.json')]
   if base+'bundle.json' not in names:continue
   state=payload[name];bundle=payload[base+'bundle.json']
   settings=bundle['settings'];candidates={c['id']:c for c in bundle['candidates']};history=[]
   wall=gpu=energy=0.
   setup=settings.get('setup_cost',{})
   manifest['campaigns'].append({'path':base,'observations':len(state['observations']),'frozen':state['frozen'],'pending_trial':(state.get('pending') or {}).get('trial_id'),'budget':settings['budget'],'setup_cost':setup,'limits':settings['limits'],'policy':settings['bo']})
   for step,o in enumerate(state['observations'],1):
    history.append(o);best=best_measured(history,settings['limits']);m=o.get('metrics',{});cost=o.get('cost',{})
    wall+=cost.get('wall_seconds',0);gpu+=cost.get('gpu_hours',0);energy+=cost.get('tuning_energy_j',0)
    c=candidates[o['candidate_id']];t=c['topology'];limits=settings['limits']
    feasible=(o['status']=='ok' and all(m[k]<=limits[k] for k in ('ttft_ms','tpot_ms','tbt_ms') if k in limits) and m['output_tps']>=limits['min_output_tps'])
    rows.append(dict(campaign=base,model=settings.get('model_id'),method=settings['bo']['method'],use_model_prior=settings['bo']['use_model_prior'],policy=settings['bo'].get('exploration_policy','legacy'),seed=settings['bo']['seed'],step=step,trial_id=o['trial_id'],candidate_id=o['candidate_id'],status=o['status'],slo_feasible=feasible,failure_reason=o.get('failure_reason',''),proposal_reason=o.get('proposal_reason',''),topology=f"{len(t['attention_gpus'])}A{len(t['expert_gpus'])}E",microbatches=c['microbatches'],**{k:t.get(k,1) for k in ('attention_dp','attention_tp','expert_dp','expert_tp','expert_ep')},**c['knobs'],**{k:m.get(k) for k in ('energy_j','ttft_ms','tpot_ms','tbt_ms','output_tps')},best_feasible_energy_j=best['metrics']['energy_j'] if best else None,**{k:cost.get(k) for k in ('wall_seconds','gpu_hours','tuning_energy_j')},cumulative_measurement_wall_seconds=wall,cumulative_gpu_hours=gpu,cumulative_tuning_energy_j=energy,setup_evaluations=setup.get('evaluations',0),charged_evaluations=setup.get('evaluations',0)+step,gpu_hours_including_setup=gpu+setup.get('gpu_hours',0),energy_scope='participating GPUs',nvml_member=base+'trials/'+o['trial_id']+'/nvml.jsonl'))
with (OUT/'measurements.csv').open('w',newline='') as f:
 if rows:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('CSV rows',len(rows),'campaigns',len(manifest['campaigns']),flush=True)

"""Watch, verify and publish the user-requested four-GPU RPS4/RPS8 experiments."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import publish_capacity_v2_suite as common

ROOT = common.ROOT
NEW_FILES = ['orchestration/legacy_max_reference.py', 'orchestration/run_capacity_v2_4gpu.py',
             'tools/publish_capacity_v2_4gpu.py']


def model_directory(cfg, rate, model):
    override = cfg.get('model_directories', {}).get(str(rate), {}).get(model)
    return Path(override) if override else Path(cfg['runs'][str(rate)]) / model


def ready(cfg):
    for model in ('qwen', 'deepseek'):
        runtime = cfg.get('model_status_runtime', {}).get(model, cfg['runtime'])
        status = common.read(Path(runtime)/(model+'-status.json'))
        if status['phase'] == 'needs_attention':
            raise RuntimeError(model+' queue needs attention: '+status.get('error',''))
        if status['phase'] != 'complete':
            return False
        if status['completed_rates'] != cfg['rates']:
            raise ValueError('Completed rate list differs')
        if status['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip() and Path('/proc/'+str(status['pid'])).exists():
            return False
    for rate in cfg['runs']:
        for model in ('qwen','deepseek'):
            p = model_directory(cfg, rate, model)
            state = common.read(p/'campaign/state.json')
            bundle = common.read(p/'campaign/bundle.json')
            if len(state['observations']) != 16 or state['pending']:
                raise ValueError('Expected exactly 16 completed search attempts')
            if common.best_measured(state['observations'], bundle['settings']['limits']) and not state['frozen']:
                raise ValueError('Best configuration has not been frozen')
            if common.read(p/'resident-session/session-cost.json').get('cleanup_error'):
                raise ValueError('Resident cleanup failed')
    return True


def summarize(cfg):
    report = dict(captured_utc=common.now(), base_commit=cfg['base_commit'],
                  method='capacity_v2', revision=cfg['revision'], models=[],
                  aborted_campaigns=cfg.get('aborted_campaigns', []),
                  interpretation='Lowest measured participating-GPU energy passing all frozen limits; seed 0; no heldout evaluation')
    rows = []
    for rate,name in cfg['runs'].items():
        for model in ('qwen','deepseek'):
            directory = model_directory(cfg, rate, model)
            ref = common.read(directory/'inputs/slo-reference.json')
            record = dict(rps=int(rate), model=model, gpu_budget=4, requests=200,
                          reference=ref['metrics'], limits=ref['limits'], methods={},
                          comparison_scope='cross_runtime_historical_reference' if rate=='4' else 'historical_matched_protocol_baseline')
            scope = directory/'inputs/reference-comparison-scope.json'
            if scope.exists():
                record['reference_differences'] = common.read(scope)
            report['models'].append(record)
            methods = [('capacity_v2', directory/'campaign')]
            if rate == '8':
                old = Path(cfg['rps8_baseline'])/model/'comparison'
                methods = [('random',old/'random-seed0'),('generic_bo',old/'generic_bo-seed0')] + methods
            for method,path in methods:
                if not (path/'state.json').exists():
                    continue
                state = common.read(path/'state.json')
                bundle = common.read(path/'bundle.json')
                limits = bundle['settings']['limits']
                if limits != ref['limits']:
                    raise ValueError('Comparison limits differ: '+str(path))
                if bundle['settings']['workload']['arrival_rate_rps'] != int(rate):
                    raise ValueError('Comparison rate differs')
                obs = state['observations']
                best = common.best_measured(obs,limits)
                item = dict(directory=str(path), attempts=len(obs), successful=sum(o['status']=='ok' for o in obs),
                            feasible_attempts=sum(bool(common.best_measured([o],limits)) for o in obs),
                            frozen=state['frozen'], best=best, budget=bundle['settings']['budget'],
                            setup_cost=bundle['settings']['setup_cost'])
                if best:
                    first = next(o for o in obs if o['candidate_id']==best['candidate_id'])
                    item['configuration'] = common.read(path/'trials'/first['trial_id']/'configuration.json')
                    item['saving_vs_reference_pct'] = 100*(1-best['metrics']['energy_j']/ref['metrics']['energy_j'])
                record['methods'][method] = item
                for i,o in enumerate(obs,1):
                    row = dict(rps=int(rate), model=model, method=method, step=i, trial_id=o['trial_id'],
                               candidate_id=o['candidate_id'], status=o['status'],
                               slo_feasible=bool(common.best_measured([o],limits)),
                               failure_reason=o.get('failure_reason',''),
                               configuration=json.dumps(common.read(path/'trials'/o['trial_id']/'configuration.json'),sort_keys=True))
                    row.update({k:o.get('metrics',{}).get(k) for k in ('energy_j','ttft_ms','tpot_ms','tbt_ms','output_tps')})
                    rows.append(row)
    return report, rows


def build(cfg):
    sys.path.insert(0,str(ROOT/'bo_dse'))
    from official import validate_context
    for rate in cfg['runs']:
        for model in ('qwen','deepseek'):
            validate_context(common.read(model_directory(cfg,rate,model)/'official-config.json'))
    repo = Path(cfg['export_checkout'])
    if not repo.exists():
        common.git(ROOT,'worktree','add','-b',cfg['branch'],str(repo),cfg['export_base'])
    if common.git(repo,'branch','--show-current') != cfg['branch']:
        raise ValueError('Unexpected export branch')
    dest = repo/'experiment-records'/cfg['snapshot']
    dest.mkdir(parents=True,exist_ok=True)
    manifest = dict(snapshot_utc=common.now(),base_commit=cfg['base_commit'],files=[],archives=[],
                    excluded=['weights','virtual environments','caches','credentials'])
    source = [ROOT/name for name in common.git(ROOT,'ls-files','-z').split('\0') if name and not name.startswith('experiment-records/')]
    source += [ROOT/name for name in NEW_FILES + ['tools/publish_capacity_v2_suite.py']]
    source += [ROOT/'environment/OFFICIAL_INSTALLED.json']
    common.archive(dest,'capacity-v2-four-gpu-source',source,manifest)
    for rate,name in cfg['runs'].items():
        common.archive(dest,'capacity-v2-rps'+rate+'-four-gpu-results',common.result_files(Path(name)),manifest)
    for rate,models in cfg.get('model_directories', {}).items():
        for model,name in models.items():
            common.archive(dest,'capacity-v2-rps'+rate+'-'+model+'-recovery-results',common.result_files(Path(name)),manifest)
    common.archive(dest,'queue-and-validation',common.result_files(Path(cfg['runtime'])),manifest)
    common.archive(dest,'historical-rps8-four-gpu-baselines',common.result_files(Path(cfg['rps8_baseline'])),manifest)
    common.write(dest/'manifest.json',manifest)
    shutil.copy2(ROOT/'experiment-records/2026-09-11-143809-rps8-200-8gpu-complete/restore.py',dest/'restore.py')
    report,rows = summarize(cfg)
    common.write(dest/'summary.json',report)
    with (dest/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=['# Capacity v2: four-GPU pools, 4/8 RPS, 200 requests','',
           'Both models ran seed 0, 16 search attempts per rate, 200 requests per measurement and 8 warmup requests. '
           'Qwen uses GPUs 0–3; DeepSeek uses GPUs 4–7. The same neighbor-reduction capacity_v2 policy as the preceding eight-GPU experiments is used. '
           'Each pool may select fewer than four GPUs. Historical MAX is reused and charged once; no new MAX measurement was performed.','',
           'RPS4 uses the user-selected `inputs/calibration-max` 200-request history. Its TTFT/TPOT p90 limits are 105% of MAX, and '
           'the throughput floor is 95% of MAX. Historical token-interval timestamps are absent, so no TBT SLO is invented. '
           'The historical modified plugin and enabled prefix cache differ from the current upstream plugin and disabled prefix cache. '
           'RPS4 energy ratios are cross-runtime historical comparisons and cannot isolate the optimizer contribution. '
           'Only historical serving-window setup cost is available; it is explicitly a lower bound.','',
           'RPS8 reuses the exact four-GPU, 200-request MAX and four SLO limits from the historical Random/BO campaigns. '
           'Their raw records are included. Best uses the unchanged repeat-aggregation rule. These are one-seed calibration results, without heldout validation.','',
           '| RPS | Model | Method | Attempts | Successful | SLO feasible | Best kJ | Saving vs historical MAX |',
           '|---:|---|---|---:|---:|---:|---:|---:|']
    for record in report['models']:
        for method,item in record['methods'].items():
            energy=f"{item['best']['metrics']['energy_j']/1000:.2f}" if item['best'] else 'none'
            saving=f"{item['saving_vs_reference_pct']:.2f}%" if item['best'] else '—'
            lines.append(f"| {record['rps']} | {record['model']} | {method} | {item['attempts']} | {item['successful']} | {item['feasible_attempts']} | {energy} | {saving} |")
    if cfg.get('aborted_campaigns'):
        lines += ['', 'DeepSeek RPS4 was restarted after five RPC-port bind failures and one interrupted launch. '
                  'The first round has 16 accounted attempts (10 successful), retained separately with its complete raw evidence and costs. '
                  'It is excluded from the replacement round\'s 16-attempt optimization results. The RPC ports were reserved against ephemeral allocation. '
                  'The first round\'s reserved-GPU session cost remains in its `resident-session/session-cost.json`; it must not be discarded or added twice.']
    lines += ['', 'All failures, raw replay/NVML logs, reference provenance, source hashes and runtime locks are retained. '
              'Run `python3 restore.py restored-records` to verify and restore every archived byte.', '']
    (dest/'README.md').write_text('\n'.join(lines))
    for name in NEW_FILES:
        target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(common.checked_bytes(ROOT/name))
    with tempfile.TemporaryDirectory(prefix='capacity-four-gpu-verify-') as td:
        subprocess.run([sys.executable,str(dest/'restore.py'),td],check=True,timeout=600)
    common.write(dest/'validation.json',dict(all_archive_and_file_hashes_verified=True,credential_scan='passed',
                                           restored_files=len(manifest['files']),verified_utc=common.now()))
    common.git(repo,'add','--','experiment-records/'+cfg['snapshot'],*NEW_FILES)
    if common.git(repo,'diff','--cached','--name-only'):
        common.git(repo,'commit','-m',
                   'Archive capacity v2 four-GPU experiments at 4 and 8 RPS with inherited 200-request MAX')
    return common.git(repo,'rev-parse','HEAD')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();cfg=common.read(args.config)
    if args.check:
        print(json.dumps(dict(ready=ready(cfg),summary=summarize(cfg)[0])));return
    runtime=Path(cfg['publisher_runtime']);runtime.mkdir(parents=True,exist_ok=True)
    with (runtime/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=runtime/'status.json';state=common.read(state_path) if state_path.exists() else {}
        if state.get('phase')=='pushed':return
        commit=state.get('commit')
        while True:
            try:
                if not ready(cfg):
                    common.write(state_path,dict(phase='waiting_for_completion',pid=os.getpid(),updated_utc=common.now()))
                    time.sleep(30);continue
                if not commit:
                    common.write(state_path,dict(phase='building_and_verifying_archive',pid=os.getpid(),updated_utc=common.now()))
                    commit=build(cfg)
                common.write(state_path,dict(phase='pushing',pid=os.getpid(),commit=commit,updated_utc=common.now()))
                url=common.push(cfg,commit)
                common.write(state_path,dict(phase='pushed',commit=commit,url=url,branch=cfg['branch'],updated_utc=common.now()))
                print('Published '+url,flush=True);return
            except Exception as error:
                common.write(state_path,dict(phase='retry_pending',pid=os.getpid(),commit=commit,error=str(error),updated_utc=common.now()))
                time.sleep(60)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Paper entry point; upstream observable-only functions are historical compatibility code.

The upstream runtime has no four-stage instrumentation. Its duration/power model
is available only through an explicit --backend legacy-observable selection.
The default command dispatches to the instrumented paper implementation.
"""
import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bo_dse/scripts/afd'))
from static_dse.campaign import (create_campaign, isolation, read_json as read, write_json as write,
                                 file_hash as sha, ask, tell, status, freeze)
from static_dse.comparison import VARIANTS, create_comparison, comparison_report
from static_dse.space import configuration, enumerate_candidates, layout, digest, KNOBS, audit
from official_space import structures
from official_worker import artifact, clean_environment
from native_service import boot_id, identity, stop_group

MODELS = {'qwen36': 'Qwen3.6-35B-A3B', 'deepseek-v2-lite': 'DeepSeek-V2-Lite-Chat'}
ECHO = ('trial_id', 'candidate_id', 'configuration_sha256', 'context_sha256', 'request_sha256', 'mode', 'selection_split')


def bo_runtime():
    return dict(python=platform.python_version(), packages={name: importlib.metadata.version(name)
                for name in ('numpy', 'scipy', 'scikit-learn', 'nvidia-ml-py')})


@contextmanager
def gpu_locks(gpus):
    handles = []
    try:
        for gpu in sorted(gpus):
            path = (Path(tempfile.gettempdir()) / f'moe-bo-gpu-{gpu}.lock')
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o666)
            handles.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for fd in reversed(handles):
            os.close(fd)


def validate_context(config):
    if config['bo_runtime'] != bo_runtime():
        raise ValueError('Frozen BO Python/package runtime changed')
    manifest = read(config['context_manifest'])
    for path, expected in manifest['files_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen source/input changed: {path}')
    for path, expected in manifest['model_files'].items():
        stat = Path(path).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError(f'Frozen model file changed: {path}')


def worker(config, action, directory, c=None):
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory/('cleanup-worker-config.json' if action == 'cleanup' else 'worker-config.json')
    write(config_path, config)
    command = [config['native_python'], str(ROOT/'bo_dse/official_worker.py'), action,
               '--config', str(config_path), '--directory', str(directory)]
    if c is not None:
        write(directory/'configuration.json', c)
        command += ['--configuration', str(directory/'configuration.json')]
    with (directory/('cleanup-worker.log' if action == 'cleanup' else 'worker.log')).open('ab') as log:
        process = subprocess.Popen(command, cwd=directory, env=clean_environment(), stdout=log, stderr=log,
                                   start_new_session=True)
        if action in ('trial', 'reference'):
            write(directory/'worker-service.json', dict(pid=process.pid, start_ticks=identity(process.pid), boot_id=boot_id()))
        try:
            process.wait(timeout=config['trial_timeout_seconds'])
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if process.returncode:
        raise RuntimeError(f'Official {action} exited {process.returncode}: {directory}/worker.log')


def attempt(config, c, directory, reference=False):
    if (directory/'worker-result.json').exists():
        result = read(directory/'worker-result.json')
        resolved = directory/'CLEANUP_RESOLVED.json'
        if result.get('cleanup_required') and (not resolved.exists() or
                read(resolved).get('original_receipt') != artifact(directory/'worker-result.json')):
            raise RuntimeError(f'Cleanup required: {directory}; run cleanup before resuming')
        return result
    if (directory/'STARTED.json').exists():
        raise RuntimeError(f'Interrupted attempt: {directory}; run recover to preserve its failure cost')
    validate_context(config)
    directory.mkdir(parents=True, exist_ok=True)
    write(directory/'STARTED.json', dict(started_wall_ns=time.time_ns(), configuration=c))
    try:
        worker(config, 'reference' if reference else 'trial', directory, c)
    except BaseException as error:
        # A cooperative worker normally publishes the failure receipt itself.
        if not (directory/'worker-result.json').exists():
            recover_attempt(config, directory, str(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    result = read(directory/'worker-result.json')
    if result.get('cleanup_required'):
        raise RuntimeError(f'Cleanup required: {directory}; no further candidate may launch')
    return result


def recover_attempt(config, directory, reason='Interrupted attempt recovered; elapsed cost is an upper bound'):
    started = read(directory/'STARTED.json')
    if (directory/'worker-service.json').exists():
        stop_group(read(directory/'worker-service.json'))
    cleanup_error = None
    try:
        worker(config, 'cleanup', directory)
    except Exception as error:
        cleanup_error = str(error)
    elapsed = max((time.time_ns()-started['started_wall_ns'])/1e9, 1e-9)
    # SIGTERM may have allowed the worker to finish its original receipt.
    if (directory/'worker-result.json').exists() and cleanup_error is None:
        return
    result = dict(status='failed', failure_reason=reason, cleanup_required=cleanup_error is not None,
                  cost=dict(wall_seconds=elapsed, gpu_hours=elapsed*len(config['gpus'])/3600, tuning_energy_j=0),
                  cost_energy_complete=False, recovered_cost_is_upper_bound=True,
                  artifacts=[artifact(directory/'STARTED.json')])
    if cleanup_error:
        result['failure_reason'] += '; cleanup: ' + cleanup_error
    write(directory/'worker-result.json', result)


def spec_for(args):
    if any(m not in (1, 2) for m in args.microbatches):
        raise ValueError('Official microbatches can only be 1 (DBO off) or 2 (DBO on)')
    spec = dict(selection_split='calibration', topologies=structures(args.gpus, args.model),
                enumerate_parallelism=False, attention_frequencies_mhz=args.frequencies,
                expert_frequencies_mhz=args.frequencies, attention_power_caps_w=args.power_caps,
                expert_power_caps_w=args.power_caps, microbatches=args.microbatches, execution_mode='eager')
    if getattr(args, 'dbo_threshold_profiles', None):
        spec['dbo_threshold_profiles'] = [list(map(int, pair.split(':'))) for pair in args.dbo_threshold_profiles]
    return spec


def initialize(args):
    spec = spec_for(args)
    candidates = enumerate_candidates(spec)
    count = len({digest(layout(c)) for c in candidates})
    if args.evaluations < 3 or not math.isfinite(args.gpu_hours) or args.gpu_hours <= 0:
        raise ValueError('Budget needs positive GPU hours and at least 3 evaluations')
    for value in (args.rps, args.trial_timeout_seconds, args.ttft_ms, args.tpot_ms, args.min_output_tps):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Rate, timeout and SLO limits must be finite and positive')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    ports = [args.api_port, args.api_port+1, args.afd_port, args.dp_rpc_port, args.dp_rpc_port+100]
    if len(set(ports)) != len(ports) or any(p < 1 or p > 65535 for p in ports):
        raise ValueError('Service ports must be distinct and in 1..65535')
    if args.dry_run:
        return dict(mode='dry_run', backend='official', runtime_lock=read(ROOT/'environment/official-runtime.lock.json'),
                    direct_search=getattr(args, 'direct_search', False),
                    exploration_policy=getattr(args, 'exploration_policy', None) or 'broad_v2',
                    output_validation=getattr(args, 'output_validation', 'exact_tokens'),
                    tbt_slo=getattr(args, 'tbt_slo', False), evaluation_requests=getattr(args, 'evaluation_requests', None), slo_mode=getattr(args, 'slo_mode', 'absolute'),
                    specification=spec, raw_candidates=len(candidates), structures=count,
                    setup_evaluations_before_filtering=1 if getattr(args, 'direct_search', False) else count+1,
                    model_filter_applied=False, candidate_status='pending_model_hardware_and_execution_validation',
                    required_evaluations_before_filtering=3 if getattr(args, 'direct_search', False) else count+3,
                    execution_verified=False, gpu_actions_performed=False,
                    mechanism_model='external_power_duration_v1', actual_microbatch_splits_available=False,
                    require_four_stage=False, paper_method=False,
                    excluded=['independent expert replicas',
                              'M>2', 'async', 'PP', 'multinode', 'graph: deferred from initial eager protocol'],
                    next_steps=['immutable split audit', 'stock token reference', 'validate every structure using calibration',
                                'freeze evidenced subspace', 'observable model calibration and BO', 'freeze measured best'])
    directory = args.directory.resolve()
    if directory.exists():
        raise FileExistsError('Use prepare/run to resume the existing directory')
    split = isolation(args.calibration, args.heldout, ['source_index', 'source_timestamp'])
    rows = [json.loads(s) for s in args.calibration.read_text().splitlines() if s.strip()]
    heldout = [json.loads(s) for s in args.heldout.read_text().splitlines() if s.strip()]
    if len(rows) < 8 or any(r.get('evaluation_split') != 'calibration' for r in rows) or any(r.get('evaluation_split') != 'heldout' for r in heldout):
        raise ValueError('Explicit calibration/heldout labels and at least 8 calibration requests required')
    arrivals = [float(r['arrival_s']) for r in rows]
    if min(arrivals) < 0 or max(arrivals) <= min(arrivals):
        raise ValueError('Calibration must have a nonzero arrival span')
    evaluation_requests = getattr(args, 'evaluation_requests', None)
    if evaluation_requests is not None and not 1 <= evaluation_requests <= len(rows):
        raise ValueError('Evaluation request count must fit the calibration source')
    selected = rows if evaluation_requests is None else rows[:evaluation_requests]
    selected_arrivals = [float(r['arrival_s']) for r in selected]
    model_path = (args.model_path or ROOT/'artifacts/models'/MODELS[args.model]).resolve()
    model = read(model_path/'config.json')
    model = model.get('text_config', model)
    experts = model.get('num_experts', model.get('n_routed_experts'))
    if not experts:
        raise ValueError('Model expert count missing')
    directory.mkdir(parents=True)
    inputs = directory/'inputs'
    inputs.mkdir()
    for name, source in (('calibration.jsonl', args.calibration), ('heldout.jsonl', args.heldout)):
        (inputs/name).write_bytes(source.read_bytes())
    (inputs/'calibration-source.jsonl').write_bytes(args.calibration.read_bytes())
    write(inputs/'source-isolation.json', isolation(inputs/'calibration-source.jsonl', inputs/'heldout.jsonl',
                                                   ['source_index', 'source_timestamp']))
    if evaluation_requests is not None:
        (inputs/'calibration.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (inputs/'warmup.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:8]))
    split = isolation(inputs/'calibration.jsonl', inputs/'heldout.jsonl', ['source_index', 'source_timestamp'])
    write(inputs/'isolation.json', dict(**split, warmup_source='first 8 calibration rows', heldout_used_for_tuning=False))
    config = dict(backend='official', bo_runtime=bo_runtime(), directory=str(directory), model=args.model, model_path=str(model_path), gpus=args.gpus,
                  native_python=str(args.native_python.absolute()), rps=args.rps, clock_url=args.clock_url,
                  api_port=args.api_port, afd_port=args.afd_port, dp_rpc_port=args.dp_rpc_port,
                  max_model_len=8192, max_output_tokens=128,
                  trial_timeout_seconds=args.trial_timeout_seconds,
                  trace=split['calibration'], heldout_trace=split['heldout'],
                  warmup_trace=str(inputs/'warmup.jsonl'), warmup_requests=8,
                  time_scale=((len(selected)-1)/(args.rps*(max(selected_arrivals)-min(selected_arrivals)))
                              if len(selected) > 1 else 1.),
                  limits=dict(ttft_ms=args.ttft_ms, tpot_ms=args.tpot_ms, min_output_tps=args.min_output_tps),
                  budget=dict(evaluations=args.evaluations, gpu_hours=args.gpu_hours), comparison=args.comparison, seeds=args.seeds,
                  comparison_methods=getattr(args, 'comparison_methods', None) or list(VARIANTS),
                  correctness_reference_path=str(inputs/'correctness-reference.json'),
                  correctness_contract=str(inputs/'correctness-contract.json'), context_manifest=str(inputs/'context.json'))
    config['output_validation'] = getattr(args, 'output_validation', 'exact_tokens')
    config['direct_search'] = getattr(args, 'direct_search', False)
    config['resident_search'] = getattr(args, 'resident_search', False)
    config['slo_mode'] = getattr(args, 'slo_mode', 'absolute')
    config['tbt_slo'] = getattr(args, 'tbt_slo', False)
    if config['tbt_slo']:
        config['tbt_protocol'] = 'p90 pooled within-request client SSE token-ID arrival intervals; coalesced tokens share arrival time'
    if config['tbt_slo'] and config['slo_mode'] != 'relative_max':
        raise ValueError('TBT SLO requires relative Max mode')
    config['evaluation_requests'] = len(selected)
    config['cohort_rule'] = 'first N calibration source rows, fixed for every candidate and Max'
    if getattr(args, 'exploration_policy', None):
        config['exploration_policy'] = args.exploration_policy
    if config['slo_mode'] == 'relative_max':
        if not config['direct_search']:
            raise ValueError('Relative Max mode requires direct search')
        config['limits'] = None  # Derived only from the newly measured, matched Max reference.
        config['slo_ratios'] = dict(ttft_ms=1.05, tpot_ms=1.05, min_output_tps=.95)
        if config['tbt_slo']:
            config['slo_ratios']['tbt_ms'] = 1.05
    if config['direct_search'] and config['output_validation'] != 'request_completion':
        raise ValueError('Direct search requires the explicit request_completion output policy')
    write(inputs/'specification.json', spec)
    prior = getattr(args, 'prior_failed_reference_directory', None)
    prior_run = getattr(args, 'prior_run_directory', None)
    if prior is not None and prior_run is not None:
        raise ValueError('Select only one cost inheritance source')
    if prior is not None:
        inherit_failed_reference(config, prior.resolve(), inputs, allow_preparation=config['direct_search'])
    if prior_run is not None:
        inherit_run_costs(config, prior_run.resolve(), inputs)
    with gpu_locks(config['gpus']):
        worker(config, 'preflight', directory/'preflight')
    config['hardware'] = read(directory/'preflight/preflight.json')['hardware']
    memory_clocks = set.intersection(*[{p['memory_mhz'] for p in d['clock_pairs']}
                                      for d in config['hardware']['devices'].values()])
    if not memory_clocks:
        raise ValueError('Selected GPUs have no common advertised memory clock')
    config['memory_clock_mhz'] = max(memory_clocks)
    write(inputs/'hardware.json', config['hardware'])
    runtime = dict(adapter='official_v026', model_id=args.model, gpu_budget=len(args.gpus), allowed_gpus=args.gpus,
                   connector='p2p_nccl', execution_modes=['eager'], memory_clock_mhz=config['memory_clock_mhz'],
                   divisibility=dict(expert_ep=experts, attention_tp=model['num_attention_heads'], expert_tp=model['num_attention_heads']),
                   structures=[dict(layout=row, status='pending_validation', evidence=[])
                               for row in {digest(layout(c)): layout(c) for c in candidates}.values()])
    if config.get('exploration_policy') == 'capacity_v2':
        from static_dse.capacity import build_profile
        profile = build_profile(model_path, candidates, config['hardware'],
                                max_model_len=config['max_model_len'])
        runtime['capacity_profile'] = profile
        write(inputs/'capacity.json', profile)
    write(inputs/'runtime-pending.json', runtime)
    space_audit = audit(candidates, config['hardware'], runtime)
    write(inputs/'space-audit.json', space_audit)
    checks = {row['id']: row for row in space_audit['candidates']}
    valid = [c for c in candidates if checks[c['id']]['status'] != 'hard_rejected']
    if not valid:
        raise ValueError('No candidate matches hardware/model capabilities; inspect preflight and specification')
    groups = {}
    for c in valid:
        key = digest(layout(c))
        if key not in groups or tuple(c['knobs'][k] for k in KNOBS) > tuple(groups[key]['knobs'][k] for k in KNOBS):
            groups[key] = c
    arm_counts = [c['evaluations'] for c in config.get('prior_arm_costs', {}).values()]
    needed = (3 if config['direct_search'] else len(groups)+3) + len(setup_receipts(directory)) + max(arm_counts, default=0)
    if args.evaluations < needed:
        raise ValueError(f'Filtered space needs {len(groups)} structure validations + 1 stock reference + '
                         'at least 2 search trials; raise --evaluations instead of dropping unverified structures')
    # Reference fixed before measurements: balanced full pool, TP=1, DBO off if included.
    reference = next(c for c in groups.values() if len(c['topology']['attention_gpus']) == len(args.gpus)//2
                     and len(c['topology']['expert_gpus']) == len(args.gpus)//2 and c['topology']['attention_tp'] == 1
                     and c['topology']['expert_tp'] == 1 and c['microbatches'] == min(args.microbatches))
    config.update(configurations=[configuration(c) for c in valid], validation_candidates=list(groups.values()),
                  reference_candidate_id=reference['id'], reference_configuration=configuration(reference))
    config['correctness'] = dict(protocol='stock_vllm_exact_tokens_v1', selection_split='calibration', trace=config['trace'],
                                 model_config=artifact(model_path/'config.json'), max_output_tokens=config['max_output_tokens'],
                                 generation=dict(temperature=0, seed=0, ignore_eos=True, generation_config='vllm'), reference_tp=2,
                                 reference_gpu_uuids=[config['hardware']['devices'][str(g)]['uuid'] for g in args.gpus[:2]])
    write(config['correctness_contract'], config['correctness'])
    if getattr(args, 'reuse_max_directory', None):
        from inherited_reference import inherit
        inherit(config, args.reuse_max_directory, inputs)
    paths = [p for folder in ('bo_dse', 'migration', 'scripts/afd', 'services') for p in (ROOT/folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.sh') and not any(s.startswith('.') or s in ('results', '__pycache__') for s in p.relative_to(ROOT).parts)]
    paths += [p for p in inputs.iterdir() if p.is_file()]
    if (inputs/'prior-preparation.json').exists():
        paths += [Path(r['artifact']['path']) for r in read(inputs/'prior-preparation.json')['receipts']]
    if (inputs/'prior-arm-costs.json').exists():
        paths += [Path(r['path']) for arm in read(inputs/'prior-arm-costs.json')['arms'].values()
                  for r in arm['receipts']]
        paths += [Path(arm['state_evidence']['path']) for arm in read(inputs/'prior-arm-costs.json')['arms'].values()]
    paths += [model_path/'config.json', ROOT/'environment/official-runtime.lock.json', ROOT/'environment/OFFICIAL_INSTALLED.json',
              ROOT/'environment/native-requirements.lock.txt']
    model_files = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in model_path.rglob('*') if p.is_file()}
    if not any(p.endswith('.safetensors') for p in model_files):
        raise ValueError('Model weights missing')
    write(config['context_manifest'], dict(files_sha256={str(p): sha(p) for p in paths}, model_files=model_files))
    write(directory/'official-config.json', config)
    return config


def setup_receipts(directory):
    prior = directory/'inputs/prior-preparation.json'
    receipts = []
    inherited = directory/'inputs/inherited-max.json'
    if inherited.exists():
        source = read(inherited)['receipt']
        if artifact(source['path']) != source:
            raise ValueError('Inherited Max receipt changed')
        receipts.append(read(source['path']))
    if prior.exists():
        for row in read(prior)['receipts']:
            if artifact(row['artifact']['path']) != row['artifact']:
                raise ValueError('Inherited preparation receipt changed')
            receipts.append(read(row['artifact']['path']))
    return receipts + [read(p) for p in sorted((directory/'validation').glob('*/worker-result.json'))]


def inherit_failed_reference(config, prior, inputs, allow_preparation=False):
    """Start a new frozen runtime after a pre-structure failure, carrying all cost."""
    old = read(prior/'official-config.json')
    if (prior/'READY.json').exists() or (prior/'comparison').exists():
        raise ValueError('Only failed stock-reference preparation can be inherited')
    for key in ('model', 'model_path', 'gpus', 'rps', 'limits', 'budget', 'comparison', 'seeds',
                'max_model_len', 'max_output_tokens', 'warmup_requests', 'time_scale'):
        if old[key] != config[key]:
            raise ValueError(f'Inherited protocol differs: {key}')
    for key in ('trace', 'heldout_trace'):
        if old[key]['sha256'] != config[key]['sha256']:
            raise ValueError(f'Inherited trace differs: {key}')
    if read(prior/'inputs/specification.json') != read(inputs/'specification.json'):
        raise ValueError('Inherited search grid differs')
    paths = sorted((prior/'validation').glob('*/worker-result.json'))
    if not paths or (not allow_preparation and paths != [prior/'validation/stock-reference/worker-result.json']):
        raise ValueError('Inheritance requires exactly one local stock-reference receipt')
    if not allow_preparation and read(paths[0])['status'] != 'failed':
        raise ValueError('Cannot repeat a successful stock reference')
    for path in paths:
        attempt(old, {}, path.parent)  # enforce cleanup acknowledgement without rerunning
    for marker in prior.rglob('STARTED.json'):
        if not (marker.parent/'worker-result.json').exists():
            raise ValueError('Recover interrupted preparation before inheritance')
    rows = []
    inherited = prior/'inputs/prior-preparation.json'
    if inherited.exists():
        setup_receipts(prior)  # verify prior receipt hashes
        rows.extend(read(inherited)['receipts'])
    rows.extend(dict(artifact=artifact(p)) for p in paths)
    if len({r['artifact']['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate inherited receipt')
    write(inputs/'prior-preparation.json', dict(previous_directory=str(prior), receipts=rows,
          reason=('User-authorized direct-search/output-policy change' if allow_preparation else 'External tooling repair')
          + '; original frozen campaign and failures retained; all costs charged per arm'))


def check_setup_budget(config, directory):
    receipts = setup_receipts(directory)
    arm_costs = list(config.get('prior_arm_costs', {}).values())
    evaluations = len(receipts) + max((c['evaluations'] for c in arm_costs), default=0)
    gpu_hours = sum(r['cost']['gpu_hours'] for r in receipts) + max((c['gpu_hours'] for c in arm_costs), default=0)
    if evaluations >= config['budget']['evaluations'] or gpu_hours >= config['budget']['gpu_hours']:
        raise RuntimeError('Preparation budget exhausted; all failures remain charged in validation/')


def inherit_run_costs(config, prior, inputs):
    """User-requested protocol change: preserve common preparation and each arm's own costs."""
    old = read(prior/'official-config.json')
    if not config.get('direct_search') or not old['comparison'] or not config['comparison']:
        raise ValueError('Protocol transition requires comparison direct search')
    for key in ('model', 'model_path', 'gpus', 'rps', 'budget', 'seeds', 'max_model_len',
                'max_output_tokens', 'warmup_requests'):
        if config[key] != old[key]:
            raise ValueError(f'Protocol transition must retain {key}')
    source = prior/'inputs/calibration-source.jsonl'
    if not source.exists():
        source = prior/'inputs/calibration.jsonl'
    if sha(source) != sha(inputs/'calibration-source.jsonl') or old['heldout_trace']['sha256'] != config['heldout_trace']['sha256']:
        raise ValueError('Protocol transition must use the same audited source splits')
    if read(prior/'inputs/specification.json') != read(inputs/'specification.json'):
        raise ValueError('Protocol transition must retain the search grid')
    for marker in prior.rglob('STARTED.json'):
        if not (marker.parent/'worker-result.json').exists():
            raise ValueError('Finish or recover every old attempt before transition')
        attempt(old, {}, marker.parent)  # checks unresolved cleanup; never reruns
    setup_receipts(prior)
    inherited = prior/'inputs/prior-preparation.json'
    common = read(inherited)['receipts'] if inherited.exists() else []
    common += [dict(artifact=artifact(p)) for p in sorted((prior/'validation').glob('*/worker-result.json'))]
    if len({r['artifact']['path'] for r in common}) != len(common):
        raise ValueError('Duplicate shared cost receipt')
    write(inputs/'prior-preparation.json', dict(previous_directory=str(prior), receipts=common,
          reason='User-authorized cohort/SLO revision; all earlier common preparation costs retained'))
    arms = {}
    previous_arms = prior/'inputs/prior-arm-costs.json'
    previous_arms = read(previous_arms)['arms'] if previous_arms.exists() else {}
    manifest = read(prior/'comparison/comparison.json')
    for arm in manifest['campaigns']:
        name = f"{arm['method']}-seed{arm['seed']}"
        path = Path(arm['directory'])
        state = read(path/'state.json')
        if state['bundle_sha256'] != sha(path/'bundle.json'):
            raise ValueError('Old campaign bundle changed')
        if state['pending'] is not None:
            raise ValueError('Account every old pending search trial before transition')
        refs = list(previous_arms.get(name, {}).get('receipts', []))
        for obs in state['observations']:
            receipt = path/'trials'/obs['trial_id']/'result.json'
            value = read(receipt)
            if value['cost'] != obs['cost'] or value['status'] != obs['status']:
                raise ValueError('Search state and original receipt disagree')
            refs.append(artifact(receipt))
        if len({r['path'] for r in refs}) != len(refs):
            raise ValueError('Duplicate arm cost receipt')
        values = []
        for ref in refs:
            if artifact(ref['path']) != ref:
                raise ValueError('Inherited arm receipt changed')
            values.append(read(ref['path']))
        cost = {key: sum(v['cost'][key] for v in values) for key in ('wall_seconds', 'gpu_hours', 'tuning_energy_j')}
        cpu_wall = (previous_arms.get(name, {}).get('cpu_wall_seconds', 0)
                    + state.get('optimizer_wall_seconds', 0) + state.get('initialization_wall_seconds', 0))
        cost['wall_seconds'] += cpu_wall
        cost['evaluations'] = len(values)
        arms[name] = dict(cost=cost, failures=sum(v['status'] != 'ok' for v in values), receipts=refs,
                          cpu_wall_seconds=cpu_wall, state_evidence=artifact(path/'state.json'))
    # Older manifests omit the field and used the original three methods.
    methods = config.get('comparison_methods') or ['v2', 'generic_bo', 'random']
    expected = {f'{method}-seed{seed}' for method in methods for seed in config['seeds']}
    if set(arms) != expected:
        raise ValueError('Inherited comparison arms differ')
    write(inputs/'prior-arm-costs.json', dict(previous_directory=str(prior), arms=arms,
          rule='Only the matching method/seed inherits its search costs; no prior search labels reused'))
    config['prior_arm_costs'] = {name: a['cost'] for name, a in arms.items()}


def prepare(config):
    directory = Path(config['directory'])
    if (directory/'READY.json').exists():
        return read(directory/'READY.json')
    validate_context(config)
    with gpu_locks(config['gpus']):
        check_setup_budget(config, directory)
        direct = config.get('direct_search', False)
        if not direct:
            receipt = attempt(config, config['reference_configuration'], directory/'validation/stock-reference', reference=True)
            if receipt['status'] != 'ok':
                raise RuntimeError('Stock correctness reference failed; inspect charged validation/stock-reference receipt')
            config['correctness_reference'] = artifact(config['correctness_reference_path'])
        runtime = read(directory/'inputs/runtime-pending.json')
        runtime['structures'] = []
        observations = []
        for candidate in config['validation_candidates']:
            if direct and candidate['id'] != config['reference_candidate_id']:
                runtime['structures'].append(dict(layout=layout(candidate), status='pending_validation', evidence=[],
                                                   scope='Measured only if selected during search; no claimed execution support'))
                continue
            path = directory/'validation'/candidate['id']
            if not (path/'worker-result.json').exists():
                check_setup_budget(config, directory)
            inherited = config.get('inherited_max') if candidate['id'] == config['reference_candidate_id'] else None
            if inherited:
                result = read(inherited['receipt']['path'])
                result.update(origin='historical_max_reuse', local_execution_verified=False)
            else:
                result = attempt(config, configuration(candidate), path)
            result.update(candidate_id=candidate['id'], selection_split='calibration')
            runtime['structures'].append(dict(layout=layout(candidate), status=(
                                               'verified' if result['status'] == 'ok' else 'validation_failed'),
                                               evidence=[inherited['receipt'] if inherited else artifact(path/'worker-result.json')],
                                               local_execution_verified=not bool(inherited) and result['status'] == 'ok',
                                               evidence_origin='historical_max_reuse' if inherited else 'local_measurement',
                                               failure_reason=result.get('failure_reason'),
                                               scope=('Historical Max only; local execution remains unverified' if inherited else
                                                      'local calibration CLI/output/telemetry validation; actual DBO split count unobserved')))
            if result['status'] == 'ok':
                observations.append(result)
        write(directory/'inputs/runtime.json', runtime)
        write(directory/'inputs/profile.json', dict(selection_split='calibration', model=config['model'], candidates=[], external_observations=observations))
        if config['reference_candidate_id'] not in {r['candidate_id'] for r in observations}:
            raise RuntimeError('Frozen reference structure failed validation; cannot silently select another reference')
        anchor = next(r for r in observations if r['candidate_id'] == config['reference_candidate_id'])
        if config.get('slo_mode') == 'relative_max':
            m = anchor['metrics']
            config['limits'] = dict(ttft_ms=m['ttft_ms']*1.05, tpot_ms=m['tpot_ms']*1.05,
                                  min_output_tps=m['output_tps']*.95)
            if config.get('tbt_slo'):
                config['limits']['tbt_ms'] = m['tbt_ms'] * 1.05
            reference_path = directory/'inputs/slo-reference.json'
            value = dict(protocol='fixed_balanced_official_afd_max_relative_v1',
                         configuration=config['reference_configuration'], trace=config['trace'],
                         metrics=m, ratios=config['slo_ratios'], limits=config['limits'],
                         receipt=(config['inherited_max']['receipt'] if config.get('inherited_max') else
                                  artifact(directory/'validation'/config['reference_candidate_id']/'worker-result.json')),
                         interpretation='Latency at most 105% of Max; throughput at least 95%; faster is allowed',
                         selection_split='calibration', heldout_evaluated=False)
            if config.get('inherited_max'):
                value.update(origin='historical_max_reuse', local_measurement_performed=False,
                             source_reference=config['inherited_max']['reference'])
            if reference_path.exists() and read(reference_path) != value:
                raise ValueError('Frozen relative Max baseline changed')
            write(reference_path, value)
            context = read(config['context_manifest'])
            context['files_sha256'][str(reference_path)] = sha(reference_path)
            write(config['context_manifest'], context)
        receipts = setup_receipts(directory)
        setup = {k: sum(r['cost'][k] for r in receipts) for k in ('wall_seconds', 'gpu_hours', 'tuning_energy_j')}
        setup['evaluations'] = len(receipts)
        write(directory/'official-config.json', config)
        settings = dict(mode='physical', selection_split='calibration', model_id=config['model'],
                        mechanism_model='external_power_duration_v1', require_four_stage=False,
                        require_output_correctness=config.get('output_validation', 'exact_tokens') == 'exact_tokens',
                        allow_structure_probes=direct, limits=config['limits'], budget=config['budget'], setup_cost=setup,
                        default_energy_j=next(r['metrics']['energy_j'] for r in observations if r['candidate_id'] == config['reference_candidate_id']),
                        reference_candidate_id=config['reference_candidate_id'],
                        workload=dict(arrival_rate_rps=config['rps'], official_config_sha256=sha(directory/'official-config.json')),
                        context_files=dict(model_config=str(Path(config['model_path'])/'config.json'),
                                           plugin_source=str(ROOT/'environment/official-runtime.lock.json'),
                                           launcher=str(ROOT/'bo_dse/official_worker.py'), context_manifest=config['context_manifest'],
                                           official_config=str(directory/'official-config.json'), correctness_reference=config['correctness_reference_path']),
                        bo=dict(seed=config['seeds'][0], initial_parameter_probes=4, initial_joint_probes=0, analytical_probe_every=0,
                                evaluation_seconds=(anchor['cost']['wall_seconds'] if config.get('slo_mode') == 'relative_max'
                                                    else max(r['cost']['wall_seconds'] for r in receipts)),
                                knob_switch_seconds=120., structure_switch_seconds=120.),
                        profile_trace_files=[config['trace']['path']], identity_fields=['source_index', 'source_timestamp'])
        if config.get('exploration_policy'):
            settings['bo']['exploration_policy'] = config['exploration_policy']
        if config.get('inherited_max'):
            settings['reference_evidence_origin'] = 'historical_max_reuse'
        if config.get('exploration_policy') == 'capacity_v2':
            settings['capacity_profile'] = read(directory/'inputs/capacity.json')
            settings['context_files']['capacity_profile'] = str(directory/'inputs/capacity.json')
        if direct:
            settings['context_files'].pop('correctness_reference')
        if config.get('slo_mode') == 'relative_max':
            settings['context_files']['slo_reference'] = str(directory/'inputs/slo-reference.json')
        if config.get('prior_arm_costs'):
            settings['prior_arm_costs'] = config['prior_arm_costs']
            settings['context_files']['prior_arm_costs'] = str(directory/'inputs/prior-arm-costs.json')
        for key, name in (('specification', 'specification.json'), ('hardware', 'hardware.json'), ('runtime', 'runtime.json'),
                          ('profile', 'profile.json'), ('calibration_trace', 'calibration.jsonl'), ('heldout_trace', 'heldout.jsonl')):
            settings[key] = str(directory/'inputs'/name)
        write(directory/'campaign-settings.json', settings)
        target = directory/('comparison' if config['comparison'] else 'campaign')
        if config['comparison']:
            create_comparison(directory/'campaign-settings.json', target, seeds=config['seeds'], resume=target.exists(), methods=config.get('comparison_methods'))
        elif not target.exists():
            create_campaign(directory/'campaign-settings.json', target)
        ready = dict(backend='official', mechanism_model='external_power_duration_v1', target=str(target),
                     verified_structures=len(observations), attempted_structures=1 if direct else len(runtime['structures']),
                     pending_structures=len(runtime['structures'])-len(observations) if direct else 0,
                     output_validation=config.get('output_validation', 'exact_tokens'), direct_search=direct,
                     slo_mode=config.get('slo_mode', 'absolute'), limits=config['limits'],
                     evaluation_requests=config['trace']['requests'], prior_arm_costs=config.get('prior_arm_costs', {}),
                     setup_cost_per_arm=setup, setup_energy_complete=all(r.get('cost_energy_complete', False) for r in receipts),
                     stage_times_available=False, heldout_evaluated=False)
        if config.get('inherited_max'):
            ready.update(verified_structures=0, attempted_structures=0,
                         pending_structures=len(runtime['structures'])-1, inherited_verified_structures=1,
                         inherited_max=True)
        write(directory/'READY.json', ready)
        return ready


def evaluate(config, request, directory):
    validate_context(config)
    if (request['mode'] != 'physical' or request['selection_split'] != 'calibration' or request['trace'] != config['trace']
            or digest({k: v for k, v in request.items() if k != 'request_sha256'}) != request['request_sha256']
            or digest(request['configuration']) != request['configuration_sha256']
            or request['configuration'] not in config['configurations']
            or request['workload']['official_config_sha256'] != sha(Path(config['directory'])/'official-config.json')):
        raise ValueError('Frozen official request/configuration mismatch')
    result = attempt(config, request['configuration'], directory)
    result.update({k: request[k] for k in ECHO}, trace_sha256=config['trace']['sha256'], origin='physical_measurement')
    write(directory/'result.json', result)
    return result


def run(config, one=False):
    directory = Path(config['directory'])
    ready = read(directory/'READY.json')
    target = Path(ready['target'])
    campaigns = read(target/'comparison.json')['campaigns'] if config['comparison'] else [dict(directory=str(target))]
    import random
    from resident_cost import Session
    if config.get('resident_search'):
        config = dict(config, resident_session_directory=str(directory/'resident-session'))
    with gpu_locks(config['gpus']), Session(config,campaigns) as resident:
        while True:
            active = False
            order = list(campaigns)
            random.Random(sum(status(c['directory'])['cost']['evaluations'] for c in order)).shuffle(order)
            for arm in order:
                path = Path(arm['directory'])
                if status(path)['frozen']:
                    continue
                request = ask(path)
                if request.get('stopped'):
                    if status(path)['best']:
                        freeze(path)
                    continue
                active = True
                result = evaluate(config, request, path/'trials'/request['trial_id'])
                result = resident.charge(result, path/'trials'/request['trial_id'])
                tell(path, result)
            if one or not active:
                break
    return comparison_report(target) if config['comparison'] else status(target)


def legacy_main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}; retain receipts and recover if needed')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description='Historical upstream-only duration/power prior; not the paper four-stage model.')
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('start')
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default='qwen36')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--reuse-max-directory', type=Path,
                   help='Reuse a compatible archived eight-GPU Max receipt and its frozen SLOs without remeasurement')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--microbatches', type=int, nargs='+', default=[1, 2])
    p.add_argument('--dbo-threshold-profiles', nargs='+', metavar='DECODE:PREFILL',
                   help='Paired threshold profiles; requires --microbatches 2')
    p.add_argument('--frequencies', type=int, nargs='+', default=[1050, 1170, 1290, 1350, 1410])
    p.add_argument('--power-caps', type=int, nargs='+', default=[200, 250, 300, 350, 400])
    p.add_argument('--rps', type=float, default=1.)
    p.add_argument('--evaluations', type=int, default=32)
    p.add_argument('--gpu-hours', type=float, default=8.)
    p.add_argument('--native-python', type=Path, default=ROOT/'.venv-official/bin/python')
    p.add_argument('--calibration', type=Path, default=ROOT/'inputs/traces/calibration-200.jsonl')
    p.add_argument('--heldout', type=Path, default=ROOT/'inputs/traces/heldout-400.jsonl')
    p.add_argument('--clock-url', default='http://127.0.0.1:9096')
    p.add_argument('--api-port', type=int, default=18000)
    p.add_argument('--afd-port', type=int, default=16000)
    p.add_argument('--dp-rpc-port', type=int, default=28000)
    p.add_argument('--trial-timeout-seconds', type=float, default=3600)
    p.add_argument('--ttft-ms', type=float, default=2000.)
    p.add_argument('--tpot-ms', type=float, default=100.)
    p.add_argument('--min-output-tps', type=float, default=1.)
    p.add_argument('--comparison', action='store_true')
    p.add_argument('--comparison-methods', nargs='+', choices=['v2','generic_bo','random','ga'],
                   help='Selected independent baselines; new comparisons default to all four')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--output-validation', choices=('exact_tokens', 'request_completion'), default='exact_tokens')
    p.add_argument('--direct-search', action='store_true',
                   help='Measure the fixed reference, then explore pending structures during search; no stock token gate')
    p.add_argument('--prior-failed-reference-directory', type=Path,
                   help='Carry all preparation cost from a frozen campaign that failed before structure validation')
    p.add_argument('--prior-run-directory', type=Path,
                   help='User-authorized protocol transition; carry common setup and matching per-arm search costs')
    p.add_argument('--evaluation-requests', type=int,
                   help='Freeze the first N calibration requests for every candidate and Max; audit the entire source')
    p.add_argument('--exploration-policy', choices=('broad_v2', 'capacity_v2'),
                   help='Policy for the model-guided arm; capacity_v2 starts at the smallest capacity-eligible high-clock structure')
    p.add_argument('--tbt-slo', action='store_true', help='Require pooled client token-interval p90 <= 105% of matched Max')
    p.add_argument('--slo-mode', choices=('absolute', 'relative_max'), default='absolute')
    p.add_argument('--resident-search', action='store_true', help='Reuse healthy identical service configurations during search')
    p.add_argument('--dry-run', action='store_true')
    for action in ('prepare', 'run', 'cleanup', 'recover'):
        p = sub.add_parser(action)
        p.add_argument('directory', type=Path)
        if action == 'run':
            p.add_argument('--one', action='store_true', help='One interleaved round when comparing')
    p = sub.add_parser('select-gpus')
    p.add_argument('--gpus', type=int, nargs='+')
    p.add_argument('--count', type=int, default=4, choices=[4])
    p.add_argument('--minimum-free-gib', type=float, default=65)
    p.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'select-gpus':
        from gpu_selection import inspect
        import pynvml
        result = inspect(pynvml, args.gpus, args.count, args.minimum_free_gib)
        if args.output:
            if args.output.exists():
                raise FileExistsError(args.output)
            write(args.output, result)
    elif args.action == 'start':
        config = initialize(args)
        result = config if args.dry_run else prepare(config)
        if not args.dry_run and not args.prepare_only:
            result = run(read(args.directory/'official-config.json'))
    else:
        config = read(args.directory/'official-config.json')
        if args.action == 'prepare':
            result = prepare(config)
        elif args.action == 'run':
            result = run(config, args.one)
        else:
            with gpu_locks(config['gpus']):
                for marker in sorted(args.directory.rglob('STARTED.json')):
                    path = marker.parent
                    if (path/'worker-result.json').exists():
                        receipt = read(path/'worker-result.json')
                        if receipt.get('cleanup_required'):
                            worker(config, 'cleanup', path)
                            # Preserve the immutable measurement receipt; append recovery evidence.
                            write(path/'CLEANUP_RESOLVED.json', dict(verified=True, original_receipt=artifact(path/'worker-result.json')))
                    elif args.action == 'recover':
                        recover_attempt(config, path)
            result = dict(action=args.action, directory=str(args.directory))
    print(json.dumps(result, indent=2, allow_nan=False))


def main(argv=None):
    from entrypoint import main as dispatch
    return dispatch(argv)


if __name__ == '__main__':
    main()

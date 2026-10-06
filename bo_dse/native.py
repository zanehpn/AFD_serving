#!/usr/bin/env python3
"""Paper ECO: measured four-stage calibration, FIFO scheduling and residual GP BO."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BO = ROOT / 'bo_dse'
sys.path.insert(0, str(ROOT / 'migration'))
sys.path.insert(0, str(BO / 'scripts/afd'))
from bo_topology import topologies
from bo_layout import deployment as layout_deployment, enumerate_layouts, replicas
from output_correctness import PROTOCOL as CORRECTNESS_PROTOCOL
from static_dse.campaign import (ask, create_campaign, file_hash, freeze, isolation,
                                 read_json, status, tell, write_json)
from static_dse.comparison import create_comparison, comparison_round, comparison_report
from static_dse.executor import run_one
from static_dse.feedback import attach_feedback, make_manifest
from static_dse.space import configuration, digest, enumerate_candidates, hard_filter, layout
from build_deepseek_v6_four_stage_profile import load_four_stage_groups, summarize_reference_workload
from four_stage_dse_v6.model import pipeline_time_ms
from four_stage_dse_v6.attention_workload import common_context_target
from static_dse.paper_model import MODEL as ENERGY_MODEL

MODELS = {'qwen36': 'Qwen3.6-35B-A3B', 'deepseek-v2-lite': 'DeepSeek-V2-Lite-Chat'}
# Dry-run metadata for the pinned models only. Physical preparation reads the
# actual model config and preflight verifies its locked SHA-256.
MODEL_EXPERTS = {'qwen36': 256, 'deepseek-v2-lite': 64}
ECHO = ('trial_id', 'candidate_id', 'configuration_sha256', 'context_sha256',
        'request_sha256', 'mode', 'selection_split')
RUNTIME_ENV_PREFIXES = ('CUDA_', 'NCCL_', 'VLLM_', 'TORCH_', 'PYTORCH_', 'OMP_', 'MKL_', 'OPENBLAS_', 'TOKENIZERS_')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_hash(path)}


def bo_runtime():
    return {'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'scipy', 'scikit-learn', 'nvidia-ml-py')}}


def environment(config):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('ECODEP_', *RUNTIME_ENV_PREFIXES)) and k not in ('PYTHONPATH', 'LD_LIBRARY_PATH')}
    env.update(config.get('runtime_environment', {}))
    env.update(config['environment'])
    env['PATH'] = str(Path(config['native_python']).parent) + os.pathsep + env.get('PATH', '')
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', VLLM_USE_V2_MODEL_RUNNER='0')
    return env


def invoke(config, arguments, **kwargs):
    return subprocess.run([config['native_python'], *map(str, arguments)],
                          cwd=ROOT, env=environment(config), check=True, **kwargs)


def validate_context(config):
    check(config.get('mechanism_model') == ENERGY_MODEL and config.get('require_four_stage') is True,
          'Paper execution requires a new four-stage FIFO campaign; historical configs cannot be silently resumed')
    check(config['bo_runtime'] == bo_runtime(), 'BO Python/package environment changed')
    manifest = read_json(config['context_manifest'])
    for name, expected in manifest['files_sha256'].items():
        check(file_hash(name) == expected, f'Frozen environment/input/source changed: {name}')


def validate_request(config, request):
    check(request['mode'] == 'physical' and request['selection_split'] == 'calibration',
          'Native BO only executes physical calibration requests')
    unsigned = {k: v for k, v in request.items() if k != 'request_sha256'}
    check(digest(unsigned) == request['request_sha256'], 'Request digest mismatch')
    check(digest(request['configuration']) == request['configuration_sha256'], 'Configuration digest mismatch')
    check(request['trace'] == config['trace'], 'Frozen calibration trace mismatch')
    check(file_hash(config['trace']['path']) == config['trace']['sha256'], 'Calibration trace changed')
    check(request['workload']['native_config_sha256'] == file_hash(config['config_path']), 'Native configuration changed')
    for item in request['context_files'].values():
        check(file_hash(item['path']) == item['sha256'], 'Request context file changed')
    validate_context(config)
    check(request['workload']['arrival_rate_rps'] == config['rps'], 'Offered load mismatch')
    c = request['configuration']
    check(c in config['configurations'], 'Candidate has no frozen native launch mapping')
    return c


def build_plan(config, request, directory):
    """Translate a single BO point into the existing native suite protocol."""
    c = validate_request(config, request)
    suite_id = 'bo-' + digest({'directory': str(directory.resolve()), 'request': request['request_sha256']})[:24]
    candidate_id = request['candidate_id']
    na, ne = len(c['attention_gpus']), len(c['expert_gpus'])
    template = read_json(ROOT / 'inputs/protocols' / config['model'] / 'calibration-max-deployment.json')
    template['plugin']['root'] = config['plugin_root']
    template['plugin']['commit'] = config['plugin_commit']
    deploy = copy.deepcopy(template)
    deploy.update(method='mechanism_assisted_bo', arm='ours', candidate_id=candidate_id,
                  evaluation_split='calibration', selection_split='calibration',
                  calibration_trace=config['trace']['path'], calibration_trace_sha256=config['trace']['sha256'],
                  formal_evaluation_eligible=False, math_dse_parameter_probe=True)
    deploy['topology'] = (layout_deployment(c) if config.get('joint_space') else
                          dict(attention_dp=na, ffn_ep=ne, attention_tp=1, expert_tp=1))
    if config.get('joint_space'):
        deploy['runtime_contract']['microbatches'] = c['microbatches']
        deploy['runtime_contract']['enable_dbo'] = c['microbatches'] == 2
        deploy['runtime_contract'].update(config['microbatch_thresholds'])
    deploy['runtime_contract']['stage_trace'] = True
    point = {'candidate_id': candidate_id, 'request_path_clock_transitions': 0}
    for role, ids in (('attention', c['attention_gpus']), ('expert', c['expert_gpus'])):
        point[role + '_mhz'] = [c[role + '_mhz']] * len(ids)
        point[role + '_power_w'] = [c[role + '_power_w']] * len(ids)
    deploy['operating_points'] = {'selection_split': 'calibration', 'candidate_id': candidate_id,
                                'calibration_trace_sha256': config['trace']['sha256'],
                                'by_rps': {str(config['rps']): point}}
    entry = {'position': 1, 'arm': candidate_id, 'repetition': 1,
             'rates_rps': [config['rps']], 'suite_id': suite_id}
    plan = {'campaign': str(directory.resolve()), 'model_tag': MODELS[config['model']],
            'model': config['model'], 'template': template, 'allocation': config['gpus'],
            'measurement_gpus': c['attention_gpus'] + c['expert_gpus'],
            'energy_boundary': 'active_gpus_only',
            'request_count': request['trace']['requests'], 'trace_path': config['trace']['path'],
            'evaluation_split': 'calibration', 'schedule': [entry],
            'candidates': [{'id': candidate_id, 'topology': f'{na}a{ne}e'}],
            'contract': {'max_output_tokens': config['max_output_tokens']}}
    if config.get('joint_space'):
        plan['configuration'] = c
        plan['microbatch_thresholds'] = config['microbatch_thresholds']
        plan['output_correctness'] = {'contract': config['correctness'], 'reference': artifact(
            Path(config['config_path']).parent / 'inputs/correctness-reference.json')}
        plan['replicas'] = replicas(c, int(config['environment']['ECODEP_API_PORT']),
                                   int(config['environment']['ECODEP_EXPERT_API_PORT']),
                                   int(config['environment']['ECODEP_AFD_PORT']),
                                   int(config['environment']['ECODEP_DP_RPC_BASE_PORT']))
    (directory / 'deployments').mkdir()
    write_json(directory / 'deployments' / (candidate_id + '.json'), deploy)
    write_json(directory / 'schedule.json', {'selection_split': 'calibration', 'entries': [entry]})
    write_json(directory / 'PLAN.json', plan)
    return plan


def suite_paths(plan):
    suite_id = plan['schedule'][0]['suite_id']
    return (ROOT / 'results/afd_suites' / plan['model_tag'] / suite_id,
            ROOT / 'results/afd_serving' / plan['model_tag'] / suite_id)


def suite_command(config, plan, request):
    c = request['configuration']
    env = environment(config)
    env.update(ECODEP_BO_FEEDBACK='1',
               ECODEP_ATTENTION_GPUS=','.join(map(str, c['attention_gpus'])),
               ECODEP_EXPERT_GPUS=','.join(map(str, c['expert_gpus'])),
               ECODEP_MEASUREMENT_GPUS=','.join(map(str, plan['measurement_gpus'])),
               ECODEP_SCHEDULE_PATH=str(Path(plan['campaign']) / 'schedule.json'),
               ECODEP_SCHEDULE_POSITION='1', ECODEP_REPETITION_ID='1',
               ECODEP_WARMUP_TRACE=config['warmup_trace'], ECODEP_WARMUP_MANIFEST=config['warmup_manifest'],
               ECODEP_MAX_OUTPUT_TOKENS=str(config['max_output_tokens']),
               ECODEP_ATTENTION_CLOCKS_MHZ=','.join(['1410'] * len(c['attention_gpus'])),
               ECODEP_EXPERT_CLOCKS_MHZ=','.join(['1410'] * len(c['expert_gpus'])),
               ECODEP_ATTENTION_POWER_W=','.join(['400'] * len(c['attention_gpus'])),
               ECODEP_EXPERT_POWER_W=','.join(['400'] * len(c['expert_gpus'])))
    if config.get('joint_space'):
        env.update({('ECODEP_' + k.upper()): str(v) for k, v in config['microbatch_thresholds'].items()})
    cmd = ['bash', str(ROOT / 'scripts/afd/run_v026_ours_rate_suite.sh'),
           str(Path(plan['campaign']) / 'deployments' / (request['candidate_id'] + '.json')),
           config['trace']['path'], plan['schedule'][0]['suite_id'], str(config['rps'])]
    return cmd, env


@contextmanager
def gpu_locks(gpus):
    # Shared host paths also coordinate different clones of this BO executor.
    handles = []
    try:
        for gpu in sorted(gpus):
            path = str(Path(tempfile.gettempdir()) / f'moe-bo-gpu-{gpu}.lock')
            try:
                # flock does not require write access. Avoid O_CREAT on an
                # existing other-user file in the shared temporary directory (protected_regular).
                handle = open(path, 'r')
            except FileNotFoundError:
                try:
                    handle = open(path, 'x')
                except FileExistsError:
                    handle = open(path, 'r')
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for handle in handles:
            handle.close()


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def stop_meter(directory):
    record = directory / 'meter-process.json'
    if not record.exists():
        return
    owner = read_json(record)
    pid, started = owner['pid'], owner['start_ticks']
    if started is None or process_identity(pid) != started:
        return
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 45
    while process_identity(pid) == started and time.monotonic() < deadline:
        time.sleep(.1)
    if process_identity(pid) == started:
        os.killpg(pid, signal.SIGKILL)


def cleanup(config, plan):
    stop_meter(Path(plan['campaign']))
    suite, run = suite_paths(plan)
    invoke(config, [ROOT / 'migration/native_service.py', 'stop', run])
    # Never reset GPUs unless this suite acquired them before starting service.
    if (suite / 'allocation-reset-before.json').exists():
        invoke(config, [ROOT / 'scripts/afd/reset_gpus.py', '--url', config['environment']['ECODEP_CLOCK_URL'],
                        '--gpus', ','.join(map(str, plan['measurement_gpus'])),
                        '--output', Path(plan['campaign']) / 'recovery-reset.json'], stdout=subprocess.DEVNULL)


def measured_suite(config, plan, request, directory):
    """Meter loading through cleanup, separately from serving-window energy."""
    cmd, env = suite_command(config, plan, request)
    meter = [config['native_python'], str(BO / 'scripts/afd/measure_command.py'),
             '--gpus', ','.join(map(str, plan['measurement_gpus'])), '--interval-ms', '100',
             '--output', str(directory / 'tuning-telemetry.json'),
             '--samples-output', str(directory / 'tuning-power.jsonl'), '--', *cmd]
    with (directory / 'native.log').open('a') as log:
        process = subprocess.Popen(meter, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        write_json(directory / 'meter-process.json', {'pid': process.pid, 'start_ticks': process_identity(process.pid)})
        try:
            returncode = process.wait(timeout=config['trial_timeout_seconds'])
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            cleanup(config, plan)
            raise
    check(returncode == 0, f'Native suite exited {returncode}; inspect {directory / "native.log"}')


def receipt_cost(config, directory):
    journal = read_json(directory / 'STARTED.json')
    active = read_json(directory / 'PLAN.json')['measurement_gpus']
    check(active and len(active) == len(set(active)) and set(active) <= set(config['gpus']),
          'Tuning energy GPU mapping differs from the allocation')
    maximum_power = dict(zip(config['gpus'], config['maximum_power_w']))
    elapsed = max(.001, time.time() - journal['started_wall_seconds'])
    path = directory / 'tuning-telemetry.json'
    if path.exists():
        telemetry = read_json(path)
        if (telemetry['sample_error_count'] == 0 and telemetry['sample_time_coverage'] >= .99
                and telemetry.get('gpu_ids') == active
                and math.isfinite(telemetry['energy_j']) and telemetry['energy_j'] > 0):
            return {'wall_seconds': elapsed, 'gpu_hours': len(config['gpus']) * elapsed / 3600,
                    'tuning_energy_j': telemetry['energy_j']}, 'nvml_full_suite_window'
    # Crash receipts must not report missing energy as zero. Explicit conservative
    # bound, not a measured label; the trial remains failed and cannot be an incumbent.
    return {'wall_seconds': elapsed, 'gpu_hours': len(config['gpus']) * elapsed / 3600,
            'tuning_energy_j': sum(maximum_power[g] for g in active) * elapsed}, 'power_limit_upper_bound_after_meter_failure'


def measured_groups(cell):
    telemetry = read_json(cell / 'telemetry.json')
    with tempfile.TemporaryDirectory(prefix='bo-bootstrap-shapes-') as directory:
        paths = []
        for source in sorted((cell / 'traces').glob('stage-*.jsonl')):
            rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            rows = [r for r in rows if telemetry['started_wall_ns'] <= r['start_wall_ns'] <= r['end_wall_ns'] <= telemetry['finished_wall_ns']]
            target = Path(directory) / source.name
            target.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            paths.append(target)
        return load_four_stage_groups(paths, minimum_complete_coverage=.95, token_scope="attention_dp_total")[0]


def execute(config, request, result_path, recover=False, bootstrap=False):
    validate_request(config, request)
    directory = result_path.parent / 'native'
    blocked = Path(config['config_path']).parent / 'CLEANUP_REQUIRED.json'
    check(not blocked.exists(), 'Previous cleanup failed. Run native.py cleanup DIRECTORY before continuing')
    if result_path.exists():
        cached = read_json(result_path)
        for item in cached.get('artifacts', []):
            check(file_hash(item['path']) == item['sha256'], 'Cached result artifact changed')
        return cached
    with gpu_locks(config['gpus']):
        if directory.exists():
            check(recover, 'Interrupted native trial retained. Use native.py recover; never overwrite or rerun it.')
            plan = read_json(directory / 'PLAN.json')
            check(read_json(directory / 'request.json') == request, 'Recovery request differs')
        else:
            check(not recover, 'No native attempt to recover')
            directory.mkdir(parents=True)
            write_json(directory / 'request.json', request)
            plan = build_plan(config, request, directory)
            write_json(directory / 'STARTED.json', {'started_wall_seconds': time.time(),
                                                   'request_sha256': request['request_sha256']})
        receipt = {key: request[key] for key in ECHO}
        receipt.update(origin='physical_measurement', trace_sha256=request['trace']['sha256'])
        try:
            if recover:
                cleanup(config, plan)
                raise RuntimeError('Interrupted attempt recovered and charged; no automatic remeasurement')
            check(os.geteuid() == 0, 'Native GPU execution requires root on the target server')
            invoke(config, [ROOT / 'migration/runtime_fingerprint.py', '--check', ROOT / 'environment/native-runtime.json'])
            invoke(config, [BO / 'native_backend.py', 'verify-platform'])
            invoke(config, [ROOT / 'scripts/afd/check_gpu_availability.py', '--gpus', ','.join(map(str, config['gpus'])),
                            '--minimum-free-gib', '65', '--stable-for-s', '2', '--wait-timeout-s', '30'],
                   stdout=subprocess.DEVNULL)
            measured_suite(config, plan, request, directory)
            invoke(config, [BO / 'native_backend.py', 'validate', '--plan', directory / 'PLAN.json',
                            '--output', directory / 'validation.json'])
            _, run = suite_paths(plan)
            cell = run / f'rps-{config["rps"]}'
            receipt.update(status='ok', requests=request['trace']['requests'],
                           completed_requests=request['trace']['requests'], failed_requests=0,
                           execution_verified=True, telemetry_valid=True)
            receipt['cost'], receipt['cost_energy_source'] = receipt_cost(config, directory)
            check(receipt['cost_energy_source'] == 'nvml_full_suite_window', 'Full tuning energy telemetry invalid')
            receipt['artifacts'] = [artifact(directory / 'validation.json'), artifact(directory / 'tuning-telemetry.json'),
                                    artifact(directory / 'PLAN.json')]
            if config.get('joint_space'):
                receipt['output_correctness'] = read_json(directory / 'validation.json')[0]['output_correctness']
                receipt['artifacts'].extend([artifact(cell / 'correctness-report.json'),
                                             plan['output_correctness']['reference']])
            if bootstrap:
                groups = measured_groups(cell)
                # Bootstrap has no frozen normalization target yet. Derive it from
                # calibration stage shapes, then verify using the exact same importer.
                request = copy.deepcopy(request)
                request['model_workload'].update(summarize_reference_workload(groups, 1.))
                request['model_workload_sha256'] = digest(request['model_workload'])
                request['request_sha256'] = digest({k: v for k, v in request.items() if k != 'request_sha256'})
                receipt['request_sha256'] = request['request_sha256']
                write_json(directory / 'normalized-bootstrap-request.json', request)
            manifest = directory / 'feedback-manifest.json'
            write_json(manifest, make_manifest(request, cell, config['layers']))
            receipt = attach_feedback(request, receipt, manifest)
            receipt['normalization_workload'] = request['model_workload']
        except Exception as error:
            cleanup_error = None
            try:
                cleanup(config, plan)
            except Exception as problem:
                cleanup_error = str(problem)
                write_json(blocked, {'plan': artifact(directory / 'PLAN.json'), 'reason': cleanup_error})
            receipt.update(status='failed', failure_reason=f'{type(error).__name__}: {error}',
                           cleanup_error=cleanup_error)
            receipt['cost'], receipt['cost_energy_source'] = receipt_cost(config, directory)
            write_json(directory / 'failure.json', {'reason': receipt['failure_reason'], 'cleanup_error': cleanup_error,
                                                   'cost_energy_source': receipt['cost_energy_source']})
            receipt['artifacts'] = [artifact(directory / 'failure.json'), artifact(directory / 'request.json')]
            receipt['artifacts'].extend(artifact(directory / name) for name in
                                        ('STARTED.json', 'PLAN.json', 'tuning-telemetry.json')
                                        if (directory / name).exists())
        # GPU allocation locks are held through validation/import and recovery.
        # Include that elapsed time as well as the native serving process lifetime.
        receipt['cost'], receipt['cost_energy_source'] = receipt_cost(config, directory)
        write_json(result_path, receipt)
        return receipt


def preflight(config):
    check(os.geteuid() == 0, 'Run native preparation as root on the A100 server')
    invoke(config, [ROOT / 'migration/preflight.py'])
    invoke(config, [ROOT / 'migration/runtime_fingerprint.py', '--output', ROOT / 'environment/native-runtime.json'])


def initial_inputs(args):
    directory = args.directory.resolve()
    check(not directory.exists(), 'Output exists; use run/resume or a new directory')
    model_config = ROOT / 'artifacts/models' / MODELS[args.model] / 'config.json'
    model_data = read_json(model_config)
    model_data = model_data.get('text_config', model_data)
    num_experts = model_data.get('num_experts', model_data.get('n_routed_experts'))
    check(num_experts is not None, 'Model config must declare its expert count')
    structures = topologies(args.gpus, num_experts)
    joint = getattr(args, 'space_mode', 'legacy') == 'joint'
    if joint:
        layouts, exclusions = enumerate_layouts(args.gpus, num_experts, args.microbatches,
                                                args.parallel_degrees, model_data)
        check(layouts, 'No native layouts match the requested parallel/microbatch space')
        reference = {k: v for k, v in structures[0].items() if k != 'id'}
        reference.update(microbatches=2 if 2 in args.microbatches else args.microbatches[0], execution_mode='eager')
        check(reference in layouts, 'Reference layout is absent from the requested degree/microbatch grid')
        structures = list({digest({k: v for k, v in c.items() if k not in ('microbatches', 'execution_mode')}):
                           {k: v for k, v in c.items() if k not in ('microbatches', 'execution_mode')} for c in layouts}.values())
    check(args.evaluations >= (4 if joint else len(structures) + 2) and args.gpu_hours > 0,
          'Budget must include stock correctness reference (joint), topology bootstrap and two BO observations')
    check(args.trial_timeout_seconds > 0, 'Trial timeout must be positive')
    audit = isolation(args.calibration, args.heldout, ['source_index', 'source_timestamp'])
    rows = [json.loads(line) for line in args.calibration.read_text().splitlines() if line.strip()]
    check(all(r.get('evaluation_split') == 'calibration' for r in rows), 'Calibration trace labels required')
    check(len(rows) > 8, 'Calibration must contain more than eight warmup requests')
    check(args.seeds and len(set(args.seeds)) == len(args.seeds) and all(s >= 0 for s in args.seeds), 'Seeds must be distinct nonnegative integers')
    config = {'schema_version': 1, 'model': args.model, 'gpus': args.gpus, 'rps': args.rps,
              'mechanism_model': ENERGY_MODEL, 'require_four_stage': True,
              'joint_space': joint,
              'energy_accounting': {'serving': 'active_gpus_only', 'tuning': 'active_gpus_only',
                                    'gpu_hours': 'reserved_allocation'},
              'bo_runtime': bo_runtime(),
              'native_python': str(args.native_python.absolute()), 'max_output_tokens': 128,
              'trial_timeout_seconds': args.trial_timeout_seconds,
              'runtime_environment': {k: v for k, v in os.environ.items()
                                      if (k.startswith(RUNTIME_ENV_PREFIXES) or k == 'LD_LIBRARY_PATH')
                                      and k != 'CUDA_VISIBLE_DEVICES'},
              'environment': {'ECODEP_CONTAINER_RUNTIME': 'native', 'ECODEP_MODELS': args.model,
                              'ECODEP_BO_ALLOCATION_SIZE': str(len(args.gpus)),
                              'ECODEP_ATTENTION_GPUS': ','.join(map(str, (reference if joint else structures[0])['attention_gpus'])),
                              'ECODEP_EXPERT_GPUS': ','.join(map(str, (reference if joint else structures[0])['expert_gpus'])),
                              'ECODEP_CLOCK_URL': args.clock_url,
                              'ECODEP_ATTENTION_CLOCK_URL': args.clock_url, 'ECODEP_EXPERT_CLOCK_URL': args.clock_url,
                              'ECODEP_API_PORT': str(args.api_port), 'ECODEP_EXPERT_API_PORT': str(args.api_port + 1),
                              'ECODEP_AFD_PORT': str(args.afd_port), 'ECODEP_DP_RPC_BASE_PORT': str(args.dp_rpc_port),
                              'ECODEP_GPU_WAIT_TIMEOUT_S': '120', 'ECODEP_GPU_STABLE_FOR_S': '2',
                              'ECODEP_MINIMUM_FREE_GIB': '65', 'ECODEP_SERVER_TIMEOUT_S': '1800'},
              'config_path': str(directory / 'native-config.json'),
              'context_manifest': str(directory / 'context.json')}
    preflight(config)
    # Installation checks precede campaign creation. No calibration has run yet.
    directory.mkdir(parents=True)
    inputs = directory / 'inputs'
    inputs.mkdir()
    for name, path in (('calibration', args.calibration), ('heldout', args.heldout)):
        (inputs / (name + '.jsonl')).write_bytes(path.read_bytes())
    warmup = inputs / 'warmup.jsonl'
    warmup.write_text(''.join(json.dumps(r) + '\n' for r in rows[:8]))
    check(len(rows) > 8, 'Warmup must differ from full calibration replay')
    config['trace'] = {**audit['calibration'], 'path': str(inputs / 'calibration.jsonl')}
    if joint:
        config['microbatch_thresholds'] = {'dbo_decode_token_threshold': 2, 'dbo_prefill_token_threshold': 12}
        config['correctness'] = {'protocol': CORRECTNESS_PROTOCOL, 'selection_split': 'calibration',
                                 'trace': config['trace'], 'model_config': artifact(model_config),
                                 'reference_tp': 2, 'reference_gpus': args.gpus[:2],
                                 'max_output_tokens': config['max_output_tokens'],
                                 'generation': {'temperature': 0, 'seed': 0, 'ignore_eos': True,
                                                'generation_config': 'vllm'}}
    config['heldout'] = str(inputs / 'heldout.jsonl')
    config['warmup_trace'], config['warmup_manifest'] = str(warmup), str(inputs / 'warmup-manifest.json')
    write_json(inputs / 'warmup-manifest.json', {'selection_split': 'calibration', 'source': config['trace']['path'],
                                               'source_sha256': config['trace']['sha256'], 'output': str(warmup),
                                               'output_sha256': file_hash(warmup)})
    from capture_hardware import capture
    import pynvml
    hardware = capture(args.gpus, pynvml)
    if joint:
        config['correctness']['reference_gpu_uuids'] = [hardware['devices'][str(g)]['uuid'] for g in args.gpus[:2]]
    selection_path = getattr(args, 'gpu_selection', None)
    if selection_path:
        selection = read_json(selection_path)
        expected = {r['index']: r['uuid'] for r in selection['devices']}
        check(all(hardware['devices'][str(i)]['uuid'] == expected[i] for i in args.gpus),
              'Physical GPU UUID changed since selection; select GPUs again')
        write_json(inputs / 'gpu-selection.json', selection)
    write_json(inputs / 'hardware.json', hardware)
    config['maximum_power_w'] = [hardware['devices'][str(g)]['max_power_w'] for g in args.gpus]
    spec = read_json(BO / 'configs/space.example.json')
    spec.update(enumerate_parallelism=False, microbatches=list(args.microbatches) if joint else [2], execution_mode='eager')
    spec['topologies'] = structures
    runtime = read_json(BO / 'configs/runtime.example.json')
    runtime.update(model_id=MODELS[args.model], gpu_budget=len(args.gpus), allowed_gpus=args.gpus, structures=[],
                   connector='p2p_nccl',
                   scope='Native P2P rate-suite: A_DP=A cards, all TP=1, E_DP=1, EP=E cards, M=2, eager')
    if joint:
        runtime.update(adapter='v026_joint_replicas', execution_modes=['eager'],
                       scope='Independent AFD replicas, native EP or intra-expert TP, configurable eager microbatches')
        write_json(inputs / 'layout-exclusions.json', exclusions)
    config['layers'] = model_data['num_hidden_layers']
    runtime['divisibility'] = {'expert_ep': num_experts}
    candidates = enumerate_candidates(spec)
    candidates = [c for c in candidates if hard_filter(c, hardware, runtime)['status'] != 'hard_rejected']
    check(candidates, 'No candidate operating points supported on this hardware')
    config['configurations'] = [configuration(c) for c in candidates]
    config['references'] = [configuration(c) for c in candidates
                            if all(c['knobs'][k] == v for k, v in (('attention_mhz', 1410), ('expert_mhz', 1410),
                                                                  ('attention_power_w', 400), ('expert_power_w', 400)))]
    if joint:
        config['references'] = [c for c in config['references'] if all(c[k] == v for k, v in reference.items())]
    check(len(config['references']) == (1 if joint else len(structures)),
          'A100 MAX reference or model-divisible EP unavailable for a requested topology')
    template = read_json(ROOT / 'inputs/protocols' / args.model / 'calibration-max-deployment.json')
    config['plugin_root'] = str(ROOT / 'third_party' / Path(template['plugin']['root']).name)
    config['plugin_commit'] = read_json(ROOT / 'environment/plugins.lock.json')['targets'][Path(config['plugin_root']).name]
    config['budget'] = {'evaluations': args.evaluations, 'gpu_hours': args.gpu_hours}
    config['comparison'], config['seeds'] = args.comparison, args.seeds
    config['limits'] = None if args.ttft_ms is None else {'ttft_ms': args.ttft_ms, 'tpot_ms': args.tpot_ms,
                                                        'min_output_tps': args.min_output_tps}
    check(config['limits'] is None or all(v is not None and math.isfinite(v) and v > 0 for v in config['limits'].values()),
          'Provide all three positive absolute SLO limits, or omit all to use baseline ratios')
    check(args.ttft_ms is not None or (args.tpot_ms is None and args.min_output_tps is None), 'Incomplete SLO limits')
    write_json(inputs / 'space.json', spec)
    write_json(inputs / 'runtime.json', runtime)
    write_json(inputs / 'profile.json', {'model': MODELS[args.model], 'selection_split': 'calibration', 'candidates': []})
    write_json(directory / 'isolation.json', audit)
    files = {model_config, ROOT / 'environment/native-runtime.json', ROOT / 'environment/native-platform.json'}
    for base in ('bo_dse', 'migration', 'scripts', 'services', 'environment'):
        files.update(p for p in (ROOT / base).rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.txt')
                     and directory not in p.parents
                     and not any(part in ('.venv', 'results', '__pycache__', '.pytest_cache') for part in p.relative_to(ROOT / base).parts))
    files.update(p for p in Path(config['plugin_root']).rglob('*.py') if '__pycache__' not in p.parts)
    files.update(inputs.glob('*.jsonl'))
    files.add(inputs / 'warmup-manifest.json')
    if selection_path:
        files.add(inputs / 'gpu-selection.json')
    files.add(ROOT / 'inputs/protocols' / args.model / 'calibration-max-deployment.json')
    write_json(directory / 'context.json', {'files_sha256': {str(p): file_hash(p) for p in sorted(files)}})
    write_json(directory / 'native-config.json', config)
    return config


def bootstrap_request(config, c, index):
    req = {'schema_version': 1, 'trial_id': f'bootstrap-{index:02d}', 'candidate_id': 'dse-' + digest(c)[:20],
           'configuration': c, 'configuration_sha256': digest(c), 'context_sha256': file_hash(config['context_manifest']),
           'mode': 'physical', 'selection_split': 'calibration', 'trace': config['trace'],
           'workload': {'four_stage_token_scope': 'attention_dp_total', 'arrival_rate_rps': config['rps'], 'native_config_sha256': file_hash(config['config_path'])},
           'model_workload': {}, 'model_workload_sha256': digest({}),
           'context_files': {'context:native': artifact(config['context_manifest'])}}
    req['request_sha256'] = digest(req)
    return req


def prepare(config, retry_failed=False):
    directory = Path(config["config_path"]).parent
    validate_context(config)
    with (directory / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "READY.json").exists():
            return read_json(directory / "campaign-config.json")
        return prepare_locked(config, retry_failed)


def prepare_locked(config, retry_failed=False):
    directory = Path(config['config_path']).parent
    validate_context(config)
    results, completed_dirs, all_attempts = [], [], []
    if config.get('joint_space'):
        import correctness_reference
        all_attempts.extend(correctness_reference.prepare(config, retry_failed, SimpleNamespace(**globals())))
    for index, c in enumerate(config['references']):
        base = directory / f'bootstrap-{index:02d}'
        paths = [base, *sorted(directory.glob(f'bootstrap-{index:02d}-retry-*'))]
        paths = [p for p in paths if p.is_dir()]
        request = bootstrap_request(config, c, index)
        for path in paths:
            if not (path / 'result.json').exists():
                check(retry_failed, f'Interrupted bootstrap at {path}; use prepare --retry-failed to recover and charge it')
                execute(config, request, path / 'result.json', recover=True, bootstrap=True)
            result = execute(config, request, path / 'result.json', bootstrap=True)
            all_attempts.append(result)
        if not paths or result['status'] != 'ok':
            check(not paths or retry_failed, f'Bootstrap failed at {paths[-1] if paths else base}; inspect logs, then use prepare --retry-failed')
            check(len(all_attempts) < config['budget']['evaluations'] - 1 and
                  sum(r['cost']['gpu_hours'] for r in all_attempts) < config['budget']['gpu_hours'],
                  'Bootstrap exhausted the frozen budget; preserve its costs and results')
            path = base if not paths else directory / f'bootstrap-{index:02d}-retry-{len(paths):03d}'
            path.mkdir()
            result = execute(config, request, path / 'result.json', bootstrap=True)
            all_attempts.append(result)
            check(result['status'] == 'ok', f'Bootstrap failed at {path}; inspect logs, then use prepare --retry-failed')
        else:
            path = paths[-1]
        results.append(result)
        completed_dirs.append(path)
    # Freeze a common actually observed shape before BO. This avoids inventing a
    # zero-token target or normalizing each topology to a different workload.
    cells = [suite_paths(read_json(path / 'native/PLAN.json'))[1] / f'rps-{config["rps"]}'
             for path in completed_dirs]
    group_sets = [measured_groups(cell) for cell in cells]
    shapes = [{(r['prefill_tokens'], r['decode_tokens']) for r in groups} for groups in group_sets]
    shared = sorted(set.intersection(*shapes))
    check(shared, 'Bootstrap topologies have no common observed token shape; cannot normalize matched stage feedback')
    decode = [shape for shape in shared if shape[0] == 0 and shape[1] > 0]
    target = (decode or shared)[len(decode or shared) // 2]
    workload = {'prefill_tokens_per_microbatch': target[0], 'decode_tokens_per_microbatch': target[1],
                'communication_bytes_scale': 1., 'routing_imbalance': 1.}
    context_target, context_status = common_context_target(group_sets, workload)
    workload.update(context_target)
    write_json(directory / 'inputs/attention-workload-normalization.json', {
        'selection_split': 'calibration', 'status': context_status, 'target': context_target,
        'token_workload': workload,
        'evidence': [artifact(path / 'result.json') for path in completed_dirs],
        'fallback': 'token-only prior if context support is missing; no fabricated zero KV load'})
    anchors = []
    candidates = enumerate_candidates(read_json(directory / 'inputs/space.json'))
    for i, (cfg, cell) in enumerate(zip(config['references'], cells)):
        request = bootstrap_request(config, cfg, i)
        request.update(model_workload=workload, model_workload_sha256=digest(workload))
        request['request_sha256'] = digest({k: v for k, v in request.items() if k != 'request_sha256'})
        item = completed_dirs[i] / 'shared-manifest.json'
        write_json(item, make_manifest(request, cell, config['layers']))
        receipt = {**results[i], 'request_sha256': request['request_sha256']}
        results[i] = attach_feedback(request, receipt, item)
        results[i]['normalization_workload'] = workload
        write_json(completed_dirs[i] / 'shared-result.json', results[i])
        stage = results[i]['four_stage']
        candidate = next(c for c in candidates if configuration(c) == cfg)
        anchors.append({**candidate, 'validation_status': 'physically_measured_anchor',
                        'parallelism_legal': True, 'memory_feasible': True,
                        'stage_models': stage['stage_models'], 'operating_state': stage['operating_state'],
                        'analytical_provisioning': stage['analytical_provisioning'],
                        'power_model': {'idle_intercept_w': stage['power_w'], 'dynamic_slope_w': 0.,
                                        'total_power_cap_w': sum(len(cfg[r + '_gpus']) * cfg[r + '_power_w'] for r in ('attention', 'expert'))},
                        'layers': stage['layers'], 'requests_per_pipeline': stage['requests_per_pipeline'],
                        'guard_pipeline_time_ms': pipeline_time_ms(stage['stage_ms'], microbatches=cfg['microbatches'], layers=stage['layers'],
                                                                  schedule_model=stage['analytical_provisioning']['schedule_model'])})
    inputs = directory / 'inputs'
    write_json(inputs / 'profile.json', {'selection_split': 'calibration', 'model': MODELS[config['model']],
                                        'reference_workload': workload, 'candidates': anchors})
    runtime = read_json(inputs / 'runtime.json')
    runtime['structures'] = [{'layout': layout(next(c for c in candidates if configuration(c) == cfg)),
                              'status': 'verified', 'evidence': [artifact(completed_dirs[i] / 'shared-result.json')]}
                             for i, cfg in enumerate(config['references'])]
    write_json(inputs / 'runtime.json', runtime)
    baseline = results[0]
    limits = config['limits'] or {'ttft_ms': baseline['metrics']['ttft_ms'] * 1.05,
                                  'tpot_ms': baseline['metrics']['tpot_ms'] * 1.05,
                                  'min_output_tps': baseline['metrics']['output_tps'] * .95}
    settings = {'selection_split': 'calibration', 'mode': 'physical', 'model_id': MODELS[config['model']],
                'mechanism_model': ENERGY_MODEL,
                **{k: str(inputs / (k + '.json')) for k in ('hardware', 'runtime', 'profile')},
                'specification': str(inputs / 'space.json'), 'calibration_trace': config['trace']['path'],
                'heldout_trace': config['heldout'], 'profile_trace_files': [config['trace']['path']],
                'context_files': {'model_config': str(ROOT / 'artifacts/models' / MODELS[config['model']] / 'config.json'),
                                  'plugin_source': config['context_manifest'], 'launcher': str(BO / 'native.py'),
                                  'native_config': config['config_path']},
                'workload': {**baseline['normalization_workload'], 'four_stage_token_scope': 'attention_dp_total', 'arrival_rate_rps': config['rps'],
                             'native_config_sha256': file_hash(config['config_path']), 'energy_boundary': 'active_serving_gpus'},
                'reference_configuration': config['references'][0], 'limits': limits,
                'default_energy_j': baseline['metrics']['energy_j'],
                'setup_cost': {'evaluations': len(all_attempts), **{k: sum(r['cost'][k] for r in all_attempts)
                              for k in ('gpu_hours', 'wall_seconds', 'tuning_energy_j')}},
                'budget': config['budget'], 'require_four_stage': True, 'allow_structure_probes': bool(config.get('joint_space')),
                'bo': {'seed': config['seeds'][0], 'evaluation_seconds': 300, 'structure_switch_seconds': 300,
                       'knob_switch_seconds': 300, 'max_repeats': 2}}
    if config.get('joint_space'):
        settings['require_output_correctness'] = True
        settings['context_files']['correctness_reference'] = str(inputs / 'correctness-reference.json')
    check(settings['setup_cost']['gpu_hours'] < config['budget']['gpu_hours'], 'Bootstrap exhausted the GPU-hour budget')
    write_json(directory / 'campaign-config.json', settings)
    if config['comparison']:
        create_comparison(directory / 'campaign-config.json', directory / 'comparison', config['seeds'], resume=True)
    else:
        campaign = directory / 'campaign'
        if campaign.exists():
            status(campaign)
        else:
            temporary = directory / '.campaign-initializing'
            if temporary.exists():
                if (temporary / 'state.json').exists():
                    state = read_json(temporary / 'state.json')
                    check(not state['observations'] and state['pending'] is None, 'Initialization contains measurements')
                temporary.rename(directory / f'.campaign-interrupted-{time.time_ns()}')
            create_campaign(directory / 'campaign-config.json', temporary)
            temporary.rename(campaign)
    write_json(directory / 'READY.json', {'native_config': artifact(config['config_path']),
                                         'campaign_config': artifact(directory / 'campaign-config.json')})
    return settings


def run(config, one=False):
    directory = Path(config['config_path']).parent
    check((directory / 'READY.json').exists(), 'Preparation did not finish; inspect bootstrap results')
    validate_context(config)
    command = [sys.executable, str(BO / 'native.py'), '--backend', 'paper', 'evaluate', '--config', config['config_path']]
    if config['comparison']:
        while True:
            result = comparison_round(directory / 'comparison', command, timeout_seconds=config['trial_timeout_seconds'] + 300)
            write_json(directory / 'comparison-report.json', comparison_report(directory / 'comparison'))
            if one or result['all_stopped']:
                return result
    else:
        campaign = directory / 'campaign'
        while True:
            if status(campaign)['frozen']:
                return status(campaign)
            result = run_one(campaign, command, timeout_seconds=config['trial_timeout_seconds'] + 300)
            if result.get('stopped'):
                if status(campaign)['best']:
                    deployment = freeze(campaign)
                    write_json(directory / 'frozen.json', deployment)
                    return deployment
                return result
            if one:
                return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('select-gpus', help='inspect physical GPU availability without loading models or setting clocks')
    p.add_argument('--count', type=int, choices=(4, 6, 8), default=4)
    p.add_argument('--gpus', nargs='+', type=int)
    p.add_argument('--minimum-free-gib', type=float, default=65.)
    p.add_argument('--output', type=Path)
    p = sub.add_parser('start', help='preflight, bootstrap, initialize, then run physical BO')
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default='qwen36')
    gpu_input = p.add_mutually_exclusive_group()
    gpu_input.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3], help='4, 6, or 8 physical GPU indices')
    gpu_input.add_argument('--gpu-selection', type=Path, help='ready JSON from select-gpus; actual availability is rechecked before execution')
    p.add_argument('--space-mode', choices=('joint', 'legacy'), default='joint')
    p.add_argument('--parallel-degrees', nargs='+', type=int, default=list(range(1, 9)))
    p.add_argument('--microbatches', nargs='+', type=int, default=[1, 2, 4])
    p.add_argument('--rps', type=int, choices=(1, 2, 4, 8, 16), default=4)
    p.add_argument('--evaluations', type=int, default=32)
    p.add_argument('--gpu-hours', type=float, default=8)
    p.add_argument('--native-python', type=Path, default=ROOT / '.venv/bin/python')
    p.add_argument('--calibration', type=Path, default=ROOT / 'inputs/traces/calibration-200.jsonl')
    p.add_argument('--heldout', type=Path, default=ROOT / 'inputs/traces/heldout-400.jsonl')
    p.add_argument('--clock-url', default='http://127.0.0.1:9096')
    p.add_argument('--api-port', type=int, default=18000)
    p.add_argument('--afd-port', type=int, default=16239)
    p.add_argument('--dp-rpc-port', type=int, default=29550)
    p.add_argument('--trial-timeout-seconds', type=float, default=3600)
    p.add_argument('--ttft-ms', type=float)
    p.add_argument('--tpot-ms', type=float)
    p.add_argument('--min-output-tps', type=float)
    p.add_argument('--comparison', action='store_true')
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--prepare-only', action='store_true', help='run real bootstrap but stop before BO trials')
    p.add_argument('--dry-run', action='store_true', help='show intended protocol without touching GPUs or outputs')
    p = sub.add_parser('run', help='resume a prepared campaign without changing its protocol')
    p.add_argument('directory', type=Path)
    p.add_argument('--one', action='store_true')
    p = sub.add_parser('prepare', help='finish interrupted preparation without repeating completed bootstrap trials')
    p.add_argument('directory', type=Path)
    p.add_argument('--retry-failed', action='store_true', help='charge interrupted attempts and create a new bootstrap attempt within the same budget')
    p = sub.add_parser('cleanup', help='retry cleanup after a failed reset or service stop')
    p.add_argument('directory', type=Path)
    p = sub.add_parser('evaluate', help=argparse.SUPPRESS)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p = sub.add_parser('recover', help='clean up an interrupted trial and charge it as failed')
    p.add_argument('directory', type=Path, help='top-level native experiment directory')
    p.add_argument('--campaign', type=Path, required=True, help='campaign or comparison arm containing the pending trial')
    args = parser.parse_args(argv)
    if args.action == 'select-gpus':
        from gpu_selection import inspect
        import pynvml
        report = inspect(pynvml, args.gpus, args.count, args.minimum_free_gib)
        if args.output:
            check(not args.output.exists(), 'GPU selection output already exists')
            write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return
    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.action == 'start':
        if args.gpu_selection:
            selection = read_json(args.gpu_selection)
            check(selection.get('status') == 'ready', 'GPU selection is not ready; do not launch')
            args.gpus = selection['gpus']
            selected_devices = {r['index']: r for r in selection['devices']}
            check(all(selected_devices[i]['eligible_now'] and selected_devices[i].get('uuid') for i in args.gpus),
                  'GPU selection lacks available device identities')
        structures = topologies(args.gpus, MODEL_EXPERTS[args.model])
        if args.dry_run:
            exclusions = []
            if args.space_mode == 'joint':
                structures, exclusions = enumerate_layouts(args.gpus, MODEL_EXPERTS[args.model],
                                                           args.microbatches, args.parallel_degrees)
            result = {'mode': 'dry_run', 'model': MODELS[args.model], 'gpus': args.gpus, 'rps': args.rps,
                      'backend': 'paper', 'mechanism_model': ENERGY_MODEL,
                      'require_four_stage': True, 'pipeline_model': 'finite_microbatch_fifo_v1',
                      'energy_correction': 'standardized_log_energy_residual_gp',
                      'missing_stage_feedback': 'failed_trial_no_external_prior_fallback',
                      'space_mode': args.space_mode, 'execution_verified': False,
                      'excluded_layouts': exclusions,
                      'supported_structures': structures,
                      'energy_boundary': 'active_gpus_only',
                      'gpu_hours_boundary': 'reserved_allocation',
                      'bootstrap_evaluations': 1 if args.space_mode == 'joint' else len(structures),
                      'correctness_reference_evaluations': 1 if args.space_mode == 'joint' else 0,
                      'budget_per_arm': {'evaluations': args.evaluations, 'gpu_hours': args.gpu_hours},
                      'comparison': args.comparison, 'steps': ['preflight',
                      'one measured MAX reference; other structures probed on budget' if args.space_mode == 'joint'
                      else 'one measured MAX calibration per topology',
                      'strict feedback validation', 'campaign init', 'ask/native replay/tell', 'freeze'],
                      'gpu_actions_performed': False}
        else:
            config = initial_inputs(args)
            result = prepare(config)
            if not args.prepare_only:
                result = run(config)
    elif args.action == 'evaluate':
        result = execute(read_json(args.config), read_json(args.request), args.result)
    elif args.action == 'run':
        result = run(read_json(args.directory / 'native-config.json'), args.one)
    elif args.action == 'prepare':
        config = read_json(args.directory / 'native-config.json')
        result = prepare(config, args.retry_failed)
    elif args.action == 'cleanup':
        config = read_json(args.directory / 'native-config.json')
        marker = args.directory / 'CLEANUP_REQUIRED.json'
        item = read_json(marker)['plan']
        check(file_hash(item['path']) == item['sha256'], 'Cleanup plan changed')
        with gpu_locks(config['gpus']):
            cleanup(config, read_json(item['path']))
            marker.rename(args.directory / f'CLEANUP_RESOLVED-{time.time_ns()}.json')
        result = {'cleanup_verified': True}
    else:
        config = read_json(args.directory / 'native-config.json')
        pending = status(args.campaign)['pending']
        check(pending is not None, 'No pending trial to recover')
        path = args.campaign / 'trials' / pending['trial_id'] / 'result.json'
        receipt = execute(config, pending, path, recover=True)
        result = tell(args.campaign, receipt)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    from entrypoint import main as dispatch
    dispatch()

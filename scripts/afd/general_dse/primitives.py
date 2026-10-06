"""Portable primitive measurements, direct algebra, and provenance verification.

The GPU runner imports torch/NVML only after an explicit probe-run invocation.
The small synthetic kernels are calibration inputs, not benchmark trace requests.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .engine import validate_hardware


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+'\n')


def verify_sources(obj):
    for name, expected in obj.get('sources', {}).items():
        if sha(name) != expected:
            raise ValueError(f'Frozen source changed: {name}')


def build(raw, source=None):
    if raw.get('selection_split') != 'calibration' or raw.get('kind') != 'independent_primitive_measurements':
        raise ValueError('Cannot promote topology stage equivalents to portable primitives')
    if not raw.get('complete'):
        raise ValueError('Primitive measurement incomplete')
    by_mhz = {}
    for f, rows in raw['device_measurements'].items():
        gemm = [dict(m=r['m'], n=r['n'], k=r['k'], flops_per_s=2*r['m']*r['n']*r['k']/r['seconds']) for r in rows['gemm']]
        bw = statistics.median(2*r['bytes']/r['seconds'] for r in rows['copy'])
        by_mhz[f] = dict(gemm=gemm, memory_bytes_per_s=bw, idle_w=rows['idle_w'],
                         active_w=max(rows['idle_w'], max(r['power_w'] for r in rows['gemm']+rows['copy'])))
    links = {}
    for key, samples in raw['links'].items():
        ordered = sorted(samples, key=lambda r:r['bytes'])
        lo, hi = ordered[0], ordered[-1]
        if hi['bytes'] <= lo['bytes']:
            raise ValueError('Two distinct communication payloads are required')
        beta = (hi['seconds']-lo['seconds'])/(hi['bytes']-lo['bytes'])
        alpha = lo['seconds']-beta*lo['bytes']
        if not math.isfinite(beta) or beta <= 0 or not math.isfinite(alpha) or alpha < 0:
            raise ValueError('Communication alpha/beta not identifiable: retain data and collect a new calibration probe')
        links[key] = dict(latency_s=alpha, bandwidth_bytes_per_s=1/beta)
    # No fabricated full-bisection scaling from a two-GPU measurement.
    fabric = raw.get('measured_fabric_bandwidth_bytes_per_s') or min((v['bandwidth_bytes_per_s'] for v in links.values()), default=1.)
    obj = dict(schema='hardware_primitives_v1', selection_split='calibration', dtype_bytes=raw['dtype_bytes'],
        gpus=raw['gpus'], device=dict(by_mhz=by_mhz, idle_reference_mhz=max(map(int,by_mhz)), power_limit_w=raw['power_limit_w']),
        links=links, fabric_bandwidth_bytes_per_s=fabric,
        fabric_scope='measured' if raw.get('measured_fabric_bandwidth_bytes_per_s') else 'conservative_shared_fabric_equal_to_slowest_measured_pair',
        sources=({str(Path(source).resolve()):sha(source)} if source else {}),
        calibration_models=[], calibration_topologies=[],
        assumptions=['GEMM and streaming-copy separately measured; no stage-time inversion',
            'homogeneous GPUs share shape-dependent primitive rates measured on first allocated GPU',
            'power is observed kernel-active power, not a worst-case bound for every model',
            'peer-copy latency/bandwidth are transport proxies; NCCL protocol/algorithm overhead requires validation',
            'unmeasured fabric concurrency uses a conservative shared bottleneck'],
        provenance=raw['provenance'])
    return validate_hardware(obj)


def probe_plan(root, gpus, frequencies):
    if not gpus or len(gpus)!=len(set(gpus)) or any(i<0 for i in gpus):
        raise ValueError('Choose distinct nonnegative physical GPU IDs')
    if not frequencies or len(frequencies)!=len(set(frequencies)) or any(f<1 for f in frequencies):
        raise ValueError('Choose distinct positive frequencies')
    root = Path(root)
    files = [* (root/'scripts/afd/general_dse').glob('*.py'), root/'migration/general_dse.py']
    return dict(schema='primitive_probe_plan_v1', selection_split='calibration', physical_gpus=gpus,
        frequencies=frequencies, dtype='bfloat16', dtype_bytes=2,
        gemm_shapes=[[m,k,k] for m in (1,32,512) for k in (1024,4096)],
        copy_bytes=[64*1024*1024,128*1024*1024], link_bytes=[4096,16*1024*1024],
        minimum_measurement_seconds=.25, idle_measurement_seconds=1.,
        sources={str(p.resolve()):sha(p) for p in files},
        request_trace_access=False, model_weights_required=False,
        placement_scope='all ordered physical pairs; no model/topology end-to-end replays')


def run(plan_path, output):
    """Root-only native CUDA microprobes; refuses occupied GPUs, restores every touched GPU."""
    import os
    import signal
    import subprocess
    plan = json.loads(Path(plan_path).read_text()); verify_sources(plan)
    if plan['schema']!='primitive_probe_plan_v1' or plan['selection_split']!='calibration':
        raise ValueError('Invalid primitive plan')
    if Path(output).exists() or Path(str(output)+'.STARTED.json').exists():
        raise FileExistsError('Probe output or partial run exists; retain it and use a new frozen plan/output')
    if os.geteuid()!=0:
        raise PermissionError('Native primitive clock control requires root')
    import pynvml as nv
    nv.nvmlInit()
    ids = plan['physical_gpus']; handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in ids]
    original_power, touched = {}, []
    def clients(handle):
        return nv.nvmlDeviceGetComputeRunningProcesses(handle) + nv.nvmlDeviceGetGraphicsRunningProcesses(handle)
    for handle in handles:
        if clients(handle):
            raise RuntimeError('Allocated GPU occupied; no clock changes made')
    names = [str(nv.nvmlDeviceGetName(h)) for h in handles]
    if len(set(names)) != 1:
        raise ValueError('This primitive profile currently requires homogeneous GPUs')
    for handle in handles:
        mem_clock = nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_MEM)
        supported = nv.nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clock)
        if not set(plan['frequencies']) <= set(supported):
            raise ValueError('Requested frequency unsupported; edit a new plan before measuring')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,ids))
    import torch
    if torch.cuda.device_count()!=len(ids):
        raise RuntimeError('Physical/logical CUDA mapping mismatch')
    for i, handle in enumerate(handles):
        # Verify UUID mapping (CUDA enumeration may differ from NVML index order).
        actual=str(torch.cuda.get_device_properties(i).uuid).replace('GPU-','').lower()
        expected=str(nv.nvmlDeviceGetUUID(handle)).replace('GPU-','').lower()
        if actual!=expected:
            raise RuntimeError('CUDA/NVML UUID mapping differs; refuse to profile another GPU')
    gpu_rows=[dict(id=i,uuid=str(nv.nvmlDeviceGetUUID(h)),name=n,memory_bytes=nv.nvmlDeviceGetMemoryInfo(h).total)
              for i,h,n in zip(ids,handles,names)]
    raw=dict(kind='independent_primitive_measurements', selection_split='calibration',complete=False,
        dtype_bytes=plan['dtype_bytes'],gpus=gpu_rows,device_measurements={},links={},
        power_limit_w=min(nv.nvmlDeviceGetPowerManagementDefaultLimit(h)/1000 for h in handles),
        provenance=dict(plan_path=str(Path(plan_path).resolve()),plan_sha256=sha(plan_path),
            code_sha256=plan['sources'],torch=torch.__version__,cuda=torch.version.cuda,
            gpu_interconnect=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)))
    write(str(output)+'.STARTED.json',dict(plan_sha256=sha(plan_path),gpus=gpu_rows))
    def interrupted(*_):
        raise KeyboardInterrupt
    old_handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    def contamination():
        if any(p.pid != os.getpid() for handle in handles for p in clients(handle)):
            raise RuntimeError('Another process entered the primitive allocation')
    def bench(op, devices):
        for _ in range(5):op()
        for d in devices:torch.cuda.synchronize(d)
        elapsed,power=[],[]; started=time.monotonic()
        while time.monotonic()-started < plan['minimum_measurement_seconds'] or len(elapsed)<5:
            t=time.perf_counter()
            for _ in range(10):op()
            for d in devices:torch.cuda.synchronize(d)
            elapsed.append((time.perf_counter()-t)/10)
            power.append(nv.nvmlDeviceGetPowerUsage(handles[devices[0]])/1000)
        contamination()
        return dict(seconds=statistics.median(elapsed),power_w=statistics.median(power))
    failure=None
    try:
        contamination()
        for i,handle in zip(ids,handles):
            original_power[i]=nv.nvmlDeviceGetPowerManagementLimit(handle)
            touched.append((i,handle))
            nv.nvmlDeviceSetPowerManagementLimit(handle,int(raw['power_limit_w']*1000))
        for freq in plan['frequencies']:
            for handle in handles:nv.nvmlDeviceSetGpuLockedClocks(handle,freq,freq)
            torch.cuda.set_device(0)
            time.sleep(plan['idle_measurement_seconds'])
            row=dict(idle_w=statistics.mean(nv.nvmlDeviceGetPowerUsage(h)/1000 for h in handles),gemm=[],copy=[])
            for m,k,n in plan['gemm_shapes']:
                a=torch.ones((m,k),device='cuda:0',dtype=torch.bfloat16)
                b=torch.ones((k,n),device='cuda:0',dtype=torch.bfloat16)
                c=torch.empty((m,n),device='cuda:0',dtype=torch.bfloat16)
                value=bench(lambda:torch.mm(a,b,out=c),[0])
                row['gemm'].append(dict(m=m,k=k,n=n,**value))
                observed=nv.nvmlDeviceGetClockInfo(handles[0],nv.NVML_CLOCK_SM)
                if abs(observed-freq)>max(15,freq*.03):
                    raise RuntimeError('Observed GPU clock differs from requested primitive frequency')
                del a,b,c
            for size in plan['copy_bytes']:
                a=torch.empty(size,device='cuda:0',dtype=torch.uint8);b=torch.empty_like(a)
                row['copy'].append(dict(bytes=size,**bench(lambda:b.copy_(a),[0])))
                del a,b
            raw['device_measurements'][str(freq)]=row
            print('Measured device primitives at',freq,'MHz',flush=True)
        max_freq=max(plan['frequencies'])
        for handle in handles:nv.nvmlDeviceSetGpuLockedClocks(handle,max_freq,max_freq)
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i==j:continue
                values=[]
                for size in plan['link_bytes']:
                    src=torch.empty(size,device=f'cuda:{i}',dtype=torch.uint8)
                    dst=torch.empty(size,device=f'cuda:{j}',dtype=torch.uint8)
                    values.append(dict(bytes=size,**bench(lambda:dst.copy_(src,non_blocking=True),[i,j])))
                    del src,dst
                raw['links'][f'{ids[i]}:{ids[j]}']=values
            print('Measured outgoing links from GPU',ids[i],flush=True)
        contamination(); raw['complete']=True
    except BaseException as exc:
        failure=exc;raw['failure']=str(exc)
    finally:
        errors=[]
        for i,handle in touched:
            try:
                nv.nvmlDeviceResetGpuLockedClocks(handle)
                nv.nvmlDeviceSetPowerManagementLimit(handle,original_power[i])
            except Exception as exc:errors.append(str(exc))
        for s,handler in old_handlers.items():signal.signal(s,handler)
        raw['restored']=not errors
        if errors:raw['complete']=False;raw['restore_errors']=errors
        write(output,raw);nv.nvmlShutdown()
    if failure:raise failure
    if not raw['complete']:raise RuntimeError('Primitive allocation restoration failed')
    return raw

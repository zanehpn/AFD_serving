"""Read-only physical GPU selection; deployment rechecks ownership under locks."""
import fcntl
import tempfile
from pathlib import Path


def inspect(nvml, requested=None, count=4, minimum_free_gib=65., lock_directory=Path(tempfile.gettempdir())):
    if count not in (4, 6, 8) or minimum_free_gib <= 0:
        raise ValueError('Select 4, 6, or 8 GPUs and a positive free-memory threshold')
    if requested is not None and (len(requested) != count or len(set(requested)) != count
                                   or any(type(i) is not int or i < 0 for i in requested)):
        raise ValueError('Explicit GPU list must contain the requested number of distinct physical indices')
    nvml.nvmlInit()
    try:
        rows = []
        available_count = nvml.nvmlDeviceGetCount()
        if requested is not None and any(i >= available_count for i in requested):
            raise ValueError('Requested physical GPU does not exist')
        for index in range(available_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            name, uuid = nvml.nvmlDeviceGetName(handle), nvml.nvmlDeviceGetUUID(handle)
            name = name.decode() if isinstance(name, bytes) else name
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            reasons = []
            pids = sorted({p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)} |
                          {p.pid for p in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)})
            if pids:
                reasons.append('foreign_or_existing_gpu_processes')
            if memory.free < minimum_free_gib * 1024 ** 3:
                reasons.append('insufficient_free_memory')
            if 'A100-SXM4-80GB' not in name:
                reasons.append('native_protocol_gpu_model_mismatch')
            # Zero process count alone does not prove the card is unreserved.
            try:
                with (lock_directory / f'moe-bo-gpu-{index}.lock').open('r') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except FileNotFoundError:
                pass  # Selection neither creates locks nor reserves cards.
            except BlockingIOError:
                reasons.append('reserved_by_other_bo_campaign')
            except PermissionError:
                reasons.append('reservation_lock_not_accessible')
            rows.append({'index': index, 'uuid': uuid, 'name': name,
                         'free_gib': memory.free / 1024 ** 3, 'process_ids': pids,
                         'eligible_now': not reasons, 'reasons': reasons})
        selected = list(requested) if requested is not None else [r['index'] for r in rows if r['eligible_now']][:count]
        ready = len(selected) == count and all(rows[i]['eligible_now'] for i in selected)
        return {'status': 'ready' if ready else 'insufficient_available_gpus', 'gpus': selected if ready else [],
                'requested_gpus': requested, 'devices': rows, 'gpu_actions_performed': False,
                'reservation_held': False, 'requires_recheck_before_launch': True}
    finally:
        nvml.nvmlShutdown()

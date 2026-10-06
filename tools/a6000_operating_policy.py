"""Validate controlled power and explicitly declared memory-clock policy."""
import math


def check_operating_point(config, candidate, gpu, state):
    role = 'attention' if gpu in candidate['attention_gpus'] else 'expert'
    if not math.isfinite(state['power_limit_w']) or abs(state['power_limit_w'] - candidate[role + '_power_w']) > 1:
        raise ValueError('Measured power cap differs from frozen operating point')
    memory = state['memory_clock_mhz']
    policy = config.get('memory_clock_policy', 'fixed')
    if policy == 'driver_managed_pstates':
        if config.get('expected_gpu_model') != 'NVIDIA RTX A6000':
            raise ValueError('Driver-managed memory policy is only validated for A6000')
        if not isinstance(memory, (float, int)) or not math.isfinite(memory) or not 0 < memory <= config['memory_clock_mhz']:
            raise ValueError('Invalid measured driver-managed memory clock')
    elif policy != 'fixed' or memory != config['memory_clock_mhz']:
        raise ValueError('Measured memory clock differs from frozen operating point')


def clock_summary(window):
    result = {}
    for row in window:
        for gpu, state in zip(row['gpu_ids'], row['operating_state']):
            values = result.setdefault(str(gpu), {'sm_mhz_samples': [], 'memory_mhz_sample_counts': {}})
            values['sm_mhz_samples'].append(state['sm_clock_mhz'])
            key = str(state['memory_clock_mhz'])
            counts = values['memory_mhz_sample_counts']
            counts[key] = counts.get(key, 0) + 1
    for values in result.values():
        samples = values.pop('sm_mhz_samples')
        values['sm_mhz'] = {'min': min(samples), 'mean': sum(samples) / len(samples), 'max': max(samples)}
    return result

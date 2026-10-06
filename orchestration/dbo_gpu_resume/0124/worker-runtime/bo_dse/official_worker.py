"""Run the frozen worker with an explicit, auditable physical GPU mapping."""
import sys
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
config_path = Path(sys.argv[sys.argv.index('--config')+1])
logical_gpus = json.loads(config_path.read_text())['gpus']
SOURCE = BASE.with_name('dbo_on_1256_v3' if logical_gpus == [1,2,5,6] else 'dbo_on_0347_v3')/'source/bo_dse'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(BASE))
import official_worker as w
from mapping import transform, load_manifest

original_trial = w.trial
original_cleanup = w.cleanup

def trial(config, candidate, directory, reference=False):
    physical, mapped = transform(config, candidate)
    w.write(directory/'execution-configuration.json', mapped)
    w.write(directory/'execution-hardware.json', physical['hardware'])
    record = dict(manifest=w.artifact(BASE/'MIGRATION.json'),
                  logical_configuration=candidate, physical_configuration=mapped,
                  note='User-requested pool migration; previous observations retained; physical hardware differs across phases')
    w.write(directory/'physical-execution.json', record)
    result = original_trial(physical, mapped, directory, reference=reference)
    result['physical_execution'] = record
    result['artifacts'] += [w.artifact(directory/name) for name in
                           ('execution-configuration.json', 'execution-hardware.json', 'physical-execution.json')]
    result['artifacts'].append(w.artifact(BASE/'MIGRATION.json'))
    w.write(directory/'worker-result.json', result)
    return result

def cleanup(config, directory):
    if not config.get('_physical_migration') and (directory/'physical-execution.json').exists():
        config = transform(config)
    return original_cleanup(config, directory)

w.trial = trial
w.cleanup = cleanup
if __name__ == '__main__':
    load_manifest()
    w.main()

"""Launch the prepared neighboring-reduction benchmark after its predecessor."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = (Path(__file__).resolve().parents[2] / 'MOE_DVFS-capacity-v2-neighbor-reduction')
OUT = ROOT / 'results/qwen-rps16-200-capacity-v2-neighbor-reduction'
PYTHON = ROOT / 'bo_dse/.venv/bin/python'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'status.json').exists():
        raise RuntimeError('Experiment already launched; inspect status instead of duplicating it')
    env = dict(os.environ, PYTHONUNBUFFERED='1',
               MOE_QWEN_EXPERIMENT=OUT.name, MOE_QWEN_REVISION='neighbor_reduction',
               MOE_QWEN_PREDECESSOR=str(ROOT.parent / 'MOE_DVFS-capacity-v2-feedback/results/qwen-rps16-200-capacity-v2-feedback/status.json'))
    with (OUT / 'queue.log').open('ab') as log:
        queue = subprocess.Popen([str(PYTHON), str(ROOT / 'orchestration/run_qwen_feedback.py'), 'run'],
                                 cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if queue.poll() is not None:
            raise RuntimeError('Queue exited during launch; inspect queue.log')
        if (OUT / 'status.json').exists():
            break
        time.sleep(1)
    else:
        raise RuntimeError('Queue has not published status; inspect process before retrying')
    with (OUT / 'comparison-monitor.log').open('ab') as log:
        monitor = subprocess.Popen([str(PYTHON), str(ROOT / 'orchestration/watch_qwen_feedback.py')],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    result = {'queue_pid': queue.pid, 'comparison_monitor_pid': monitor.pid,
              'status': str(OUT / 'status.json'), 'source_directory': str(ROOT)}
    (OUT / 'launcher.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()

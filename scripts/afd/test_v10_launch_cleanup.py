"""Exercise shell cleanup with mocked clock calls; never access real GPUs."""
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize('runner_name,reset_calls', [('run_v026_causal_dvfs_v10_rate_suite.sh', 2), ('run_v026_causal_dvfs_rate_suite.sh', 2), ('run_v026_ours_rate_suite.sh', 1)])
@pytest.mark.parametrize('acquired', [0, 1])
def test_cleanup_only_resets_acquired_allocation(tmp_path, acquired, runner_name, reset_calls):
    runner = Path(__file__).with_name(runner_name).read_text()
    body = 'cleanup() {' + runner.split('cleanup() {', 1)[1].split('\non_exit()', 1)[0]
    setup = '''set -eu
cleanup_done=0
cleanup_status=0
controller_pid=""
watcher_pid=""
server_started=0
CONTAINER_RUNTIME=apptainer
MONITOR_OUTPUT="$TEST_ROOT/no-output"
MONITOR_FAILURE="$TEST_ROOT/no-failure"
SUITE_DIR="$TEST_ROOT"
ECODEP_ROOT="$TEST_ROOT"
CLOCK_URL=unused
GPU_CLOCK_URL_MAP=unused
GPUS=4,5,6,7
ATTENTION_CLOCK_URL=unused
EXPERT_CLOCK_URL=unused
ATTENTION_GPUS=4,5
EXPERT_GPUS=6,7
MAX_ATTENTION_CLOCKS=1410,1410
MAX_EXPERT_CLOCKS=1410,1410
python3() { echo called >> "$TEST_ROOT/calls"; }
'''
    env = dict(os.environ, TEST_ROOT=str(tmp_path), gpu_allocation_acquired=str(acquired))
    subprocess.run(['bash', '-c', setup + body + '\ncleanup\ncleanup\n'], env=env, check=True)
    calls = tmp_path / 'calls'
    assert (len(calls.read_text().splitlines()) if calls.exists() else 0) == acquired * reset_calls

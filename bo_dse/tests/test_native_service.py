"""CPU processes exercise the actual service CLI and orphan-worker cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SERVICE = Path(__file__).resolve().parents[2] / 'migration/native_service.py'
spec = importlib.util.spec_from_file_location('native_service', SERVICE)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def cli(directory, *args):
    return subprocess.run([sys.executable, str(SERVICE), args[0], str(directory), *args[1:]],
                          capture_output=True, text=True, timeout=15)


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('CPU service did not reach expected state')


@pytest.mark.parametrize('leader_dead', [False, True])
@pytest.mark.parametrize('ignore_term', [False, True])
def test_stop_waits_for_workers_and_preserves_other_groups(tmp_path, leader_dead, ignore_term):
    ready = tmp_path / 'worker.json'
    worker = (
        'import os, signal, time; from pathlib import Path; '
        + ('signal.signal(signal.SIGTERM, signal.SIG_IGN); ' if ignore_term else '')
        + f'Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(120)'
    )
    leader = f'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", {worker!r}]); time.sleep(120)'
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)
    state = None
    try:
        result = cli(tmp_path, 'start', '--', sys.executable, '-c', leader)
        assert result.returncode == 0, result.stderr
        state = json.loads((tmp_path / 'native-service.json').read_text())
        wait_for(ready.exists)
        worker_pid = int(ready.read_text())
        if leader_dead:
            os.kill(state['pid'], signal.SIGKILL)
            wait_for(lambda: service.identity(state['pid']) is None)
        duplicate = cli(tmp_path, 'start', '--', 'true')
        assert duplicate.returncode != 0 and 'process group is still alive' in duplicate.stderr
        assert json.loads((tmp_path / 'native-service.json').read_text()) == state
        stopped = cli(tmp_path, 'stop', '--stop-grace', '0.15')
        assert stopped.returncode == 0, stopped.stderr
        assert service.identity(worker_pid) is None
        assert service.group_members(state) == []
        assert unrelated.poll() is None
        assert json.loads((tmp_path / 'native-service.json').read_text())['stopped'] is True
        assert cli(tmp_path, 'stop').returncode == 0
    finally:
        if state:
            service.stop_group(state, grace=0)
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_reused_leader_pid_is_never_signalled(monkeypatch):
    monkeypatch.setattr(service, 'process_info', lambda pid: {'start_ticks': '200', 'state': 'Z'})
    monkeypatch.setattr(service, 'signal_members', lambda *a: pytest.fail('unrelated process was signalled'))
    with pytest.raises(RuntimeError, match='PID was reused'):
        service.stop_group({'pid': 123, 'start_ticks': '100'})


def test_failed_cleanup_cannot_report_success_or_mark_stopped(tmp_path, monkeypatch):
    state = {'pid': 123, 'start_ticks': '100'}
    path = tmp_path / 'native-service.json'
    path.write_text(json.dumps(state))
    monkeypatch.setattr(service, 'group_members', lambda _: [(123, '100')])
    monkeypatch.setattr(service, 'signal_members', lambda *a: None)
    # Advance the deadline without waiting for the default kill timeout.
    ticks = iter(range(0, 1000, 100))
    monkeypatch.setattr(service.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(sys, 'argv', [str(SERVICE), 'stop', str(tmp_path), '--stop-grace', '0'])
    with pytest.raises(RuntimeError, match='cleanup incomplete'):
        service.main()
    assert json.loads(path.read_text()) == state


def test_metadata_from_previous_boot_cannot_signal_processes(monkeypatch):
    monkeypatch.setattr(service, 'boot_id', lambda: 'current-boot')
    monkeypatch.setattr(service, 'process_info', lambda _: pytest.fail('stale process ownership used'))
    assert service.group_members({'pid': 123, 'start_ticks': '100', 'boot_id': 'old-boot'}) == []

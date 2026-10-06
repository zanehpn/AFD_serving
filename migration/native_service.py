#!/usr/bin/env python3
"""Own a native process group; record PID start time to prevent PID-reuse kills."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def process_info(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': data[0], 'pgid': int(data[2]), 'sid': int(data[3]), 'start_ticks': data[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity(pid):
    info = process_info(pid)
    return info['start_ticks'] if info and info['state'] not in ('Z', 'X') else None


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def group_members(service):
    """A setsid launch owns its group even after its original leader exits."""
    if service.get('stopped') or service.get('boot_id', boot_id()) != boot_id():
        return []
    pid = service['pid']
    leader = process_info(pid)
    # Check zombies too: a reused leader PID never authorizes a group signal.
    if leader and leader['start_ticks'] != service['start_ticks']:
        raise RuntimeError('service PID was reused; refusing to signal another process group')
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        info = process_info(int(path.name))
        if not info or info['pgid'] != pid or info['state'] in ('Z', 'X'):
            continue
        if info['sid'] != pid or int(info['start_ticks']) < int(service['start_ticks']):
            raise RuntimeError('process group ownership mismatch')
        members.append((int(path.name), info['start_ticks']))
    return members


def signal_members(members, sig):
    for pid, start in members:
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            # The descriptor prevents PID reuse between verification and signal.
            if identity(pid) == start:
                signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            pass
        finally:
            os.close(fd)


def stop_group(service, grace=30, kill_timeout=5):
    for sig, timeout in ((signal.SIGTERM, grace), (signal.SIGKILL, kill_timeout)):
        deadline = time.monotonic() + timeout
        signalled = set()
        while True:
            members = group_members(service)
            if not members:
                return
            signal_members([m for m in members if m not in signalled], sig)
            signalled.update(members)
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
    remaining = group_members(service)
    if remaining:
        raise RuntimeError(f'native service cleanup incomplete; live workers: {remaining}')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['start', 'stop', 'wait'])
    p.add_argument('directory', type=Path)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--stop-grace', type=float, default=30)
    args, command = p.parse_known_args()
    if args.stop_grace < 0:
        p.error('--stop-grace must be nonnegative')
    d = args.directory.resolve()
    state = d / 'native-service.json'
    if args.action == 'start':
        if command[:1] == ['--']:
            command = command[1:]
        if not command:
            p.error('start needs a command after --')
        if state.exists():
            previous = json.loads(state.read_text())
            if group_members(previous):
                raise RuntimeError('existing service process group is still alive')
        d.mkdir(parents=True, exist_ok=True)
        with (d / 'launcher.log').open('ab') as out:
            child = subprocess.Popen(command, stdout=out, stderr=out, start_new_session=True)
        start = identity(child.pid)
        if start is None:
            raise RuntimeError('service exited immediately; inspect launcher.log')
        state.write_text(json.dumps({'pid': child.pid, 'start_ticks': start, 'boot_id': boot_id(),
                                     'command': command}) + '\n')
        (d / 'native-service.pid').write_text(str(child.pid) + '\n')
        return
    if not state.exists():
        if args.action == 'stop':
            return
        raise RuntimeError('service metadata missing')
    service = json.loads(state.read_text())
    pid = service['pid']
    alive = lambda: identity(pid) == service['start_ticks']
    if args.action == 'stop':
        stop_group(service, grace=args.stop_grace)
        service['stopped'] = True
        state.write_text(json.dumps(service) + '\n')
        return
    until = time.monotonic() + args.timeout
    while time.monotonic() < until:
        if not alive():
            raise RuntimeError('native service exited before readiness; inspect launcher.log')
        try:
            cfg = json.loads((d / 'launch_config.json').read_text())
            ready = (d / 'attention.log').read_text(errors='replace').count('Application startup complete.')
            if ready >= cfg.get('attention_dp', cfg['attention_ranks'] // cfg.get('attention_tp', 1)):
                with urllib.request.urlopen(f"http://127.0.0.1:{cfg.get('api_port', 18000)}/health", timeout=2) as response:
                    if response.status == 200:
                        time.sleep(2)
                        return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError('native service readiness timeout')


if __name__ == '__main__':
    main()

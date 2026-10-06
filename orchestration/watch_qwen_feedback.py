"""Refresh the comparison artifact as the revised Qwen campaign progresses."""
import json
import time
from pathlib import Path

from compare_qwen_feedback import NEW, compare


def main():
    previous = None
    while True:
        status = json.loads((NEW.parent/'status.json').read_text())
        state_path = NEW/'campaign/state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            progress = (len(state['observations']), status['phase'])
            if progress != previous:
                compare()
                previous = progress
        if status['phase'] in ('complete', 'needs_attention', 'paused_by_user'):
            return
        if not Path('/proc', str(status['pid'])).exists():
            raise RuntimeError('Experiment queue exited without a terminal status')
        time.sleep(15)


if __name__ == '__main__':
    main()

"""Exercise the real streaming parser, including empty-text token deltas."""
import asyncio
import json
from pathlib import Path
import sys
import time

import pytest

pytest.importorskip('aiohttp', reason='replay runs in the native environment')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/afd'))
from replay_trace import replay_one


@pytest.mark.parametrize('damage', [None, 'absent', 'malformed'])
def test_streamed_output_ids_and_missing_id_failure(damage, monkeypatch):
    now = [0.]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])
    events = [{'choices': [{'text': '', 'token_ids': [10]}]},
              {'choices': [{'text': 'answer', 'token_ids': [11, 12]}]},
              {'choices': [], 'usage': {'completion_tokens': 3}}]
    if damage == 'absent':
        for event in events[:2]:
            event['choices'][0].pop('token_ids')
    elif damage == 'malformed':
        events[0]['choices'][0]['token_ids'] = [True]

    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        @property
        def content(self):
            async def lines():
                for event, timestamp in zip(events, [.01, .11, .21]):
                    now[0] = timestamp
                    yield ('data: ' + json.dumps(event) + '\n').encode()
                yield b'data: [DONE]\n'
            return lines()

    class Session:
        def post(self, endpoint, json, headers):
            assert json['return_token_ids'] is True and json['seed'] == 0
            assert json['temperature'] == 0 and json['ignore_eos'] is True
            assert headers['X-Request-Id'] == 'ecodep-7'
            return Response()

    row = {'request_id': 7, 'input_tokens': 4, 'output_tokens': 3, 'arrival_s': 0}
    result = asyncio.run(replay_one(Session(), 'unused', 'model', row, time.perf_counter(), 1, 3, 400, 120, True))
    if damage is None:
        assert result['error'] is None and result['actual_output_token_ids'] == [10, 11, 12]
        assert result['first_token_wall_ns'] is not None
        assert len(result['token_arrival_s']) == 3
        assert result['token_event_sizes'] == [1, 2]
        assert len(result['tbt_ms']) == 2
        assert result['tbt_ms'] == pytest.approx([100., 0.])
        assert result['token_arrival_s'][1] == result['token_arrival_s'][2]
    else:
        assert result['error'] is not None

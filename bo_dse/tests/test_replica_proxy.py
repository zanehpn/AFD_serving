"""Loopback-only check: the replica proxy preserves streaming latency and IDs."""
import asyncio
from pathlib import Path
import sys
import pytest

aiohttp = pytest.importorskip('aiohttp', reason='proxy runs in the native serving environment')
from aiohttp import web
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/afd'))
from replica_proxy import application


def test_streaming_round_robin_and_request_identity():
    async def exercise():
        runners, seen, resets = [], [], []
        release = asyncio.Event()
        async def serve(app):
            runner=web.AppRunner(app);await runner.setup();runners.append(runner)
            site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
            return site._server.sockets[0].getsockname()[1]
        def backend(index):
            app=web.Application()
            async def completion(request):
                seen.append((index,request.headers.get('X-Request-Id')))
                response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
                await response.prepare(request);await response.write(b'data: first\n\n')
                await release.wait();await response.write(b'data: [DONE]\n\n');await response.write_eof()
                return response
            async def health(request): return web.Response(text='ok')
            async def reset(request): resets.append(index);return web.json_response({'success':True})
            app.router.add_post('/v1/completions',completion)
            app.router.add_get('/health',health)
            app.router.add_post('/reset_prefix_cache',reset)
            return app
        try:
            ports=[await serve(backend(i)) for i in range(2)]
            proxy=await serve(application(ports))
            async with aiohttp.ClientSession() as session:
                for i in range(2):
                    async with session.post(f'http://127.0.0.1:{proxy}/v1/completions',json={'stream':True},
                                            headers={'X-Request-Id':f'request-{i}'}) as response:
                        first=await asyncio.wait_for(response.content.readline(),2)
                        assert first==b'data: first\n'
                        release.set()
                        assert b'[DONE]' in await response.read()
                async with session.post(f'http://127.0.0.1:{proxy}/reset_prefix_cache') as response:
                    assert response.status==200
                    assert (await response.json())['success'] is True
                async with session.get(f'http://127.0.0.1:{proxy}/health') as response:
                    assert response.status==200
            assert seen==[(0,'request-0'),(1,'request-1')]
            assert resets==[0,1]
        finally:
            release.set()
            for runner in reversed(runners): await runner.cleanup()
    asyncio.run(exercise())


@pytest.mark.parametrize('bad_reply', [
    (200, '{"success":false}'), (200, '{}'), (200, 'invalid json'),
    (200, '{"success":1}'), (200, '{"success":"true"}'), (200, '[]'),
    (503, '{"success":true}'), 'disconnect',
])
@pytest.mark.parametrize('bad_index', [0, 1])
def test_cache_reset_requires_every_replica_to_succeed(bad_reply, bad_index):
    async def exercise():
        runners, resets = [], []

        async def serve(app):
            runner = web.AppRunner(app)
            await runner.setup()
            runners.append(runner)
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            return site._server.sockets[0].getsockname()[1]

        def backend(index):
            app = web.Application()

            async def reset(request):
                resets.append((index, request.query.get('reset_running_requests')))
                if index != bad_index:
                    return web.json_response({'success': True})
                if bad_reply == 'disconnect':
                    request.transport.close()
                    return web.Response()
                return web.Response(status=bad_reply[0], text=bad_reply[1], content_type='application/json')

            app.router.add_post('/reset_prefix_cache', reset)
            return app

        try:
            ports = [await serve(backend(i)) for i in range(3)]
            proxy = await serve(application(ports))
            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://127.0.0.1:{proxy}/reset_prefix_cache?reset_running_requests=true') as response:
                    assert response.status == 502
                    result = await response.json()
            assert result['success'] is False
            assert [r['success'] for r in result['replicas']] == [i != bad_index for i in range(3)]
            assert resets == [(i, 'true') for i in range(3)]
        finally:
            for runner in reversed(runners):
                await runner.cleanup()

    asyncio.run(exercise())


@pytest.mark.parametrize('path', ['/start_profile', '/stop_profile'])
def test_profile_broadcast_accepts_empty_success_response(path):
    async def exercise():
        runners, calls = [], []

        async def serve(app):
            runner = web.AppRunner(app)
            await runner.setup()
            runners.append(runner)
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            return site._server.sockets[0].getsockname()[1]

        def backend(index):
            app = web.Application()

            async def profile(request):
                calls.append(index)
                return web.Response()

            app.router.add_post(path, profile)
            return app

        try:
            ports = [await serve(backend(i)) for i in range(2)]
            proxy = await serve(application(ports))
            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://127.0.0.1:{proxy}{path}') as response:
                    assert response.status == 200 and await response.read() == b''
            assert calls == [0, 1]
        finally:
            for runner in reversed(runners):
                await runner.cleanup()

    asyncio.run(exercise())

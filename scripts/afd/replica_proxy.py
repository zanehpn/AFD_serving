"""Single local API for independent AFD replicas; no retries or hidden replay."""
import argparse
import asyncio
import json
from itertools import count
from aiohttp import ClientError, ClientSession, ClientTimeout, web


def application(backends):
    app = web.Application(client_max_size=32 * 1024 * 1024)
    order = count()
    session_key = web.AppKey('session', ClientSession)

    async def lifecycle(app):
        async with ClientSession(timeout=ClientTimeout(total=3600)) as session:
            app[session_key] = session
            yield
    app.cleanup_ctx.append(lifecycle)

    async def health(request):
        async def one(port):
            try:
                async with app[session_key].get(f'http://127.0.0.1:{port}/health', timeout=ClientTimeout(total=2)) as response:
                    return response.status == 200
            except (OSError, asyncio.TimeoutError):
                return False
        healthy = all(await asyncio.gather(*(one(p) for p in backends)))
        return web.json_response({'healthy': healthy}, status=200 if healthy else 503)

    async def forward(request):
        # Development cache-reset/profile operations apply to every replica.
        broadcast = request.path in ('/reset_prefix_cache', '/start_profile', '/stop_profile')
        ports = backends if broadcast else [backends[next(order) % len(backends)]]
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() in ('content-type', 'x-request-id')}
        if not broadcast:
            async with app[session_key].request(request.method, f'http://127.0.0.1:{ports[0]}{request.rel_url}',
                                             data=body, headers=headers) as upstream:
                response = web.StreamResponse(status=upstream.status,
                                              headers={'Content-Type': upstream.headers.get('Content-Type', 'application/octet-stream')})
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        replies = []
        resets = []
        for port in ports:
            try:
                async with app[session_key].request(request.method, f'http://127.0.0.1:{port}{request.rel_url}',
                                                 data=body, headers=headers) as response:
                    payload = await response.read()
                    replies.append((response.status, payload, response.content_type))
                    if request.path == '/reset_prefix_cache':
                        try:
                            result = json.loads(payload)
                            success = (200 <= response.status < 300 and isinstance(result, dict)
                                       and result.get('success') is True)
                        except (ValueError, UnicodeDecodeError):
                            success = False
                        resets.append({'port': port, 'status': response.status, 'success': success})
            except (ClientError, OSError, asyncio.TimeoutError) as exc:
                if request.path != '/reset_prefix_cache':
                    raise
                # Still attempt every replica; do not hide a partial reset or retry it.
                resets.append({'port': port, 'success': False, 'error': str(exc)})
        if request.path == '/reset_prefix_cache':
            success = bool(resets) and all(r['success'] for r in resets)
            return web.json_response({'success': success, 'replicas': resets}, status=200 if success else 502)
        selected = next((r for r in replies if r[0] >= 400), replies[0])
        return web.Response(status=selected[0], body=selected[1], content_type=selected[2])

    app.router.add_get('/health', health)
    app.router.add_route('*', '/{tail:.*}', forward)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--backends', nargs='+', type=int, required=True)
    args = parser.parse_args()
    web.run_app(application(args.backends), host='127.0.0.1', port=args.port)

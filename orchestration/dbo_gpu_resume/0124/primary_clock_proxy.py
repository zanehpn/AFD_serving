"""Forward only this experiment's four GPUs to existing authorized daemons."""
import json
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

ROUTES = {0: 9095, 1: 9095, 2: 9095, 4: 9096}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, value):
        payload = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != '/health':
            return self.send(404, {'error': 'unsupported endpoint'})
        try:
            for port in set(ROUTES.values()):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
                    health = json.load(response)
                assert health['protocol'] == 'nvcontrold.applied_ack.v2'
                assert {g for g, p in ROUTES.items() if p == port} <= set(health['allowed'])
            self.send(200, {'ok': True, 'allowed': list(ROUTES), 'protocol': 'nvcontrold.applied_ack.v2'})
        except Exception as error:
            self.send(503, {'error': str(error)})

    def do_POST(self):
        try:
            if self.path not in ('/set_clock', '/set_power_limit', '/reset'):
                return self.send(404, {'error': 'unsupported endpoint'})
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            gpu = body['gpu']
            if type(gpu) is not int or gpu not in ROUTES:
                return self.send(403, {'error': 'GPU outside experiment allocation'})
            request = urllib.request.Request(f'http://127.0.0.1:{ROUTES[gpu]}{self.path}',
                        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=10) as response:
                value = json.load(response)
            self.send(200, value)
        except Exception as error:
            self.send(502, {'error': str(error)})


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Keep acknowledgements alive during parent cleanup.
    ThreadingHTTPServer(('127.0.0.1', 19101), Handler).serve_forever()

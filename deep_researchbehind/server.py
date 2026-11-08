#!/usr/bin/env python3
"""Local HTTP bridge from the HAE-GEO runtime to Xunsearch BM25."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

try:
    from .backend import Backend
except ImportError:
    from backend import Backend


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.send_json(404, dict(success=False, error='Unknown route'))
            return self.send_json(200, dict(success=True, backend='xunsearch-bm25',
                index_revision=backend.manifest['index_revision'],
                environments=list(backend.manifest['environments'])))

        def do_POST(self):
            if self.path != '/search':
                return self.send_json(404, dict(success=False, error='Unknown route'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('Expected a JSON object')
                result = backend.search(**request)
            except KeyError:
                return self.send_json(404, dict(success=False, error='Unknown environment'))
            except (ValueError, TypeError) as error:
                return self.send_json(400, dict(success=False, error=str(error)))
            except Exception as error:
                return self.send_json(502, dict(success=False, error=str(error)))
            self.send_json(200, result)

        def log_message(self, format_string, *args):
            return
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--php', default='php')
    args = parser.parse_args()
    backend = Backend(args.manifest, php=args.php)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    print(f'Xunsearch adapter listening on http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

"""
server.py
ClearSight v4 — Local Development Server

Serves the static frontend with proper MIME types.
Run:  python server.py
Then open: http://localhost:8080
"""

import http.server
import socketserver
import os
import sys

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class ClearSightHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with correct MIME types and CORS headers."""

    extensions_map = {
        '':      'application/octet-stream',
        '.html': 'text/html',
        '.css':  'text/css',
        '.js':   'application/javascript',
        '.json': 'application/json',
        '.png':  'image/png',
        '.jpg':  'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.svg':  'image/svg+xml',
        '.ico':  'image/x-icon',
        '.woff': 'font/woff',
        '.woff2':'font/woff2',
        '.ttf':  'font/ttf',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Allow cross-origin requests (needed for fonts, API calls)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, anthropic-version, x-api-key')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        # Suppress noisy 304 logs, keep 200/404/500
        if args and args[1] not in ('304',):
            super().log_message(fmt, *args)


def run():
    with socketserver.TCPServer(('', PORT), ClearSightHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f'\n  ╔══════════════════════════════════════╗')
        print(f'  ║   ClearSight v4 — Dev Server         ║')
        print(f'  ║   http://localhost:{PORT}               ║')
        print(f'  ║   Press Ctrl+C to stop               ║')
        print(f'  ╚══════════════════════════════════════╝\n')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n  Server stopped.')
            sys.exit(0)


if __name__ == '__main__':
    run()

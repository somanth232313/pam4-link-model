"""Web entrypoint for the interactive demo (Vercel Python runtime, WSGI).

    GET  /              -> public/index.html
    GET  /api/health    -> {"ok": true}
    GET  /api/defaults  -> parameter defaults and ranges
    POST /api/simulate  -> JSON parameters in, simulation results out

Run locally:  python app.py   (then open http://localhost:8000)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pam4link.service import PARAMS, run  # noqa: E402

INDEX = ROOT / "public" / "index.html"
MAX_BODY = 4096


def _respond(start_response, status: str, body: bytes, content_type: str, extra=()):
    headers = [("Content-Type", content_type), ("Content-Length", str(len(body))), *extra]
    start_response(status, headers)
    return [body]


def _json(start_response, status: str, payload) -> list[bytes]:
    return _respond(start_response, status, json.dumps(payload).encode(), "application/json",
                    [("Cache-Control", "no-store")])


def app(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/") or "/"

    if path in ("/", "/index.html") and method == "GET":
        return _respond(start_response, "200 OK", INDEX.read_bytes(), "text/html; charset=utf-8")

    if path == "/api/health":
        return _json(start_response, "200 OK", {"ok": True})

    if path == "/api/defaults" and method == "GET":
        return _json(start_response, "200 OK",
                     {k: {"default": d, "min": lo, "max": hi} for k, (d, lo, hi) in PARAMS.items()})

    if path == "/api/simulate":
        if method != "POST":
            return _json(start_response, "405 Method Not Allowed", {"error": "use POST"})
        try:
            length = min(int(environ.get("CONTENT_LENGTH") or 0), MAX_BODY)
            raw = json.loads(environ["wsgi.input"].read(length) or b"{}") if length else {}
            if not isinstance(raw, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            return _json(start_response, "400 Bad Request", {"error": str(exc)})
        t0 = time.perf_counter()
        result = run(raw)
        result["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return _json(start_response, "200 OK", result)

    return _json(start_response, "404 Not Found", {"error": "not found"})


if __name__ == "__main__":
    from wsgiref.simple_server import make_server

    with make_server("127.0.0.1", 8000, app) as server:
        print("Serving on http://127.0.0.1:8000")
        server.serve_forever()

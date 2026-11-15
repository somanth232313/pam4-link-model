"""Tests for the web demo service and WSGI entrypoint."""
import io
import json
import sys
from pathlib import Path

import pytest

from pam4link.service import PARAMS, parse_params, run

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web  # noqa: E402


def test_parse_params_fills_defaults_and_clamps():
    p = parse_params({"rx_power_dbm": 99, "ctle_peaking_db": "bad", "ffe_taps": 1})
    assert p["rx_power_dbm"] == PARAMS["rx_power_dbm"][2]
    assert p["ctle_peaking_db"] == PARAMS["ctle_peaking_db"][0]
    assert p["ffe_taps"] == 0
    assert set(p) == set(PARAMS)


def test_run_returns_complete_json_serializable_result():
    out = run({})
    json.dumps(out)  # must not raise (no NaN objects / numpy types)
    assert out["metrics"]["passes_kp4"]
    assert len(out["bathtub"]["ber"]) == len(out["bathtub"]["phase_ui"])
    eye = out["eye_eq"]
    assert len(eye["density"]) == eye["rows"] * eye["cols"]


def test_run_without_ffe_has_no_equalized_eye():
    out = run({"ffe_taps": 0})
    assert out["eye_eq"] is None and out["ffe"]["taps"] and out["ffe"]["pre"] == 0


def call(method, path, body=b""):
    status = {}
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "CONTENT_LENGTH": str(len(body)),
           "wsgi.input": io.BytesIO(body)}
    chunks = web.app(env, lambda s, h: status.update(code=int(s.split()[0]), headers=dict(h)))
    return status["code"], status["headers"], b"".join(chunks)


@pytest.mark.parametrize("path", ["/", "/api/health", "/api/defaults"])
def test_get_routes(path):
    code, _, body = call("GET", path)
    assert code == 200 and body


def test_simulate_route_and_errors():
    code, headers, body = call("POST", "/api/simulate", json.dumps({"rx_power_dbm": -8}).encode())
    assert code == 200 and headers["Content-Type"] == "application/json"
    assert json.loads(body)["params"]["rx_power_dbm"] == -8
    assert call("POST", "/api/simulate", b"not json")[0] == 400
    assert call("POST", "/api/simulate", b"[1, 2]")[0] == 400
    assert call("GET", "/api/simulate")[0] == 405
    assert call("GET", "/nope")[0] == 404

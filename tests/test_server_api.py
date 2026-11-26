import json
import threading
import urllib.request
import urllib.error

import pytest

from stochastic_medium.server import Handler
from http.server import ThreadingHTTPServer


@pytest.fixture(scope="module")
def server_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.status, json.loads(r.read().decode())


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_schema_endpoint_drives_controls(server_url):
    status, d = _get(server_url + "/api/schema")
    assert status == 200
    ids = [c["id"] for c in d["schema"]]
    assert {"scene", "kappa", "policy", "partition", "spp"} <= set(ids)
    assert "complex" in d["scenes"]


def test_static_index_and_script(server_url):
    with urllib.request.urlopen(server_url + "/", timeout=10) as r:
        html = r.read().decode()
    assert "app.js" in html and "内核" in html
    with urllib.request.urlopen(server_url + "/app.js", timeout=10) as r:
        assert b"/api/render" in r.read()


def test_render_accumulate_and_trace(server_url):
    cfg = {"scene": "constant", "kappa": 0.8, "spp": 3,
           "width": 48, "height": 32, "partition": "global"}
    status, d = _post(server_url + "/api/render", cfg)
    assert status == 200, d
    assert d["image"].startswith("data:image/png;base64,")
    assert d["weights_image"].startswith("data:image/png")
    assert d["total_spp"] == 3

    status, d2 = _post(server_url + "/api/render",
                       dict(cfg, accumulate=True))
    assert status == 200
    assert d2["accumulated_passes"] == 2
    assert d2["total_spp"] == 6

    status, tr = _post(server_url + "/api/trace",
                       dict(cfg, fx=0.5, fy=0.5))
    assert status == 200
    assert tr["tau_analytic"] >= 0
    assert 0.0 <= tr["transmittance_analytic"] <= 1.0
    assert "events" in tr["result"]


def test_majorant_failure_and_feedback_recovery(server_url):
    bad = {"scene": "complex", "partition": "flawed_grid",
           "policy": "strict", "grid_res": 8, "spp": 1,
           "width": 48, "height": 32}
    status, d = _post(server_url + "/api/render", bad)
    assert status == 422
    assert d["error"]["type"] == "majorant_violation"
    assert "repartition" in d["error"]["allowed_actions"]

    status, fb = _post(server_url + "/api/feedback",
                       {"action": "repartition", "config": bad})
    assert status == 200, fb
    assert fb["image"].startswith("data:image/png")
    assert fb["config"]["policy"] == "repartition"
    assert fb["kernel_probe"]["ok"] is True


def test_param_error_is_structured(server_url):
    status, d = _post(server_url + "/api/render", {"kappa": 999})
    assert status == 400 or status == 422
    assert d["error"]["type"] == "param_error"


def test_validation_endpoint_contract(server_url):
    status, d = _get(server_url + "/api/validation")
    assert status == 200
    assert isinstance(d.get("available"), bool)
    if d["available"]:
        assert "all_statistical_checks_pass" in d
        assert d["figure"].startswith("data:image/png;base64,")

"""Stdlib JSON + static server for the stochastic-medium experiment window.

Run::

    python3 -m stochastic_medium.cli serve --host 127.0.0.1 --port 8765

The browser is a thin view: every control descriptor comes from
:func:`stochastic_medium.params.control_schema`, every image/weight is
produced by the stochastic kernel, and majorant/budget failures are returned
as structured contexts so the UI can feed a chosen repair action back into
the kernel via ``/api/feedback``.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import numpy as np

from .batch import (heatmap_png_base64, render_multi_scalar, render_single,
                    tonemap_png_base64)
from .errors import MediumKernelError
from .params import RenderConfig, control_schema
from .scenes import FRAME, SCENE_FACTORIES, light_towards, LIGHT
from .tracking import estimate_transmittance

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

_MIME = {".html": "text/html; charset=utf-8",
         ".js": "application/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8",
         ".png": "image/png",
         ".json": "application/json; charset=utf-8",
         ".svg": "image/svg+xml"}


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def error_context(exc: BaseException) -> Dict[str, Any]:
    if isinstance(exc, MediumKernelError):
        try:
            return exc.to_context()
        except Exception:  # pragma: no cover - defensive
            return {"type": exc.error_type, "message": str(exc)}
    return {"type": "internal_error", "message": str(exc),
            "traceback": traceback.format_exc(limit=4)}


class KernelSession:
    """Holds the live kernel state: cached fields, accumulator, feedback log."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.fields: Dict[str, Any] = {}
        self.feedback_log: list[Dict[str, Any]] = []
        self.last_error: Optional[Dict[str, Any]] = None
        self.last_config: Optional[Dict[str, Any]] = None
        self.renders = 0
        self.traces = 0
        self.accumulator: Optional[Dict[str, Any]] = None
        self.started_at = time.time()

    def field(self, scene: str):
        if scene not in self.fields:
            self.fields[scene] = SCENE_FACTORIES[scene]()
        return self.fields[scene]

    def reset(self) -> None:
        with self.lock:
            self.accumulator = None
            self.last_error = None
            self.feedback_log.clear()
            self.fields.clear()

    # -- progressive accumulation ----------------------------------------
    def _acc_key(self, cfg: RenderConfig) -> str:
        d = cfg.to_dict()
        for k in ("seed",):
            d.pop(k, None)
        return json.dumps(d, sort_keys=True, default=_json_default)

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = RenderConfig.from_request(data)
        with self.lock:
            self.last_config = cfg.to_dict()
            field = self.field(cfg.scene)
            key = self._acc_key(cfg)
            want_acc = bool(data.get("accumulate", False))
            acc = self.accumulator if (want_acc and self.accumulator
                                       and self.accumulator.get("key") == key) \
                else None
            offset = acc["passes"] if acc else 0

            t0 = time.time()
            if cfg.mode == "multi":
                out = render_multi_scalar(field, cfg, seed_offset=offset)
                native_spp = 1
            else:
                out = render_single(field, cfg, seed_offset=offset)
                native_spp = cfg.spp
            elapsed = time.time() - t0

            rgb, trans, cand = out["rgb"], out["trans"], out["cand"]
            if acc is None:
                acc = {"key": key, "passes": 0, "rgb": np.zeros_like(rgb),
                       "trans": np.zeros_like(trans),
                       "cand": np.zeros_like(cand),
                       "width": rgb.shape[1], "height": rgb.shape[0],
                       "native_spp": native_spp,
                       "mode": cfg.mode}
            acc["rgb"] += rgb
            acc["trans"] += trans
            acc["cand"] += cand
            acc["passes"] += 1
            self.accumulator = acc
            self.renders += 1

            p = acc["passes"]
            mean_rgb = acc["rgb"] / p
            mean_trans = acc["trans"] / p
            mean_cand = acc["cand"] / p
            total_spp = p * native_spp

            weights_url = heatmap_png_base64(mean_trans, 0.0, 1.0)
            cmax = float(np.percentile(mean_cand, 99.0))
            cand_url = heatmap_png_base64(mean_cand, 0.0, max(1.0, cmax))
            image_url = tonemap_png_base64(mean_rgb, cfg.exposure)

            stats = dict(out["stats"])
            stats.update({
                "elapsed_seconds_last_pass": round(elapsed, 3),
                "accumulated_passes": p,
                "total_spp": total_spp,
                "weight_min": float(mean_trans.min()),
                "weight_max": float(mean_trans.max()),
                "candidate_heatmap_vmax": max(1.0, cmax),
            })
            return {
                "image": image_url,
                "weights_image": weights_url,
                "candidates_image": cand_url,
                "stats": stats,
                "config": cfg.to_dict(),
                "width": acc["width"],
                "height": acc["height"],
                "total_spp": total_spp,
                "accumulated_passes": p,
            }

    # -- single-ray inspection -------------------------------------------
    def trace(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from .integrators import single_scattering_radiance
        cfg = RenderConfig.from_request(data.get("config", data))
        fx = float(data.get("fx", 0.5))
        fy = float(data.get("fy", 0.5))
        fx = min(1.0, max(0.0, fx))
        fy = min(1.0, max(0.0, fy))
        with self.lock:
            self.last_config = cfg.to_dict()
            self.traces += 1
            field = self.field(cfg.scene)
            W, H = cfg.width, cfg.height
            ix = int(round(fx * (W - 1)))
            iy = int(round(fy * (H - 1)))
            xs = np.linspace(FRAME["x_min"], FRAME["x_max"], W)
            zs = np.linspace(FRAME["z_min"], FRAME["z_max"], H)
            x, z = float(xs[ix]), float(zs[iy])
            o = np.array([x, FRAME["camera_y"], z])
            d = np.array([0.0, 1.0, 0.0])
            L = float(field.aabb_max[1]) - FRAME["camera_y"]

            if cfg.mode == "multi":
                path = __import__("stochastic_medium.integrators",
                                  fromlist=["trace_path"]).trace_path(
                    field, o, d, L, kappa=cfg.kappa, albedo=cfg.albedo,
                    g=cfg.g, light_dir=light_towards(),
                    light_color=LIGHT["color"],
                    light_intensity=LIGHT["intensity"],
                    max_bounces=cfg.max_bounces, partition=cfg.partition,
                    grid_res=cfg.grid_res, policy=cfg.policy,
                    seed=int(cfg.seed) + ix * 131 + iy * 17)
                tau = cfg.kappa * field.optical_depth(o, d, 0.0, L)
                return {"mode": "multi", "pixel": {"ix": ix, "iy": iy,
                                                   "x": x, "z": z},
                        "length": L, "tau_analytic": float(tau),
                        "transmittance_analytic": float(np.exp(-min(tau, 700))),
                        "trace": path, "config": cfg.to_dict()}

            out = single_scattering_radiance(
                field, o, d, L, kappa=cfg.kappa, albedo=cfg.albedo, g=cfg.g,
                light_dir=light_towards(), light_color=LIGHT["color"],
                light_intensity=LIGHT["intensity"], partition=cfg.partition,
                grid_res=cfg.grid_res, policy=cfg.policy,
                seed=int(cfg.seed) + ix * 131 + iy * 17)
            res = out["result"]
            tau = float(res.tau_analytic)
            return {
                "mode": "single",
                "pixel": {"ix": ix, "iy": iy, "x": x, "z": z},
                "length": L,
                "tau_analytic": tau,
                "transmittance_analytic": float(np.exp(-min(tau, 700))),
                "radiance": [float(v) for v in out["radiance"]],
                "background": [float(v) for v in out["background"]],
                "result": res.to_dict(),
                "config": cfg.to_dict(),
            }

    # -- failure feedback -------------------------------------------------
    def feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        action = str(data.get("action", "repartition"))
        base = dict(data.get("config") or self.last_config or {})
        majorant_actions = ("repartition", "discard_resample", "clamp_demo")

        probe: Dict[str, Any] = {}
        render_payload: Optional[Dict[str, Any]] = None

        with self.lock:
            err_ctx = self.last_error
            field = self.field(str(base.get("scene", "complex")))

            def _probe_ray(**kw):
                x, z = 0.0, 2.0
                pos = (err_ctx or {}).get("position")
                if pos and len(pos) >= 3:
                    x, z = float(pos[0]), float(pos[2])
                o = np.array([x, FRAME["camera_y"], z])
                d = np.array([0.0, 1.0, 0.0])
                Ll = float(field.aabb_max[1]) - FRAME["camera_y"]
                return x, z, estimate_transmittance(
                    field, o, d, Ll, estimator="ratio", seed=31337,
                    record=True, **kw)

            if action in majorant_actions:
                base["policy"] = action
                if action == "repartition" and base.get("partition") == "global":
                    # scalar wrapper upgrades global -> certified grid;
                    # for the wavefront frame the repartition is explicit
                    base["partition"] = "certified_grid"
                cfg = RenderConfig.from_request(base)
                try:
                    x, z, r = _probe_ray(
                        kappa=cfg.kappa, partition=cfg.partition,
                        grid_res=cfg.grid_res, policy=action)
                    probe = {"ok": True, "x": x, "z": z,
                             "estimate": r.estimate,
                             "n_candidates": r.n_candidates,
                             "n_discarded": r.n_discarded,
                             "attempts": r.attempts,
                             "repairs": r.repairs,
                             "tau_analytic": r.tau_analytic}
                except MediumKernelError as exc:
                    probe = {"ok": False, "error": error_context(exc)}

            elif action == "lower_density":
                base["kappa"] = max(0.05,
                                    float(base.get("kappa", 1.7)) * 0.6)
                cfg = RenderConfig.from_request(base)
                probe = {"ok": True, "changed": "kappa",
                         "new_kappa": cfg.kappa}

            elif action == "raise_budget":
                base["max_iterations"] = min(
                    20000, int(base.get("max_iterations", 4096)) * 2)
                cfg = RenderConfig.from_request(base)
                probe = {"ok": True, "changed": "max_iterations",
                         "new_max_iterations": cfg.max_iterations}

            elif action == "analytic_handoff":
                cfg = RenderConfig.from_request(base)
                try:
                    x, z, r = _probe_ray(
                        kappa=cfg.kappa, partition=cfg.partition,
                        grid_res=cfg.grid_res, policy=cfg.policy,
                        max_candidates=25,
                        budget_action="analytic_handoff")
                    probe = {"ok": True, "x": x, "z": z,
                             "estimate": r.estimate,
                             "analytic_handoffs": r.analytic_handoffs,
                             "n_candidates": r.n_candidates,
                             "note": "候选预算耗尽后用解析光学深度反演精确采样自由程尾部"}
                except (MediumKernelError, TypeError) as exc:
                    probe = {"ok": False, "error": error_context(exc)
                             if isinstance(exc, MediumKernelError)
                             else {"message": str(exc)}}
                self.feedback_log.append({
                    "time": time.strftime("%H:%M:%S"), "action": action,
                    "error_type": (err_ctx or {}).get("type"),
                    "probe": probe})
                return {"probe_only": True, "kernel_probe": probe}
            else:
                cfg = RenderConfig.from_request(base)

            entry = {"time": time.strftime("%H:%M:%S"), "action": action,
                     "error_type": (err_ctx or {}).get("type"),
                     "probe": probe}
            self.feedback_log.append(entry)
            self.feedback_log = self.feedback_log[-30:]

            # feedback changes kernel parameters/policy -> restart accumulation
            self.accumulator = None
            render_payload = self.render(base)
            render_payload["kernel_probe"] = probe
            render_payload["feedback_entry"] = entry
            return render_payload

    def diagnostics(self) -> Dict[str, Any]:
        with self.lock:
            report_path = OUTPUT_DIR / "validation_report.json"
            validation = None
            if report_path.exists():
                try:
                    validation = json.loads(report_path.read_text())
                except Exception:
                    validation = None
            return {
                "renders": self.renders,
                "traces": self.traces,
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "last_error": self.last_error,
                "feedback_log": self.feedback_log,
                "accumulating": self.accumulator is not None,
                "accumulated_passes": (self.accumulator or {}).get("passes", 0),
                "scenes_loaded": list(self.fields.keys()),
                "validation_pass": (validation or {}).get(
                    "all_statistical_checks_pass"),
                "validation_elapsed": (validation or {}).get("elapsed_seconds"),
            }


SESSION = KernelSession()


class Handler(BaseHTTPRequestHandler):
    server_version = "StochasticMedium/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet local server
        return

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=_json_default).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": {"message": f"not found: {path.name}"}}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",
                         _MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length > 4_000_000:
            raise ValueError("request too large")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/schema":
                self._send_json({"schema": control_schema(),
                                 "scenes": list(SCENE_FACTORIES.keys()),
                                 "frame": FRAME})
            elif path == "/api/diagnostics":
                self._send_json(SESSION.diagnostics())
            elif path == "/api/validation":
                rp = OUTPUT_DIR / "validation_report.json"
                if not rp.exists():
                    self._send_json({"available": False})
                else:
                    rep = json.loads(rp.read_text())
                    rep["available"] = True
                    fig = OUTPUT_DIR / "transmittance_validation.png"
                    if fig.exists():
                        import base64
                        rep["figure"] = (
                            "data:image/png;base64,"
                            + base64.b64encode(fig.read_bytes()).decode())
                    self._send_json(rep)
            elif path.startswith("/outputs/"):
                name = Path(path.removeprefix("/outputs/")).name
                self._send_file(OUTPUT_DIR / name)
            elif path in ("/", "/index.html"):
                self._send_file(WEB_DIR / "index.html")
            elif path in ("/app.js", "/style.css"):
                self._send_file(WEB_DIR / path.lstrip("/"))
            else:
                self._send_json({"error": {"message": "not found"}}, 404)
        except Exception as exc:  # pragma: no cover - network boundary
            self._send_json({"error": error_context(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            data = self._read_json()
            if path == "/api/render":
                try:
                    self._send_json(SESSION.render(data))
                except MediumKernelError as exc:
                    ctx = error_context(exc)
                    SESSION.last_error = ctx
                    self._send_json({"error": ctx}, 422)
            elif path == "/api/trace":
                try:
                    self._send_json(SESSION.trace(data))
                except MediumKernelError as exc:
                    ctx = error_context(exc)
                    SESSION.last_error = ctx
                    self._send_json({"error": ctx}, 422)
            elif path == "/api/feedback":
                try:
                    self._send_json(SESSION.feedback(data))
                except MediumKernelError as exc:
                    ctx = error_context(exc)
                    SESSION.last_error = ctx
                    self._send_json({"error": ctx}, 422)
            elif path == "/api/reset":
                SESSION.reset()
                self._send_json({"ok": True})
            else:
                self._send_json({"error": {"message": "not found"}}, 404)
        except Exception as exc:  # pragma: no cover - network boundary
            self._send_json({"error": error_context(exc)}, 400)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Stochastic-medium experiment window: "
          f"http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":  # pragma: no cover
    serve()

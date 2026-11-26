"""Independent validation harness and artifact generation.

Runs the stochastic kernel against:

1. the analytic constant-density result T = exp(-kappa * length), including
   Bernoulli mean/variance checks and Poisson candidate-count formulas;
2. analytic piecewise-linear layered density vs scipy quad;
3. sparse volumes (hit rays and empty gaps);
4. the majorant failure / repair matrix, including a quantitatively biased
   clamp demonstration;
5. image generation for the complex scene plus sampling-weight maps.

Outputs JSON + PNG artifacts under ``outputs/``.  Run with::

    python3 -m stochastic_medium.cli validate
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .batch import heatmap_png_base64, render_single, tonemap_png_base64
from .errors import MajorantViolation
from .fields import CompositeField, UniformBox
from .params import RenderConfig
from .reference import quad_optical_depth, quad_transmittance
from .scenes import FRAME, load_scene
from .tracking import estimate_transmittance

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

CONST_FIELD = CompositeField([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
O = np.array([0.0, 0.0, -3.0])
D = np.array([0.0, 0.0, 1.0])
L = 4.0
PATH_LEN = 2.0


class LiedSegments(CompositeField):
    """Deliberately under-declares positive segment majorants (bad partition)."""

    def build_segments(self, *a, **k):
        segs = super().build_segments(*a, **k)
        for s in segs:
            if s.density_majorant > 0:
                s.density_majorant = 0.5
        return segs


def _z_mean(samples: np.ndarray, exact: float, theory_var: float) -> float:
    n = len(samples)
    return (float(np.mean(samples)) - exact) / math.sqrt(theory_var / n)


def _bernoulli_var_z(samples: np.ndarray, p: float) -> float:
    n = len(samples)
    var = float(np.var(samples, ddof=1))
    target = p * (1.0 - p)
    mu4 = target * (1.0 - 3.0 * target) + target * target
    var_of_var = max((mu4 - target * target) / n, 1e-12)
    return (var - target) / math.sqrt(var_of_var)


def validate_constant(n_per_kappa: int = 8000) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for kappa in (0.05, 0.3, 1.0, 2.5, 5.0):
        exact = math.exp(-kappa * PATH_LEN)
        rng = random.Random(7000 + int(kappa * 100))
        binary = np.array([
            estimate_transmittance(
                CONST_FIELD, O, D, L, kappa=kappa,
                estimator="binary", rng=rng).estimate
            for _ in range(n_per_kappa)])
        rng2 = random.Random(8000 + int(kappa * 100))
        ratio = np.array([
            estimate_transmittance(
                CONST_FIELD, O, D, L, kappa=kappa,
                estimator="ratio", rng=rng2).estimate
            for _ in range(n_per_kappa // 2)])
        zb = _z_mean(binary, exact, exact * (1 - exact))
        zr = _z_mean(ratio, exact, exact * (1 - exact))
        vz = _bernoulli_var_z(binary, exact)

        rng3 = random.Random(9000 + int(kappa * 100))
        counts = np.array([
            estimate_transmittance(
                CONST_FIELD, O, D, L, kappa=kappa,
                estimator="ratio", rng=rng3).n_candidates
            for _ in range(4000)])
        count_theory = kappa * PATH_LEN
        count_z = ((counts.mean() - count_theory)
                   / (math.sqrt(count_theory) / math.sqrt(len(counts))))
        rows.append({
            "kappa": kappa,
            "exact_transmittance": exact,
            "binary_mean": float(binary.mean()),
            "binary_mean_z": zb,
            "ratio_mean": float(ratio.mean()),
            "ratio_mean_z": zr,
            "variance_z": vz,
            "candidate_mean": float(counts.mean()),
            "candidate_theory": count_theory,
            "candidate_count_z": float(count_z),
            "pass": abs(zb) < 5.0 and abs(zr) < 5.5 and abs(vz) < 5.0
                    and abs(count_z) < 5.0,
        })
    return {"n_per_kappa": n_per_kappa, "rows": rows,
            "pass": all(r["pass"] for r in rows)}


def validate_layered_sparse(n: int = 6000) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    field = load_scene("layers")
    rays = [
        ([-2.0, FRAME["camera_y"], 1.1], [0.05, 1, 0.02], 6.4),
        ([1.5, FRAME["camera_y"], 3.3], [-0.12, 1, -0.05], 6.4),
        ([0.2, FRAME["camera_y"], -0.5], [0.2, 0.98, 0.08], 6.2),
    ]
    for j, (o0, d0, Ll) in enumerate(rays):
        dd = np.array(d0, float)
        dd /= np.linalg.norm(dd)
        oa = np.array(o0, float)
        ref = quad_transmittance(field, oa, dd, Ll, kappa=1.4)
        rng = random.Random(300 + j)
        s = np.array([
            estimate_transmittance(
                field, oa, dd, Ll, kappa=1.4, estimator="binary",
                partition="layers", rng=rng).estimate
            for _ in range(n)])
        z = _z_mean(s, ref["transmittance"],
                    ref["transmittance"] * (1 - ref["transmittance"]))
        checks.append({"scene": "layers", "ray": j,
                       "quad_transmittance": ref["transmittance"],
                       "mc_mean": float(s.mean()), "z": z,
                       "pass": abs(z) < 5.0})

    sparse = load_scene("sparse")
    blob = sparse.components[0]
    c = blob.center
    o_hit = np.array([c[0], FRAME["camera_y"], c[2]])
    dy = np.array([0.0, 1.0, 0.0])
    ref = quad_transmittance(sparse, o_hit, dy, FRAME["ray_length"], kappa=1.1)
    rng = random.Random(9)
    s = np.array([
        estimate_transmittance(
            sparse, o_hit, dy, FRAME["ray_length"], kappa=1.1,
            estimator="binary", partition="certified_grid", grid_res=22,
            rng=rng).estimate
        for _ in range(n)])
    z = _z_mean(s, ref["transmittance"],
                ref["transmittance"] * (1 - ref["transmittance"]))
    checks.append({"scene": "sparse_hit",
                   "quad_transmittance": ref["transmittance"],
                   "mc_mean": float(s.mean()), "z": z,
                   "pass": abs(z) < 5.0})

    o_gap = np.array([4.85, FRAME["camera_y"], 5.35])
    ref0 = quad_transmittance(sparse, o_gap, dy, FRAME["ray_length"], kappa=1.1)
    rng = random.Random(10)
    s0 = np.array([
        estimate_transmittance(
            sparse, o_gap, dy, FRAME["ray_length"], kappa=1.1,
            estimator="binary", partition="certified_grid", grid_res=22,
            rng=rng).estimate
        for _ in range(1500)])
    checks.append({"scene": "sparse_empty_gap",
                   "quad_transmittance": ref0["transmittance"],
                   "mc_mean": float(s0.mean()), "z": 0.0,
                   "pass": float(s0.mean()) == 1.0 and ref0["tau"] < 1e-6})
    return {"checks": checks, "pass": all(c["pass"] for c in checks)}


def validate_analytic_references() -> Dict[str, Any]:
    rays = []
    for x in (-3.2, -1.2, 0.0, 1.7, 3.6):
        for z in (-0.8, 0.8, 2.6, 4.1):
            d = np.array([0.1 * math.sin(x), 1.0, 0.07 * math.cos(z)])
            d /= np.linalg.norm(d)
            rays.append((np.array([x, FRAME["camera_y"], z]), d, 6.2))
    rows = []
    for name in ("constant", "layers", "sparse", "complex"):
        field = load_scene(name)
        worst = 0.0
        for o, d, Ll in rays:
            ta = field.optical_depth(o, d, 0.0, Ll)
            tq, err = quad_optical_depth(field, o, d, Ll)
            scale = max(1.0, abs(tq))
            rel = abs(ta - tq) / scale
            worst = max(worst, rel)
        rows.append({"scene": name, "worst_relative_error_vs_quad": worst,
                     "pass": worst < 2e-4})
    return {"rows": rows, "pass": all(r["pass"] for r in rows)}


def validate_majorant_policies(n: int = 6000) -> Dict[str, Any]:
    bad = LiedSegments([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
    true_t = math.exp(-2.0)
    wrong_t = math.exp(-1.0)

    rng = random.Random(103)
    clamp_samples = np.array([
        estimate_transmittance(bad, O, D, L, kappa=1.0,
                               policy="clamp_demo", rng=rng).estimate
        for _ in range(n)])
    rng = random.Random(101)
    discard_samples = np.array([
        estimate_transmittance(bad, O, D, L, kappa=1.0,
                               policy="discard_resample", rng=rng).estimate
        for _ in range(n)])

    raised = False
    ctx = None
    try:
        estimate_transmittance(bad, O, D, L, kappa=1.0,
                               policy="strict", seed=3)
    except MajorantViolation as e:
        raised = True
        ctx = e.to_context()

    z_discard = _z_mean(discard_samples, true_t, true_t * (1 - true_t))
    clamp_mean = float(clamp_samples.mean())
    out = {
        "strict_raises": raised,
        "violation_context": ctx,
        "honest_transmittance": true_t,
        "clamp_demo_mean": clamp_mean,
        "clamp_demo_predicted_biased": wrong_t,
        "clamp_is_biased": abs(clamp_mean - wrong_t) < 0.03
                           and abs(clamp_mean - true_t) > 0.15,
        "discard_mean": float(discard_samples.mean()),
        "discard_z": z_discard,
        "pass": raised and abs(z_discard) < 5.0
                and abs(clamp_mean - wrong_t) < 0.03,
    }
    return out


def make_figures(report: Dict[str, Any]) -> List[str]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    paths: List[str] = []

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (1) constant transmittance curve
    ax = axes[0, 0]
    rows = report["constant"]["rows"]
    ks = np.linspace(0, 5.5, 200)
    ax.plot(ks, np.exp(-ks * PATH_LEN), "-", lw=2,
            label="analytic $e^{-2\\kappa}$")
    kv = [r["kappa"] for r in rows]
    mv = [r["binary_mean"] for r in rows]
    err = [1.96 * math.sqrt(r["exact_transmittance"]
                            * (1 - r["exact_transmittance"])
                            / report["constant"]["n_per_kappa"]) for r in rows]
    ax.errorbar(kv, mv, yerr=err, fmt="o", capsize=3, label="binary MC")
    ax.set_xlabel(r"$\kappa$ [wu$^{-1}$]")
    ax.set_ylabel("mean transmittance")
    ax.set_title("Constant density: mean vs analytic")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # (2) z scores mean / variance
    ax = axes[0, 1]
    x = np.arange(len(rows))
    w = 0.25
    ax.bar(x - w, [r["binary_mean_z"] for r in rows], w, label="mean z")
    ax.bar(x, [r["variance_z"] for r in rows], w, label="variance z")
    ax.bar(x + w, [r["candidate_count_z"] for r in rows], w, label="count z")
    ax.axhline(5, color="r", ls="--", lw=1)
    ax.axhline(-5, color="r", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["kappa"]) for r in rows])
    ax.set_title("Statistical z-scores (all |z| < 5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # (3) candidate counts
    ax = axes[0, 2]
    ax.plot(kv, [r["candidate_theory"] for r in rows], "k-",
            label="Poisson theory $2\\kappa$")
    ax.plot(kv, [r["candidate_mean"] for r in rows], "o", label="MC mean")
    ax.set_xlabel(r"$\kappa$ [wu$^{-1}$]")
    ax.set_ylabel("candidate events / ray")
    ax.set_title("Ratio-tracking candidate counts")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # (4) layered / sparse means vs quad
    ax = axes[1, 0]
    checks = report["layered_sparse"]["checks"]
    labels = [f"{c['scene']}#{c.get('ray', '')}" for c in checks]
    exact = [c["quad_transmittance"] for c in checks]
    mc = [c["mc_mean"] for c in checks]
    y = np.arange(len(checks))
    ax.barh(y - .18, exact, .36, label="quad reference")
    ax.barh(y + .18, mc, .36, label="stochastic MC")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Layered / sparse validation")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # (5) majorant policy bias / repair
    ax = axes[1, 1]
    mp = report["majorant_policies"]
    names = ["honest\n$e^{-2}$", "clamp_demo\n(biased)",
             "discard+\nresample"]
    vals = [mp["honest_transmittance"], mp["clamp_demo_mean"],
            mp["discard_mean"]]
    colors = ["#4477aa", "#cc3311", "#228833"]
    ax.bar(names, vals, color=colors)
    ax.axhline(mp["honest_transmittance"], color="k", ls=":", lw=1)
    ax.axhline(mp["clamp_demo_predicted_biased"], color="#cc3311",
               ls="--", lw=1, label="predicted $e^{-1}$")
    ax.set_ylabel("mean transmittance")
    ax.set_title("Low majorant: clamping hides bias")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # (6) analytic vs quad errors
    ax = axes[1, 2]
    ar = report["analytic_references"]["rows"]
    ax.bar([r["scene"] for r in ar],
           [max(r["worst_relative_error_vs_quad"], 1e-12) for r in ar])
    ax.set_yscale("log")
    ax.set_ylabel("worst relative error vs scipy quad")
    ax.set_title("Analytic optical-depth segments")
    ax.grid(alpha=.3, which="both")

    fig.tight_layout()
    p = OUTPUT_DIR / "transmittance_validation.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))
    return paths


def render_scene_artifacts() -> Dict[str, str]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    paths: Dict[str, str] = {}

    cfg = RenderConfig(scene="complex", kappa=1.7, spp=48, width=384,
                       height=256, partition="certified_grid", grid_res=16,
                       policy="strict", exposure=1.25, seed=20261126)
    t0 = time.time()
    out = render_single(load_scene("complex"), cfg)
    render_seconds = time.time() - t0

    from PIL import Image
    import base64
    import io

    def _save_data_url(url: str, name: str) -> str:
        raw = url.split(",", 1)[1]
        p = OUTPUT_DIR / name
        p.write_bytes(base64.b64decode(raw))
        return str(p)

    paths["complex_png"] = _save_data_url(
        tonemap_png_base64(out["rgb"], cfg.exposure), "complex_scene.png")
    paths["weights_png"] = _save_data_url(
        heatmap_png_base64(out["trans"], 0.0, 1.0),
        "complex_transmittance_weights.png")
    cmax = float(np.percentile(out["cand"], 99))
    paths["candidates_png"] = _save_data_url(
        heatmap_png_base64(out["cand"], 0.0, max(1.0, cmax)),
        "complex_candidate_counts.png")

    # four-scene contact sheet
    thumbs = []
    for name in ("constant", "layers", "sparse", "complex"):
        c = RenderConfig(scene=name, spp=6, width=192, height=128,
                         partition=("certified_grid" if name == "sparse"
                                    else ("layers" if name == "layers"
                                          else "global")),
                         grid_res=18, seed=5)
        o = render_single(load_scene(name), c)
        url = tonemap_png_base64(o["rgb"], c.exposure)
        raw = url.split(",", 1)[1]
        thumbs.append(Image.open(io.BytesIO(base64.b64decode(raw))))
    sheet = Image.new("RGB", (192 * 2, 128 * 2), (20, 24, 34))
    sheet.paste(thumbs[0], (0, 0))
    sheet.paste(thumbs[1], (192, 0))
    sheet.paste(thumbs[2], (0, 128))
    sheet.paste(thumbs[3], (192, 128))
    sheet_path = OUTPUT_DIR / "scene_contact_sheet.png"
    sheet.save(sheet_path)
    paths["contact_sheet_png"] = str(sheet_path)
    paths["render_seconds"] = render_seconds
    paths["render_stats"] = out["stats"]
    return paths


def run(output_json: str | Path | None = None,
        quick: bool = False) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    n_const = 2500 if quick else 8000
    n_other = 2000 if quick else 6000

    report: Dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    report["analytic_references"] = validate_analytic_references()
    report["constant"] = validate_constant(n_const)
    report["layered_sparse"] = validate_layered_sparse(n_other)
    report["majorant_policies"] = validate_majorant_policies(n_other)
    report["all_statistical_checks_pass"] = all([
        report["analytic_references"]["pass"],
        report["constant"]["pass"],
        report["layered_sparse"]["pass"],
        report["majorant_policies"]["pass"],
    ])
    report["elapsed_seconds"] = time.time() - t0

    report["figures"] = make_figures(report)
    report["artifacts"] = render_scene_artifacts()

    out_path = Path(output_json) if output_json else OUTPUT_DIR / "validation_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                   default=_json_default))
    report["json_path"] = str(out_path)
    return report


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


if __name__ == "__main__":  # pragma: no cover
    rep = run()
    print(json.dumps({"pass": rep["all_statistical_checks_pass"],
                      "json": rep["json_path"],
                      "elapsed": rep["elapsed_seconds"]}, indent=2))

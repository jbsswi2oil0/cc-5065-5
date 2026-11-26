import math

import numpy as np
import pytest

from stochastic_medium.batch import render_single
from stochastic_medium.errors import MajorantViolation
from stochastic_medium.params import RenderConfig
from stochastic_medium.scenes import FRAME, load_scene


def _interior_mask(W, H, xlo=-2.0, xhi=2.0, zlo=0.6, zhi=3.4):
    xs = np.linspace(FRAME["x_min"], FRAME["x_max"], W)
    zs = np.linspace(FRAME["z_min"], FRAME["z_max"], H)
    gx, gz = np.meshgrid(xs, zs)
    return ((gx > xlo) & (gx < xhi) & (gz > zlo) & (gz < zhi),
            (gz < -0.6) | (gz > 4.6))


@pytest.mark.parametrize("kappa", [0.3, 1.0, 2.0])
def test_batch_constant_transmittance_interior(kappa):
    W, H, spp = 41, 31, 48
    cfg = RenderConfig(scene="constant", kappa=kappa, spp=spp,
                       width=W, height=H, seed=7)
    out = render_single(load_scene("constant"), cfg)
    inside, outside = _interior_mask(W, H)
    n_rays = int(inside.sum()) * spp
    exact = math.exp(-4.0 * kappa)  # camera rays cross y in [-2, 2]
    mean = float(out["trans"][inside].mean())
    z = (mean - exact) / math.sqrt(exact * (1 - exact) / n_rays)
    assert abs(z) < 5.0, (mean, exact, z)
    assert float(out["trans"][outside].mean()) == 1.0


def test_batch_candidate_count_matches_poisson_rate():
    W, H, spp = 41, 31, 24
    cfg = RenderConfig(scene="constant", kappa=1.4, spp=spp,
                       width=W, height=H, seed=8)
    out = render_single(load_scene("constant"), cfg)
    inside, _ = _interior_mask(W, H)
    mean_c = float(out["cand"][inside].mean())
    assert abs(mean_c - 4.0 * 1.4) < 0.25


def test_batch_grid_and_global_agree_on_constant():
    kw = dict(scene="constant", kappa=0.8, spp=32, width=31, height=23)
    a = render_single(load_scene("constant"),
                      RenderConfig(partition="global", seed=11, **kw))
    b = render_single(load_scene("constant"),
                      RenderConfig(partition="certified_grid", grid_res=10,
                                   seed=11, **kw))
    inside, _ = _interior_mask(31, 23)
    assert abs(float(a["trans"][inside].mean())
               - float(b["trans"][inside].mean())) < 0.06


def test_batch_flawed_grid_strict_refuses_frame():
    cfg = RenderConfig(scene="complex", spp=1, width=48, height=32,
                       partition="flawed_grid", policy="strict", grid_res=10)
    with pytest.raises(MajorantViolation):
        render_single(load_scene("complex"), cfg)


def test_batch_clamp_demo_is_visibly_biased():
    # under-declared global majorant (x0.55) + clipping: mean becomes e^{-0.55*4k}
    W, H, spp = 41, 31, 64
    cfg = RenderConfig(scene="constant", kappa=1.0, spp=spp, width=W,
                       height=H, seed=13, policy="clamp_demo")
    out = render_single(load_scene("constant"), cfg)
    inside, _ = _interior_mask(W, H)
    mean = float(out["trans"][inside].mean())
    biased_exact = math.exp(-0.55 * 4.0)
    honest = math.exp(-4.0)
    assert abs(mean - biased_exact) < 0.05
    assert mean > 5 * honest

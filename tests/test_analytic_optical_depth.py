import numpy as np
import pytest

from stochastic_medium.reference import (quad_optical_depth,
                                         simpson_optical_depth)
from stochastic_medium.scenes import FRAME, load_scene


def _rays():
    rays = []
    for x in (-3.5, -1.2, 0.0, 1.7, 3.8):
        for z in (-0.8, 0.6, 2.4, 4.2):
            d = np.array([0.10 * np.sin(x), 1.0, 0.07 * np.cos(z)])
            d /= np.linalg.norm(d)
            rays.append((np.array([x, FRAME["camera_y"], z]), d,
                         FRAME["ray_length"]))
    # oblique / almost grazing rays
    rays.append((np.array([0.2, -3.0, 2.0]),
                 np.array([0.9, 0.35, 0.25]) / np.linalg.norm([0.9, .35, .25]),
                 5.0))
    rays.append((np.array([-4.0, -2.8, 3.1]),
                 np.array([0.2, 0.98, -0.05]), 5.5))
    return rays


@pytest.mark.parametrize("scene", ["constant", "layers", "sparse", "complex"])
def test_analytic_tau_matches_scipy_quad(scene):
    field = load_scene(scene)
    worst = 0.0
    for o, d, L in _rays():
        tau_a = field.optical_depth(o, d, 0.0, L)
        tau_q, err = quad_optical_depth(field, o, d, L)
        assert tau_a >= -1e-9
        scale = max(1.0, abs(tau_q))
        worst = max(worst, abs(tau_a - tau_q) / scale)
        # quadrature at density discontinuities can underestimate its error
        assert abs(tau_a - tau_q) <= 2e-5 * scale + 10 * err, (
            scene, tau_a, tau_q, err)
    assert worst < 1e-4


def test_simpson_independent_crosscheck():
    field = load_scene("complex")
    o = np.array([-1.3, -3.2, 3.4])
    d = np.array([0.1, 1.0, 0.05])
    d /= np.linalg.norm(d)
    a = field.optical_depth(o, d, 0, 6.4)
    s = simpson_optical_depth(field, o, d, 6.4, n=40001)
    q, _ = quad_optical_depth(field, o, d, 6.4)
    assert abs(a - s) < 1e-4 * max(1.0, a)
    assert abs(q - s) < 1e-4 * max(1.0, q)


@pytest.mark.parametrize("scene,x,z", [
    ("complex", -3.94, 1.21),   # exact +y ray above mist stack top z=1.1
    ("complex", 2.45, 1.21),
    ("complex", 0.0, 2.5),
    ("complex", 4.2, -1.6),
    ("layers", 0.0, -1.6),      # below all layers
    ("layers", 0.0, 5.6),       # above all layers
    ("layers", 0.0, 3.0),
])
def test_parallel_y_ray_outside_layer_range_is_zero_or_matches_quad(scene, x, z):
    field = load_scene(scene)
    o = np.array([x, FRAME["camera_y"], z])
    d = np.array([0.0, 1.0, 0.0])
    tau_a = field.optical_depth(o, d, 0.0, FRAME["ray_length"])
    tau_q, err = quad_optical_depth(field, o, d, FRAME["ray_length"])
    assert abs(tau_a - tau_q) <= 2e-5 * max(1.0, abs(tau_q)) + 10 * err, (
        scene, x, z, tau_a, tau_q)


def test_parallel_xy_outside_bounds_gives_zero_tau():
    field = load_scene("layers")
    # x outside [-5,5], ray parallel to y and z -> never enters xy bounds
    o = np.array([6.0, FRAME["camera_y"], 1.0])
    d = np.array([0.0, 1.0, 0.0])
    assert field.optical_depth(o, d, 0.0, 8.0) == 0.0
    # constant y below the xy y-bounds, ray along x
    o2 = np.array([0.0, -3.2, 1.0])
    d2 = np.array([1.0, 0.0, 0.0])
    assert field.optical_depth(o2, d2, 0.0, 12.0) == 0.0


def test_batch_and_scalar_optical_depth_agree_on_parallel_rays():
    field = load_scene("complex")
    rays = [
        ([-3.94, -3.2, 1.21], [0, 1, 0]),
        ([0.0, -3.2, 2.5], [0, 1, 0]),
        ([6.0, -3.2, 1.0], [0, 1, 0]),
        ([0.2, -3.2, 0.3], [0, 1, 0]),
    ]
    o = np.array([r[0] for r in rays], float)
    d = np.array([r[1] for r in rays], float)
    tb = field.optical_depth_batch(o, d, 0.0,
                                   np.full(len(rays), FRAME["ray_length"]))
    for i, (oo, dd) in enumerate(zip(o, d)):
        ts = field.optical_depth(oo, dd, 0.0, FRAME["ray_length"])
        assert abs(tb[i] - ts) < 1e-12


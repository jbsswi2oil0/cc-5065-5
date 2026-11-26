import math
import random

import numpy as np

from stochastic_medium.fields import CompositeField, UniformBox
from stochastic_medium.integrators import (hg_phase, sample_hg_direction,
                                           shadow_transmittance, trace_path,
                                           single_scattering_radiance)

F = CompositeField([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
LIGHT = np.array([0.2, -0.3, 0.93])
LIGHT /= np.linalg.norm(LIGHT)


def test_hg_isotropic_normalization_value():
    assert abs(hg_phase(0.3, 0.0) - 1.0 / (4 * math.pi)) < 1e-12


def test_hg_sample_mean_cosine_equals_g():
    rng = random.Random(31)
    incident = np.array([0.0, 1.0, 0.0])
    for g in (-0.6, 0.0, 0.7):
        cosines = []
        for _ in range(12000):
            w = sample_hg_direction(incident, g, rng)
            cosines.append(float(np.dot(w, incident)))
        mean_cos = float(np.mean(cosines))
        assert abs(mean_cos - g) < 0.03, (g, mean_cos)


def test_shadow_transmittance_analytic():
    eps = 1e-5
    t = shadow_transmittance(F, np.array([0.0, 0.0, 0.0]),
                             np.array([0, 0, 1.0]), kappa=0.9, epsilon=eps)
    assert abs(t - math.exp(-0.9 * (1.0 - eps))) < 1e-9
    t2 = shadow_transmittance(F, np.array([0.0, 0.0, 0.0]),
                              np.array([0, 0, -1.0]), kappa=0.9,
                              epsilon=eps)
    assert abs(t2 - math.exp(-0.9 * (1.0 - eps))) < 1e-9


def test_trace_path_bounded_and_finite():
    rng = random.Random(41)
    out = trace_path(
        F, np.array([0.0, -3.0, 0.0]), np.array([0, 1.0, 0]), 4.0,
        kappa=1.1, albedo=0.85, g=0.3, light_dir=LIGHT,
        light_color=np.ones(3), light_intensity=2.0,
        max_bounces=3, rng=rng)
    rad = np.array(out["radiance"])
    assert np.all(np.isfinite(rad)) and np.all(rad >= 0)
    assert len(out["bounces"]) <= 5
    assert out["bounces"][-1]["type"] in ("background", "real_scattering")


def test_more_bounces_do_not_blow_up():
    rng = random.Random(42)
    out = trace_path(
        F, np.array([0.2, -3.0, 0.1]), np.array([0.05, 1, 0.02]), 4.2,
        kappa=2.5, albedo=0.95, g=0.6, light_dir=LIGHT,
        light_color=np.ones(3), light_intensity=3.0,
        max_bounces=4, rng=rng)
    rad = np.array(out["radiance"])
    assert np.all(np.isfinite(rad)) and np.all(rad < 1e4)


def test_single_scattering_estimator_returns_weights():
    rng = random.Random(43)
    out = single_scattering_radiance(
        F, np.array([0.0, -3.0, 0.0]), np.array([0, 1.0, 0]), 4.0,
        kappa=0.8, albedo=0.9, g=0.2, light_dir=LIGHT,
        light_color=np.ones(3), light_intensity=2.0, rng=rng)
    rad = out["radiance"]
    assert np.all(np.isfinite(rad)) and np.all(rad >= 0)
    res = out["result"]
    assert len(res.events) == res.n_candidates
    assert all(0.0 <= e.p_applied <= 1.0 for e in res.events)

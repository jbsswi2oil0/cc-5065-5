import math
import random

import numpy as np
import pytest

from stochastic_medium.errors import EventBudgetExceeded
from stochastic_medium.fields import (CompositeField, LayerStack, UniformBox)
from stochastic_medium.tracking import (analytic_free_flight,
                                        estimate_transmittance)
from tests.statsutil import assert_unbiased


def box_field(value=1.0):
    return CompositeField([UniformBox([-1, -1, -1], [1, 1, 1], value)])


def test_ray_pointing_away_is_vacuum():
    f = box_field()
    r = estimate_transmittance(f, [0, 0, -3], [0, 0, -1], 4.0, seed=1)
    assert r.estimate == 1.0
    assert r.n_candidates == 0


def test_zero_length_and_negative_safe():
    f = box_field()
    assert estimate_transmittance(f, [0, 0, 0], [0, 0, 1], 0.0).estimate == 1.0


def test_origin_on_entry_plane_and_exit_at_L():
    f = box_field()
    samples = [
        estimate_transmittance(f, [0, 0, -1], [0, 0, 1], 2.0,
                               kappa=0.8, seed=100 + i).estimate
        for i in range(8000)]
    assert_unbiased(np.array(samples), math.exp(-1.6))


def test_origin_inside_medium():
    f = box_field()
    samples = [
        estimate_transmittance(f, [0, 0, 0.2], [0, 0, 1], 0.8,
                               kappa=1.4, seed=200 + i).estimate
        for i in range(8000)]
    assert_unbiased(np.array(samples), math.exp(-1.4 * 0.8))


def test_ray_parallel_to_layer_boundaries():
    layers = LayerStack([
        (0.0, 1.0, 0.3, 0.0),
        (1.0, 2.0, 1.1, 0.0),
    ]).with_xy_bounds(-5, -5, 5, 5)
    f = CompositeField([layers], [-5, -5, 0], [5, 5, 2])
    # horizontal ray at z=1.5 -> constant density 1.1, length 4
    o = np.array([-2.0, 0.0, 1.5])
    d = np.array([1.0, 0.0, 0.0])
    tau = f.optical_depth(o, d, 0, 4.0)
    assert abs(tau - 4.4) < 1e-9
    samples = [
        estimate_transmittance(f, o, d, 4.0, kappa=0.5,
                               partition="layers", seed=300 + i).estimate
        for i in range(8000)]
    assert_unbiased(np.array(samples), math.exp(-0.5 * 4.4))


def test_extremely_thin_layer_segment_handled():
    eps = 1e-13
    layers = LayerStack([
        (0.0, 1.0, 0.2, 0.0),
        (1.0, 1.0 + eps, 5.0, 0.0),
        (1.0 + eps, 2.0, 0.4, 0.0),
    ]).with_xy_bounds(-5, -5, 5, 5)
    f = CompositeField([layers], [-5, -5, 0], [5, 5, 2])
    o = np.array([0.0, 0.0, -1.0])
    d = np.array([0.0, 0.0, 1.0])
    L = 3.0
    tau = f.optical_depth(o, d, 0, L)
    assert math.isfinite(tau) and tau > 0
    r = estimate_transmittance(f, o, d, L, partition="layers",
                               kappa=1.0, seed=1)
    assert math.isfinite(r.estimate)
    assert any(abs(s["t1"] - s["t0"]) <= 1e-12 for s in r.segments)


def test_zero_extinction_is_exact_vacuum():
    f = box_field()
    r = estimate_transmittance(f, [0, 0, -3], [0, 0, 1], 4, kappa=0.0)
    assert r.estimate == 1.0
    assert r.vacuum_fast_path and r.n_candidates == 0


def test_near_zero_extinction_converges_to_vacuum():
    f = box_field()
    # deep vacuum fast path is numerically exactly the vacuum limit
    r = estimate_transmittance(f, [0, 0, -3], [0, 0, 1], 4, kappa=1e-13)
    assert r.vacuum_fast_path
    assert abs(r.estimate - 1.0) < 1e-11
    # small but stochastic regime: mean converges to 1 at first order
    samples = [
        estimate_transmittance(f, [0, 0, -3], [0, 0, 1], 4, kappa=1e-5,
                               seed=500 + i).estimate
        for i in range(20000)]
    exact = math.exp(-2e-5)
    assert_unbiased(np.array(samples), exact, zmax=5.0,
                    theory_var=exact * (1 - exact))
    assert abs(float(np.mean(samples)) - 1.0) < 5e-4


def test_extremely_dense_raises_instead_of_looping():
    f = box_field()
    with pytest.raises(EventBudgetExceeded):
        estimate_transmittance(f, [0, 0, -3], [0, 0, 1], 4,
                               kappa=1e6, max_candidates=200,
                               budget_action="error", seed=1)


def test_analytic_handoff_cap_zero_is_exact_for_ratio():
    f = box_field()
    r = estimate_transmittance(f, [0, 0, -3], [0, 0, 1], 4, kappa=1.37,
                               estimator="ratio", max_candidates=0,
                               budget_action="analytic_handoff", seed=1)
    assert r.analytic_handoffs >= 1
    assert abs(r.estimate - math.exp(-2 * 1.37)) < 1e-10


def test_analytic_handoff_binary_is_unbiased():
    f = box_field()
    samples = [
        estimate_transmittance(f, [0, 0, -3], [0, 0, 1], 4, kappa=0.8,
                               estimator="binary", max_candidates=1,
                               budget_action="analytic_handoff",
                               seed=400 + i).estimate
        for i in range(10000)]
    assert_unbiased(np.array(samples), math.exp(-1.6), zmax=5.0)


def test_analytic_free_flight_inversion_distribution():
    f = box_field()
    rng = random.Random(55)
    hits = 0
    n = 20000
    for _ in range(n):
        t = analytic_free_flight(f, np.array([0, 0, -3.0]),
                                 np.array([0, 0, 1.0]), 0.0, 4.0, 0.8, rng)
        hits += t is not None
    p_hit = 1.0 - math.exp(-1.6)
    z = (hits / n - p_hit) / math.sqrt(p_hit * (1 - p_hit) / n)
    assert abs(z) < 5.0

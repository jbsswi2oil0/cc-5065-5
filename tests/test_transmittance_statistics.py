import math
import random

import numpy as np
import pytest

from stochastic_medium.reference import quad_transmittance
from stochastic_medium.scenes import FRAME, load_scene
from stochastic_medium.fields import UniformBox, CompositeField
from tests.statsutil import assert_unbiased, bernoulli_variance_z, sample_means


CONST_FIELD = CompositeField([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
O = np.array([0.0, 0.0, -3.0])
D = np.array([0.0, 0.0, 1.0])
L = 4.0
PATH_LEN = 2.0  # through the box along z


@pytest.mark.parametrize("kappa", [0.05, 0.3, 1.0, 2.5, 5.0])
@pytest.mark.parametrize("estimator", ["binary", "ratio"])
def test_constant_density_mean_transmittance(kappa, estimator):
    exact = math.exp(-kappa * PATH_LEN)
    n = 40000 if kappa >= 5.0 else 12000
    s = sample_means(CONST_FIELD, O, D, L, n,
                     seed=1000 + int(kappa * 100) + (7 if estimator == "ratio" else 0),
                     estimator=estimator, kappa=kappa)
    # with an exact majorant both estimators are Bernoulli in constant media
    assert_unbiased(s, exact, zmax=5.0, theory_var=exact * (1 - exact))


def test_binary_estimator_variance_is_bernoulli():
    exact = math.exp(-2.0)
    s = sample_means(CONST_FIELD, O, D, L, 30000, seed=4242,
                     estimator="binary", kappa=1.0)
    z = bernoulli_variance_z(s, exact)
    assert abs(z) < 5.0, z
    assert abs(float(np.mean(s)) - exact) < 0.01


def test_binary_candidate_count_formula():
    kappa = 1.3
    rng = random.Random(77)
    counts = []
    for _ in range(30000):
        r = __import__("stochastic_medium.tracking", fromlist=["estimate_transmittance"]) \
            .estimate_transmittance(CONST_FIELD, O, D, L, kappa=kappa,
                                    estimator="binary", rng=rng)
        counts.append(r.n_candidates)
    counts = np.array(counts)
    # E[N] = M/sigma (1 - exp(-sigma l)) = 1 - exp(-kappa l), M = sigma
    theory = 1.0 - math.exp(-kappa * PATH_LEN)
    z = (counts.mean() - theory) / (counts.std(ddof=1) / math.sqrt(len(counts)))
    assert abs(z) < 5.0, (counts.mean(), theory, z)


def test_ratio_candidate_count_is_poisson_rate_times_length():
    kappa = 0.8
    rng = random.Random(78)
    counts = []
    from stochastic_medium.tracking import estimate_transmittance
    for _ in range(30000):
        r = estimate_transmittance(CONST_FIELD, O, D, L, kappa=kappa,
                                   estimator="ratio", rng=rng)
        counts.append(r.n_candidates)
    counts = np.array(counts)
    theory = kappa * PATH_LEN
    z = (counts.mean() - theory) / (math.sqrt(theory) / math.sqrt(len(counts)))
    assert abs(z) < 5.0, (counts.mean(), theory, z)


@pytest.mark.parametrize("kappa", [0.7, 2.2])
def test_layered_density_matches_independent_quad(kappa):
    field = load_scene("layers")
    rays = [
        (np.array([-2.0, FRAME["camera_y"], 1.1]), np.array([0.05, 1, 0.02]), 6.4),
        (np.array([1.5, FRAME["camera_y"], 3.3]), np.array([-0.12, 1, -0.05]), 6.4),
        (np.array([0.2, FRAME["camera_y"], -0.5]), np.array([0.2, 0.98, 0.08]), 6.2),
    ]
    for j, (o, d0, Ll) in enumerate(rays):
        d = d0 / np.linalg.norm(d0)
        ref = quad_transmittance(field, o, d, Ll, kappa=kappa)
        # binary estimator: exact Bernoulli variance, no heavy-tail SEM issue
        s = sample_means(field, o, d, Ll, 16000, seed=300 + j,
                         estimator="binary", kappa=kappa,
                         partition="layers")
        assert_unbiased(s, ref["transmittance"], zmax=5.5,
                        theory_var=ref["transmittance"]
                        * (1 - ref["transmittance"]))


def test_sparse_volume_hits_and_empty_gaps():
    field = load_scene("sparse")
    blob = field.components[0]
    c = blob.center
    o_hit = np.array([c[0], FRAME["camera_y"], c[2]])
    d = np.array([0.0, 1.0, 0.0])
    ref = quad_transmittance(field, o_hit, d, FRAME["ray_length"], kappa=1.1)
    assert ref["tau"] > 0.05
    s = sample_means(field, o_hit, d, FRAME["ray_length"], 12000, seed=9,
                     estimator="binary", kappa=1.1,
                     partition="certified_grid", grid_res=22)
    assert_unbiased(s, ref["transmittance"], zmax=5.5,
                    theory_var=ref["transmittance"]
                    * (1 - ref["transmittance"]))

    # a ray through an empty gap: analytic tau == 0 and every sample is 1
    o_gap = np.array([4.85, FRAME["camera_y"], 5.35])
    ref0 = quad_transmittance(field, o_gap, d, FRAME["ray_length"], kappa=1.1)
    assert ref0["tau"] < 1e-6
    s0 = sample_means(field, o_gap, d, FRAME["ray_length"], 2000, seed=10,
                      estimator="binary", kappa=1.1,
                      partition="certified_grid", grid_res=22)
    assert float(np.mean(s0)) == 1.0

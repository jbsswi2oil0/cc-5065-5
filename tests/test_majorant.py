import math
import random

import numpy as np
import pytest

from stochastic_medium.errors import MajorantViolation
from stochastic_medium.fields import CompositeField, GaussianBlob, UniformBox
from stochastic_medium.scenes import FRAME, load_scene
from stochastic_medium.tracking import estimate_transmittance
from tests.statsutil import assert_unbiased

O = np.array([0.0, 0.0, -3.0])
D = np.array([0.0, 0.0, 1.0])
L = 4.0


class LiedSegments(CompositeField):
    """Test double: keeps certified global/box bounds honest but lies in
    the declared segment majorants (simulates an external bad partition)."""

    def build_segments(self, *a, **k):
        segs = super().build_segments(*a, **k)
        for s in segs:
            if s.density_majorant > 0:
                s.density_majorant = 0.5
        return segs


def test_certified_partitions_never_violate_on_scenes():
    for name in ["constant", "layers", "sparse", "complex"]:
        field = load_scene(name)
        rng = random.Random(11)
        for partition in ["global", "certified_grid", "layers"]:
            for x in (-3.2, 0.0, 2.7):
                o = np.array([x, FRAME["camera_y"], 1.5 + 0.3 * x])
                d = np.array([0.08, 1.0, 0.03])
                d /= np.linalg.norm(d)
                kw = dict(partition=partition, grid_res=12,
                          policy="strict", kappa=1.3, rng=rng)
                r = estimate_transmittance(field, o, d, 6.0, **kw)
                assert math.isfinite(r.estimate)
                assert not any(x.get("kind", "").startswith("audit")
                               and "BIASED" in x.get("kind", "")
                               for x in r.repairs)


def test_flawed_grid_is_detected_by_audit():
    field = CompositeField(
        [GaussianBlob([0, 0, 0], [0.25, 0.25, 0.25], 2.0, 3.0)],
        [-1, -1, -1], [1, 1, 1])
    with pytest.raises(MajorantViolation) as ei:
        estimate_transmittance(field, O, D, L, partition="flawed_grid",
                               grid_res=3, policy="strict", seed=1)
    ctx = ei.value.to_context()
    assert ctx["safe_density_majorant"] > ctx["declared_majorant"]
    assert ctx["raw_accept_probability"] >= 0.0
    assert ctx["detected_by"] == "audit"
    assert "repartition" in ctx["allowed_actions"]


def test_runtime_check_detects_uncertified_low_majorant():
    field = LiedSegments([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
    raised = False
    for seed in range(60):
        try:
            estimate_transmittance(field, O, D, L, policy="strict",
                                   audit=False, seed=seed)
        except MajorantViolation as e:
            assert e.raw_accept_probability > 1.0
            raised = True
            break
    assert raised


def test_audit_discard_resample_is_unbiased():
    field = LiedSegments([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
    rng = random.Random(101)
    samples = [
        estimate_transmittance(field, O, D, L, kappa=1.0,
                               policy="discard_resample", rng=rng).estimate
        for _ in range(12000)]
    assert_unbiased(np.array(samples), math.exp(-2.0), zmax=5.0)


def test_repartition_is_unbiased():
    field = CompositeField(
        [GaussianBlob([0, 0, 0], [0.25, 0.25, 0.25], 2.0, 3.0)],
        [-1, -1, -1], [1, 1, 1])
    tau = field.optical_depth(O, D, 0, L)
    rng = random.Random(102)
    samples = []
    attempts = set()
    for _ in range(12000):
        r = estimate_transmittance(field, O, D, L, kappa=1.0,
                                   partition="flawed_grid", grid_res=3,
                                   policy="repartition", rng=rng)
        samples.append(r.estimate)
        attempts.add(r.attempts)
    assert_unbiased(np.array(samples), math.exp(-tau), zmax=5.0)
    assert 2 in attempts  # at least the first attempt failed and repartitioned


def test_clamp_after_violation_is_demonstrably_biased():
    field = LiedSegments([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
    rng = random.Random(103)
    samples = np.array([
        estimate_transmittance(field, O, D, L, kappa=1.0,
                               policy="clamp_demo", rng=rng).estimate
        for _ in range(16000)])
    true_t = math.exp(-2.0)
    wrong_t = math.exp(-1.0)  # clamped p=1 under M=0.5 over length 2
    assert abs(samples.mean() - wrong_t) < 0.02
    assert abs(samples.mean() - true_t) > 0.15


def test_batch_level_invalidation_discards_all_bad_samples():
    """If a strict batch finds a runtime violation, the unbiased action is
    to discard the whole batch and resample with a certified majorant."""
    bad = LiedSegments([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
    n = 4000
    invalid = False
    try:
        for i in range(50):  # likely to hit a candidate violation quickly
            estimate_transmittance(bad, O, D, L, policy="strict",
                                   audit=False, seed=i)
    except MajorantViolation:
        invalid = True
    assert invalid

    rng = random.Random(202)
    good = CompositeField([UniformBox([-1, -1, -1], [1, 1, 1], 1.0)])
    samples = [
        estimate_transmittance(good, O, D, L, kappa=1.0, rng=rng).estimate
        for _ in range(n)]
    assert_unbiased(np.array(samples), math.exp(-2.0), zmax=5.0)

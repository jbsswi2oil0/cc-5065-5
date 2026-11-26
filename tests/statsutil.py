import math
import random
from typing import Callable, Sequence

import numpy as np

from stochastic_medium.tracking import estimate_transmittance


def sample_means(field, o, d, L, n: int, *, seed: int = 1,
                 estimator: str = "binary", **kwargs):
    rng = random.Random(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = estimate_transmittance(
            field, o, d, L, estimator=estimator, rng=rng, **kwargs).estimate
    return out


def assert_unbiased(samples: np.ndarray, exact: float, zmax: float = 5.5,
                    theory_var: float | None = None):
    n = len(samples)
    mean = float(np.mean(samples))
    sample_sem = float(np.std(samples, ddof=1) / math.sqrt(n))
    if theory_var is not None:
        sem = math.sqrt(max(theory_var, 0.0) / n)
    else:
        sem = sample_sem
    if sem == 0.0:
        # deterministic estimator: must match closely
        assert abs(mean - exact) < 1e-8, (mean, exact)
        return mean, 0.0
    z = abs(mean - exact) / sem
    assert z <= zmax, (
        f"mean={mean:.6g} exact={exact:.6g} z={z:.2f} > {zmax} "
        f"(sem={sem:.2e}, sample_sem={sample_sem:.2e}, n={n})")
    return mean, z


def bernoulli_variance_z(samples: np.ndarray, p: float) -> float:
    """z-score for sample variance against Bernoulli variance p(1-p)."""
    n = len(samples)
    var = float(np.var(samples, ddof=1))
    target = p * (1.0 - p)
    # variance of the sample variance for a Bernoulli population
    mu4 = target * (1.0 - 3.0 * target) + target * target
    var_of_var = max((mu4 - target * target) / n, 1e-12)
    return (var - target) / math.sqrt(var_of_var)

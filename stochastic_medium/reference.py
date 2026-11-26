"""Independent numerical references for optical depth / transmittance.

The stochastic tests compare against scipy adaptive quadrature of the raw
density definition, which is an independent integration path from the
analytic segment formulas in fields.py.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import quad


def quad_optical_depth(field_obj, origin, direction, L: float,
                       kappa: float = 1.0,
                       epsabs: float = 1e-9,
                       epsrel: float = 2e-9
                       ) -> Tuple[float, float]:
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)

    def integrand(t):
        p = o + d * t
        return kappa * float(field_obj.density(p))

    tau, err = quad(integrand, 0.0, float(L), epsabs=epsabs,
                    epsrel=epsrel, limit=300)
    return tau, err


def quad_transmittance(field_obj, origin, direction, L: float,
                       kappa: float = 1.0) -> dict:
    tau, err = quad_optical_depth(field_obj, origin, direction, L, kappa)
    return {"tau": tau, "tau_quad_error": err,
            "transmittance": float(np.exp(-min(tau, 700.0)))}


def simpson_optical_depth(field_obj, origin, direction, L: float,
                          kappa: float = 1.0, n: int = 20001) -> float:
    """Dense composite Simpson reference (second independent method)."""
    if n % 2 == 0:
        n += 1
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    ts = np.linspace(0.0, float(L), n)
    pts = o[None, :] + d[None, :] * ts[:, None]
    vals = kappa * np.asarray(field_obj.density(pts), float)
    h = L / (n - 1)
    return float(h / 3.0 * (vals[0] + vals[-1]
                            + 4.0 * vals[1:-1:2].sum()
                            + 2.0 * vals[2:-2:2].sum()))

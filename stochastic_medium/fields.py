"""Density fields, analytic optical depths and partition majorants.

Every field exposes:

* ``density(p)`` -- dimensionless relative density d(x) >= 0;
* ``density_majorant()`` -- certified global upper bound of d;
* ``box_density_bound(lo, hi)`` -- certified bound over an AABB cell;
* ``optical_depth*`` -- analytic integral of density along a ray segment.

The stochastic kernel multiplies density-side values by the single
``kappa`` scale, so extinction sigma_t, scattering sigma_s and the
majorant M all stay in consistent 1/length units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import erf


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def ray_aabb_interval(o: np.ndarray, d: np.ndarray,
                      lo: np.ndarray, hi: np.ndarray
                      ) -> Optional[Tuple[float, float]]:
    """Scalar ray vs AABB. Returns (t_near, t_far), possibly negative."""
    t_near = -math.inf
    t_far = math.inf
    for i in range(3):
        oi, di = float(o[i]), float(d[i])
        if abs(di) < 1e-15:
            if oi < lo[i] - 1e-12 or oi > hi[i] + 1e-12:
                return None
            continue
        t1 = (float(lo[i]) - oi) / di
        t2 = (float(hi[i]) - oi) / di
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far + 1e-12:
            return None
    if t_far < t_near - 1e-12:
        return None
    return t_near, t_far


def ray_aabb_interval_batch(o: np.ndarray, d: np.ndarray,
                            lo: np.ndarray, hi: np.ndarray):
    """Vectorised slab intersection for o (N,3), d (N,3) or (3,).

    Returns t_near, t_far arrays; a ray that misses has t_near > t_far.
    """
    n = o.shape[0]
    dd = d if d.ndim == 2 else np.broadcast_to(d, o.shape)
    t_near = np.full(n, -np.inf)
    t_far = np.full(n, np.inf)
    for axis in range(3):
        di = dd[:, axis]
        same = np.abs(di) < 1e-15
        safe_di = np.where(same, 1.0, di)
        t1 = (lo[axis] - o[:, axis]) / safe_di
        t2 = (hi[axis] - o[:, axis]) / safe_di
        ta = np.minimum(t1, t2)
        tb = np.maximum(t1, t2)
        # parallel axes do not constrain the interval unless the origin is
        # outside that slab (handled below)
        ta = np.where(same, -np.inf, ta)
        tb = np.where(same, np.inf, tb)
        t_near = np.maximum(t_near, ta)
        t_far = np.minimum(t_far, tb)
        outside_parallel = same & (
            (o[:, axis] < lo[axis] - 1e-12)
            | (o[:, axis] > hi[axis] + 1e-12))
        # impossible interval (+inf, -inf): stays inverted through all clips
        t_near = np.where(outside_parallel, np.inf, t_near)
        t_far = np.where(outside_parallel, -np.inf, t_far)
    return t_near, t_far


def _quadratic_interval(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                        level: float = 1.0):
    """Roots of A t^2 + B t + C = level, vectorised over leading batch."""
    disc = B * B - 4.0 * A * (C - level)
    valid = disc >= 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    inv = 0.5 / np.maximum(A, 1e-300)
    r1 = (-B - sq) * inv
    r2 = (-B + sq) * inv
    return valid, r1, r2


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class Component:
    tag = "component"

    def __init__(self, aabb_min: Sequence[float], aabb_max: Sequence[float]):
        self.aabb_min = np.asarray(aabb_min, dtype=float)
        self.aabb_max = np.asarray(aabb_max, dtype=float)

    def density(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def density_majorant(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def box_density_bound(self, lo: np.ndarray, hi: np.ndarray) -> float:
        raise NotImplementedError

    def optical_depth_batch(self, o: np.ndarray, d: np.ndarray,
                            t0, t1) -> np.ndarray:
        raise NotImplementedError

    def optical_depth(self, o, d, t0: float, t1: float) -> float:
        o = np.asarray(o, dtype=float).reshape(1, 3)
        d = np.asarray(d, dtype=float).reshape(1, 3)
        return float(self.optical_depth_batch(o, d, t0, t1)[0])

    # -- shared conservative AABB overlap bound ----------------------------
    def _aabb_overlap_density(self, lo: np.ndarray, hi: np.ndarray,
                              value: float) -> float:
        if np.all(hi >= self.aabb_min) and np.all(lo <= self.aabb_max):
            return float(value)
        return 0.0


class UniformBox(Component):
    """Constant density inside an axis-aligned box."""

    tag = "uniform_box"

    def __init__(self, aabb_min, aabb_max, value: float):
        super().__init__(aabb_min, aabb_max)
        self.value = float(value)

    def density(self, p):
        inside = np.all((p >= self.aabb_min) & (p <= self.aabb_max), axis=-1)
        return np.where(inside, self.value, 0.0)

    def density_majorant(self):
        return self.value

    def box_density_bound(self, lo, hi):
        return self._aabb_overlap_density(lo, hi, self.value)

    def optical_depth_batch(self, o, d, t0, t1):
        t_near, t_far = ray_aabb_interval_batch(
            o, d, self.aabb_min, self.aabb_max)
        t0a = np.broadcast_to(np.asarray(t0, float), o.shape[0])
        t1a = np.broadcast_to(np.asarray(t1, float), o.shape[0])
        ta = np.maximum(t0a, t_near)
        tb = np.minimum(t1a, t_far)
        length = np.maximum(tb - ta, 0.0)
        return self.value * length


class UniformEllipsoid(Component):
    """Constant density inside an ellipsoid (analytic chord length)."""

    tag = "uniform_ellipsoid"

    def __init__(self, center, radii, value: float):
        c = np.asarray(center, float)
        r = np.asarray(radii, float)
        super().__init__(c - r, c + r)
        self.center = c
        self.radii = r
        self.value = float(value)

    def _quadratic(self, p_or_o, d=None):
        if d is None:  # points
            rel = (p_or_o - self.center) / self.radii
            return np.sum(rel * rel, axis=-1)
        rel_o = (p_or_o - self.center) / self.radii
        rel_d = d / self.radii
        A = np.sum(rel_d * rel_d, axis=-1)
        B = 2.0 * np.sum(rel_o * rel_d, axis=-1)
        C = np.sum(rel_o * rel_o, axis=-1)
        return A, B, C

    def density(self, p):
        q = self._quadratic(p)
        return np.where(q <= 1.0, self.value, 0.0)

    def density_majorant(self):
        return self.value

    def box_density_bound(self, lo, hi):
        # Minimum ellipsoid quadratic form over the box is attained at the
        # centre clamped into the box; q_min <= 1 iff they overlap.
        closest = np.clip(self.center, lo, hi)
        qmin = float(np.sum(((closest - self.center) / self.radii) ** 2))
        return self.value if qmin <= 1.0 + 1e-12 else 0.0

    def optical_depth_batch(self, o, d, t0, t1):
        dd = d if d.ndim == 2 else np.broadcast_to(d, o.shape)
        A, B, C = self._quadratic(o, dd)
        valid, r1, r2 = _quadratic_interval(A, B, C, 1.0)
        t0 = np.broadcast_to(np.asarray(t0, float), o.shape[0])
        t1arr = np.broadcast_to(np.asarray(t1, float), o.shape[0])
        ta = np.maximum.reduce([t0, r1])
        tb = np.minimum.reduce([t1arr, r2])
        length = np.maximum(tb - ta, 0.0)
        return np.where(valid, self.value * length, 0.0)


class GaussianBlob(Component):
    """Truncated Gaussian density blob with analytic ray line integral."""

    tag = "gaussian_blob"

    def __init__(self, center, radii, amplitude: float, cutoff_sigma: float = 3.0):
        c = np.asarray(center, float)
        r = np.asarray(radii, float)
        self.cut = float(cutoff_sigma)
        super().__init__(c - self.cut * r, c + self.cut * r)
        self.center = c
        self.radii = r
        self.amplitude = float(amplitude)

    def _coeffs(self, o, d):
        rel_o = (o - self.center) / self.radii
        rel_d = d / self.radii
        A = np.sum(rel_d * rel_d, axis=-1)
        B = 2.0 * np.sum(rel_o * rel_d, axis=-1)
        C = np.sum(rel_o * rel_o, axis=-1)
        return A, B, C

    def density(self, p):
        rel = (p - self.center) / self.radii
        q = np.sum(rel * rel, axis=-1)
        return np.where(q <= self.cut * self.cut,
                        self.amplitude * np.exp(-q), 0.0)

    def density_majorant(self):
        return self.amplitude

    def box_density_bound(self, lo, hi):
        closest = np.clip(self.center, lo, hi)
        qmin = float(np.sum(((closest - self.center) / self.radii) ** 2))
        if qmin > self.cut * self.cut:
            return 0.0
        return self.amplitude * math.exp(-qmin)

    def optical_depth_batch(self, o, d, t0, t1):
        dd = d if d.ndim == 2 else np.broadcast_to(d, o.shape)
        A, B, C = self._coeffs(o, dd)
        level = self.cut * self.cut
        valid, r1, r2 = _quadratic_interval(A, B, C, level)

        t0a = np.broadcast_to(np.asarray(t0, float), o.shape[0])
        t1a = np.broadcast_to(np.asarray(t1, float), o.shape[0])
        ta = np.maximum.reduce([t0a, r1])
        tb = np.minimum.reduce([t1a, r2])
        nonzero = valid & (tb > ta) & (A > 1e-18)

        qmin = C - B * B / (4.0 * A)
        rootA = np.sqrt(A)
        pref = self.amplitude * np.exp(-qmin) * math.sqrt(math.pi) / (2.0 * rootA)
        sa = rootA * ta + B / (2.0 * rootA)
        sb = rootA * tb + B / (2.0 * rootA)
        tau = pref * (erf(sb) - erf(sa))

        # Pathological A ~ 0: density is effectively constant along the ray.
        flat = valid & (tb > ta) & (A <= 1e-18)
        tau_flat = self.amplitude * np.exp(-C) * (tb - ta)
        return np.where(nonzero, tau, 0.0) + np.where(flat, tau_flat, 0.0)


class LayerStack(Component):
    """Horizontal (z) layers with constant or linear-in-z density.

    layer specs: (z_bottom, z_top, a, b) with d(z) = a + b*z.
    """

    tag = "layers"

    def __init__(self, layers: Sequence[Tuple[float, float, float, float]]):
        self.layers = list(layers)
        z0 = min(l[0] for l in self.layers)
        z1 = max(l[1] for l in self.layers)
        # x/y unbounded in principle; give a large finite box supplied later
        super().__init__([-1e4, -1e4, z0], [1e4, 1e4, z1])
        self.z_edges = np.array([self.layers[0][0]] + [l[1] for l in self.layers],
                                dtype=float)
        self.a = np.array([l[2] for l in self.layers], dtype=float)
        self.b = np.array([l[3] for l in self.layers], dtype=float)
        self._layer_majors = [
            max(self._d_at(l, l[0]), self._d_at(l, l[1]), 0.0)
            for l in self.layers
        ]

    @staticmethod
    def _d_at(layer, z):
        return layer[2] + layer[3] * z

    def with_xy_bounds(self, xmin, ymin, xmax, ymax) -> "LayerStack":
        self.aabb_min = np.array([xmin, ymin, self.aabb_min[2]])
        self.aabb_max = np.array([xmax, ymax, self.aabb_max[2]])
        return self

    def density(self, p):
        z = p[..., 2]
        idx = np.searchsorted(self.z_edges, z, side="right") - 1
        idx_c = np.clip(idx, 0, len(self.layers) - 1)
        val = self.a[idx_c] + self.b[idx_c] * z
        valid_z = (idx >= 0) & (idx < len(self.layers))
        inside_xy = np.all((p[..., :2] >= self.aabb_min[:2])
                           & (p[..., :2] <= self.aabb_max[:2]), axis=-1)
        valid = valid_z & inside_xy
        return np.where(valid, np.maximum(val, 0.0), 0.0)

    def density_majorant(self):
        return float(max(self._layer_majors))

    def layer_index_at_z(self, z: float) -> int:
        idx = int(np.searchsorted(self.z_edges, z, side="right") - 1)
        if idx < 0:
            return 0
        return min(idx, len(self.layers) - 1)

    def layer_majorant(self, idx: int) -> float:
        return float(self._layer_majors[idx])

    def box_density_bound(self, lo, hi):
        zlo, zhi = float(lo[2]), float(hi[2])
        best = 0.0
        for k, layer in enumerate(self.layers):
            a, b = layer[0], layer[1]
            za, zb = max(zlo, a), min(zhi, b)
            if zb < za:
                continue
            best = max(best,
                       max(self._d_at(layer, za), self._d_at(layer, zb)))
        return max(best, 0.0)

    def optical_depth_batch(self, o, d, t0, t1):
        n = o.shape[0]
        # honour the finite xy (and z) AABB of the layer stack
        t_near, t_far = ray_aabb_interval_batch(
            o, d, self.aabb_min, self.aabb_max)
        t0a = np.maximum(np.broadcast_to(np.asarray(t0, float), (n,)), t_near)
        t1a = np.minimum(np.broadcast_to(np.asarray(t1, float), (n,)), t_far)

        dz = d[:, 2] if d.ndim == 2 else np.full(n, float(d[2]))
        oz = o[:, 2]
        out = np.zeros(n)
        hit = t_far >= t_near - 1e-12
        for layer in self.layers:
            ze0, ze1, aa, bb = layer
            with np.errstate(divide="ignore", invalid="ignore"):
                t_z0 = (ze0 - oz) / dz
                t_z1 = (ze1 - oz) / dz
            enter = np.minimum(t_z0, t_z1)
            leave = np.maximum(t_z0, t_z1)
            ta = np.maximum(t0a, enter)
            tb = np.minimum(t1a, leave)

            parallel = np.abs(dz) < 1e-15
            inside_z = (oz >= ze0) & (oz <= ze1)
            # parallel ray: either z stays inside this layer for the whole
            # clipped interval, or the layer is never hit (zero length).
            ta = np.where(parallel, t0a, ta)
            tb = np.where(parallel, np.where(inside_z, t1a, ta), tb)

            alpha = aa + bb * oz
            beta = bb * dz
            with np.errstate(divide="ignore", invalid="ignore"):
                seg = tb - ta
                contrib = alpha * seg + 0.5 * beta * (tb * tb - ta * ta)
            valid_seg = hit & (tb > ta)
            out += np.where(valid_seg, contrib, 0.0)
        return np.maximum(out, 0.0)


# ---------------------------------------------------------------------------
# Composite field
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    t0: float
    t1: float
    density_majorant: float
    tag: str = ""
    index: int = -1
    cell_lo: Optional[np.ndarray] = None
    cell_hi: Optional[np.ndarray] = None

    @property
    def length(self) -> float:
        return self.t1 - self.t0


class CompositeField:
    """Sum of components; majorant is the sum of component majorants."""

    def __init__(self, components: Sequence[Component],
                 aabb_min: Optional[Sequence[float]] = None,
                 aabb_max: Optional[Sequence[float]] = None,
                 name: str = "composite"):
        self.components = list(components)
        self.name = name
        if aabb_min is not None and aabb_max is not None:
            self.aabb_min = np.asarray(aabb_min, float)
            self.aabb_max = np.asarray(aabb_max, float)
        else:
            self.aabb_min = np.min([c.aabb_min for c in components], axis=0)
            self.aabb_max = np.max([c.aabb_max for c in components], axis=0)

    def density(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        total = np.zeros(p.shape[:-1] if p.ndim >= 1 else ())
        for c in self.components:
            total = total + c.density(p)
        return total

    def density_majorant(self) -> float:
        return float(sum(c.density_majorant() for c in self.components))

    def box_density_bound(self, lo, hi) -> float:
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        return float(sum(c.box_density_bound(lo, hi) for c in self.components))

    def optical_depth_batch(self, o, d, t0=0.0, t1=None):
        o = np.asarray(o, float)
        if o.ndim == 1:
            o = o.reshape(1, 3)
        d = np.asarray(d, float)
        if t1 is None:
            raise ValueError("t1 required")
        total = np.zeros(o.shape[0])
        for c in self.components:
            total += c.optical_depth_batch(o, d, t0, t1)
        return total

    def optical_depth(self, o, d, t0: float, t1: float) -> float:
        return float(self.optical_depth_batch(o, d, t0, t1)[0])

    def certified_segment_majorant(self, seg: "Segment") -> Optional[float]:
        """Trusted local bound for a segment, if one can be certified.

        Grid segments carry their AABB cell, so the analytic cell bound is
        available.  Returns None for declarations that can only be checked
        at candidate time.
        """
        if seg.cell_lo is not None and seg.cell_hi is not None:
            return self.box_density_bound(seg.cell_lo, seg.cell_hi)
        return None

    # -- partitions ---------------------------------------------------------
    def build_segments(self, o, d, L: float,
                       partition: str = "global",
                       grid_res: int = 14,
                       flawed: Optional[bool] = None) -> List[Segment]:
        """Ordered segments covering ray interval [0, L].

        Vacuum ranges get density majorant 0 and are advanced directly.
        """
        o = np.asarray(o, float)
        d = np.asarray(d, float)
        if flawed is None:
            flawed = partition == "flawed_grid"
        if partition in ("flawed_grid", "certified_grid"):
            partition_effective = "grid"
        else:
            partition_effective = partition

        hit = ray_aabb_interval(o, d, self.aabb_min, self.aabb_max)
        if hit is None or L <= 0.0:
            return [Segment(0.0, float(L), 0.0, "vacuum", 0)] if L > 0 else []
        entry = max(hit[0], 0.0)
        exit_ = min(hit[1], float(L))

        segments: List[Segment] = []
        idx = 0

        def add(a, b, m, tag, cell_lo=None, cell_hi=None):
            nonlocal idx
            if b - a <= 1e-15:
                return
            segments.append(Segment(float(a), float(b), float(max(m, 0.0)),
                                    tag, idx, cell_lo, cell_hi))
            idx += 1

        if exit_ <= entry:
            add(0.0, L, 0.0, "vacuum")
            return segments

        if entry > 0.0:
            add(0.0, entry, 0.0, "vacuum")

        if partition_effective == "global":
            add(entry, exit_, self.density_majorant(), "global_majorant",
                self.aabb_min.copy(), self.aabb_max.copy())

        elif partition_effective == "layers":
            self._build_layer_segments(o, d, entry, exit_, add)

        elif partition_effective == "grid":
            self._build_grid_segments(o, d, entry, exit_, grid_res,
                                      flawed, add)
        else:
            raise ValueError(f"unknown partition {partition}")

        if exit_ < L:
            add(exit_, L, 0.0, "vacuum")
        return segments

    def _build_layer_segments(self, o, d, entry, exit_, add):
        cuts = [entry, exit_]
        layer_stack = next((c for c in self.components
                            if isinstance(c, LayerStack)), None)
        if layer_stack is not None and abs(d[2]) > 1e-15:
            z_edges = layer_stack.z_edges
            for z in z_edges[1:-1]:
                t = (float(z) - float(o[2])) / float(d[2])
                if entry + 1e-12 < t < exit_ - 1e-12:
                    cuts.append(t)
        cuts = sorted(set(cuts))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a <= 1e-15:
                continue
            za = float(o[2] + d[2] * a)
            zb = float(o[2] + d[2] * b)
            zlo, zhi = min(za, zb), max(za, zb)
            box_lo = np.array([self.aabb_min[0], self.aabb_min[1], zlo])
            box_hi = np.array([self.aabb_max[0], self.aabb_max[1], zhi])
            major = self.box_density_bound(box_lo, box_hi)
            zmid = 0.5 * (zlo + zhi)
            if layer_stack is not None:
                k = layer_stack.layer_index_at_z(zmid)
                tag = f"layer_{k}"
            else:
                tag = "layer_slab"
            add(a, b, major, tag, box_lo, box_hi)

    def _build_grid_segments(self, o, d, entry, exit_, res, flawed, add):
        lo = self.aabb_min
        hi = self.aabb_max
        h = (hi - lo) / res
        cuts = [entry, exit_]
        for axis in range(3):
            if abs(d[axis]) < 1e-15:
                continue
            for k in range(1, res):
                plane = lo[axis] + k * h[axis]
                t = (plane - o[axis]) / d[axis]
                if entry + 1e-12 < t < exit_ - 1e-12:
                    cuts.append(float(t))
        cuts.sort()
        # remove exact duplicates only
        unique = [cuts[0]]
        for t in cuts[1:]:
            if abs(t - unique[-1]) > 1e-12:
                unique.append(t)
        if unique[-1] != cuts[-1]:
            unique.append(cuts[-1])

        for a, b in zip(unique[:-1], unique[1:]):
            if b - a <= 1e-15:
                continue
            tm = 0.5 * (a + b)
            mid = o + d * tm
            cell_lo = np.floor((mid - lo) / h).astype(int)
            cell_lo = np.clip(cell_lo, 0, res - 1)
            c0 = lo + cell_lo * h
            c1 = c0 + h
            if flawed:
                major = self._corner_majorant(c0, c1)
                tag = "flawed_cell"
            else:
                major = self.box_density_bound(c0, c1)
                tag = "certified_cell"
            add(a, b, major, tag, c0.copy(), c1.copy())

    def _corner_majorant(self, c0: np.ndarray, c1: np.ndarray) -> float:
        """Naive declared bound: sample 8 corners only (can underestimate!)."""
        best = 0.0
        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    p = np.array([c0[0] if ix == 0 else c1[0],
                                  c0[1] if iy == 0 else c1[1],
                                  c0[2] if iz == 0 else c1[2]])
                    best = max(best, float(self.density(p)))
        return best

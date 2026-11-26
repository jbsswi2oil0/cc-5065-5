"""Density edits, brick majorants and their maintenance.

The kernel integrates a density field against *declared* regional majorants
``M = kappa * d_bar`` (units: 1/length, identical to ``sigma_t = kappa * d``).
A declaration is only trustworthy while it upper-bounds the *evaluated* density
of the region it covers.  Editing the density invalidates some declarations;
this module records those edits reproducibly and keeps/recomputes the
declarations.

Concepts
--------
``DensityEdit`` / ``DensityEditLog``
    Append-only, hashed, replayable list of additive box edits on top of a
    declared base density.  The log is the only source of "what changed", so
    the evaluated density, its certified per-brick supremum and the edit
    history all derive from it -- never from a hand-edited number.

``BrickMajorantMaintenance`` -- two maintenance strategies, compared on a
    sharp spike (see :func:`spiky_fixture`):

    ``brick``
        Every density brick owns an independent majorant, declared from its
        eight corners.  On an edit only the *primary* brick the edit is
        assigned to is refreshed.  Cheap, fully local, but a spike that sits
        strictly between corners, or is carried into a neighbouring brick by
        the edit's footprint, is invisible -- the declaration stays too low.

    ``recompute``
        The edit's bounding box defines the affected brick set; every affected
        brick is recomputed with the analytic cell supremum
        (``box_density_bound``).  A correct bounding box therefore restores a
        certified bound even across a brick boundary.

``maintenance_scan``
    The operational check a designer runs after editing: read the edit log,
    find the first brick whose declared majorant is below the certified
    supremum of the evaluated field, and report it.  It is a whole-brick
    supremum check; individual rays that happen not to cross the spike can
    never prove a declaration safe, because a candidate-time ``sigma_t <= M``
    check only samples points that happen to be visited.

``DeclaredBrickField``
    Feeds the *declared* brick table to the real scalar kernel while exposing
    the true evaluated density and the certified supremum, so the same
    ``estimate_transmittance`` path that ships in the package can demonstrate
    the change in transmittance before and after repair.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import (Component, CompositeField, UniformBox)


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------

@dataclass
class DensityEdit:
    """One additive box edit on the base density.

    The evaluated density is ``base(p) + sum(value for edits with p in box)``.
    A negative value removes material.  ``provenance`` records who/what made
    the edit so a stale declaration can be traced to its cause.
    """

    lo: Sequence[float]
    hi: Sequence[float]
    value: float
    provenance: str = "user_edit"
    version: int = 0

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, float)
        self.hi = np.asarray(self.hi, float)

    def component(self) -> UniformBox:
        return UniformBox(self.lo, self.hi, self.value)

    def primary_brick(self, res: int, aabb_min, aabb_max) -> int:
        """Brick that owns the edit's centre (the local-maintenance target)."""
        centre = 0.5 * (self.lo + self.hi)
        h = (np.asarray(aabb_max, float) - np.asarray(aabb_min, float)) / res
        idx = np.floor((centre - aabb_min) / h).astype(int)
        idx = np.clip(idx, 0, res - 1)
        return int(np.ravel_multi_index(tuple(idx), (res, res, res)))

    def affected_bricks(self, res: int, aabb_min, aabb_max) -> List[int]:
        """Bricks whose closed cell overlaps the edit box (conservative).

        One cell of slack is added on each side of the computed range so a box
        boundary that sits exactly on a cell plane still includes both
        neighbours; the result is a safe over-approximation of the footprint.
        """
        amin = np.asarray(aabb_min, float)
        amax = np.asarray(aabb_max, float)
        h = (amax - amin) / res
        ranges = []
        for axis in range(3):
            k0 = int(math.floor((self.lo[axis] - amin[axis]) / h[axis]))
            k1 = int(math.ceil((self.hi[axis] - amin[axis]) / h[axis])) - 1
            k0 = max(0, min(res - 1, k0))
            k1 = max(0, min(res - 1, k1))
            if k1 < k0:
                k1 = k0
            # closed-cell overlap: include the neighbour on a shared plane
            if k0 > 0 and abs(self.lo[axis]
                              - (amin[axis] + k0 * h[axis])) < 1e-12:
                k0 -= 1
            if k1 < res - 1 and abs(self.hi[axis]
                                    - (amin[axis] + (k1 + 1) * h[axis])) < 1e-12:
                k1 += 1
            ranges.append((k0, k1))
        out: List[int] = []
        for i in range(ranges[0][0], ranges[0][1] + 1):
            for j in range(ranges[1][0], ranges[1][1] + 1):
                for k in range(ranges[2][0], ranges[2][1] + 1):
                    out.append(int(np.ravel_multi_index((i, j, k),
                                                        (res, res, res))))
        return out


@dataclass
class DensityEditLog:
    """Reproducible density = base + logged edits, with a stable signature."""

    base_density: Any                     # Component or CompositeField
    aabb_min: Sequence[float]
    aabb_max: Sequence[float]
    res: int
    scene: str
    seed: int
    edits: List[DensityEdit] = field(default_factory=list)
    signature: str = ""

    def __post_init__(self) -> None:
        self.aabb_min = np.asarray(self.aabb_min, float)
        self.aabb_max = np.asarray(self.aabb_max, float)
        self.res = int(self.res)
        for n, e in enumerate(self.edits, start=1):
            e.version = n
        self.signature = self._hash()

    # -- construction -----------------------------------------------------
    def append(self, lo, hi, value: float, provenance: str = "user_edit"
               ) -> DensityEdit:
        edit = DensityEdit(lo, hi, float(value), provenance)
        edit.version = len(self.edits) + 1
        self.edits.append(edit)
        self.signature = self._hash()
        return edit

    def _hash(self) -> str:
        base_tag = getattr(self.base_density, "tag",
                           type(self.base_density).__name__)
        payload = {
            "base_tag": base_tag,
            "base_majorant": float(self.base_density.density_majorant()),
            "base_aabb_min": [float(x) for x in self.base_density.aabb_min],
            "base_aabb_max": [float(x) for x in self.base_density.aabb_max],
            "scene": self.scene,
            "seed": int(self.seed),
            "res": int(self.res),
            "aabb_min": [float(x) for x in self.aabb_min],
            "aabb_max": [float(x) for x in self.aabb_max],
            "edits": [{"lo": [float(x) for x in e.lo],
                       "hi": [float(x) for x in e.hi],
                       "value": float(e.value),
                       "provenance": e.provenance,
                       "version": int(e.version)} for e in self.edits],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def manifest(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "seed": int(self.seed),
            "res": int(self.res),
            "aabb_min": [float(x) for x in self.aabb_min],
            "aabb_max": [float(x) for x in self.aabb_max],
            "base_tag": getattr(self.base_density, "tag", "base"),
            "signature": self.signature,
            "n_edits": len(self.edits),
            "edits": [{"version": e.version,
                       "lo": [float(x) for x in e.lo],
                       "hi": [float(x) for x in e.hi],
                       "value": float(e.value),
                       "provenance": e.provenance} for e in self.edits],
        }

    # -- evaluated field --------------------------------------------------
    def base_field(self) -> CompositeField:
        return CompositeField([self.base_density],
                              self.aabb_min, self.aabb_max,
                              name=f"{self.scene}_base")

    def starts_from(self, other: "DensityEditLog") -> bool:
        """True when ``other`` is a prefix of this log (same base and prefix).

        Used to reject a stale sample whose edits no longer match the current
        scene: only a log that begins with the same edits can be the current
        material, and even then a superseded log must be replayed.
        """
        if other._hash_base() != self._hash_base():
            return False
        if len(other.edits) > len(self.edits):
            return False
        for a, b in zip(other.edits, self.edits):
            if (not np.allclose(a.lo, b.lo) or not np.allclose(a.hi, b.hi)
                    or abs(a.value - b.value) > 1e-12):
                return False
        return True

    def _hash_base(self) -> str:
        payload = {
            "base_tag": getattr(self.base_density, "tag",
                                type(self.base_density).__name__),
            "base_aabb_min": [float(x) for x in self.base_density.aabb_min],
            "base_aabb_max": [float(x) for x in self.base_density.aabb_max],
            "scene": self.scene, "res": int(self.res),
            "aabb_min": [float(x) for x in self.aabb_min],
            "aabb_max": [float(x) for x in self.aabb_max],
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def evaluated_field(self, upto: int = -1) -> CompositeField:
        comps = [self.base_density]
        for e in self.edits:
            if upto >= 0 and e.version > upto:
                break
            comps.append(e.component())
        return CompositeField(comps, self.aabb_min, self.aabb_max,
                              name=f"{self.scene}_evaluated")

    # -- geometry ---------------------------------------------------------
    def cell_bounds(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        res = self.res
        i, j, k = np.unravel_index(int(index), (res, res, res))
        h = (self.aabb_max - self.aabb_min) / res
        c0 = self.aabb_min + np.array([i, j, k], float) * h
        return c0, c0 + h


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def corner_extreme(field_obj: Any, c0: np.ndarray, c1: np.ndarray) -> float:
    """Max evaluated density over the eight corners of a cell."""
    best = 0.0
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                p = np.array([c0[0] if ix == 0 else c1[0],
                              c0[1] if iy == 0 else c1[1],
                              c0[2] if iz == 0 else c1[2]], float)
                best = max(best, float(field_obj.density(p)))
    return best


def certified_sup(field_obj: Any, c0: np.ndarray, c1: np.ndarray) -> float:
    """Analytic cell supremum (``box_density_bound`` of the evaluated field)."""
    return max(float(field_obj.box_density_bound(c0, c1)), 0.0)


# ---------------------------------------------------------------------------
# Maintenance strategies
# ---------------------------------------------------------------------------

class BrickMajorantMaintenance:
    """Per-brick majorant table maintained by one of two strategies."""

    def __init__(self, log: DensityEditLog, strategy: str = "recompute"):
        if strategy not in ("brick", "recompute"):
            raise ValueError(f"未知维护策略 {strategy!r}")
        self.log = log
        self.strategy = strategy
        self.res = log.res
        self.base_field = log.base_field()
        self.declared = self._full_build(self.base_field)
        self.versions = np.zeros((self.res,) * 3, dtype=int)
        self._applied = 0
        self._last_changed: List[int] = []

    # -- bound primitives -------------------------------------------------
    def _brick_bound(self, field_obj: Any, index: int) -> float:
        c0, c1 = self.log.cell_bounds(index)
        if self.strategy == "brick":
            return corner_extreme(field_obj, c0, c1)
        return certified_sup(field_obj, c0, c1)

    def _full_build(self, field_obj: Any) -> np.ndarray:
        table = np.zeros((self.res,) * 3)
        n = self.res ** 3
        for b in range(n):
            table.reshape(-1)[b] = self._brick_bound(field_obj, b)
        return table

    # -- edits ------------------------------------------------------------
    def apply_edits(self) -> List[int]:
        """Maintain the table after new edits; return bricks refreshed."""
        changed: List[int] = []
        for edit in self.log.edits[self._applied:]:
            field_v = self.log.evaluated_field(upto=edit.version)
            if self.strategy == "brick":
                targets = [edit.primary_brick(self.res, self.log.aabb_min,
                                              self.log.aabb_max)]
            else:
                targets = edit.affected_bricks(self.res, self.log.aabb_min,
                                               self.log.aabb_max)
            for b in targets:
                self.declared.reshape(-1)[b] = self._brick_bound(field_v, b)
                self.versions.reshape(-1)[b] = edit.version
                changed.append(b)
            self._applied = edit.version
        self._last_changed = changed
        return changed

    def stale_bricks(self) -> List[int]:
        """Bricks touched by an edit but not refreshed since that edit."""
        stale = set()
        for edit in self.log.edits:
            for b in edit.affected_bricks(self.res, self.log.aabb_min,
                                          self.log.aabb_max):
                if self.versions.reshape(-1)[b] < edit.version:
                    stale.add(int(b))
        return sorted(stale)

    def handle_stale_materials(self) -> List[int]:
        """Recompute every stale brick with its certified supremum.

        This is the safe disposition of stale material: because the edit log
        fully determines the evaluated density, the certified cell supremum is
        available without a hand edit, and the table is brought up to date.
        """
        changed: List[int] = []
        for edit in self.log.edits:
            field_v = self.log.evaluated_field(upto=edit.version)
            for b in edit.affected_bricks(self.res, self.log.aabb_min,
                                          self.log.aabb_max):
                if self.versions.reshape(-1)[b] < edit.version:
                    self.declared.reshape(-1)[b] = certified_sup(field_v, b)
                    self.versions.reshape(-1)[b] = edit.version
                    changed.append(int(b))
        self._applied = len(self.log.edits)
        self._last_changed = changed
        return changed

    def table(self) -> np.ndarray:
        return self.declared.copy()

    def declared_bound(self, index: int) -> float:
        return float(self.declared.reshape(-1)[int(index)])

    def refresh_brick(self, index: int) -> Dict[str, Any]:
        """Recompute one brick from the currently applied evaluated field."""
        old = self.declared_bound(index)
        field_v = self.log.evaluated_field()
        new = certified_sup(field_v, *self.log.cell_bounds(index))
        self.declared.reshape(-1)[index] = new
        return {"index": int(index), "old_density_bound": old,
                "new_density_bound": new, "provenance": "certified_sup"}


# ---------------------------------------------------------------------------
# Maintenance scan
# ---------------------------------------------------------------------------

def maintenance_scan(log: DensityEditLog,
                     maintenance: BrickMajorantMaintenance,
                     tol: float = 1e-9) -> Dict[str, Any]:
    """Find the first brick whose declared majorant is below the certified
    supremum of the evaluated field.

    The comparison is over the whole brick (analytic cell supremum vs declared
    bound), so it is a proof of validity, not a sample of visited points.
    """
    field_obj = log.evaluated_field()
    violations: List[Dict[str, Any]] = []
    first: Optional[int] = None
    for b in range(log.res ** 3):
        c0, c1 = log.cell_bounds(b)
        certified = certified_sup(field_obj, c0, c1)
        centre = 0.5 * (c0 + c1)
        centre_val = float(field_obj.density(centre))
        corners = corner_extreme(field_obj, c0, c1)
        declared = maintenance.declared_bound(b)
        if declared + tol * max(1.0, certified) < certified:
            violations.append({
                "index": b,
                "declared": declared,
                "certified_sup": certified,
                "centre_value": centre_val,
                "corner_extreme": corners,
                "cell_lo": [float(x) for x in c0],
                "cell_hi": [float(x) for x in c1],
            })
            if first is None:
                first = b
    return {
        "strategy": maintenance.strategy,
        "first_invalidated_brick": first,
        "n_invalidated": len(violations),
        "violations": violations,
        "check_kind": "whole_brick_supremum",
        "signature": log.signature,
    }


# ---------------------------------------------------------------------------
# Declared-majorant field for the real kernel
# ---------------------------------------------------------------------------

class DeclaredBrickField(CompositeField):
    """The evaluated (true) field paired with a *declared* brick bound table.

    ``density`` and analytic ``optical_depth`` come from the true evaluated
    field; ``build_segments`` stamps each grid segment with the declared brick
    majorant; ``certified_segment_majorant`` (inherited) returns the true cell
    supremum, so ``estimate_transmittance`` sees exactly the declared-vs-true
    gap that maintenance is supposed to close.
    """

    def __init__(self, true_field: CompositeField, table: np.ndarray,
                 res: int, aabb_min, aabb_max) -> None:
        super().__init__(true_field.components, aabb_min, aabb_max,
                         name="declared_bricks")
        self.table = np.asarray(table, float).reshape(res, res, res)
        self.res = int(res)
        self.h = (self.aabb_max - self.aabb_min) / self.res

    def brick_index(self, point: np.ndarray) -> Tuple[int, int, int]:
        idx = np.clip(np.floor((np.asarray(point, float)
                                - self.aabb_min) / self.h).astype(int),
                      0, self.res - 1)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def build_segments(self, o, d, L: float, partition: str = "global",
                       grid_res: Optional[int] = None,
                       flawed: Optional[bool] = None) -> List[Any]:
        # Ignore the requested partition: this field is used to exercise the
        # declared brick table, which is aligned to grid cells at ``res``.
        segs = super().build_segments(o, d, L,
                                      partition="certified_grid",
                                      grid_res=self.res)
        for s in segs:
            if s.cell_lo is None or s.cell_hi is None:
                continue
            mid = 0.5 * (np.asarray(s.cell_lo, float)
                         + np.asarray(s.cell_hi, float))
            i, j, k = self.brick_index(mid)
            s.density_majorant = float(self.table[i, j, k])
            s.tag = f"{s.tag}_declared"
        return segs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def spiky_fixture(res: int = 8) -> Dict[str, Any]:
    """Reproducible medium with a sharp spike straddling two bricks.

    Base density 0.1 everywhere; a 2.0 spike sits strictly *between* the eight
    corners of both bricks it touches, so an eight-corner declaration cannot
    see it while the analytic supremum can.  The edit's centre lies in brick
    ``(4,4,4)`` (the primary brick); its footprint reaches into ``(5,4,4)``.
    """
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    base_value = 0.1
    spike_value = 2.0
    spike_lo = np.array([0.18, 0.10, 0.10])
    spike_hi = np.array([0.26, 0.16, 0.16])
    base = CompositeField([UniformBox(lo, hi, base_value)], lo, hi,
                          name="spiky_base")
    log = DensityEditLog(base_density=base, aabb_min=lo, aabb_max=hi,
                         res=res, scene="spiky_medium", seed=0)
    edit = log.append(spike_lo, spike_hi, spike_value,
                      provenance="designer_adds_spike")
    true_field = log.evaluated_field()
    primary = edit.primary_brick(res, lo, hi)
    affected = edit.affected_bricks(res, lo, hi)
    return {
        "log": log,
        "true_field": true_field,
        "aabb_min": lo,
        "aabb_max": hi,
        "res": res,
        "spike_lo": spike_lo,
        "spike_hi": spike_hi,
        "spike_value": spike_value,
        "base_value": base_value,
        "primary_brick": primary,
        "affected_bricks": affected,
        "spike_sup": base_value + spike_value,
    }


def affected_rays(fixture: Dict[str, Any], n_rays: int = 4,
                  seed: int = 0) -> List[Dict[str, Any]]:
    """+y orthographic rays whose x/z footprints cross the spike.

    Also returns one control ray in a spike-free column, so tests can show
    that "no ray violated" is not the same as "no declaration is invalid".
    """
    rng = np.random.default_rng(seed)
    lo = fixture["spike_lo"]
    hi = fixture["spike_hi"]
    domain_y0 = float(fixture["aabb_min"][1])
    domain_y1 = float(fixture["aabb_max"][1])
    start_y = domain_y0 - 0.5
    L = (domain_y1 - domain_y0) + 1.0
    rays: List[Dict[str, Any]] = []
    for _ in range(n_rays):
        x = float(rng.uniform(lo[0] + 1e-4, hi[0] - 1e-4))
        z = float(rng.uniform(lo[2] + 1e-4, hi[2] - 1e-4))
        rays.append({"origin": [x, start_y, z], "direction": [0.0, 1.0, 0.0],
                     "L": L, "kind": "crosses_spike"})
    # control: a column that only sees the background
    rays.append({"origin": [float(lo[0]) - 0.4, start_y, float(lo[2]) + 0.02],
                 "direction": [0.0, 1.0, 0.0], "L": L,
                 "kind": "background_only"})
    return rays


# ---------------------------------------------------------------------------
# Transmittance evidence before / after repair
# ---------------------------------------------------------------------------

def analytic_transmittance(fixture: Dict[str, Any], ray: Dict[str, Any],
                           kappa: float) -> float:
    tau = fixture["true_field"].optical_depth(
        np.asarray(ray["origin"], float),
        np.asarray(ray["direction"], float), 0.0, float(ray["L"]))
    return float(math.exp(-min(kappa * tau, 700.0)))


def transmittance_evidence(fixture: Dict[str, Any],
                           maintenance: BrickMajorantMaintenance,
                           kappa: float = 1.0, n_samples: int = 2000,
                           seed: int = 1) -> Dict[str, Any]:
    """Per-ray transmittance before vs after closing the declaration gap.

    ``before`` uses the maintenance table as-is; a strict run must reject any
    ray whose spike declaration is too low, and a clamp demo shows the numeric
    bias if someone were to clamp instead.  ``after`` rebuilds the table with
    the certified supremum for the affected bricks and re-runs the same kernel
    path, which must become unbiased against the analytic transmittance.
    """
    from .tracking import estimate_transmittance
    from .errors import MajorantViolation

    rays = affected_rays(fixture, seed=seed)
    table_before = maintenance.table()
    table_after = table_before.copy()
    for b in fixture["affected_bricks"]:
        table_after.reshape(-1)[b] = certified_sup(
            fixture["true_field"], *fixture["log"].cell_bounds(b))

    before = DeclaredBrickField(fixture["true_field"], table_before,
                                fixture["res"], fixture["aabb_min"],
                                fixture["aabb_max"])
    after = DeclaredBrickField(fixture["true_field"], table_after,
                               fixture["res"], fixture["aabb_min"],
                               fixture["aabb_max"])

    records: List[Dict[str, Any]] = []
    for j, ray in enumerate(rays):
        o = np.asarray(ray["origin"], float)
        d = np.asarray(ray["direction"], float)
        L = float(ray["L"])
        analytic = analytic_transmittance(fixture, ray, kappa)

        strict_raised = False
        violation = None
        try:
            estimate_transmittance(before, o, d, L, kappa=kappa,
                                   estimator="ratio", policy="strict",
                                   seed=2000 + j)
        except MajorantViolation as exc:
            strict_raised = True
            violation = {"ray_t": exc.ray_t,
                         "declared_majorant": exc.declared_majorant,
                         "actual_sigma_t": exc.actual_sigma_t}

        # numeric bias from clamping (demonstration only; never a proof)
        clamp_samples = np.array([
            estimate_transmittance(before, o, d, L, kappa=kappa,
                                   estimator="binary", policy="clamp_demo",
                                   seed=3000 + j * 97 + i).estimate
            for i in range(max(64, n_samples // 8))])
        clamp_mean = float(clamp_samples.mean())

        # repaired: strict must not raise, and the mean must match analytic
        after_samples = np.array([
            estimate_transmittance(after, o, d, L, kappa=kappa,
                                   estimator="binary", policy="strict",
                                   seed=5000 + j * 131 + i).estimate
            for i in range(n_samples)])
        after_mean = float(after_samples.mean())
        var = analytic * (1.0 - analytic)
        sem = math.sqrt(max(var, 0.0) / n_samples)
        z = 0.0 if sem == 0 else abs(after_mean - analytic) / sem

        records.append({
            "kind": ray["kind"],
            "origin": ray["origin"],
            "analytic_transmittance": analytic,
            "before_strict_raised": strict_raised,
            "before_violation": violation,
            "before_clamp_mean": clamp_mean,
            "before_clamp_abs_error": abs(clamp_mean - analytic),
            "after_mean": after_mean,
            "after_z": z,
            "after_strict_raised": False,
        })
    return {
        "kappa": kappa,
        "n_samples_after": n_samples,
        "rays": records,
        "n_before_raised": sum(r["before_strict_raised"] for r in records),
        "n_after_raised": 0,
        "worst_after_z": max(abs(r["after_z"]) for r in records),
        "note": "射线证据仅展示后果；有效性由上界逐砖 supremum 校验保证。",
    }

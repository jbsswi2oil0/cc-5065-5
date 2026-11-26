"""Reproducible procedural random medium that a designer can edit.

A random medium is fully determined by its seed: the component set is drawn
from a seeded generator, so any caller that holds the manifest can rebuild an
identical density (``manifest["signature"]`` is the check).  Density edits are
recorded in a :class:`~stochastic_medium.maintenance.DensityEditLog`, so the
evaluated medium is always ``base(seed) + logged edits`` and never depends on
hand-edited values.

Stale material handling
-----------------------
A cached medium is stale when the manifest it was built from no longer matches
the requested seed/signature (for example the page kept an old sample after a
density edit).  :class:`RandomMediumCache` detects the signature mismatch, drops
the stale field and rebuilds from the seed, and reports the rebuild so callers
can restart accumulation rather than blend old and new samples.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .fields import (CompositeField, GaussianBlob, LayerStack, UniformBox,
                     UniformEllipsoid)
from .maintenance import (BrickMajorantMaintenance, DensityEditLog,
                          certified_sup)
from .scenes import FRAME

RANDOM_AABB_MIN = np.array([-5.0, -2.5, -1.5])
RANDOM_AABB_MAX = np.array([5.0, 2.5, 5.5])


def _component_descriptors(seed: int, n_blobs: int, n_boxes: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    desc: List[Dict[str, Any]] = []
    for _ in range(n_blobs):
        x = float(rng.uniform(-4.2, 4.2))
        y = float(rng.uniform(-1.7, 1.7))
        z = float(rng.uniform(-0.7, 4.9))
        rx = float(rng.uniform(0.35, 1.1))
        ry = float(rng.uniform(0.3, 0.9))
        rz = float(rng.uniform(0.35, 1.0))
        amp = float(rng.uniform(0.4, 1.2))
        cut = float(rng.uniform(2.2, 3.0))
        desc.append({"type": "gaussian_blob", "center": [x, y, z],
                     "radii": [rx, ry, rz], "amplitude": amp,
                     "cutoff_sigma": cut})
    for _ in range(n_boxes):
        cx = float(rng.uniform(-4.0, 4.0))
        cy = float(rng.uniform(-1.6, 1.6))
        cz = float(rng.uniform(-0.6, 4.6))
        sx = float(rng.uniform(0.18, 0.5))
        sy = float(rng.uniform(0.18, 0.5))
        sz = float(rng.uniform(0.18, 0.5))
        val = float(rng.uniform(0.6, 1.5))
        desc.append({"type": "uniform_box",
                     "lo": [cx - sx, cy - sy, cz - sz],
                     "hi": [cx + sx, cy + sy, cz + sz], "value": val})
    # a low mist layer keeps the medium non-empty at the bottom of the frame
    desc.append({"type": "mist_layer"})
    return desc


def components_from_descriptors(desc: Sequence[Dict[str, Any]]) -> List[Any]:
    comps: List[Any] = []
    for d in desc:
        t = d["type"]
        if t == "gaussian_blob":
            comps.append(GaussianBlob(d["center"], d["radii"], d["amplitude"],
                                      d.get("cutoff_sigma", 3.0)))
        elif t == "uniform_box":
            comps.append(UniformBox(d["lo"], d["hi"], d["value"]))
        elif t == "uniform_ellipsoid":
            comps.append(UniformEllipsoid(d["center"], d["radii"], d["value"]))
        elif t == "mist_layer":
            comps.append(LayerStack([
                (-1.5, -0.2, 0.28, 0.0),
                (-0.2, 0.9, 0.28, -0.32),
            ]).with_xy_bounds(-5.0, -2.5, 5.0, 2.5))
        else:  # pragma: no cover - manifests are validated on load
            raise ValueError(f"未知随机介质分量类型 {t!r}")
    return comps


@dataclass
class RandomMedium:
    """A seeded procedural medium plus its reproducible edit log."""

    seed: int = 20261126
    n_blobs: int = 7
    n_boxes: int = 4
    res: int = 10
    strategy: str = "recompute"
    log: Optional[DensityEditLog] = None
    _descriptors: List[Dict[str, Any]] = field(default_factory=list)
    rebuilds: int = 0

    def __post_init__(self) -> None:
        self._descriptors = _component_descriptors(self.seed, self.n_blobs,
                                                   self.n_boxes)
        base = CompositeField(components_from_descriptors(self._descriptors),
                              RANDOM_AABB_MIN, RANDOM_AABB_MAX,
                              name="random_base")
        if self.log is None:
            self.log = DensityEditLog(
                base_density=base, aabb_min=RANDOM_AABB_MIN,
                aabb_max=RANDOM_AABB_MAX, res=self.res,
                scene="random", seed=self.seed)
        else:
            self.log.scene = "random"
            self.log.seed = self.seed
            self.log.signature = self.log._hash()

    # -- description / reproducibility -----------------------------------
    def manifest(self) -> Dict[str, Any]:
        m = self.log.manifest()
        m.update({
            "generator": "stochastic_medium.random_medium",
            "seed": self.seed,
            "n_blobs": self.n_blobs,
            "n_boxes": self.n_boxes,
            "res": self.res,
            "components": self._descriptors,
        })
        return m

    def to_json(self) -> str:
        return json.dumps(self.manifest(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any]) -> "RandomMedium":
        """Rebuild an identical medium from a manifest and verify its hash.

        Optional keys are tolerated so an older sample manifest still replays;
        the final assert guarantees the rebuilt density is the documented one
        rather than a hand-edited one.
        """
        medium = cls(seed=int(manifest["seed"]),
                     n_blobs=int(manifest.get("n_blobs", 7)),
                     n_boxes=int(manifest.get("n_boxes", 4)),
                     res=int(manifest.get("res", 10)))
        for edit in manifest.get("edits", []):
            medium.log.append(edit["lo"], edit["hi"], float(edit["value"]),
                              provenance=edit.get("provenance", "replayed"))
        expected = manifest.get("signature")
        if expected is not None:
            assert medium.log.signature == expected, (
                "随机介质复现失败：签名不一致，拒绝把陈旧材料当作当前介质")
        return medium

    @property
    def signature(self) -> str:
        return self.log.signature

    # -- edits ------------------------------------------------------------
    def edit(self, lo, hi, value: float, provenance: str = "ui_density_edit"
             ) -> Dict[str, Any]:
        e = self.log.append(lo, hi, value, provenance=provenance)
        return {"version": e.version, "lo": [float(x) for x in e.lo],
                "hi": [float(x) for x in e.hi], "value": float(e.value),
                "signature": self.signature}

    def edit_global_brick(self, delta: float, provenance: str = "ui_delta"
                          ) -> Dict[str, Any]:
        """Add ``delta`` to the whole lower half of the frame (a bulk edit)."""
        return self.edit(RANDOM_AABB_MIN,
                         [RANDOM_AABB_MAX[0], 0.5, RANDOM_AABB_MAX[2]],
                         delta, provenance=provenance)

    def reset_edits(self) -> None:
        self.log.edits.clear()
        self.log.signature = self.log._hash()

    # -- fields / maintenance --------------------------------------------
    def field(self) -> CompositeField:
        return self.log.evaluated_field()

    def maintenance(self) -> BrickMajorantMaintenance:
        m = BrickMajorantMaintenance(self.log, strategy=self.strategy)
        m.apply_edits()
        return m

    def refresh(self) -> None:
        """Rebuild the base components from the seed (dispose stale material)."""
        self._descriptors = _component_descriptors(self.seed, self.n_blobs,
                                                   self.n_boxes)
        base = CompositeField(components_from_descriptors(self._descriptors),
                              RANDOM_AABB_MIN, RANDOM_AABB_MAX,
                              name="random_base")
        self.log.base_density = base
        self.log.signature = self.log._hash()
        self.rebuilds += 1


class RandomMediumCache:
    """Keeps one medium per seed and disposes it when the signature changes.

    Stale disposal: a cached medium whose signature no longer matches the
    requested manifest is dropped and rebuilt *from the manifest*, so the
    evaluated density is the documented one and never a blend of old edits.
    A manifest-less request with a mismatched signature falls back to the seed.
    """

    def __init__(self) -> None:
        self._media: Dict[int, RandomMedium] = {}
        self.disposals = 0

    def get(self, seed: int, manifest: Optional[Dict[str, Any]] = None
            ) -> RandomMedium:
        expected = (manifest or {}).get("signature")
        medium = self._media.get(seed)
        if medium is not None and expected is not None \
                and medium.signature != expected:
            del self._media[seed]
            self.disposals += 1
            medium = None
        if medium is None:
            medium = (RandomMedium.from_manifest(manifest)
                      if manifest is not None else RandomMedium(seed=seed))
            self._media[seed] = medium
        return medium

    def invalidate(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self._media.clear()
        else:
            self._media.pop(seed, None)

    def signatures(self) -> Dict[int, str]:
        return {s: m.signature for s, m in self._media.items()}


def random_medium_field(seed: int = 20261126) -> CompositeField:
    """Convenience factory: the evaluated density of a seeded random medium."""
    return RandomMedium(seed=seed).field()

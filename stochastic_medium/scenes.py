"""Declared scenes: constant slab, layers, sparse volume, complex composite."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .fields import (CompositeField, GaussianBlob, LayerStack,
                     UniformBox, UniformEllipsoid)

# Orthographic camera frame used by the web renderer.
FRAME = {
    "x_min": -5.0, "x_max": 5.0,
    "z_min": -1.5, "z_max": 5.5,
    "camera_y": -3.2, "ray_length": 6.4,  # ends at y = 3.2
}

LIGHT = {
    # direction *towards* the directional light
    "direction": np.array([-0.45, -0.55, 0.70], dtype=float),
    "color": np.array([1.0, 0.93, 0.80]),
    "intensity": 2.6,
}


def linear_layer(z0: float, z1: float, d0: float, d1: float
                 ) -> Tuple[float, float, float, float]:
    b = (d1 - d0) / (z1 - z0)
    a = d0 - b * z0
    return (z0, z1, a, b)


def constant_scene() -> CompositeField:
    box = UniformBox([-3.0, -2.0, 0.0], [3.0, 2.0, 4.0], 1.0)
    return CompositeField([box], name="constant")


def layered_scene() -> CompositeField:
    layers = LayerStack([
        linear_layer(-1.5, 0.2, 0.10, 0.22),
        linear_layer(0.2, 1.6, 0.22, 0.95),
        (1.6, 2.7, 0.85, 0.0),          # constant d = 0.85
        linear_layer(2.7, 5.5, 0.55, 0.04),
    ]).with_xy_bounds(-5.0, -2.5, 5.0, 2.5)
    return CompositeField([layers],
                          [-5.0, -2.5, -1.5], [5.0, 2.5, 5.5],
                          name="layers")


def sparse_scene(seed: int = 7) -> CompositeField:
    rng = np.random.default_rng(seed)
    comps = []
    for _ in range(16):
        x = rng.uniform(-4.4, 4.4)
        z = rng.uniform(-0.8, 5.0)
        y = rng.uniform(-1.8, 1.8)
        r = rng.uniform(0.22, 0.55)
        comps.append(UniformEllipsoid(
            [x, y, z],
            [r * rng.uniform(0.8, 1.5), r, r * rng.uniform(0.8, 1.4)],
            float(rng.uniform(0.9, 1.7)),
        ))
    for _ in range(5):
        cx = rng.uniform(-4.2, 4.2)
        cz = rng.uniform(-0.5, 4.8)
        cy = rng.uniform(-1.8, 1.8)
        s = rng.uniform(0.18, 0.4)
        comps.append(UniformBox(
            [cx - s, cy - s, cz - s],
            [cx + s, cy + s, cz + s],
            float(rng.uniform(0.7, 1.4)),
        ))
    return CompositeField(comps,
                          [-5.0, -2.5, -1.5], [5.0, 2.5, 5.5],
                          name="sparse")


def complex_scene() -> CompositeField:
    comps = []
    # ground mist with a soft upper edge
    mist = LayerStack([
        linear_layer(-1.5, -0.1, 0.55, 0.42),
        linear_layer(-0.1, 1.1, 0.42, 0.0),
    ]).with_xy_bounds(-5.0, -2.5, 5.0, 2.5)
    comps.append(mist)

    clouds = [
        ((-2.6, 0.2, 3.5), (1.9, 1.25, 0.95), 0.85),
        ((0.4, -0.4, 4.1), (2.3, 1.45, 1.15), 1.00),
        ((3.0, 0.5, 2.9), (1.5, 1.0, 0.85), 0.80),
        ((-0.9, 0.9, 2.2), (1.3, 0.9, 0.7), 0.70),
        ((3.6, -0.8, 4.6), (1.2, 0.85, 0.75), 0.65),
    ]
    for c, r, a in clouds:
        comps.append(GaussianBlob(c, r, a, cutoff_sigma=2.7))

    # two dense compact cores
    comps.append(UniformEllipsoid((-1.4, -0.6, 1.1), (0.55, 0.5, 0.6), 1.5))
    comps.append(UniformEllipsoid((2.0, 0.7, 1.5), (0.45, 0.55, 0.5), 1.35))

    return CompositeField(comps,
                          [-5.0, -2.5, -1.5], [5.0, 2.5, 5.5],
                          name="complex")


SCENE_FACTORIES = {
    "constant": constant_scene,
    "layers": layered_scene,
    "sparse": sparse_scene,
    "complex": complex_scene,
}


def load_scene(name: str) -> CompositeField:
    if name not in SCENE_FACTORIES:
        raise KeyError(f"unknown scene {name}")
    return SCENE_FACTORIES[name]()


def light_towards() -> np.ndarray:
    d = LIGHT["direction"].astype(float).copy()
    return d / np.linalg.norm(d)

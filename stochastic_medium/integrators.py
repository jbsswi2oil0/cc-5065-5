"""Lighting integration: weighted single scattering and finite bounces."""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np

from .fields import ray_aabb_interval
from .tracking import CandidateRecord, estimate_transmittance

SCATTER_TINT = np.array([0.82, 0.92, 1.0])


def hg_phase(cos_theta: float, g: float) -> float:
    g = max(-0.999, min(0.999, g))
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * math.pi * denom ** 1.5)


def sample_hg_direction(incident: np.ndarray, g: float,
                        rng: random.Random) -> np.ndarray:
    """Sample an outgoing direction around incident via HG phase function."""
    u1 = rng.random()
    u2 = rng.random()
    if abs(g) < 1e-6:
        cos_theta = 1.0 - 2.0 * u1
    else:
        gg = g * g
        cos_theta = ((1.0 + gg
                      - ((1.0 - gg) / (1.0 - g + 2.0 * g * u1)) ** 2)
                     / (2.0 * g))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * u2

    w = np.asarray(incident, float)
    w = w / np.linalg.norm(w)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(w, helper))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(w, helper)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    direction = (sin_theta * math.cos(phi) * u
                 + sin_theta * math.sin(phi) * v
                 + cos_theta * w)
    return direction / np.linalg.norm(direction)


def shadow_transmittance(field_obj, position, to_light, kappa: float,
                         epsilon: float = 1e-5) -> float:
    o = np.asarray(position, float) + epsilon * np.asarray(to_light, float)
    d = np.asarray(to_light, float)
    hit = ray_aabb_interval(o, d, field_obj.aabb_min, field_obj.aabb_max)
    if hit is None:
        return 1.0
    t_end = max(hit[1], 0.0)
    if t_end <= 0.0:
        return 1.0
    tau = kappa * field_obj.optical_depth(o, d, 0.0, t_end)
    return math.exp(-min(tau, 700.0))


def background_pixel(x: float, z: float) -> np.ndarray:
    z0, z1 = -1.5, 5.5
    m = min(1.0, max(0.0, (z - z0) / (z1 - z0)))
    bottom = np.array([0.93, 0.87, 0.76])
    top = np.array([0.55, 0.68, 0.85])
    col = bottom * (1.0 - m) + top * m
    glow = math.exp(-(((x + 2.3) / 2.2) ** 2 + ((z - 3.7) / 1.7) ** 2))
    col = col + glow * np.array([0.55, 0.42, 0.22])
    return np.clip(col, 0.0, 1.4)


def background_direction(d: np.ndarray) -> np.ndarray:
    m = min(1.0, max(0.0, 0.5 + 0.5 * float(d[2])))
    bottom = np.array([0.93, 0.87, 0.76])
    top = np.array([0.55, 0.68, 0.85])
    sun = np.array([-0.45, -0.55, 0.70])
    sun /= np.linalg.norm(sun)
    glow = max(0.0, float(np.dot(d, sun))) ** 18
    col = bottom * (1.0 - m) + top * m + glow * np.array([1.2, 0.9, 0.55])
    return np.clip(col, 0.0, 1.6)


def single_scattering_radiance(field_obj, origin, direction, L: float, *,
                               kappa: float, albedo: float, g: float,
                               light_dir: np.ndarray,
                               light_color: np.ndarray,
                               light_intensity: float,
                               partition: str = "global",
                               grid_res: int = 14,
                               policy: str = "strict",
                               rng: Optional[random.Random] = None,
                               seed: Optional[int] = None,
                               ) -> Dict[str, Any]:
    """Weighted-candidate single-scattering estimator for one ray."""
    result = estimate_transmittance(
        field_obj, origin, direction, L, kappa=kappa,
        estimator="ratio", partition=partition, grid_res=grid_res,
        policy=policy, rng=rng, seed=seed, record=True)

    d = np.asarray(direction, float)
    radiance = np.zeros(3)
    for ev in result.events:
        if ev.kind not in ("real", "null"):
            continue
        cos_theta = float(np.dot(d, light_dir))
        phase = hg_phase(cos_theta, g)
        shadow = shadow_transmittance(field_obj, ev.position, light_dir, kappa)
        # sigma_s/M = p_applied * albedo * tint
        local = (ev.weight_before * ev.p_applied * albedo * SCATTER_TINT
                 * phase * shadow * light_color * light_intensity)
        radiance += local
        ev.scattering = [float(x) for x in np.atleast_1d(local)]

    bg = background_pixel(float(origin[0]), float(origin[2]))
    radiance += result.estimate * bg
    return {"radiance": radiance, "result": result, "background": bg}


def trace_path(field_obj, origin, direction, L: float, *,
               kappa: float, albedo: float, g: float,
               light_dir: np.ndarray, light_color: np.ndarray,
               light_intensity: float,
               max_bounces: int = 2,
               partition: str = "global", grid_res: int = 14,
               policy: str = "strict",
               rng: Optional[random.Random] = None,
               seed: Optional[int] = None,
               record_candidates: bool = True) -> Dict[str, Any]:
    """Finite-bounce delta-tracking path with next-event estimation."""
    if rng is None:
        rng = random.Random(seed)
    o = np.asarray(origin, float).copy()
    d = np.asarray(direction, float).copy()
    L = float(L)

    throughput = np.ones(3)
    radiance = np.zeros(3)
    bounces: List[Dict[str, Any]] = []
    all_repairs: List[Dict[str, Any]] = []

    for bounce in range(max_bounces + 1):
        res = estimate_transmittance(
            field_obj, o, d, L, kappa=kappa, estimator="binary",
            partition=partition, grid_res=grid_res, policy=policy,
            rng=rng, record=record_candidates)
        all_repairs.extend(res.repairs)

        if res.survived or res.collision_t is None:
            bg = background_direction(d)
            contrib = throughput * bg
            radiance += contrib
            bounces.append({
                "bounce": bounce, "type": "background",
                "direction": [float(x) for x in d],
                "contribution": [float(x) for x in contrib],
                "n_candidates": res.n_candidates,
                "estimate": res.estimate,
            })
            break

        t_col = res.collision_t
        pos = np.asarray(o, float) + d * t_col
        cos_theta = float(np.dot(d, light_dir))
        phase = hg_phase(cos_theta, g)
        shadow = shadow_transmittance(field_obj, pos, light_dir, kappa)
        direct = (throughput * albedo * SCATTER_TINT * phase * shadow
                  * light_color * light_intensity)
        radiance += direct

        real_event = next((e for e in reversed(res.events)
                           if e.kind in ("real", "analytic")), None)
        bounces.append({
            "bounce": bounce,
            "type": "real_scattering",
            "position": [float(x) for x in pos],
            "ray_t": float(t_col),
            "cos_theta": cos_theta,
            "phase": phase,
            "shadow_transmittance": shadow,
            "direct_contribution": [float(x) for x in direct],
            "throughput_before": [float(x) for x in throughput],
            "n_candidates": res.n_candidates,
            "n_real": res.n_real,
            "candidates": [e.to_dict() for e in res.events]
            if record_candidates else [],
        })

        if bounce == max_bounces:
            bounces[-1]["terminated"] = "max_bounces"
            break

        throughput = throughput * albedo * SCATTER_TINT
        d = sample_hg_direction(d, g, rng)
        o = pos + 1e-5 * d
        hit = ray_aabb_interval(o, d, field_obj.aabb_min, field_obj.aabb_max)
        if hit is None or hit[1] <= 0.0:
            bg = background_direction(d)
            radiance += throughput * bg
            bounces.append({"bounce": bounce + 1, "type": "background",
                            "reason": "ray_outside_medium"})
            break
        L = max(float(hit[1]), 1e-9)

    return {
        "radiance": [float(x) for x in radiance],
        "bounces": bounces,
        "repairs": all_repairs,
        "max_bounces": max_bounces,
        "throughput_final": [float(x) for x in throughput],
    }

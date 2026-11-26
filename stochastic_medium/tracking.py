"""Scalar stochastic-tracking kernel (Woodcock/delta + ratio tracking).

This is the reference implementation used by the tests and the inspection
endpoint.  It is deliberately explicit:

* candidates are sampled from Poisson rates given by *declared* segment
  majorants M = kappa * d_bar;
* a candidate is real with probability sigma_t(x) / M;
* sigma_t(x) > M is a ``MajorantViolation`` -- never silently clamped;
* violation policies are strict failure, segment discard + resample with a
  safe majorant, or whole-ray repartition + retry;
* candidate budgets are hard bounded; the dense-tail action is an exact
  analytic optical-depth handoff (unbiased Rao-Blackwellisation), never an
  infinite loop.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .errors import EventBudgetExceeded, MajorantViolation, MediumKernelError
from .params import PROB_REL_TOL, VACUUM_TAU_EPS


# ---------------------------------------------------------------------------
# records / results
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    t: float
    position: List[float]
    segment: int
    tag: str
    density: float
    sigma_t: float
    majorant: float
    p_raw: float
    p_applied: float
    kind: str                     # 'real' | 'null' | 'analytic'
    accept_u: Optional[float] = None
    weight_before: Optional[float] = None
    weight_factor: Optional[float] = None
    weight_after: Optional[float] = None
    scattering: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TransmittanceResult:
    estimate: float
    estimator: str
    survived: bool
    kappa: float
    partition: str
    policy: str
    n_candidates: int = 0
    n_real: int = 0
    n_null: int = 0
    n_discarded: int = 0
    n_clamped_tolerance: int = 0
    analytic_handoffs: int = 0
    vacuum_fast_path: bool = False
    tau_analytic: float = 0.0
    declared_bound_tau: float = 0.0
    collision_t: Optional[float] = None
    events: List[CandidateRecord] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    repairs: List[Dict[str, Any]] = field(default_factory=list)
    attempts: int = 1

    def to_dict(self, include_events: bool = True) -> Dict[str, Any]:
        d = {
            "estimate": self.estimate,
            "estimator": self.estimator,
            "survived": self.survived,
            "kappa": self.kappa,
            "partition": self.partition,
            "policy": self.policy,
            "n_candidates": self.n_candidates,
            "n_real": self.n_real,
            "n_null": self.n_null,
            "n_discarded": self.n_discarded,
            "n_clamped_tolerance": self.n_clamped_tolerance,
            "analytic_handoffs": self.analytic_handoffs,
            "vacuum_fast_path": self.vacuum_fast_path,
            "tau_analytic": self.tau_analytic,
            "declared_bound_tau": self.declared_bound_tau,
            "collision_t": self.collision_t,
            "segments": self.segments,
            "repairs": self.repairs,
            "attempts": self.attempts,
        }
        if include_events:
            d["events"] = [e.to_dict() for e in self.events]
        return d


def _segments_json(segments, kappa) -> List[Dict[str, Any]]:
    return [{
        "index": s.index, "t0": s.t0, "t1": s.t1,
        "density_majorant": s.density_majorant,
        "majorant": kappa * s.density_majorant,
        "tag": s.tag,
    } for s in segments]


def _violation(field_obj, o, d, t, seg, kappa, dens, sigma, M, p_raw,
               partition, detected_by: str = "candidate") -> MajorantViolation:
    pos = np.asarray(o, float) + np.asarray(d, float) * t
    safe_d = field_obj.certified_segment_majorant(seg)
    if safe_d is None:
        safe_d = field_obj.density_majorant()
    return MajorantViolation(
        f"主界过低({detected_by}): sigma_t={sigma:.6g} > M={M:.6g} "
        f"(p={p_raw:.6g}>1)，位置 t={t:.6g}, 段 {seg.index}({seg.tag})。"
        f"禁止夹紧隐瞒偏差。",
        position=[float(x) for x in pos],
        ray_t=float(t),
        segment_index=seg.index,
        declared_majorant=float(M),
        actual_sigma_t=float(sigma),
        raw_accept_probability=float(p_raw),
        density=float(dens),
        density_majorant_declared=float(seg.density_majorant),
        safe_density_majorant=float(kappa * safe_d),
        partition=partition,
        extra={"detected_by": detected_by},
    )


def audit_majorants(field_obj, o, d, segments, kappa, partition,
                    policy: str) -> List[Dict[str, Any]]:
    """Pre-flight check of declared segment bounds against certified bounds.

    A declared M=0 segment would be skipped without ever sampling a
    candidate, so candidate-time checks alone cannot detect that class of
    under-declaration.  Grid segments therefore carry a certifiable AABB
    bound that we audit here.
    """
    repairs: List[Dict[str, Any]] = []
    for seg in segments:
        if seg.length <= 0.0:
            continue
        certified_d = field_obj.certified_segment_majorant(seg)
        if certified_d is None:
            continue
        declared = seg.density_majorant
        if declared + 1e-10 * max(1.0, certified_d) >= certified_d:
            continue
        tm = 0.5 * (seg.t0 + seg.t1)
        pos = np.asarray(o, float) + np.asarray(d, float) * tm
        dens = float(field_obj.density(pos))
        viol = _violation(field_obj, o, d, tm, seg, kappa, dens,
                          kappa * dens, kappa * declared,
                          dens / max(declared, 1e-300) if declared > 0
                          else float("inf"),
                          partition, detected_by="audit")
        if policy == "strict" or policy == "repartition":
            raise viol
        if policy == "discard_resample":
            repairs.append({
                "kind": "audit_discard_resample",
                "segment": seg.index,
                "old_density_majorant": declared,
                "new_density_majorant": certified_d,
            })
            seg.density_majorant = certified_d
        elif policy == "clamp_demo":
            # Deliberately leave the bad declaration in place so the caller
            # can demonstrate the resulting bias.
            repairs.append({
                "kind": "clamp_demo_BIASED_audit",
                "segment": seg.index,
                "declared_density_majorant": declared,
                "certified_density_majorant": certified_d,
                "warning": "已发现欠界仍选择夹紧：结果有偏，仅用于演示",
            })
    return repairs


def _check_p(p_raw: float) -> Tuple[float, bool]:
    if p_raw < -1e-12:
        raise MediumKernelError(
            f"密度场给出负值，接受概率 p={p_raw}；密度必须 >= 0")
    if p_raw > 1.0 + PROB_REL_TOL:
        return p_raw, False
    if p_raw > 1.0:
        return 1.0, True
    if p_raw < 0.0:
        return 0.0, True
    return p_raw, False


def analytic_free_flight(field_obj, o, d, t0: float, L: float,
                         kappa: float, rng: random.Random
                         ) -> Optional[float]:
    """Exact inverse-CDF free flight using analytic optical depth.

    Returns collision ray-parameter t*, or None if the ray survives to L.
    """
    tau_total = kappa * field_obj.optical_depth(o, d, t0, L)
    target = -math.log(1.0 - rng.random())
    if tau_total <= target:
        return None
    lo, hi = float(t0), float(L)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        tau = kappa * field_obj.optical_depth(o, d, t0, mid)
        if tau < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-11 * max(1.0, L):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_transmittance(field_obj, origin, direction, L: float, *,
                           kappa: float = 1.0,
                           estimator: str = "ratio",
                           partition: str = "global",
                           grid_res: int = 14,
                           policy: str = "strict",
                           budget_action: str = "error",
                           max_candidates: int = 2_000_000,
                           vacuum_eps: float = VACUUM_TAU_EPS,
                           rng: Optional[random.Random] = None,
                           seed: Optional[int] = None,
                           record: bool = False,
                           audit: bool = True,
                           ) -> TransmittanceResult:
    """Estimate T = exp(-integral_0^L sigma_t dt) for one ray."""
    if estimator not in ("ratio", "binary"):
        raise ValueError("estimator must be 'ratio' or 'binary'")
    if policy == "repartition":
        return _with_repartition(
            field_obj, origin, direction, L, kappa=kappa,
            estimator=estimator, partition=partition, grid_res=grid_res,
            budget_action=budget_action, max_candidates=max_candidates,
            vacuum_eps=vacuum_eps, rng=rng, seed=seed, record=record,
            audit=audit)

    if rng is None:
        rng = random.Random(seed)

    o = np.asarray(origin, dtype=float)
    dd = np.asarray(direction, dtype=float)
    L = float(L)

    segments = field_obj.build_segments(o, dd, L, partition=partition,
                                        grid_res=grid_res)
    if audit:
        audit_repairs = audit_majorants(field_obj, o, dd, segments, kappa,
                                        partition, policy)
    else:
        audit_repairs = []
    tau_analytic_d = field_obj.optical_depth(o, dd, 0.0, L)
    tau_analytic = kappa * tau_analytic_d
    bound_tau = kappa * sum(s.density_majorant * s.length for s in segments)
    seg_json = _segments_json(segments, kappa)

    base = TransmittanceResult(
        estimate=1.0, estimator=estimator, survived=True, kappa=kappa,
        partition=partition, policy=policy, tau_analytic=tau_analytic,
        declared_bound_tau=bound_tau, segments=seg_json)

    if L <= 0.0 or kappa <= 0.0 or bound_tau <= vacuum_eps:
        base.vacuum_fast_path = kappa <= 0.0 or bound_tau <= vacuum_eps
        base.estimate = math.exp(-min(tau_analytic, 700.0))
        base.survived = True
        return base

    events: List[CandidateRecord] = []
    n_real = n_null = n_discarded = n_round = handoffs = 0
    n_drawn = 0
    repairs: List[Dict[str, Any]] = list(audit_repairs)
    weight = 1.0
    collision_t: Optional[float] = None

    si = 0
    finished = False
    while si < len(segments) and not finished:
        seg = segments[si]
        M = kappa * seg.density_majorant
        t = seg.t0
        weight_in = weight
        anchor = len(events)
        seg_real = seg_null = seg_kept = 0

        if M <= 0.0:
            si += 1
            continue

        restart_segment = False
        while True:
            n_drawn += 1
            if n_drawn > max_candidates:
                ctx = {"ray_t": t, "segment_index": seg.index,
                       "n_drawn": n_drawn, "kappa": kappa,
                       "majorant": M}
                if budget_action == "error":
                    raise EventBudgetExceeded(
                        f"候选事件数超过上限 {max_candidates}（极浓介质？），"
                        f"已确定中止而非无限循环。", ctx)
                # exact analytic tail: unbiased handoff
                ct = analytic_free_flight(field_obj, o, dd, t, L, kappa, rng)
                handoffs += 1
                if estimator == "ratio":
                    tail_tau = kappa * field_obj.optical_depth(o, dd, t, L)
                    weight *= math.exp(-min(tail_tau, 700.0))
                else:
                    if ct is not None:
                        collision_t = ct
                        base.survived = False
                        pos = o + dd * ct
                        events.append(CandidateRecord(
                            t=ct, position=[float(x) for x in pos],
                            segment=seg.index, tag="analytic_handoff",
                            density=float(field_obj.density(pos)),
                            sigma_t=kappa * float(field_obj.density(pos)),
                            majorant=M, p_raw=1.0, p_applied=1.0,
                            kind="analytic"))
                    base.survived = ct is None
                finished = True
                break

            # Poisson candidate step under current segment majorant
            t += -math.log(1.0 - rng.random()) / M
            if t >= seg.t1:
                break

            pos = o + dd * t
            dens = float(field_obj.density(pos))
            sigma = kappa * dens
            p_raw = sigma / M
            p_applied, rounded = _check_p(p_raw)

            if p_raw > 1.0 + PROB_REL_TOL:
                viol = _violation(field_obj, o, dd, t, seg, kappa, dens,
                                  sigma, M, p_raw, partition)
                if policy == "strict":
                    raise viol
                if policy == "clamp_demo":
                    p_applied = 1.0
                    rounded = True
                    repairs.append({
                        "kind": "clamp_demo_BIASED",
                        "segment": seg.index, "ray_t": t,
                        "p_raw": p_raw,
                        "warning": "越界后夹紧到 1.0：该估计有偏，仅作反面演示",
                    })
                elif policy == "discard_resample":
                    certified_d = field_obj.certified_segment_majorant(seg)
                    safe_d = (certified_d if certified_d is not None
                              else field_obj.density_majorant())
                    old_d = seg.density_majorant
                    already_patched = any(
                        r.get("segment") == seg.index
                        and r.get("kind") in ("discard_resample",
                                              "audit_discard_resample")
                        for r in repairs)
                    if already_patched or safe_d <= old_d * (1.0 + 1e-12):
                        # The certified bound itself is violated: no
                        # unbiased majorant exists from this field contract.
                        raise viol
                    if len(repairs) > 256:
                        raise MediumKernelError(
                            "主界修复次数超过 256，停止以避免无限重试")
                    seg.density_majorant = max(safe_d, old_d)
                    M = kappa * seg.density_majorant
                    repairs.append({
                        "kind": "discard_resample",
                        "segment": seg.index,
                        "ray_t": t,
                        "old_density_majorant": old_d,
                        "new_density_majorant": seg.density_majorant,
                        "discarded_candidates": seg_kept,
                    })
                    n_discarded += seg_kept
                    n_real -= seg_real
                    n_null -= seg_null
                    if record:
                        del events[anchor:]
                    weight = weight_in
                    restart_segment = True
                    break
                else:  # pragma: no cover
                    raise viol

            if rounded and p_raw <= 1.0 + PROB_REL_TOL:
                n_round += 1

            accept_u = rng.random()
            is_real = accept_u < p_applied
            kind = "real" if is_real else "null"
            if is_real:
                seg_real += 1
                n_real += 1
            else:
                seg_null += 1
                n_null += 1
            seg_kept += 1

            w_before = weight
            factor = 1.0 - p_applied
            if estimator == "ratio":
                weight *= factor

            rec = CandidateRecord(
                t=float(t), position=[float(x) for x in pos],
                segment=seg.index, tag=seg.tag, density=dens,
                sigma_t=sigma, majorant=M, p_raw=float(p_raw),
                p_applied=float(p_applied), kind=kind,
                accept_u=float(accept_u), weight_before=w_before,
                weight_factor=factor, weight_after=weight)
            if record:
                events.append(rec)

            if estimator == "binary" and is_real:
                collision_t = float(t)
                base.survived = False
                finished = True
                break

        if restart_segment:
            continue
        si += 1

    if estimator == "ratio":
        estimate = weight
        survived = True  # ratio estimate is a weight, not a survival event
    else:
        estimate = 1.0 if base.survived else 0.0
        survived = base.survived

    base.estimate = float(estimate)
    base.survived = survived
    base.n_candidates = n_real + n_null
    base.n_real = n_real
    base.n_null = n_null
    base.n_discarded = n_discarded
    base.n_clamped_tolerance = n_round
    base.analytic_handoffs = handoffs
    base.events = events if record else []
    base.repairs = repairs
    base.collision_t = collision_t
    base.segments = _segments_json(segments, kappa)
    return base


def _with_repartition(field_obj, origin, direction, L, **kw
                      ) -> TransmittanceResult:
    """Retry with certified, progressively refined partitions."""
    partition = kw.pop("partition", "global")
    grid_res = int(kw.pop("grid_res", 14))
    rng = kw.pop("rng", None)
    seed = kw.pop("seed", None)
    if rng is None:
        rng = random.Random(seed)

    attempt_log: List[Dict[str, Any]] = []
    last_error: Optional[MajorantViolation] = None
    part = partition
    res = grid_res
    for attempt in range(7):
        try:
            result = estimate_transmittance(
                field_obj, origin, direction, L,
                partition=part, grid_res=res, policy="strict",
                rng=rng, **kw)
            result.policy = "repartition"
            result.attempts = attempt + 1
            result.repairs = attempt_log + result.repairs
            return result
        except MajorantViolation as exc:
            last_error = exc
            attempt_log.append({
                "kind": "repartition",
                "attempt": attempt + 1,
                "failed_partition": part,
                "failed_grid_res": res,
                "ray_t": exc.ray_t,
                "raw_p": exc.raw_accept_probability,
            })
            if part == "global":
                part = "certified_grid"
                res = max(res, 10)
            elif part in ("flawed_grid", "certified_grid"):
                part = "certified_grid"
                res = min(res * 2, 160)
            else:
                part = "global"
    # guaranteed-safe fallback: global majorant with segment discard
    result = estimate_transmittance(
        field_obj, origin, direction, L,
        partition="global", policy="discard_resample",
        rng=rng, **kw)
    result.policy = "repartition"
    result.attempts = 8
    result.repairs = attempt_log + result.repairs
    return result

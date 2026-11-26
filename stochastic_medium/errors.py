"""Typed failures for the stochastic-medium kernel.

The UI and validation harness rely on these structured errors to feed
failures back into the kernel instead of silently clamping bad values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class MediumKernelError(Exception):
    """Base class for every deterministic kernel failure."""

    error_type = "medium_kernel_error"

    def to_context(self) -> Dict[str, Any]:
        return {"type": self.error_type, "message": str(self)}


class ParamError(MediumKernelError):
    error_type = "param_error"


@dataclass
class MajorantViolation(MediumKernelError):
    """Raised when sigma_t(x) exceeds the *declared* majorant.

    Clamping p = sigma_t / majorant to 1 would hide this condition and
    bias the estimator, so the kernel fails loudly instead.
    """

    message: str
    position: Optional[list] = None
    ray_t: Optional[float] = None
    segment_index: Optional[int] = None
    declared_majorant: Optional[float] = None
    actual_sigma_t: Optional[float] = None
    raw_accept_probability: Optional[float] = None
    density: Optional[float] = None
    density_majorant_declared: Optional[float] = None
    safe_density_majorant: Optional[float] = None
    partition: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    error_type = "majorant_violation"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message

    def to_context(self) -> Dict[str, Any]:
        ctx = {
            "type": self.error_type,
            "message": self.message,
            "position": self.position,
            "ray_t": self.ray_t,
            "segment_index": self.segment_index,
            "declared_majorant": self.declared_majorant,
            "actual_sigma_t": self.actual_sigma_t,
            "raw_accept_probability": self.raw_accept_probability,
            "density": self.density,
            "density_majorant_declared": self.density_majorant_declared,
            "safe_density_majorant": self.safe_density_majorant,
            "partition": self.partition,
            "allowed_actions": ["repartition", "discard_resample", "clamp_demo"],
        }
        ctx.update(self.extra)
        return ctx


class EventBudgetExceeded(MediumKernelError):
    error_type = "event_budget_exceeded"

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

    def to_context(self) -> Dict[str, Any]:
        ctx = {"type": self.error_type, "message": str(self)}
        ctx.update(self.context)
        ctx["allowed_actions"] = ["analytic_handoff", "lower_density", "raise_budget"]
        return ctx

"""Kernel-driven parameter schema and validated configuration.

Unit convention (kept identical by fields, extinction and majorants):

* world distances are in world units (treated as metres);
* ``density(x)`` is a dimensionless relative density field;
* the density scale ``kappa`` is the extinction coefficient at density 1,
  in world-unit^-1;
* ``sigma_t(x) = kappa * density(x)`` and the declared majorant
  ``M = kappa * density_majorant`` therefore share units of 1/length;
* candidate acceptance probability is ``sigma_t / M = density / d_bar``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .errors import ParamError

# Numerical tolerance for sigma_t <= M checks (relative).
PROB_REL_TOL = 1e-10
# Upper-bound optical depth below which we take the exact vacuum limit.
VACUUM_TAU_EPS = 1e-12

MAJORANT_POLICIES = ("strict", "discard_resample", "repartition", "clamp_demo")
PARTITIONS = ("global", "certified_grid", "layers", "flawed_grid")
SCENES = ("constant", "layers", "sparse", "complex")
RENDER_MODES = ("single", "multi")


def control_schema() -> List[Dict[str, Any]]:
    """Control descriptors consumed verbatim by the web UI."""
    return [
        {
            "id": "scene",
            "label": "密度场景",
            "type": "select",
            "options": [
                {"value": "complex", "label": "复杂复合介质（云团+分层雾）"},
                {"value": "layers", "label": "分层密度（解析）"},
                {"value": "sparse", "label": "稀疏体积（分离团块+空区）"},
                {"value": "constant", "label": "常密度板（解析基准）"},
            ],
            "default": "complex",
        },
        {
            "id": "kappa",
            "label": "介质密度 κ（d=1 时的消光系数）",
            "type": "range",
            "min": 0.0,
            "max": 12.0,
            "step": 0.05,
            "default": 1.7,
            "unit": "wu⁻¹",
            "hint": "σ_t(x)=κ·d(x)，主界 M=κ·d̄，单位一致",
        },
        {
            "id": "albedo",
            "label": "反照率 σ_s/σ_t",
            "type": "range",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "default": 0.9,
        },
        {
            "id": "g",
            "label": "Henyey–Greenstein 各向异性 g",
            "type": "range",
            "min": -0.9,
            "max": 0.9,
            "step": 0.05,
            "default": 0.45,
        },
        {
            "id": "spp",
            "label": "每像素样本数",
            "type": "range",
            "min": 1,
            "max": 32,
            "step": 1,
            "default": 4,
        },
        {
            "id": "max_bounces",
            "label": "有限多次散射最大反弹",
            "type": "range",
            "min": 0,
            "max": 4,
            "step": 1,
            "default": 1,
        },
        {
            "id": "mode",
            "label": "积分模式",
            "type": "select",
            "options": [
                {"value": "single", "label": "单次散射·候选权重估计"},
                {"value": "multi", "label": "有限多次散射·路径跟踪"},
            ],
            "default": "single",
        },
        {
            "id": "partition",
            "label": "主界/分区声明",
            "type": "select",
            "options": [
                {"value": "global", "label": "全局保守主界"},
                {"value": "certified_grid", "label": "认证网格局部主界（重分区）"},
                {"value": "layers", "label": "分层精确主界"},
                {"value": "flawed_grid", "label": "粗糙角点网格（故意欠界演示）"},
            ],
            "default": "certified_grid",
        },
        {
            "id": "policy",
            "label": "发现主界越界时",
            "type": "select",
            "options": [
                {"value": "strict", "label": "严格失败并反馈"},
                {"value": "repartition", "label": "自动重分区后重试"},
                {"value": "discard_resample", "label": "丢弃该段并按安全主界重采样"},
                {"value": "clamp_demo", "label": "反面演示：夹紧（有偏！）"},
            ],
            "default": "strict",
        },
        {
            "id": "grid_res",
            "label": "网格分区分辨率",
            "type": "range",
            "min": 2,
            "max": 48,
            "step": 1,
            "default": 14,
        },
        {
            "id": "exposure",
            "label": "曝光",
            "type": "range",
            "min": 0.2,
            "max": 3.0,
            "step": 0.05,
            "default": 1.15,
        },
    ]


@dataclass
class RenderConfig:
    scene: str = "complex"
    kappa: float = 1.7
    albedo: float = 0.9
    g: float = 0.45
    spp: int = 4
    max_bounces: int = 1
    mode: str = "single"
    partition: str = "global"
    policy: str = "strict"
    grid_res: int = 14
    exposure: float = 1.15
    width: int = 288
    height: int = 192
    seed: int = 20261126
    max_iterations: int = 4096
    jitter: bool = True

    @classmethod
    def from_request(cls, data: Dict[str, Any] | None) -> "RenderConfig":
        data = dict(data or {})
        schema = {c["id"]: c for c in control_schema()}
        kwargs: Dict[str, Any] = {}
        for key, default in asdict(cls()).items():
            if key not in data:
                kwargs[key] = default
                continue
            value = data[key]
            if key in schema and schema[key]["type"] == "range":
                spec = schema[key]
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise ParamError(f"参数 {key} 不是数值: {value!r}")
                if value < spec["min"] - 1e-12 or value > spec["max"] + 1e-12:
                    raise ParamError(
                        f"参数 {key}={value} 超出内核声明范围 "
                        f"[{spec['min']}, {spec['max']}]"
                    )
                if isinstance(default, int) and key not in ("kappa", "g", "exposure", "albedo"):
                    value = int(round(value))
            elif key in ("spp", "max_bounces", "grid_res", "width", "height", "seed", "max_iterations"):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise ParamError(f"参数 {key} 不是整数: {value!r}")
            kwargs[key] = value

        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.scene not in SCENES:
            raise ParamError(f"未知场景 {self.scene!r}，可选 {SCENES}")
        if self.mode not in RENDER_MODES:
            raise ParamError(f"未知积分模式 {self.mode!r}")
        if self.partition not in PARTITIONS:
            raise ParamError(f"未知分区 {self.partition!r}")
        if self.policy not in MAJORANT_POLICIES:
            raise ParamError(f"未知主界策略 {self.policy!r}")
        if self.kappa < 0:
            raise ParamError("kappa 必须非负")
        if not (0.0 <= self.albedo <= 1.0):
            raise ParamError("albedo 必须在 [0,1]")
        if self.width < 8 or self.height < 8:
            raise ParamError("渲染分辨率过小")
        if self.spp < 1:
            raise ParamError("spp 至少为 1")
        if self.grid_res < 1:
            raise ParamError("grid_res 至少为 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

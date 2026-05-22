from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    mode: str
    direct_strategy: str = "local"
    surrogate_mode: str = "norm_rms_mean"
    selection_strategy: str = "dynamic"
    dynamic_regioning: bool = True
    dynamic_allocator: str = "surrogate_kv"
    dynamic_surrogate_variant: str = ""
    dynamic_anchor_width: int = 4
    score_method: str = "attention"
    head_score_fusion: str = "mean"
    protected_sink: bool = False
    null_fastpath: bool = False
    anchor_residual: bool = False

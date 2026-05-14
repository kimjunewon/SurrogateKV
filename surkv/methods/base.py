from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    mode: str
    protected_sink: bool = False
    null_fastpath: bool = False
    direct_strategy: str = "local"
    surrogate_mode: str = "mean"
    selection_strategy: str = ""
    weighted_mapping: bool = False
    anchor_residual: bool = False
    dynamic_regioning: bool = False
    dynamic_allocator: str = "legacy"
    dynamic_surrogate_variant: str = ""
    dynamic_victim_max_len: int = 0
    dynamic_micro_risk: str = "max"
    dynamic_anchor_width: int = 0
    dynamic_coverage_block: int = 512

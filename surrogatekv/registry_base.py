from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    mode: str
    kind: str
    surrogate_mode: str = "mean"
    protected_sink: bool = False
    null_fastpath: bool = False
    dynamic_regioning: bool = False
    dynamic_allocator: str = ""
    dynamic_anchor_width: int = 0


__all__ = ["MethodSpec"]

from __future__ import annotations

from ..base import MethodSpec


SPEC = MethodSpec(
    name="SurrogateKV-Drop",
    mode="surrogate_kv_drop",
    kind="drop",
    null_fastpath=True,
    dynamic_allocator="surrogate_drop",
    dynamic_anchor_width=4,
)

from __future__ import annotations

from ..base import MethodSpec


SPEC = MethodSpec(
    name="SurrogateKV-Drop",
    mode="surrogate_kv_drop",
    protected_sink=False,
    null_fastpath=True,
    direct_strategy="null",
    dynamic_allocator="surrogate_drop",
    dynamic_anchor_width=4,
)

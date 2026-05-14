from __future__ import annotations

from ..base import MethodSpec


SPEC = MethodSpec(
    name="SurrogateKV",
    mode="surrogate_kv",
    direct_strategy="local",
    surrogate_mode="norm_rms_mean",
    selection_strategy="dynamic",
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=4,
)

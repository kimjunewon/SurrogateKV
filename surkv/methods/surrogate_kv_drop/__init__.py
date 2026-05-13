from __future__ import annotations

from ..base import MethodSpec
from .allocation import plan_drop_only_cache
from .prototypes import build_zero_surrogates


SPEC = MethodSpec(
    name="SurrogateKV-Drop",
    mode="surrogate_kv_drop",
    build_surrogates=build_zero_surrogates,
    protected_sink=False,
    null_fastpath=True,
    plan_chunks=plan_drop_only_cache,
    direct_strategy="null",
    dynamic_allocator="surrogate_drop",
    dynamic_anchor_width=4,
)

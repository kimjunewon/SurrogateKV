from __future__ import annotations

from ..base import MethodSpec
from .allocation import plan_surrogate_cache
from .prototypes import build_chunk_exact_surrogates


SPEC = MethodSpec(
    name="SurrogateKV",
    mode="surrogate_kv",
    build_surrogates=build_chunk_exact_surrogates,
    plan_chunks=plan_surrogate_cache,
    direct_strategy="local",
    surrogate_mode="exact",
    selection_strategy="dynamic",
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=4,
)

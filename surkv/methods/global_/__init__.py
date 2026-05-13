from __future__ import annotations

from ..base import MethodSpec
from .allocation import plan_global_cache
from .prototypes import build_global_mean_surrogates


SPEC = MethodSpec(
    name="SurKVGlobal",
    mode="global",
    build_surrogates=build_global_mean_surrogates,
    plan_chunks=plan_global_cache,
    direct_strategy="global",
)

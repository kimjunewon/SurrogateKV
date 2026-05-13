from __future__ import annotations

from ..base import MethodSpec
from .allocation import plan_null_cache
from .prototypes import build_zero_surrogates


SPEC = MethodSpec(
    name="SurKVNull",
    mode="null",
    build_surrogates=build_zero_surrogates,
    null_fastpath=True,
    plan_chunks=plan_null_cache,
    direct_strategy="null",
)

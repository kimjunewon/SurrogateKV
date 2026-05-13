from __future__ import annotations

from ..base import MethodSpec
from .allocation import plan_local_cache
from .prototypes import build_local_mean_surrogates


SPEC = MethodSpec(
    name="SurKVLocal",
    mode="local",
    build_surrogates=build_local_mean_surrogates,
    plan_chunks=plan_local_cache,
    direct_strategy="local",
)

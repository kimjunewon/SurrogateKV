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


def local_context_pair(
    *,
    batch_idx: int,
    chunk_idx: int,
    key_states,
    value_states,
    chunk_slices,
    fallback_key,
    fallback_value,
    local_radius: int,
):
    del batch_idx, chunk_idx, key_states, value_states, chunk_slices, local_radius
    return fallback_key.mean(dim=2, keepdim=True), fallback_value.mean(dim=2, keepdim=True)

from __future__ import annotations

from ..base import MethodSpec


SPEC = MethodSpec(
    name="SurKVWeighted",
    mode="weighted",
    direct_strategy="weighted",
    protected_sink=False,
    weighted_mapping=True,
    surrogate_mode="weighted_mean",
)


def weighted_chunk_pair(*, chunk_key, chunk_value, chunk_token_scores, prototype_fn):
    return prototype_fn(chunk_key, chunk_value, chunk_token_scores)

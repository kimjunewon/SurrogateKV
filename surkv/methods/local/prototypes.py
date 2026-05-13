from __future__ import annotations

from ..base import SurrogateContext
from ..utils.prototypes import chunk_mean_prototypes


def build_local_mean_surrogates(context: SurrogateContext):
    return (
        chunk_mean_prototypes(context.key_states, context.chunk_slices),
        chunk_mean_prototypes(context.value_states, context.chunk_slices),
    )

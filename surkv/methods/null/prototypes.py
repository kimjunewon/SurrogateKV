from __future__ import annotations

from ..base import SurrogateContext


def build_zero_surrogates(context: SurrogateContext):
    batch_size, num_heads, _, head_dim = context.key_states.shape
    chunk_count = len(context.chunk_slices)
    shape = (batch_size, num_heads, chunk_count, head_dim)
    return context.key_states.new_zeros(shape), context.value_states.new_zeros(shape)

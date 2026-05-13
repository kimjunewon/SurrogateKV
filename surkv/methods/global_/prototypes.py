from __future__ import annotations

import torch

from ..base import SurrogateContext


def build_global_mean_surrogates(context: SurrogateContext):
    return (
        _selected_context_mean(context.key_states, context),
        _selected_context_mean(context.value_states, context),
    )


def _selected_context_mean(states: torch.Tensor, context: SurrogateContext) -> torch.Tensor:
    batch_size, num_heads, _, head_dim = states.shape
    chunk_count = len(context.chunk_slices)
    if chunk_count <= 0:
        return states.new_empty((batch_size, num_heads, 0, head_dim))

    fallback_chunks = list(range(chunk_count))
    per_batch = []
    for batch_idx in range(batch_size):
        selected_chunks = torch.nonzero(context.replace_mask[batch_idx], as_tuple=False).flatten().tolist()
        source_chunks = selected_chunks or fallback_chunks
        source_tokens = [
            states[batch_idx : batch_idx + 1, :, int(start) : int(end), :]
            for chunk_idx in source_chunks
            for start, end in [context.chunk_slices[int(chunk_idx)]]
            if int(end) > int(start)
        ]
        if source_tokens:
            prototype = torch.cat(source_tokens, dim=2).mean(dim=2, keepdim=True)
        else:
            prototype = states.new_zeros((1, num_heads, 1, head_dim))
        per_batch.append(prototype)

    return torch.cat(per_batch, dim=0).expand(-1, -1, chunk_count, -1).contiguous()

from __future__ import annotations

from typing import Sequence

import torch

from ..base import ChunkSlice


def chunk_mean_prototypes(states: torch.Tensor, chunk_slices: Sequence[ChunkSlice]) -> torch.Tensor:
    batch_size, num_heads, _, head_dim = states.shape
    chunk_count = len(chunk_slices)
    if chunk_count <= 0:
        return states.new_empty((batch_size, num_heads, 0, head_dim))

    if not _uses_regular_spans(chunk_slices):
        return torch.stack(
            [states[:, :, int(start) : int(end), :].mean(dim=2) for start, end in chunk_slices],
            dim=2,
        )

    first_start = int(chunk_slices[0][0])
    chunk_width = int(chunk_slices[0][1] - chunk_slices[0][0])
    tail_width = int(chunk_slices[-1][1] - chunk_slices[-1][0])
    regular_count = chunk_count if tail_width == chunk_width else chunk_count - 1
    prototypes = []

    if regular_count > 0:
        regular_tokens = regular_count * chunk_width
        regular = states[:, :, first_start : first_start + regular_tokens, :].reshape(
            batch_size,
            num_heads,
            regular_count,
            chunk_width,
            head_dim,
        )
        prototypes.append(regular.mean(dim=3))

    if tail_width != chunk_width:
        tail_start = first_start + regular_count * chunk_width
        prototypes.append(states[:, :, tail_start : tail_start + tail_width, :].mean(dim=2, keepdim=True))

    return torch.cat(prototypes, dim=2)


def _uses_regular_spans(chunk_slices: Sequence[ChunkSlice]) -> bool:
    if len(chunk_slices) <= 1:
        return True

    first_start = int(chunk_slices[0][0])
    chunk_width = int(chunk_slices[0][1] - chunk_slices[0][0])
    if chunk_width <= 0:
        return False

    for idx, (start, end) in enumerate(chunk_slices):
        start = int(start)
        end = int(end)
        expected_start = first_start + idx * chunk_width
        span_width = end - start
        if start != expected_start:
            return False
        if idx < len(chunk_slices) - 1 and span_width != chunk_width:
            return False
        if idx == len(chunk_slices) - 1 and not (0 < span_width <= chunk_width):
            return False
    return True

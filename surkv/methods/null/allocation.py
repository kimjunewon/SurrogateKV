from __future__ import annotations

from ..base import AllocationPlan, SurrogateContext
from ..utils.selection import select_low_score_chunks


def plan_null_cache(context: SurrogateContext) -> AllocationPlan:
    replace_mask = select_low_score_chunks(
        chunk_scores=context.chunk_scores,
        chunk_lengths=context.chunk_lengths,
        surrogate_lengths=context.surrogate_lengths,
        tokens_to_save=context.tokens_to_save,
    )
    return AllocationPlan(
        chunk_slices=context.chunk_slices,
        chunk_lengths=context.chunk_lengths,
        replace_mask=replace_mask,
        surrogate_lengths=context.surrogate_lengths,
    )

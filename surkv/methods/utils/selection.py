from __future__ import annotations

import torch


def select_chunks_fast(
    *,
    chunk_scores,
    chunk_lengths,
    surrogate_lengths,
    tokens_to_save: int,
    exclusion_radius: int = 0,
):
    if tokens_to_save <= 0:
        return torch.zeros_like(chunk_scores, dtype=torch.bool)
    savings_per_chunk = torch.clamp(chunk_lengths.unsqueeze(0) - surrogate_lengths, min=0)
    ordering = torch.argsort(chunk_scores, dim=-1, descending=False)
    return _select_until_budget(
        ordered_chunk_ids=ordering,
        removable_tokens=savings_per_chunk,
        target_tokens=int(tokens_to_save),
        neighbor_radius=int(exclusion_radius),
    )


def _select_until_budget(
    *,
    ordered_chunk_ids: torch.Tensor,
    removable_tokens: torch.Tensor,
    target_tokens: int,
    neighbor_radius: int,
) -> torch.Tensor:
    batch_size, chunk_count = ordered_chunk_ids.shape
    selected = torch.zeros((batch_size, chunk_count), device=ordered_chunk_ids.device, dtype=torch.bool)
    if target_tokens <= 0 or chunk_count <= 0:
        return selected

    ordered_savings = removable_tokens.gather(1, ordered_chunk_ids)
    if neighbor_radius <= 0:
        positive = ordered_savings > 0
        cumulative = torch.cumsum(torch.where(positive, ordered_savings, torch.zeros_like(ordered_savings)), dim=1)
        reached_target = cumulative >= target_tokens
        reached_any = reached_target.any(dim=1)
        prefix_len = torch.where(
            reached_any,
            reached_target.to(dtype=torch.long).argmax(dim=1) + 1,
            positive.sum(dim=1, dtype=torch.long),
        )
        ranks = torch.arange(chunk_count, device=ordered_chunk_ids.device, dtype=torch.long).unsqueeze(0)
        ordered_selected = positive & (ranks < prefix_len.unsqueeze(1))
        selected.scatter_(1, ordered_chunk_ids, ordered_selected)
        return selected

    # Kept only for API completeness; current SurrogateKV methods call this with radius 0.
    saved_tokens = removable_tokens.new_zeros((batch_size,))
    blocked = torch.zeros_like(selected)
    batch_ids = torch.arange(batch_size, device=ordered_chunk_ids.device, dtype=torch.long)
    indices = torch.arange(chunk_count, device=ordered_chunk_ids.device, dtype=torch.long)
    neighbor_mask = (indices[:, None] - indices[None, :]).abs() <= int(neighbor_radius)
    for rank in range(chunk_count):
        chunk_ids = ordered_chunk_ids[:, rank]
        savings = ordered_savings[:, rank]
        can_select = (savings > 0) & (saved_tokens < target_tokens) & ~blocked[batch_ids, chunk_ids]
        if not can_select.any():
            continue
        selected[batch_ids[can_select], chunk_ids[can_select]] = True
        saved_tokens = saved_tokens + torch.where(can_select, savings, torch.zeros_like(savings))
        blocked[can_select] |= neighbor_mask.index_select(0, chunk_ids[can_select])
    return selected

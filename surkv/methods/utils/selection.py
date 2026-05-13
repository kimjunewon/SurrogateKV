from __future__ import annotations

import torch


_NEIGHBOR_MASK_CACHE = {}


def _device_key(device: torch.device) -> str:
    index = "" if device.index is None else str(device.index)
    return f"{device.type}:{index}"


def _cached_neighbor_mask(*, chunk_count: int, radius: int, device: torch.device):
    if radius <= 0 or chunk_count <= 0:
        return None

    cache_key = (_device_key(device), int(chunk_count), int(radius))
    cached = _NEIGHBOR_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    indices = torch.arange(chunk_count, device=device, dtype=torch.long)
    mask = (indices[:, None] - indices[None, :]).abs() <= int(radius)
    if len(_NEIGHBOR_MASK_CACHE) >= 64:
        _NEIGHBOR_MASK_CACHE.clear()
    _NEIGHBOR_MASK_CACHE[cache_key] = mask
    return mask


def select_low_score_chunks(
    *,
    chunk_scores: torch.Tensor,
    chunk_lengths: torch.Tensor,
    surrogate_lengths: torch.Tensor,
    tokens_to_save: int,
    neighbor_radius: int = 0,
) -> torch.Tensor:
    if tokens_to_save <= 0:
        return torch.zeros_like(chunk_scores, dtype=torch.bool)

    removable_tokens = torch.clamp(chunk_lengths.unsqueeze(0) - surrogate_lengths, min=0)
    lowest_score_first = torch.argsort(chunk_scores, dim=-1, descending=False)
    return _select_until_budget(
        ordered_chunk_ids=lowest_score_first,
        removable_tokens=removable_tokens,
        target_tokens=int(tokens_to_save),
        neighbor_radius=int(neighbor_radius),
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

    saved_tokens = removable_tokens.new_zeros((batch_size,))
    blocked = torch.zeros_like(selected)
    deferred = torch.zeros_like(selected)
    batch_ids = torch.arange(batch_size, device=ordered_chunk_ids.device, dtype=torch.long)
    neighbor_mask = _cached_neighbor_mask(
        chunk_count=chunk_count,
        radius=neighbor_radius,
        device=ordered_chunk_ids.device,
    )

    for rank in range(chunk_count):
        chunk_ids = ordered_chunk_ids[:, rank]
        savings = ordered_savings[:, rank]
        can_help = savings > 0
        needs_more = saved_tokens < target_tokens
        blocked_now = blocked[batch_ids, chunk_ids]
        choose_now = can_help & needs_more & ~blocked_now

        selected[batch_ids, chunk_ids] |= choose_now
        saved_tokens = saved_tokens + savings * choose_now.to(dtype=savings.dtype)
        deferred[:, rank] = can_help & needs_more & blocked_now
        blocked |= neighbor_mask.index_select(0, chunk_ids) & choose_now.unsqueeze(1)

    for rank in range(chunk_count):
        chunk_ids = ordered_chunk_ids[:, rank]
        savings = ordered_savings[:, rank]
        choose_now = deferred[:, rank] & ~selected[batch_ids, chunk_ids] & (saved_tokens < target_tokens)
        selected[batch_ids, chunk_ids] |= choose_now
        saved_tokens = saved_tokens + savings * choose_now.to(dtype=savings.dtype)

    return selected

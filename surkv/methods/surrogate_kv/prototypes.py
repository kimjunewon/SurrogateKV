from __future__ import annotations

from collections import defaultdict

import torch

from ..base import SurrogateContext


def build_chunk_mean_surrogates(context: SurrogateContext):
    return build_chunk_surrogates(context, surrogate_mode="mean")


def build_chunk_exact_surrogates(context: SurrogateContext):
    return build_chunk_surrogates(context, surrogate_mode="exact")


def build_chunk_surrogates(context: SurrogateContext, *, surrogate_mode: str):
    if surrogate_mode not in {"mean", "exact", "norm_rms_mean"}:
        raise ValueError(f"Unsupported surrogate prototype mode: {surrogate_mode}")

    groups = selected_region_groups(context)
    batch_size, num_heads, _, head_dim = context.key_states.shape
    chunk_count = len(context.chunk_slices)
    key_bank = context.key_states.new_zeros((batch_size, num_heads, chunk_count, head_dim))
    value_bank = context.value_states.new_zeros((batch_size, num_heads, chunk_count, head_dim))
    if chunk_count <= 0 or not groups:
        return key_bank, value_bank

    exact_norm_restore = surrogate_mode in {"exact", "norm_rms_mean"}
    for chunk_ids, token_ids, length, group_size in groups:
        key_chunk = _gather_group(context.key_states, token_ids, group_size=group_size, length=length)
        value_chunk = _gather_group(context.value_states, token_ids, group_size=group_size, length=length)
        key_proto = key_chunk.mean(dim=3)
        value_proto = value_chunk.mean(dim=3)

        if exact_norm_restore:
            key_proto = _restore_mean_key_norm(key_proto, key_chunk, token_dim=3)
            value_proto = _restore_rms_value_norm(value_proto, value_chunk, token_dim=3)

        key_bank.index_copy_(2, chunk_ids, key_proto)
        value_bank.index_copy_(2, chunk_ids, value_proto)

    return key_bank, value_bank


def selected_region_groups(context: SurrogateContext):
    selected_mask = (context.replace_mask[0] & (context.surrogate_lengths[0] > 0)).detach().cpu().tolist()
    by_length: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for chunk_idx, selected in enumerate(selected_mask):
        if not selected:
            continue
        start, end = context.chunk_slices[chunk_idx]
        length = int(end) - int(start)
        if length > 0:
            by_length[length].append((chunk_idx, int(start)))

    groups = []
    device = context.key_states.device
    for length, entries in by_length.items():
        chunk_ids = torch.tensor([idx for idx, _ in entries], device=device, dtype=torch.long)
        starts = torch.tensor([start for _, start in entries], device=device, dtype=torch.long)
        token_ids = (starts[:, None] + torch.arange(length, device=device, dtype=torch.long)[None, :]).reshape(-1)
        groups.append((chunk_ids, token_ids, int(length), len(entries)))
    return groups


def _gather_group(states: torch.Tensor, token_ids: torch.Tensor, *, group_size: int, length: int) -> torch.Tensor:
    batch_size, num_heads, _, head_dim = states.shape
    return states.index_select(2, token_ids).reshape(batch_size, num_heads, group_size, length, head_dim)


def _restore_mean_key_norm(key_proto: torch.Tensor, key_source: torch.Tensor, *, token_dim: int) -> torch.Tensor:
    source = key_source.to(dtype=torch.float32)
    proto = key_proto.to(dtype=torch.float32)
    target_norm = source.norm(dim=-1).mean(dim=token_dim, keepdim=False).unsqueeze(-1)
    current_norm = proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (proto * (target_norm / current_norm)).to(dtype=key_proto.dtype)


def _restore_rms_value_norm(value_proto: torch.Tensor, value_source: torch.Tensor, *, token_dim: int) -> torch.Tensor:
    source = value_source.to(dtype=torch.float32)
    proto = value_proto.to(dtype=torch.float32)
    source_norm_sq = source.square().sum(dim=-1)
    target_norm = source_norm_sq.mean(dim=token_dim, keepdim=False).clamp_min(1e-12).sqrt().unsqueeze(-1)
    current_norm = proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (proto * (target_norm / current_norm)).to(dtype=value_proto.dtype)

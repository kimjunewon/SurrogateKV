from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .selection import select_chunks_fast


_CHUNK_SLICE_CACHE: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
_RECENT_MASK_CACHE: Dict[Tuple[str, int, str], torch.Tensor] = {}


def _device_key(device: torch.device) -> str:
    index = "" if device.index is None else str(device.index)
    return f"{device.type}:{index}"


def _cache_put(cache: dict, key, value, *, max_size: int = 256):
    if len(cache) >= max_size:
        cache.clear()
    cache[key] = value
    return value


def _rank01(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 0:
        return scores.to(dtype=torch.float32)
    num_items = scores.shape[-1]
    if num_items <= 1:
        return torch.zeros_like(scores, dtype=torch.float32)
    ordering = torch.argsort(scores.to(dtype=torch.float32), dim=-1, descending=False)
    rank_values = torch.linspace(
        0.0,
        1.0,
        num_items,
        device=scores.device,
        dtype=torch.float32,
    ).view(1, num_items)
    ranks = torch.empty_like(ordering, dtype=torch.float32)
    ranks.scatter_(1, ordering, rank_values.expand_as(ranks))
    return ranks


def _restore_mean_key_norm(key_proto: torch.Tensor, key_source: torch.Tensor, *, token_dim: int) -> torch.Tensor:
    target_norm = key_source.to(dtype=torch.float32).norm(dim=-1).mean(dim=token_dim, keepdim=False).unsqueeze(-1)
    current_norm = key_proto.to(dtype=torch.float32).norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (key_proto.to(dtype=torch.float32) * (target_norm / current_norm)).to(dtype=key_proto.dtype)


def _restore_rms_value_norm(value_proto: torch.Tensor, value_source: torch.Tensor, *, token_dim: int) -> torch.Tensor:
    source_norm_sq = value_source.to(dtype=torch.float32).square().sum(dim=-1)
    target_norm = source_norm_sq.mean(dim=token_dim, keepdim=False).clamp_min(1e-12).sqrt().unsqueeze(-1)
    current_norm = value_proto.to(dtype=torch.float32).norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (value_proto.to(dtype=torch.float32) * (target_norm / current_norm)).to(dtype=value_proto.dtype)


__all__ = [name for name in globals() if not name.startswith("__")]

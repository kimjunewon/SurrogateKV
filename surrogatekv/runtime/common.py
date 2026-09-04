from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

_CHUNK_SLICE_CACHE: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
_FAST_PACK_METADATA_CACHE: Dict[Tuple[str, int, int, Tuple[int, ...]], Dict[str, torch.Tensor]] = {}
_RECENT_MASK_CACHE: Dict[Tuple[str, int, str], torch.Tensor] = {}
_SURROGATE_SCORE_WEIGHT_MODES = {
    "weighted_mean",
    "asym_key_weighted",
    "asym_value_weighted",
    "norm_value_weighted",
    "pivot_value_weighted",
    "value_sqrt_weighted_rms",
}
_SURROGATE_KEY_WEIGHT_MODES = {"weighted_mean", "asym_key_weighted"}
_SURROGATE_VALUE_WEIGHT_MODES = {
    "weighted_mean",
    "asym_value_weighted",
    "norm_value_weighted",
    "pivot_value_weighted",
    "value_sqrt_weighted_rms",
}
_SURROGATE_LIGHT_VALUE_WEIGHT_MODES = {"value_sqrt_weighted_rms"}
_SURROGATE_NORM_KEY_MODES = {"norm_value_weighted", "norm_rms_mean", "value_sqrt_weighted_rms"}
_SURROGATE_PIVOT_KEY_MODES = {"pivot_value_weighted"}
_SURROGATE_RMS_VALUE_MODES = {"norm_rms_mean", "value_sqrt_weighted_rms"}
_SURROGATE_PADDED_MODES = _SURROGATE_SCORE_WEIGHT_MODES | {"mean", "norm_rms_mean"}
_SURKV_PROFILE_TIMING = str(os.environ.get("SURKV_PROFILE_TIMING", "")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SURKV_SCORE_METHOD = str(os.environ.get("SURKV_SCORE_METHOD", "")).strip().lower()
_SURKV_HEAD_SCORE_FUSION = str(os.environ.get("SURKV_HEAD_SCORE_FUSION", "")).strip().lower()
_SURKV_DIAGNOSTIC_STATS = str(os.environ.get("SURKV_DIAGNOSTIC_STATS", "")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


_SURKV_HEADWISE_ADA_OVERLAY = _env_flag("SURKV_HEADWISE_ADA_OVERLAY", False)


# Adapted and modified from transformers.models.llama.modeling_llama.repeat_kv
# (Apache-2.0). See THIRD_PARTY_NOTICES.md.
def _repeat_kv_heads(states: torch.Tensor, groups: int) -> torch.Tensor:
    groups = max(1, int(groups))
    if groups == 1:
        return states
    bsz, heads, seqlen, dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(int(bsz), int(heads), int(groups), int(seqlen), int(dim))
        .reshape(int(bsz), int(heads) * int(groups), int(seqlen), int(dim))
    )


_SCORE_WEIGHTED_SURROGATE_MODES = _SURROGATE_SCORE_WEIGHT_MODES
_KEY_WEIGHTED_SURROGATE_MODES = _SURROGATE_KEY_WEIGHT_MODES
_VALUE_WEIGHTED_SURROGATE_MODES = _SURROGATE_VALUE_WEIGHT_MODES
_NORM_RESTORED_KEY_MODES = _SURROGATE_NORM_KEY_MODES
_PIVOT_KEY_MODES = _SURROGATE_PIVOT_KEY_MODES
_LIGHT_VALUE_WEIGHT_MODES = _SURROGATE_LIGHT_VALUE_WEIGHT_MODES
_RMS_RESTORED_VALUE_MODES = _SURROGATE_RMS_VALUE_MODES


def _restore_mean_key_norm(key_proto: torch.Tensor, key_source: torch.Tensor, *, token_dim: int) -> torch.Tensor:
    target_norm = key_source.to(dtype=torch.float32).norm(dim=-1).mean(dim=token_dim, keepdim=False).unsqueeze(-1)
    current_norm = key_proto.to(dtype=torch.float32).norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = _safe_key_norm_scale(target_norm=target_norm, current_norm=current_norm)
    return (key_proto.to(dtype=torch.float32) * scale).to(dtype=key_proto.dtype)


def _safe_key_norm_scale(*, target_norm: torch.Tensor, current_norm: torch.Tensor) -> torch.Tensor:
    scale = target_norm.to(dtype=torch.float32) / current_norm.to(dtype=torch.float32).clamp_min(1e-6)
    if _env_flag("SURKV_ALLOW_SURROGATE_KEY_NORM_BOOST", False):
        return scale
    # A low-norm mean key is often the correct RoPE cancellation signal.  Boosting
    # it back to the average token norm creates off-manifold surrogate keys that
    # can dominate short-context decoding.
    return torch.minimum(scale, torch.ones_like(scale))


def _restore_rms_value_norm(value_proto: torch.Tensor, value_source: torch.Tensor, *, token_dim: int) -> torch.Tensor:
    source_norm_sq = value_source.to(dtype=torch.float32).square().sum(dim=-1)
    target_norm = source_norm_sq.mean(dim=token_dim, keepdim=False).clamp_min(1e-12).sqrt().unsqueeze(-1)
    current_norm = value_proto.to(dtype=torch.float32).norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (value_proto.to(dtype=torch.float32) * (target_norm / current_norm)).to(dtype=value_proto.dtype)


def _device_key(device: torch.device) -> str:
    index = "" if device.index is None else str(device.index)
    return f"{device.type}:{index}"


def _cache_put(cache: dict, key, value, *, max_size: int = 256):
    if len(cache) >= max_size:
        cache.clear()
    cache[key] = value
    return value


_RANK01_VALUE_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def _rank01(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 0:
        return scores.to(dtype=torch.float32)
    num_items = scores.shape[-1]
    if num_items <= 1:
        return torch.zeros_like(scores, dtype=torch.float32)
    ordering = torch.argsort(scores.to(dtype=torch.float32), dim=-1, descending=False)
    cache_key = (_device_key(scores.device), int(num_items))
    rank_values = _RANK01_VALUE_CACHE.get(cache_key)
    if rank_values is None or rank_values.device != scores.device:
        rank_values = torch.linspace(
            0.0,
            1.0,
            num_items,
            device=scores.device,
            dtype=torch.float32,
        ).view(1, num_items)
        rank_values = _cache_put(_RANK01_VALUE_CACHE, cache_key, rank_values, max_size=64)
    ranks = torch.empty_like(ordering, dtype=torch.float32)
    ranks.scatter_(1, ordering, rank_values.expand_as(ranks))
    return ranks

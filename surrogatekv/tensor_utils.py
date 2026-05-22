from __future__ import annotations

import math

import torch


def normalize_token_weights(token_scores, *, dtype, device):
    weights = torch.clamp(token_scores.to(device=device, dtype=torch.float32), min=1e-6)
    weights = weights / torch.clamp(weights.sum(), min=1e-6)
    return weights.to(dtype=dtype)


def mapping_alpha_from_weights(weights) -> float:
    flat = weights.detach().to(dtype=torch.float32).reshape(-1)
    if flat.numel() <= 1:
        return 1.0
    uniform = 1.0 / float(flat.numel())
    peak = float(flat.max().item())
    normalized_peak = (peak - uniform) / max(1e-6, 1.0 - uniform)
    normalized_peak = max(0.0, min(1.0, normalized_peak))
    return math.sqrt(normalized_peak)


def token_weight_stats(weights) -> dict[str, float]:
    flat = weights.detach().to(dtype=torch.float32).reshape(-1)
    if flat.numel() <= 0:
        return {"weight_entropy": 0.0, "weight_max": 0.0, "mapping_alpha": 0.0}
    if flat.numel() == 1:
        return {
            "weight_entropy": 0.0,
            "weight_max": float(flat.max().item()),
            "mapping_alpha": 1.0,
        }

    entropy = -(flat * torch.log(torch.clamp(flat, min=1e-12))).sum()
    entropy = entropy / math.log(flat.numel())
    return {
        "weight_entropy": float(entropy.item()),
        "weight_max": float(flat.max().item()),
        "mapping_alpha": float(mapping_alpha_from_weights(flat)),
    }


def prototype_pair(key_tensor, value_tensor, token_scores, *, surrogate_mode: str):
    if surrogate_mode in {"weighted_mean", "asym_key_weighted", "asym_value_weighted"}:
        weights = normalize_token_weights(
            token_scores,
            dtype=key_tensor.dtype,
            device=key_tensor.device,
        )
        weight_view = weights.view(1, 1, -1, 1)
        if surrogate_mode in {"weighted_mean", "asym_key_weighted"}:
            key_proto = (key_tensor * weight_view).sum(dim=2, keepdim=True)
        else:
            key_proto = key_tensor.mean(dim=2, keepdim=True)
        if surrogate_mode in {"weighted_mean", "asym_value_weighted"}:
            value_proto = (value_tensor * weight_view).sum(dim=2, keepdim=True)
        else:
            value_proto = value_tensor.mean(dim=2, keepdim=True)
        return key_proto, value_proto, token_weight_stats(weights)

    return (
        key_tensor.mean(dim=2, keepdim=True),
        value_tensor.mean(dim=2, keepdim=True),
        None,
    )

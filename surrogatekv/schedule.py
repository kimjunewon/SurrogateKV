from __future__ import annotations

import math

import torch


def adaptive_entropy_keep_ratio(*, base_keep_ratio: float, chunk_scores, q_len: int) -> float:
    safe_q_len = max(1, int(q_len))
    min_keep_ratio = 1.0 / safe_q_len
    keep_ratio = min(1.0, max(min_keep_ratio, float(base_keep_ratio)))
    if chunk_scores is None or chunk_scores.numel() <= 0 or chunk_scores.shape[-1] <= 1:
        return keep_ratio

    weights = torch.clamp(chunk_scores.to(dtype=torch.float32), min=1e-6)
    weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
    entropy = -(weights * torch.log(torch.clamp(weights, min=1e-12))).sum(dim=-1)
    entropy = entropy / math.log(weights.shape[-1])
    entropy_mean = float(entropy.mean().item())
    keep_ratio = min_keep_ratio + (keep_ratio - min_keep_ratio) * entropy_mean
    return min(1.0, max(min_keep_ratio, keep_ratio))


def resolve_layerwise_capacity(
    *,
    q_len: int,
    base_capacity_prompt: int,
    layer_idx: int,
    num_hidden_layers: int,
    scheduler: str,
    keep_high: float,
    keep_mid: float,
    keep_low: float,
    r_max: float,
    r_min: float,
):
    safe_q_len = max(1, int(q_len))
    base_ratio = min(1.0, max(1.0 / safe_q_len, float(base_capacity_prompt) / safe_q_len))

    if scheduler == "uniform":
        keep_ratio = base_ratio
    elif scheduler == "three_band":
        derived_high = keep_high if keep_high >= 0.0 else base_ratio
        derived_mid = keep_mid if keep_mid >= 0.0 else base_ratio
        derived_low = keep_low if keep_low >= 0.0 else base_ratio
        lower_end = max(1, num_hidden_layers // 3)
        middle_end = max(lower_end + 1, (2 * num_hidden_layers) // 3) if num_hidden_layers > 1 else 1
        if layer_idx < lower_end:
            keep_ratio = derived_high
        elif layer_idx < middle_end:
            keep_ratio = derived_mid
        else:
            keep_ratio = derived_low
    elif scheduler == "linear_decay":
        max_ratio = r_max if r_max >= 0.0 else base_ratio
        min_ratio = r_min if r_min >= 0.0 else base_ratio
        if num_hidden_layers <= 1:
            keep_ratio = max_ratio
        else:
            progress = float(layer_idx) / float(max(1, num_hidden_layers - 1))
            keep_ratio = max_ratio - progress * (max_ratio - min_ratio)
    elif scheduler == "adaptive_entropy":
        keep_ratio = base_ratio
    else:
        raise ValueError(f"Unsupported SurKV layer scheduler: {scheduler}")

    keep_ratio = min(1.0, max(1.0 / safe_q_len, float(keep_ratio)))
    capacity_prompt = max(1, min(safe_q_len, int(round(safe_q_len * keep_ratio))))
    return capacity_prompt, keep_ratio


def resolve_scheduler_kind(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in {"uniform", "three_band", "linear_decay", "adaptive_entropy"}:
        raise ValueError(f"Unsupported SurKV layer scheduler: {raw}")
    return normalized


def layer_capacity_schedule(
    *,
    num_layers: int,
    prompt_tokens: int,
    base_capacity: int,
    scheduler: str,
    keep_high: float,
    keep_mid: float,
    keep_low: float,
    r_max: float,
    r_min: float,
):
    keep_ratios = []
    capacities = []
    for layer_idx in range(max(1, int(num_layers))):
        capacity, keep_ratio = resolve_layerwise_capacity(
            q_len=prompt_tokens,
            base_capacity_prompt=base_capacity,
            layer_idx=layer_idx,
            num_hidden_layers=max(1, int(num_layers)),
            scheduler=scheduler,
            keep_high=keep_high,
            keep_mid=keep_mid,
            keep_low=keep_low,
            r_max=r_max,
            r_min=r_min,
        )
        keep_ratios.append(keep_ratio)
        capacities.append(capacity)
    return keep_ratios, capacities

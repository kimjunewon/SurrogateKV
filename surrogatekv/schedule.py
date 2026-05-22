from __future__ import annotations

import math

import torch


def normalize_token_weights(token_scores, *, dtype, device):
    weights = torch.clamp(token_scores.to(device=device, dtype=torch.float32), min=1e-6)
    weights = weights / torch.clamp(weights.sum(), min=1e-6)
    return weights.to(dtype=dtype)


def token_weight_summary(token_scores) -> tuple[float, float]:
    if token_scores.numel() <= 0:
        return 0.0, 0.0
    weights = normalize_token_weights(token_scores, dtype=torch.float32, device=token_scores.device)
    max_weight = float(weights.max().item())
    if weights.numel() <= 1:
        return 0.0, max_weight
    entropy = -(weights * torch.log(torch.clamp(weights, min=1e-12))).sum()
    entropy = entropy / math.log(weights.numel())
    return float(entropy.item()), max_weight


def selected_token_weight_stats(*, token_scores, chunk_slices, replace_mask) -> tuple[float | None, float | None]:
    entropies = []
    max_weights = []
    for batch_idx in range(replace_mask.shape[0]):
        selected_indices = torch.nonzero(replace_mask[batch_idx], as_tuple=False).flatten().tolist()
        for chunk_idx in selected_indices:
            start, end = chunk_slices[chunk_idx]
            entropy, max_weight = token_weight_summary(token_scores[batch_idx, start:end])
            entropies.append(entropy)
            max_weights.append(max_weight)
    if not entropies:
        return None, None
    return sum(entropies) / len(entropies), sum(max_weights) / len(max_weights)


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


def _method_key(method: str | None) -> str:
    return str(method or "").strip().lower().replace("_", "-")


def surkv_method_family(method: str | None) -> str:
    key = _method_key(method)
    if key in {"surrogatekv", "surkv", "surrogate-kv", "surrogatekv-snap", "surkv-snap", "surrogate-kv-snap"}:
        return "snap"
    if key in {"surrogatekv-ada", "surkv-ada", "surrogate-kv-ada"}:
        return "ada"
    if key in {
        "surrogatekv-dynamic",
        "surkv-dynamic",
        "surrogate-kv-dynamic",
        "surrogatekv-dynamic-layer",
        "surkv-dynamic-layer",
        "surrogate-kv-dynamic-layer",
        "surkv-layer-dynamic",
        "surrogatekv-layer-dynamic",
    }:
        return "dynamic"
    if key in {"surrogatekv-pyramid", "surkv-pyramid", "surrogate-kv-pyramid"}:
        return "pyramid"
    return "surrogate"


def _profile_window_size(*, family: str, hparam_profile: str, base_capacity: int) -> int:
    profile = str(hparam_profile or "").strip().lower()
    if family in {"dynamic", "pyramid"}:
        window_size = 8
    elif family in {"snap", "ada"} and (
        profile.startswith("official_repo_longbench_external")
        or profile.startswith("paper_original_external")
        or profile in {"niah", "needle", "needle_in_haystack"}
    ):
        window_size = 32
    else:
        window_size = 8
    return max(0, min(int(window_size), max(0, int(base_capacity) - 1)))


def _pyramid_capacity_schedule(
    *,
    num_layers: int,
    prompt_tokens: int,
    base_capacity: int,
    window_size: int,
    beta: int = 20,
) -> list[int]:
    layers = max(1, int(num_layers))
    q_len = max(1, int(prompt_tokens))
    base_capacity = max(1, min(q_len, int(base_capacity)))
    window_size = max(0, min(int(window_size), base_capacity - 1))
    base_nonwindow = max(1, base_capacity - window_size)
    if q_len < base_capacity or q_len < base_nonwindow * 2:
        return [base_capacity for _ in range(layers)]

    min_num = base_nonwindow // max(1, int(beta))
    max_num = base_nonwindow * 2 - min_num
    past_tokens = max(1, q_len - window_size)
    if max_num >= past_tokens:
        max_num = past_tokens
        min_num = base_nonwindow * 2 - max_num

    if layers <= 1:
        nonwindow_caps = [max_num]
    else:
        step = (max_num - min_num) // max(1, layers - 1)
        nonwindow_caps = [max_num - layer_idx * step for layer_idx in range(layers)]
    return [max(1, min(q_len, int(capacity) + window_size)) for capacity in nonwindow_caps]


def method_capacity_profile(
    *,
    method: str | None,
    num_layers: int,
    prompt_tokens: int,
    base_capacity: int,
    scheduler: str,
    keep_high: float,
    keep_mid: float,
    keep_low: float,
    r_max: float,
    r_min: float,
    hparam_profile: str = "",
    requested_pooling: str = "maxpool",
):
    family = surkv_method_family(method)
    layers = max(1, int(num_layers))
    q_len = max(1, int(prompt_tokens))
    base_capacity = max(1, min(q_len, int(base_capacity)))
    scheduler_kind = resolve_scheduler_kind(scheduler)
    window_size = _profile_window_size(
        family=family,
        hparam_profile=hparam_profile,
        base_capacity=base_capacity,
    )

    if family == "pyramid":
        capacities = _pyramid_capacity_schedule(
            num_layers=layers,
            prompt_tokens=q_len,
            base_capacity=base_capacity,
            window_size=window_size,
        )
        keep_ratios = [float(capacity) / float(q_len) for capacity in capacities]
    else:
        keep_ratios, capacities = layer_capacity_schedule(
            num_layers=layers,
            prompt_tokens=q_len,
            base_capacity=base_capacity,
            scheduler=scheduler_kind,
            keep_high=keep_high,
            keep_mid=keep_mid,
            keep_low=keep_low,
            r_max=r_max,
            r_min=r_min,
        )

    return {
        "family": family,
        "scheduler": scheduler_kind,
        "window_sizes": [int(window_size)] * layers,
        "pooling": "avgpool" if family == "dynamic" else str(requested_pooling or "maxpool"),
        "keep_ratios": keep_ratios,
        "capacities": capacities,
    }

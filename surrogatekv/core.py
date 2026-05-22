from __future__ import annotations

import bisect
import heapq
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .registry import METHOD_TO_MODE as SURKV_METHOD_TO_MODE, MODE_TO_SPEC
from .registry_base import MethodSpec
from .schedule import adaptive_entropy_keep_ratio
from .tensor_utils import prototype_pair


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


def _device_key(device: torch.device) -> str:
    index = "" if device.index is None else str(device.index)
    return f"{device.type}:{index}"


def _cache_put(cache: dict, key, value, *, max_size: int = 256):
    if len(cache) >= max_size:
        cache.clear()
    cache[key] = value
    return value


def _dynamic_peak_selection_scores(chunk_scores: torch.Tensor, chunk_max_scores: torch.Tensor) -> torch.Tensor:
    peak_delta = torch.clamp(chunk_max_scores - chunk_scores, min=0)
    scale = peak_delta / torch.clamp(chunk_scores.abs() + peak_delta, min=1e-6)
    return chunk_scores + scale * peak_delta


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


class SurKVCluster:
    _GLOBAL_BUDGET_LEDGER: Dict[str, object] = {
        "enabled": False,
    }
    _GLOBAL_LAYER_DYNAMIC_STATE: Dict[str, object] = {
        "enabled": False,
        "records": [],
    }


    def __init__(
        self,
        *,
        mode: str,
        window_size: int = 8,
        max_capacity_prompt: int = 320,
        kernel_size: int = 5,
        pooling: str = "maxpool",
        chunk_size: int = 32,
        local_radius: int = 1,
        sink_tokens: int = 4,
        layer_keep_ratio: float | None = None,
        layer_scheduler: str = "uniform",
        global_budget_ledger: bool = False,
        global_budget_layer_idx: int = -1,
        global_budget_num_layers: int = 0,
        global_budget_total_capacity: int = 0,
        global_budget_base_capacity: int = 0,
        global_layer_allocator: bool = False,
        score_method: str | None = None,
        head_score_fusion: str | None = None,
    ) -> None:
        self.last_stats = {}
        self.last_layout_meta = None
        self._zero_pair_cache = {}
        self._save_layout_meta = False  # opt-in only; building it forces GPU->CPU syncs.
        self._save_surrogates = False  # opt-in flag for surrogate saving
        self._last_surrogates = {}  # surrogate cache: {f"k_l{l}_h{h}": array, ...}
        self._last_allocator_stats = {}
        self._last_score_stats = {}
        self._last_ada_head_capacities = None
        self._last_fast_pack_plan = None
        self._global_layer_dynamic_finalizing_capacity = None
        self._pending_global_budget_ledger_stats = {}
        self._last_layer_budget_signal = 0.0
        self._set_config(
            mode=mode,
            window_size=window_size,
            max_capacity_prompt=max_capacity_prompt,
            kernel_size=kernel_size,
            pooling=pooling,
            chunk_size=chunk_size,
            local_radius=local_radius,
            sink_tokens=sink_tokens,
            layer_keep_ratio=layer_keep_ratio,
            layer_scheduler=layer_scheduler,
            global_budget_ledger=global_budget_ledger,
            global_budget_layer_idx=global_budget_layer_idx,
            global_budget_num_layers=global_budget_num_layers,
            global_budget_total_capacity=global_budget_total_capacity,
            global_budget_base_capacity=global_budget_base_capacity,
            global_layer_allocator=global_layer_allocator,
            score_method=score_method,
            head_score_fusion=head_score_fusion,
        )

    def reset(
        self,
        *,
        mode: str,
        window_size: int = 8,
        max_capacity_prompt: int = 320,
        kernel_size: int = 5,
        pooling: str = "maxpool",
        chunk_size: int = 32,
        local_radius: int = 1,
        sink_tokens: int = 4,
        layer_keep_ratio: float | None = None,
        layer_scheduler: str = "uniform",
        global_budget_ledger: bool = False,
        global_budget_layer_idx: int = -1,
        global_budget_num_layers: int = 0,
        global_budget_total_capacity: int = 0,
        global_budget_base_capacity: int = 0,
        global_layer_allocator: bool = False,
        score_method: str | None = None,
        head_score_fusion: str | None = None,
    ) -> None:
        self._last_layer_budget_signal = 0.0
        self._last_layer_budget_curve = None
        self._last_fast_pack_plan = None
        self._last_ada_head_capacities = None
        self._set_config(
            mode=mode,
            window_size=window_size,
            max_capacity_prompt=max_capacity_prompt,
            kernel_size=kernel_size,
            pooling=pooling,
            chunk_size=chunk_size,
            local_radius=local_radius,
            sink_tokens=sink_tokens,
            layer_keep_ratio=layer_keep_ratio,
            layer_scheduler=layer_scheduler,
            global_budget_ledger=global_budget_ledger,
            global_budget_layer_idx=global_budget_layer_idx,
            global_budget_num_layers=global_budget_num_layers,
            global_budget_total_capacity=global_budget_total_capacity,
            global_budget_base_capacity=global_budget_base_capacity,
            global_layer_allocator=global_layer_allocator,
            score_method=score_method,
            head_score_fusion=head_score_fusion,
        )

    def _set_config(
        self,
        *,
        mode: str,
        window_size: int,
        max_capacity_prompt: int,
        kernel_size: int,
        pooling: str,
        chunk_size: int,
        local_radius: int,
        sink_tokens: int,
        layer_keep_ratio: float | None,
        layer_scheduler: str,
        global_budget_ledger: bool,
        global_budget_layer_idx: int,
        global_budget_num_layers: int,
        global_budget_total_capacity: int,
        global_budget_base_capacity: int,
        global_layer_allocator: bool,
        score_method: str | None,
        head_score_fusion: str | None,
    ) -> None:
        self.mode = mode
        self.spec: MethodSpec = MODE_TO_SPEC[mode]
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        if self.spec.mode == "surrogate_kv_dynamic_layer" and str(pooling).strip().lower() == "maxpool":
            pooling = "avgpool"
        self.pooling = pooling
        self.chunk_size = chunk_size
        self.local_radius = local_radius
        self.sink_tokens = sink_tokens
        self.neighbor_exclusion_radius = 0
        self.two_surrogate_min_tokens = 48
        self.layer_keep_ratio = None if layer_keep_ratio is None else min(1.0, max(0.0, float(layer_keep_ratio)))
        self.layer_scheduler = layer_scheduler.strip().lower()
        self.global_budget_ledger = bool(global_budget_ledger)
        self.global_budget_layer_idx = int(global_budget_layer_idx)
        self.global_budget_num_layers = max(0, int(global_budget_num_layers))
        self.global_budget_total_capacity = max(0, int(global_budget_total_capacity))
        self.global_budget_base_capacity = max(0, int(global_budget_base_capacity))
        self.global_layer_allocator = bool(global_layer_allocator)
        self.gqa_support = False
        self.num_key_value_groups = 1
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None
        self.cu_qlen = None
        self.cu_head_offset = None
        self.layer_qlens = None
        self.qlen_sum = 0
        raw_score_method = score_method or _SURKV_SCORE_METHOD or getattr(self.spec, "score_method", "attention")
        raw_head_fusion = head_score_fusion or _SURKV_HEAD_SCORE_FUSION or getattr(self.spec, "head_score_fusion", "mean")
        self.score_method = str(raw_score_method or "attention").strip().lower()
        self.head_score_fusion = str(raw_head_fusion or "mean").strip().lower()
        try:
            self.ada_floor_ratio = max(0.0, min(0.95, float(os.environ.get("SURKV_ADA_FLOOR_RATIO", "0.2"))))
        except (TypeError, ValueError):
            self.ada_floor_ratio = 0.2
        self._last_layer_budget_curve = None

    def _ada_exact_query_head_cache_enabled(self) -> bool:
        return _env_flag("SURKV_ADA_EXACT_QUERY_HEADS", True)

    @classmethod
    def reset_global_budget_ledger(
        cls,
        *,
        enabled: bool,
        total_layers: int,
        total_capacity: int,
        prompt_tokens: int,
    ) -> None:
        cls._GLOBAL_BUDGET_LEDGER = {
            "enabled": bool(enabled),
            "total_layers": max(0, int(total_layers)),
            "remaining_layers": max(0, int(total_layers)),
            "total_capacity": max(0, int(total_capacity)),
            "remaining_capacity": max(0, int(total_capacity)),
            "total_base_capacity": max(0, int(total_capacity)),
            "remaining_base_capacity": max(0, int(total_capacity)),
            "prompt_tokens": max(0, int(prompt_tokens)),
            "signal_ema": 0.0,
            "true_dynamic_marginal_ema": 0.0,
            "seen_layers": 0,
        }

    @classmethod
    def reset_global_layer_dynamic(
        cls,
        *,
        enabled: bool,
        total_layers: int,
        total_capacity: int,
        prompt_tokens: int,
    ) -> None:
        cls._GLOBAL_LAYER_DYNAMIC_STATE = {
            "enabled": bool(enabled),
            "total_layers": max(0, int(total_layers)),
            "total_capacity": max(0, int(total_capacity)),
            "prompt_tokens": max(0, int(prompt_tokens)),
            "records": [],
            "finalized": False,
        }

    @classmethod
    def _global_layer_dynamic_active(cls) -> bool:
        state = cls._GLOBAL_LAYER_DYNAMIC_STATE
        return bool(state.get("enabled", False))

    def _register_global_layer_dynamic_record(
        self,
        *,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        token_scores: torch.Tensor,
        q_len: int,
        recent_len: int,
        sink_len: int,
        compressible_start: int,
        past_len: int,
        base_capacity_prompt: int,
        num_key_value_groups: int,
    ) -> None:
        state = type(self)._GLOBAL_LAYER_DYNAMIC_STATE
        if not bool(state.get("enabled", False)):
            return
        layer_idx = int(self.global_budget_layer_idx)
        records = list(state.get("records", []))
        records = [record for record in records if int(record.get("layer_idx", -1)) != layer_idx]
        chunk_slices = [(int(compressible_start), int(past_len))]
        chunk_lengths = torch.tensor(
            [max(0, int(past_len) - int(compressible_start))],
            device=key_states.device,
            dtype=torch.long,
        )
        shadow_signal = 1.0
        shadow_stats: Dict[str, float] = {}
        if str(getattr(self, "mode", "")) == "surrogate_kv_dynamic_layer":
            support_signal, support_stats = self._estimate_global_layer_support_signal(
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                target_compressed_tokens=int(base_capacity_prompt),
                sink_len=int(sink_len),
                recent_len=int(recent_len),
            )
            market_signal, market_stats = self._estimate_global_layer_market_signal(
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                target_compressed_tokens=int(base_capacity_prompt),
                sink_len=int(sink_len),
                recent_len=int(recent_len),
            )
            support_signal = float(support_signal) if math.isfinite(float(support_signal)) else 1.0
            market_signal = float(market_signal) if math.isfinite(float(market_signal)) else 1.0
            shadow_signal = math.sqrt(max(1e-9, support_signal) * max(1e-9, market_signal))
            shadow_stats = dict(support_stats or {})
            shadow_stats.update(market_stats or {})
            shadow_stats.update(
                {
                    "surrogate_kv_layer_support_signal": float(support_signal),
                    "surrogate_kv_layer_surrogate_aware_signal": float(market_signal),
                    "surrogate_kv_layer_combined_signal": float(shadow_signal),
                }
            )
        records.append(
            {
                "cluster": self,
                "layer_idx": layer_idx,
                "mode": str(getattr(self, "mode", "")),
                "key_states": key_states,
                "query_states": query_states,
                "value_states": value_states,
                "token_scores": token_scores,
                "q_len": int(q_len),
                "recent_len": int(recent_len),
                "sink_len": int(sink_len),
                "chunk_slices": chunk_slices,
                "chunk_lengths": chunk_lengths,
                "base_capacity": int(base_capacity_prompt),
                "num_key_value_groups": int(num_key_value_groups),
                "shadow_signal": float(shadow_signal),
                "shadow_stats": dict(shadow_stats),
            }
        )
        records.sort(key=lambda record: int(record.get("layer_idx", -1)))
        state["records"] = records
        state["finalized"] = False
        type(self)._GLOBAL_LAYER_DYNAMIC_STATE = state

    @staticmethod
    def _global_layer_dynamic_capacity_grid(
        *,
        q_len: int,
        min_capacity: int,
        base_capacity: int,
    ) -> List[int]:
        q_len = max(1, int(q_len))
        min_capacity = max(1, min(q_len, int(min_capacity)))
        base_capacity = max(min_capacity, min(q_len, int(base_capacity)))
        multipliers = (
            (1, 3),
            (1, 2),
            (2, 3),
            (4, 5),
            (1, 1),
            (5, 4),
            (3, 2),
            (2, 1),
            (5, 2),
        )
        capacities = {min_capacity, base_capacity, q_len}
        for num, den in multipliers:
            capacities.add(max(min_capacity, min(q_len, int(round(float(base_capacity) * float(num) / float(den))))))
        return sorted(capacities)

    @staticmethod
    def _global_layer_dynamic_objective(stats: Dict[str, object]) -> float:
        for key in (
            "ks_run_objective_value",
            "surrogate_kv_solver_score",
            "surrogate_kv_selected_value",
        ):
            value = stats.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        raw = stats.get("ks_run_raw_value", 0.0)
        sur = stats.get("ks_run_surrogate_projection_value", 0.0)
        try:
            return float(raw) + float(sur)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _evaluate_global_layer_dynamic_curve(cls, record: Dict[str, object]) -> List[Dict[str, float]]:
        cluster = record["cluster"]
        q_len = int(record["q_len"])
        min_capacity = max(1, int(record["recent_len"]) + int(record["sink_len"]) + 1)
        capacities = cls._global_layer_dynamic_capacity_grid(
            q_len=q_len,
            min_capacity=min_capacity,
            base_capacity=int(record["base_capacity"]),
        )
        old_stats = dict(getattr(cluster, "_last_allocator_stats", {}) or {})
        old_pack_plan = getattr(cluster, "_last_fast_pack_plan", None)
        curve: List[Dict[str, float]] = []
        for capacity in capacities:
            cluster._last_allocator_stats = {}
            cluster._last_fast_pack_plan = None
            allocated = cluster._dynamic_surrogate_kv_allocation(
                token_scores=record["token_scores"],
                chunk_slices=record["chunk_slices"],
                chunk_lengths=record["chunk_lengths"],
                target_compressed_tokens=int(capacity),
                sink_len=int(record["sink_len"]),
                recent_len=int(record["recent_len"]),
            )
            stats = dict(getattr(cluster, "_last_allocator_stats", {}) or {})
            utility = cls._global_layer_dynamic_objective(stats)
            if allocated is None:
                utility = 0.0
            actual_entries = stats.get("ks_run_used_entries")
            if actual_entries is None:
                actual_capacity = int(capacity)
            else:
                actual_capacity = int(actual_entries) + int(record["sink_len"]) + int(record["recent_len"])
            actual_capacity = max(min_capacity, min(q_len, int(actual_capacity)))
            curve.append(
                {
                    "capacity": int(actual_capacity),
                    "target_capacity": int(capacity),
                    "utility": float(utility),
                }
            )
        cluster._last_allocator_stats = old_stats
        cluster._last_fast_pack_plan = old_pack_plan
        best_by_capacity: Dict[int, Dict[str, float]] = {}
        for point in curve:
            capacity = int(point["capacity"])
            previous = best_by_capacity.get(capacity)
            if previous is None or float(point["utility"]) > float(previous.get("utility", -math.inf)):
                best_by_capacity[capacity] = dict(point)
        compact = [point for _capacity, point in sorted(best_by_capacity.items())]
        filtered: List[Dict[str, float]] = []
        best_utility = -math.inf
        for point in compact:
            utility = float(point["utility"])
            if utility >= best_utility:
                filtered.append(point)
                best_utility = utility
        return filtered

    @classmethod
    def _select_global_layer_dynamic_capacities(
        cls,
        *,
        records: List[Dict[str, object]],
        total_capacity: int,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[str, float]], int]:
        curves = {
            int(record["layer_idx"]): cls._evaluate_global_layer_dynamic_curve(record)
            for record in records
        }
        selected_index = {layer_idx: 0 for layer_idx in curves}
        selected_cost_capacity = {
            layer_idx: int(points[0]["capacity"]) if points else int(records[idx]["base_capacity"])
            for idx, (layer_idx, points) in enumerate(curves.items())
        }
        selected_capacity = {
            layer_idx: int(points[0].get("target_capacity", points[0]["capacity"])) if points else int(records[idx]["base_capacity"])
            for idx, (layer_idx, points) in enumerate(curves.items())
        }
        used_capacity = sum(int(value) for value in selected_cost_capacity.values())
        remaining = max(0, int(total_capacity) - int(used_capacity))
        heap: List[Tuple[float, int, int, int]] = []
        for layer_idx, points in curves.items():
            if len(points) < 2:
                continue
            cost_delta = int(points[1]["capacity"]) - int(points[0]["capacity"])
            value_delta = float(points[1]["utility"]) - float(points[0]["utility"])
            if cost_delta > 0:
                heapq.heappush(heap, (-value_delta / float(cost_delta), int(layer_idx), 0, 1))
        while heap and remaining > 0:
            neg_density, layer_idx, from_idx, to_idx = heapq.heappop(heap)
            points = curves[layer_idx]
            if int(selected_index[layer_idx]) != int(from_idx):
                continue
            cost_delta = int(points[to_idx]["capacity"]) - int(points[from_idx]["capacity"])
            if cost_delta <= 0:
                continue
            if cost_delta > remaining:
                continue
            selected_index[layer_idx] = int(to_idx)
            selected_cost_capacity[layer_idx] = int(points[to_idx]["capacity"])
            selected_capacity[layer_idx] = int(points[to_idx].get("target_capacity", points[to_idx]["capacity"]))
            remaining -= int(cost_delta)
            next_idx = int(to_idx) + 1
            if next_idx < len(points):
                next_cost = int(points[next_idx]["capacity"]) - int(points[to_idx]["capacity"])
                next_value = float(points[next_idx]["utility"]) - float(points[to_idx]["utility"])
                if next_cost > 0:
                    heapq.heappush(heap, (-next_value / float(next_cost), int(layer_idx), int(to_idx), int(next_idx)))
        selected_points = {
            layer_idx: dict(curves[layer_idx][int(selected_index[layer_idx])])
            for layer_idx in curves
        }
        return selected_capacity, selected_points, remaining

    @staticmethod
    def _select_global_layer_shadow_price_capacities(
        *,
        records: List[Dict[str, object]],
        total_capacity: int,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[str, float]], int]:
        """Allocate layer budgets with one common shadow price.

        Each layer contributes a single scalar demand signal from the same
        SurrogateKV RAW/SUR/DROP market.  Capacity is then water-filled under a
        shared price, so no layer receives budget because it happened to run
        earlier in the stack.
        """
        if not records:
            return {}, {}, int(total_capacity)

        layer_ids = np.asarray([int(record["layer_idx"]) for record in records], dtype=np.int64)
        base = np.asarray([max(1, int(record.get("base_capacity", 1))) for record in records], dtype=np.float64)
        q_lens = np.asarray([max(1, int(record.get("q_len", 1))) for record in records], dtype=np.float64)
        mins = np.asarray(
            [
                max(1, min(int(record.get("q_len", 1)), int(record.get("recent_len", 0)) + int(record.get("sink_len", 0)) + 1))
                for record in records
            ],
            dtype=np.float64,
        )
        maxs = np.maximum(mins, q_lens)
        target_total = int(round(float(total_capacity)))
        target_total = max(int(np.sum(mins)), min(int(np.sum(maxs)), int(target_total)))

        signals = np.asarray(
            [max(1e-9, float(record.get("shadow_signal", 1.0) or 1.0)) for record in records],
            dtype=np.float64,
        )
        log_signal = np.log(np.maximum(signals, 1e-9))
        center = float(np.mean(log_signal)) if log_signal.size else 0.0
        dispersion = float(np.std(log_signal)) if log_signal.size > 1 else 0.0
        elasticity = float(dispersion / (1.0 + dispersion)) if math.isfinite(dispersion) else 0.0
        prices = np.exp(np.clip(float(elasticity) * (log_signal - float(center)), -6.0, 6.0))
        demand = np.maximum(1e-9, base * prices)

        def capacities_for_lambda(lambda_value: float) -> np.ndarray:
            lam = max(1e-9, float(lambda_value))
            return np.minimum(maxs, np.maximum(mins, demand / lam))

        low = 1e-9
        high = max(1e-9, float(np.max(demand / np.maximum(mins, 1.0))))
        while float(np.sum(capacities_for_lambda(high))) > float(target_total) and high < 1e18:
            high *= 2.0
        for _ in range(64):
            mid = (low + high) * 0.5
            if float(np.sum(capacities_for_lambda(mid))) > float(target_total):
                low = mid
            else:
                high = mid

        raw_capacity = capacities_for_lambda(high)
        floors = np.floor(raw_capacity).astype(np.int64)
        min_int = mins.astype(np.int64)
        max_int = maxs.astype(np.int64)
        floors = np.minimum(max_int, np.maximum(min_int, floors))
        remainder = int(target_total) - int(np.sum(floors))
        fractions = raw_capacity - np.floor(raw_capacity)

        if remainder > 0:
            order = np.lexsort((layer_ids, -fractions))
            cursor = 0
            while remainder > 0 and cursor < len(order) * 4:
                idx = int(order[cursor % len(order)])
                if floors[idx] < max_int[idx]:
                    floors[idx] += 1
                    remainder -= 1
                cursor += 1
        elif remainder < 0:
            order = np.lexsort((layer_ids, fractions))
            cursor = 0
            while remainder < 0 and cursor < len(order) * 4:
                idx = int(order[cursor % len(order)])
                if floors[idx] > min_int[idx]:
                    floors[idx] -= 1
                    remainder += 1
                cursor += 1

        selected_capacity = {int(layer_ids[idx]): int(floors[idx]) for idx in range(len(records))}
        selected_points: Dict[int, Dict[str, float]] = {}
        shadow_price = float(high)
        for idx, record in enumerate(records):
            layer_idx = int(layer_ids[idx])
            capacity = int(selected_capacity[layer_idx])
            min_capacity = int(min_int[idx])
            signal = float(signals[idx])
            utility = float(signal * math.log1p(max(0, capacity - min_capacity)))
            selected_points[layer_idx] = {
                "capacity": int(capacity),
                "target_capacity": int(capacity),
                "utility": float(utility),
                "shadow_signal": float(signal),
                "shadow_price": float(shadow_price),
                "shadow_elasticity": float(elasticity),
                "shadow_dispersion": float(dispersion),
                "shadow_demand": float(demand[idx]),
            }
        remaining = int(target_total) - int(sum(selected_capacity.values()))
        return selected_capacity, selected_points, remaining

    @classmethod
    def finalize_global_layer_dynamic_records(cls):
        state = cls._GLOBAL_LAYER_DYNAMIC_STATE
        records = list(state.get("records", []))
        if not bool(state.get("enabled", False)) or not records:
            return []
        total_capacity = int(state.get("total_capacity", 0))
        if total_capacity <= 0:
            total_capacity = sum(int(record.get("base_capacity", 0)) for record in records)
        use_shadow_price = any(str(record.get("mode", "")) == "surrogate_kv_dynamic_layer" for record in records)
        if bool(use_shadow_price):
            selected_capacity, selected_points, remaining = cls._select_global_layer_shadow_price_capacities(
                records=records,
                total_capacity=total_capacity,
            )
        else:
            selected_capacity, selected_points, remaining = cls._select_global_layer_dynamic_capacities(
                records=records,
                total_capacity=total_capacity,
            )
        updates = []
        for record in records:
            cluster = record["cluster"]
            layer_idx = int(record["layer_idx"])
            target_capacity = int(selected_capacity.get(layer_idx, int(record["base_capacity"])))
            cluster._global_layer_dynamic_finalizing_capacity = int(target_capacity)
            try:
                key_out, value_out = cluster.update_kv(
                    record["key_states"],
                    record["query_states"],
                    record["value_states"],
                    None,
                    int(record["num_key_value_groups"]),
                )
            finally:
                cluster._global_layer_dynamic_finalizing_capacity = None
            stats = dict(getattr(cluster, "last_stats", {}) or {})
            point = selected_points.get(layer_idx, {})
            stats.update(
                {
                    "surkv_layer_dynamic_global_allocator": 1,
                    "surkv_layer_dynamic_layer_idx": int(layer_idx),
                    "surkv_layer_dynamic_base_capacity": int(record.get("base_capacity", target_capacity)),
                    "surkv_layer_dynamic_total_capacity": int(total_capacity),
                    "surkv_layer_dynamic_selected_capacity": int(target_capacity),
                    "surkv_layer_dynamic_selected_utility": float(point.get("utility", 0.0)),
                    "surkv_layer_dynamic_curve_capacity": int(point.get("capacity", target_capacity)),
                    "surkv_layer_dynamic_curve_target_capacity": int(point.get("target_capacity", target_capacity)),
                    "surkv_layer_dynamic_global_layers": int(len(selected_points)),
                    "surkv_layer_dynamic_budget_remaining": int(remaining),
                }
            )
            if bool(use_shadow_price):
                stats.update(
                    {
                        "surrogate_kv_dynamic_layer_shadow_allocator": 1,
                        "surrogate_kv_dynamic_layer_shadow_layer_idx": int(layer_idx),
                        "surrogate_kv_dynamic_layer_shadow_base_capacity": int(record.get("base_capacity", target_capacity)),
                        "surrogate_kv_dynamic_layer_shadow_selected_capacity": int(target_capacity),
                        "surrogate_kv_dynamic_layer_shadow_total_capacity": int(total_capacity),
                        "surrogate_kv_dynamic_layer_shadow_signal": float(point.get("shadow_signal", record.get("shadow_signal", 1.0))),
                        "surrogate_kv_dynamic_layer_shadow_price": float(point.get("shadow_price", 0.0)),
                        "surrogate_kv_dynamic_layer_shadow_elasticity": float(point.get("shadow_elasticity", 0.0)),
                        "surrogate_kv_dynamic_layer_shadow_dispersion": float(point.get("shadow_dispersion", 0.0)),
                        "surrogate_kv_dynamic_layer_shadow_demand": float(point.get("shadow_demand", 0.0)),
                    }
                )
                stats.update(dict(record.get("shadow_stats", {}) or {}))
            cluster.last_stats = stats
            updates.append((layer_idx, key_out, value_out))
        state["records"] = []
        state["finalized"] = True
        cls._GLOBAL_LAYER_DYNAMIC_STATE = state
        return updates

    def finalize_global_layer_dynamic(self):
        return type(self).finalize_global_layer_dynamic_records()

    def export_layout_meta(self):
        return self.last_layout_meta

    def enable_layout_meta(self, enable: bool = True):
        self._save_layout_meta = bool(enable)
        if not enable:
            self.last_layout_meta = None

    def export_pd_state(self):
        return {
            "mode": self.mode,
            "stats": dict(self.last_stats or {}),
            "layout_meta": self.last_layout_meta,
        }

    def enable_surrogate_saving(self, enable: bool = True):
        """Enable or disable surrogate saving. Default: False (zero overhead)."""
        self._save_surrogates = bool(enable)
        if not enable:
            self._last_surrogates.clear()

    def get_last_surrogates(self):
        """
        Return last saved surrogates as numpy arrays.
        Format: {f"k_l{l}_h{h}": (num_prompts, num_chunks, head_dim), ...}
        Only populated if enable_surrogate_saving(True) was called.
        """
        if not self._last_surrogates:
            return {}
        import numpy as np
        return {
            key: (
                np.array(val, dtype=np.float32)
                if isinstance(val, list)
                else np.asarray(val, dtype=np.float32)
                if isinstance(val, np.ndarray)
                else val.detach().cpu().numpy().astype(np.float32)
            )
            for key, val in self._last_surrogates.items()
        }

    def _record_saved_surrogate(self, *, batch_idx: int, chunk_idx: int, surrogate_key, surrogate_value):
        if not self._save_surrogates:
            return
        key_name = f"surrogate_k_b{batch_idx}_c{chunk_idx}"
        val_name = f"surrogate_v_b{batch_idx}_c{chunk_idx}"
        self._last_surrogates[key_name] = surrogate_key.detach().cpu().numpy().astype(np.float32)
        self._last_surrogates[val_name] = surrogate_value.detach().cpu().numpy().astype(np.float32)

    def _global_budget_ledger_active(self) -> bool:
        if not self.global_budget_ledger:
            return False
        ledger = type(self)._GLOBAL_BUDGET_LEDGER
        if not bool(ledger.get("enabled", False)):
            return False
        if self.global_budget_num_layers <= 0 or self.global_budget_layer_idx < 0:
            return False
        return True

    def _apply_global_budget_ledger(
        self,
        *,
        q_len: int,
        base_capacity_prompt: int,
        min_capacity_prompt: int,
    ) -> int:
        if not self._global_budget_ledger_active():
            return int(base_capacity_prompt)

        ledger = type(self)._GLOBAL_BUDGET_LEDGER
        remaining_layers = max(1, int(ledger.get("remaining_layers", self.global_budget_num_layers) or 1))
        remaining_capacity = max(1, int(ledger.get("remaining_capacity", 0) or 0))
        remaining_base_capacity = max(1, int(ledger.get("remaining_base_capacity", 0) or 0))
        base_capacity = max(1, int(base_capacity_prompt))
        signal = float(getattr(self, "_last_layer_budget_signal", 0.0) or 0.0)
        if not math.isfinite(signal) or signal <= 0.0:
            signal = 1.0
        keep_frac = max(0.0, min(1.0, float(base_capacity) / max(1.0, float(q_len))))
        compression_pressure = max(0.0, min(1.0, 1.0 - float(keep_frac)))
        pressure_relaxed_ledger = (
            str(getattr(self, "mode", "")) == "surrogate_kv_dynamic_layer"
            and not bool(self.global_layer_allocator)
        )
        ledger_strength = float(compression_pressure * compression_pressure) if bool(pressure_relaxed_ledger) else 1.0

        seen_layers = max(0, int(ledger.get("seen_layers", 0) or 0))
        signal_ema = float(ledger.get("signal_ema", 0.0) or 0.0)
        if not math.isfinite(signal_ema) or signal_ema <= 0.0:
            signal_ema = signal
        relative_pressure = math.log(max(signal, 1e-9) / max(signal_ema, 1e-9))

        future_layers_for_bounds = max(0, int(remaining_layers) - 1)
        hard_min_capacity = max(
            int(min_capacity_prompt),
            int(remaining_capacity) - int(future_layers_for_bounds) * int(q_len),
        )
        if bool(pressure_relaxed_ledger):
            relaxed_min = int(
                round(
                    float(min_capacity_prompt)
                    + (float(hard_min_capacity) - float(min_capacity_prompt)) * float(ledger_strength)
                )
            )
            hard_min_capacity = max(int(min_capacity_prompt), min(int(hard_min_capacity), int(relaxed_min)))
        hard_max_capacity = min(
            int(q_len),
            max(int(hard_min_capacity), int(remaining_capacity) - int(future_layers_for_bounds) * int(min_capacity_prompt)),
        )
        hard_min_capacity = max(1, min(int(q_len), int(hard_min_capacity)))
        hard_max_capacity = max(int(hard_min_capacity), min(int(q_len), int(hard_max_capacity)))

        if remaining_layers <= 1:
            if bool(self.global_layer_allocator):
                target_capacity = remaining_capacity
            else:
                surplus = int(remaining_capacity) - int(base_capacity)
                target_capacity = int(base_capacity) + int(round(float(surplus) * float(ledger_strength)))
        elif (
            bool(self.global_layer_allocator)
            and bool(getattr(self, "global_layer_allocator_curve", False))
            and getattr(self, "_last_layer_budget_curve", None)
        ):
            curve = list(getattr(self, "_last_layer_budget_curve", []) or [])
            future_layers = max(0, int(remaining_layers) - 1)
            future_min = int(future_layers) * int(min_capacity_prompt)
            target_max = max(int(min_capacity_prompt), int(remaining_capacity) - int(future_min))
            target_max = min(int(q_len), int(target_max))
            target_min = max(1, min(int(target_max), int(min_capacity_prompt)))
            valid_points = [
                point
                for point in curve
                if int(target_min) <= int(point.get("capacity", 0)) <= int(target_max)
            ]
            if not valid_points:
                target_capacity = max(int(target_min), min(int(target_max), int(round(float(remaining_capacity) / float(remaining_layers)))))
            else:
                current_signal = float(getattr(self, "_last_layer_budget_signal", 0.0) or 0.0)
                marginal_ema = float(ledger.get("true_dynamic_marginal_ema", 0.0) or 0.0)
                if not math.isfinite(marginal_ema) or marginal_ema <= 0.0:
                    marginal_ema = max(float(current_signal), 1e-9)
                best_point = max(
                    valid_points,
                    key=lambda point: (
                        float(point.get("utility", 0.0)) - float(marginal_ema) * float(point.get("capacity", 0)),
                        float(point.get("utility", 0.0)),
                        -abs(int(point.get("capacity", 0)) - int(round(float(remaining_capacity) / float(remaining_layers)))),
                    ),
                )
                target_capacity = int(best_point.get("capacity", target_min))
                target_capacity = max(int(target_min), min(int(target_max), int(target_capacity)))

                selected_marginal = float(best_point.get("avg_marginal", 0.0) or 0.0)
                if not math.isfinite(selected_marginal) or selected_marginal <= 0.0:
                    selected_marginal = float(current_signal)
                next_marginal_ema = (
                    float(selected_marginal)
                    if seen_layers <= 0 or marginal_ema <= 0.0
                    else 0.75 * float(marginal_ema) + 0.25 * float(selected_marginal)
                )
                ledger["true_dynamic_marginal_ema"] = float(next_marginal_ema)
                ledger_stats_extra = {
                    "surrogate_kv_true_dynamic_layer_allocator": 1,
                    "surrogate_kv_true_dynamic_curve_points": int(len(curve)),
                    "surrogate_kv_true_dynamic_target_min": int(target_min),
                    "surrogate_kv_true_dynamic_target_max": int(target_max),
                    "surrogate_kv_true_dynamic_selected_capacity": int(target_capacity),
                    "surrogate_kv_true_dynamic_selected_utility": float(best_point.get("utility", 0.0)),
                    "surrogate_kv_true_dynamic_selected_marginal": float(selected_marginal),
                    "surrogate_kv_true_dynamic_marginal_ema": float(next_marginal_ema),
                }
                self._last_allocator_stats.update(ledger_stats_extra)
        elif bool(self.global_layer_allocator):
            future_layers = max(0, int(remaining_layers) - 1)
            expected_future_signal = max(float(signal_ema), 1e-9)
            denom = max(1e-9, float(signal) + float(future_layers) * float(expected_future_signal))
            target_capacity = int(round(float(remaining_capacity) * float(signal) / float(denom)))
            future_min = int(future_layers) * int(min_capacity_prompt)
            target_max = int(remaining_capacity) - int(future_min)
            if int(target_max) < int(min_capacity_prompt):
                target_max = int(remaining_capacity)
            target_capacity = max(int(hard_min_capacity), min(int(hard_max_capacity), int(target_capacity)))
        else:
            credit_per_layer = (
                (float(remaining_capacity) - float(remaining_base_capacity))
                / float(remaining_layers)
                * float(ledger_strength)
            )
            # One scalar signal, one bounded online adjustment.  This mirrors
            # DynamicKV's high pooled-attention mass without another global
            # top-k over all layers.
            swing_limit = max(0, int(round(float(base_capacity) * 0.08 * float(ledger_strength))))
            signal_swing = int(round(max(-1.0, min(1.0, relative_pressure)) * float(swing_limit)))
            target_capacity = int(round(float(base_capacity) + credit_per_layer + float(signal_swing)))

            future_base = max(0, int(remaining_base_capacity) - int(base_capacity))
            future_floor = int(round(float(future_base) * 0.88))
            future_ceiling = int(round(float(future_base) * 1.12))
            exact_upper = max(1, int(remaining_capacity) - future_floor)
            exact_lower = int(remaining_capacity) - future_ceiling
            if target_capacity > exact_upper:
                target_capacity = int(round(float(target_capacity) - (float(target_capacity) - float(exact_upper)) * float(ledger_strength)))
            if target_capacity < exact_lower:
                target_capacity = int(round(float(target_capacity) + (float(exact_lower) - float(target_capacity)) * float(ledger_strength)))

        target_capacity = max(int(hard_min_capacity), min(int(hard_max_capacity), int(target_capacity)))
        target_capacity = max(1, min(int(q_len), int(target_capacity)))

        next_ema = signal if seen_layers <= 0 else (0.875 * signal_ema + 0.125 * signal)
        ledger["signal_ema"] = float(next_ema)
        ledger["seen_layers"] = int(seen_layers + 1)
        ledger_stats = {
            "surrogate_kv_global_ledger_enabled": 1,
            "surrogate_kv_global_ledger_layer_idx": int(self.global_budget_layer_idx),
            "surrogate_kv_global_ledger_base_capacity": int(base_capacity),
            "surrogate_kv_global_ledger_planned_capacity": int(target_capacity),
            "surrogate_kv_global_ledger_delta": int(target_capacity) - int(base_capacity),
            "surrogate_kv_global_ledger_signal": float(signal),
            "surrogate_kv_global_ledger_signal_ema": float(next_ema),
            "surrogate_kv_global_ledger_strength": float(ledger_strength),
            "surrogate_kv_global_layer_allocator": int(bool(self.global_layer_allocator)),
            "surrogate_kv_global_ledger_remaining_layers_before": int(remaining_layers),
            "surrogate_kv_global_ledger_remaining_capacity_before": int(remaining_capacity),
        }
        for key, value in dict(getattr(self, "_last_allocator_stats", {}) or {}).items():
            if str(key).startswith("surrogate_kv_true_dynamic_"):
                ledger_stats[key] = value
        for key, value in dict(getattr(self, "_last_allocator_stats", {}) or {}).items():
            if str(key).startswith("surrogate_kv_layer_market_"):
                ledger_stats[key] = value
            elif str(key).startswith("surrogate_kv_layer_support_"):
                ledger_stats[key] = value
            elif str(key).startswith("surrogate_kv_layer_surrogate_aware_"):
                ledger_stats[key] = value
            elif str(key).startswith("surrogate_kv_layer_combined_"):
                ledger_stats[key] = value
        self._pending_global_budget_ledger_stats = dict(ledger_stats)
        self._last_allocator_stats.update(ledger_stats)
        return int(target_capacity)

    def _commit_global_budget_ledger(
        self,
        *,
        base_capacity_prompt: int,
        planned_capacity_prompt: int,
        actual_capacity_prompt: int,
    ) -> None:
        if not self._global_budget_ledger_active():
            return
        ledger = type(self)._GLOBAL_BUDGET_LEDGER
        remaining_layers = max(1, int(ledger.get("remaining_layers", self.global_budget_num_layers) or 1))
        remaining_capacity = int(ledger.get("remaining_capacity", 0) or 0)
        remaining_base_capacity = int(ledger.get("remaining_base_capacity", 0) or 0)
        next_remaining_layers = max(0, remaining_layers - 1)
        next_remaining_capacity = max(0, remaining_capacity - int(actual_capacity_prompt))
        next_remaining_base = max(0, remaining_base_capacity - int(base_capacity_prompt))
        ledger["remaining_layers"] = int(next_remaining_layers)
        ledger["remaining_capacity"] = int(next_remaining_capacity)
        ledger["remaining_base_capacity"] = int(next_remaining_base)
        ledger_stats = dict(getattr(self, "_pending_global_budget_ledger_stats", {}) or {})
        ledger_stats.update(
            {
                "surrogate_kv_global_ledger_actual_capacity": int(actual_capacity_prompt),
                "surrogate_kv_global_ledger_planned_actual_gap": int(actual_capacity_prompt) - int(planned_capacity_prompt),
                "surrogate_kv_global_ledger_remaining_layers_after": int(next_remaining_layers),
                "surrogate_kv_global_ledger_remaining_capacity_after": int(next_remaining_capacity),
            }
        )
        self._last_allocator_stats.update(ledger_stats)

    def _estimate_global_layer_market_signal(
        self,
        *,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        target_compressed_tokens: int,
        sink_len: int,
        recent_len: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Cheap RAW/SUR/DROP marginal market signal for online layer budgeting."""
        if not chunk_slices or token_scores.shape[0] != 1:
            return 1.0, {}
        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return 1.0, {}
        base_start, base_end = span
        if int(base_end) <= int(base_start):
            return 1.0, {}

        micro_len = max(1, int(self.spec.dynamic_anchor_width or (int(self.chunk_size) // 4)))
        atom_start_arr = np.arange(int(base_start), int(base_end), int(micro_len), dtype=np.int64)
        if atom_start_arr.size <= 0:
            return 1.0, {}
        atom_end_arr = np.minimum(atom_start_arr + int(micro_len), int(base_end)).astype(np.int64)
        atom_len_int_arr = (atom_end_arr - atom_start_arr).astype(np.int64)
        scores = token_scores[:, int(base_start) : int(base_end)].detach().to(dtype=torch.float32)
        regular_atoms = int((int(base_end) - int(base_start)) // int(micro_len))
        mean_parts = []
        peak_parts = []
        if regular_atoms > 0:
            regular_tokens = int(regular_atoms) * int(micro_len)
            regular = scores[:, :regular_tokens].reshape(scores.shape[0], int(regular_atoms), int(micro_len))
            mean_parts.append(regular.mean(dim=(0, 2)))
            peak_parts.append(regular.max(dim=2).values.max(dim=0).values)
        if regular_atoms < int(atom_start_arr.size):
            segment = scores[:, int(regular_atoms) * int(micro_len) :]
            mean_parts.append(segment.mean(dim=(0, 1), keepdim=False).view(1))
            peak_parts.append(segment.max(dim=1).values.max(dim=0).values.view(1))
        if not mean_parts:
            return 1.0, {}

        atom_mean = torch.cat(mean_parts, dim=0).view(1, -1)
        atom_peak = torch.cat(peak_parts, dim=0).view(1, -1)
        atom_mean_abs_arr = atom_mean[0].detach().cpu().numpy().astype(np.float64)
        atom_peak_abs_arr = atom_peak[0].detach().cpu().numpy().astype(np.float64)
        mean_rank_t = _rank01(atom_mean)[0]
        peak_rank_t = _rank01(atom_peak)[0]
        atom_risk_t = torch.maximum(mean_rank_t, peak_rank_t)
        mean_risk_arr = mean_rank_t.detach().cpu().numpy().astype(np.float64) + 1e-6
        atom_risk_arr = atom_risk_t.detach().cpu().numpy().astype(np.float64) + 1e-6

        num_atoms = int(atom_start_arr.size)
        budget_entries = max(1, int(target_compressed_tokens) - int(sink_len) - int(recent_len))
        full_cost = int(atom_len_int_arr.sum())
        tail_floor = 1.0 / float(max(2, num_atoms + 1))
        mean_signal_arr = -np.log(np.maximum(float(tail_floor), 1.0 - np.clip(mean_risk_arr, 0.0, 1.0)))
        atom_signal_arr = -np.log(np.maximum(float(tail_floor), 1.0 - np.clip(atom_risk_arr, 0.0, 1.0)))
        atom_len_arr = np.maximum(1.0, atom_len_int_arr.astype(np.float64))
        raw_value_arr = atom_signal_arr * atom_signal_arr * atom_len_arr
        raw_density_arr = raw_value_arr / np.maximum(atom_len_arr, 1.0)
        atom_indices_arr = np.arange(num_atoms, dtype=np.int64)
        raw_drop_order = np.lexsort((atom_indices_arr, atom_risk_arr))
        raw_keep_order = np.lexsort((atom_indices_arr, -atom_risk_arr))

        actions = np.full((num_atoms,), 2, dtype=np.int8)
        current_cost = int(full_cost)
        for atom_idx in raw_drop_order.tolist():
            if current_cost <= budget_entries:
                break
            actions[int(atom_idx)] = 0
            current_cost -= int(atom_len_int_arr[int(atom_idx)])

        kept_density = raw_density_arr[actions == 2]
        dropped_density = raw_density_arr[actions == 0]
        remove_price = float(kept_density.min()) if kept_density.size else 0.0
        add_price = float(dropped_density.max()) if dropped_density.size else remove_price
        region_open_cost = max(float(add_price), float(remove_price))

        prefix_len = np.concatenate(([0], np.cumsum(atom_len_int_arr))).astype(np.int64)
        prefix_mass = np.concatenate(([0.0], np.cumsum(mean_signal_arr * atom_len_arr))).astype(np.float64)
        prefix_energy = np.concatenate(([0.0], np.cumsum(mean_signal_arr * mean_signal_arr * atom_len_arr))).astype(np.float64)

        def run_boundaries(actions_ref: np.ndarray):
            if actions_ref.size <= 0:
                return [], [], []
            changes = np.flatnonzero(actions_ref[1:] != actions_ref[:-1]).astype(np.int64) + 1
            starts = np.concatenate((np.asarray([0], dtype=np.int64), changes))
            ends = np.concatenate((changes, np.asarray([actions_ref.size], dtype=np.int64)))
            return starts.tolist(), ends.tolist(), actions_ref[starts].tolist()

        best_surrogate_bid = 0.0
        candidate_count = 0
        starts, ends, run_actions = run_boundaries(actions)
        for run_idx, run_action in enumerate(run_actions):
            if int(run_action) != 0:
                continue
            if int(run_idx) <= 0 or int(run_idx) + 1 >= len(run_actions):
                continue
            if int(run_actions[int(run_idx) - 1]) != 2 or int(run_actions[int(run_idx) + 1]) != 2:
                continue
            start_idx = int(starts[int(run_idx)])
            end_idx = int(ends[int(run_idx)])
            token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
            if token_len <= 0:
                continue
            mass = float(prefix_mass[end_idx] - prefix_mass[start_idx])
            energy = float(prefix_energy[end_idx] - prefix_energy[start_idx])
            coherent_mass = float(mass * mass / max(1.0, float(token_len)))
            value = max(0.0, min(float(energy), float(2.0 * coherent_mass - energy)))
            gain = float(value) - float(region_open_cost)
            if gain <= 0.0:
                continue
            candidate_count += 1
            best_surrogate_bid = max(float(best_surrogate_bid), float(gain) / max(1.0, float(token_len)))

        abs_mean = float(np.maximum(atom_mean_abs_arr, 0.0).mean()) if atom_mean_abs_arr.size else 0.0
        abs_peak = float(np.maximum(atom_peak_abs_arr, 0.0).mean()) if atom_peak_abs_arr.size else 0.0
        sharpness = float(abs_peak / max(abs_mean, 1e-9)) if abs_peak > 0.0 else 1.0
        score_scale = max(1e-9, float(abs_mean + abs_peak) * (1.0 + math.log1p(max(1.0, float(sharpness)))))
        signal = max(float(add_price), float(best_surrogate_bid), 1e-9) * float(score_scale)
        stats = {
            "surrogate_kv_layer_market_signal": float(signal),
            "surrogate_kv_layer_market_add_price": float(add_price),
            "surrogate_kv_layer_market_remove_price": float(remove_price),
            "surrogate_kv_layer_market_surrogate_bid": float(best_surrogate_bid),
            "surrogate_kv_layer_market_score_scale": float(score_scale),
            "surrogate_kv_layer_market_abs_mean": float(abs_mean),
            "surrogate_kv_layer_market_abs_peak": float(abs_peak),
            "surrogate_kv_layer_market_sharpness": float(sharpness),
            "surrogate_kv_layer_market_candidate_count": int(candidate_count),
            "surrogate_kv_layer_market_budget_entries": int(budget_entries),
            "surrogate_kv_layer_market_full_cost": int(full_cost),
            "surrogate_kv_layer_market_used_entries": int(current_cost),
        }
        return float(signal), stats

    def _estimate_global_layer_support_signal(
        self,
        *,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        target_compressed_tokens: int,
        sink_len: int,
        recent_len: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Cheap DynamicKV-style layer signal from the already computed scores."""
        if not chunk_slices or token_scores.shape[0] != 1:
            return 1.0, {}
        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return 1.0, {}
        base_start, base_end = span
        if int(base_end) <= int(base_start):
            return 1.0, {}

        scores = token_scores[:, int(base_start) : int(base_end)].detach().to(dtype=torch.float32)
        positive = torch.clamp(scores, min=0.0).reshape(-1)
        token_count = int(positive.numel())
        if token_count <= 0:
            return 1.0, {}

        total_mass_t = positive.sum()
        total_mass = float(total_mass_t.item())
        mean_score = float(positive.mean().item())
        peak_score = float(positive.max().item())
        std_score = float(positive.std(unbiased=False).item()) if token_count > 1 else 0.0
        budget_entries = max(1, int(target_compressed_tokens) - int(sink_len) - int(recent_len))
        top_k = max(1, min(int(token_count), int(budget_entries)))
        if total_mass <= 1e-12:
            signal = 1.0
            support_frac = 1.0
            effective_support = float(token_count)
            top_mean = 0.0
            top_mass_frac = 0.0
        else:
            top_values = torch.topk(positive, k=int(top_k), largest=True, sorted=False).values
            top_mean = float(top_values.mean().item())
            top_mass_frac = float((top_values.sum() / torch.clamp(total_mass_t, min=1e-12)).item())
            weights = positive / torch.clamp(total_mass_t, min=1e-12)
            entropy = -torch.sum(weights * torch.log(torch.clamp(weights, min=1e-12)))
            effective_support = float(torch.exp(entropy).item())
            support_frac = max(1.0 / float(token_count), min(1.0, effective_support / float(token_count)))
            evidence_scale = float(top_mean) + 0.5 * float(mean_score) + 0.25 * float(std_score)
            support_scale = math.sqrt(float(support_frac)) * math.log1p(float(effective_support))
            signal = max(1e-9, float(evidence_scale) * float(support_scale))

        sharpness = float(peak_score / max(mean_score, 1e-9)) if peak_score > 0.0 else 1.0
        stats = {
            "surrogate_kv_layer_market_signal": float(signal),
            "surrogate_kv_layer_market_add_price": float(top_mean),
            "surrogate_kv_layer_market_remove_price": float(mean_score),
            "surrogate_kv_layer_market_surrogate_bid": float(support_frac),
            "surrogate_kv_layer_market_score_scale": float(effective_support),
            "surrogate_kv_layer_market_abs_mean": float(mean_score),
            "surrogate_kv_layer_market_abs_peak": float(peak_score),
            "surrogate_kv_layer_market_sharpness": float(sharpness),
            "surrogate_kv_layer_market_candidate_count": int(top_k),
            "surrogate_kv_layer_market_budget_entries": int(budget_entries),
            "surrogate_kv_layer_market_full_cost": int(token_count),
            "surrogate_kv_layer_market_used_entries": int(top_k),
            "surrogate_kv_layer_support_effective_tokens": float(effective_support),
            "surrogate_kv_layer_support_fraction": float(support_frac),
            "surrogate_kv_layer_support_top_mass_fraction": float(top_mass_frac),
        }
        return float(signal), stats

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        update_start = time.perf_counter()
        del attention_mask
        assert key_states.shape[-2] == query_states.shape[-2]
        self.last_layout_meta = None
        self._last_allocator_stats = {}
        self._last_score_stats = {}
        self._last_fast_pack_plan = None
        self._pending_global_budget_ledger_stats = {}
        self._last_layer_budget_signal = 0.0
        self._last_layer_budget_curve = None
        self._headwise_cache_query_repeated = False

        bsz, _, q_len, head_dim = query_states.shape
        timing_breakdown = {
            "score": 0.0,
            "planning": 0.0,
            "prototype": 0.0,
            "packing": 0.0,
        }
        update_fine_timing: Dict[str, float] = {}
        profile_timing = bool(_SURKV_PROFILE_TIMING)

        def record_update_timing(name: str, start_time: float) -> None:
            if not profile_timing:
                return
            key = f"surrogate_kv_timing_update_{name}_seconds"
            update_fine_timing[key] = update_fine_timing.get(key, 0.0) + float(time.perf_counter() - start_time)

        def merge_update_timing() -> None:
            if update_fine_timing:
                stats = dict(self._last_allocator_stats or {})
                stats.update(update_fine_timing)
                self._last_allocator_stats = stats

        layer_dynamic_finalizing_capacity = getattr(self, "_global_layer_dynamic_finalizing_capacity", None)
        deferred_global_layer_mode = (
            self.spec.dynamic_allocator == "surkv_layer_dynamic"
            or str(getattr(self, "mode", "")) == "surrogate_kv_dynamic_layer"
        )
        configured_keep_ratio = min(1.0, float(self.max_capacity_prompt) / max(float(q_len), 1.0))
        if self.layer_keep_ratio is not None:
            configured_keep_ratio = min(1.0, max(1.0 / max(q_len, 1), float(self.layer_keep_ratio)))
        effective_capacity_prompt = max(1, min(q_len, int(round(q_len * configured_keep_ratio))))
        if bool(deferred_global_layer_mode) and layer_dynamic_finalizing_capacity is not None:
            effective_capacity_prompt = max(1, min(q_len, int(layer_dynamic_finalizing_capacity)))
            configured_keep_ratio = min(1.0, float(effective_capacity_prompt) / max(float(q_len), 1.0))
        ledger_base_capacity_prompt = int(effective_capacity_prompt)
        ledger_planned_capacity_prompt = int(effective_capacity_prompt)
        recent_len = min(self.window_size, q_len)
        if q_len <= effective_capacity_prompt or recent_len <= 0:
            self.last_stats = self._stats(
                full_tokens=q_len,
                compressed_tokens=q_len,
                recent_tokens=recent_len,
                selected_chunks=0,
                selected_runs=0,
                num_chunks=0,
                chunk_size=0,
                sink_tokens=0,
                two_surrogate_chunks=0,
                mode_counts={},
                op_seconds=time.perf_counter() - update_start,
                configured_keep_ratio=configured_keep_ratio,
            )
            return key_states, value_states

        past_len = q_len - recent_len
        if past_len <= 0:
            self.last_stats = self._stats(
                full_tokens=q_len,
                compressed_tokens=q_len,
                recent_tokens=recent_len,
                selected_chunks=0,
                selected_runs=0,
                num_chunks=0,
                chunk_size=0,
                sink_tokens=0,
                two_surrogate_chunks=0,
                mode_counts={},
                op_seconds=time.perf_counter() - update_start,
                configured_keep_ratio=configured_keep_ratio,
            )
            return key_states, value_states

        sink_len = min(self._protected_sink_tokens(), past_len)
        compressible_start = sink_len
        compressible_len = past_len - compressible_start
        if compressible_len <= 0:
            self.last_stats = self._stats(
                full_tokens=q_len,
                compressed_tokens=q_len,
                recent_tokens=recent_len,
                selected_chunks=0,
                selected_runs=0,
                num_chunks=0,
                chunk_size=0,
                sink_tokens=sink_len,
                two_surrogate_chunks=0,
                mode_counts={},
                op_seconds=time.perf_counter() - update_start,
                configured_keep_ratio=configured_keep_ratio,
            )
            return key_states, value_states

        budget_past_total = max(1, effective_capacity_prompt - recent_len)
        budget_compressible = max(0, budget_past_total - sink_len)
        tokens_to_save = max(0, compressible_len - budget_compressible)
        adaptive_chunk_size = self._adaptive_chunk_size(
            compressible_len=compressible_len,
            budget_compressible=budget_compressible,
            tokens_to_save=tokens_to_save,
        )
        chunk_slices = [
            (compressible_start + start, compressible_start + end)
            for start, end in self._chunk_slices(compressible_len, adaptive_chunk_size)
        ]

        if budget_compressible >= compressible_len:
            self.last_stats = self._stats(
                full_tokens=q_len,
                compressed_tokens=q_len,
                recent_tokens=recent_len,
                selected_chunks=0,
                selected_runs=0,
                num_chunks=len(chunk_slices),
                chunk_size=adaptive_chunk_size,
                sink_tokens=sink_len,
                two_surrogate_chunks=0,
                mode_counts={},
                op_seconds=time.perf_counter() - update_start,
                configured_keep_ratio=configured_keep_ratio,
            )
            return key_states, value_states

        stage_start = time.perf_counter()
        record_update_timing("pre_score_setup", update_start)
        precomputed_token_scores = getattr(self, "_precomputed_token_scores", None)
        if (
            isinstance(precomputed_token_scores, torch.Tensor)
            and precomputed_token_scores.ndim == 2
            and precomputed_token_scores.shape[0] == int(bsz)
            and precomputed_token_scores.shape[1] == int(past_len)
        ):
            token_scores = precomputed_token_scores.to(device=key_states.device, dtype=torch.float32)
            residual_scores = getattr(self, "_precomputed_surrogate_residual_token_scores", None)
            if (
                isinstance(residual_scores, torch.Tensor)
                and residual_scores.ndim == 2
                and residual_scores.shape[0] == int(bsz)
                and residual_scores.shape[1] == int(past_len)
            ):
                self._last_surrogate_residual_token_scores = residual_scores.to(
                    device=key_states.device,
                    dtype=torch.float32,
                ).detach()
            else:
                self._last_surrogate_residual_token_scores = token_scores.detach()
            score_stats = dict(getattr(self, "_precomputed_score_stats", {}) or {})
            score_stats["surrogate_kv_headwise_precomputed_scores"] = 1
            self._last_score_stats = dict(score_stats)
            self._last_allocator_stats.update(score_stats)
            self._precomputed_token_scores = None
            self._precomputed_surrogate_residual_token_scores = None
            self._precomputed_score_stats = None
        else:
            token_scores = self._past_token_scores(
                key_states=key_states,
                query_states=query_states,
                value_states=value_states,
                recent_len=recent_len,
                past_len=past_len,
                head_dim=head_dim,
                num_key_value_groups=num_key_value_groups,
                base_capacity_prompt=effective_capacity_prompt,
                sink_len=sink_len,
            )
        record_update_timing("score_stage_total", stage_start)
        timing_breakdown["score"] += time.perf_counter() - stage_start
        post_score_start = time.perf_counter()
        if (
            bool(deferred_global_layer_mode)
            and layer_dynamic_finalizing_capacity is None
            and type(self)._global_layer_dynamic_active()
        ):
            self._register_global_layer_dynamic_record(
                key_states=key_states,
                query_states=query_states,
                value_states=value_states,
                token_scores=token_scores,
                q_len=q_len,
                recent_len=recent_len,
                sink_len=sink_len,
                compressible_start=compressible_start,
                past_len=past_len,
                base_capacity_prompt=ledger_base_capacity_prompt,
                num_key_value_groups=num_key_value_groups,
            )
            self._last_allocator_stats.update(
                {
                    "surkv_layer_dynamic_deferred": 1,
                    "surkv_layer_dynamic_base_capacity": int(ledger_base_capacity_prompt),
                    "surkv_layer_dynamic_total_capacity": int(self.global_budget_total_capacity),
                    "surkv_layer_dynamic_layer_idx": int(self.global_budget_layer_idx),
                    "surrogate_kv_dynamic_layer_shadow_deferred": int(
                        str(getattr(self, "mode", "")) == "surrogate_kv_dynamic_layer"
                    ),
                }
            )
            self.last_stats = self._stats(
                full_tokens=q_len,
                compressed_tokens=q_len,
                recent_tokens=recent_len,
                selected_chunks=0,
                selected_runs=0,
                num_chunks=len(chunk_slices),
                chunk_size=adaptive_chunk_size,
                sink_tokens=sink_len,
                two_surrogate_chunks=0,
                mode_counts={},
                op_seconds=time.perf_counter() - update_start,
                configured_keep_ratio=configured_keep_ratio,
                timing_breakdown=timing_breakdown,
            )
            return key_states, value_states
        if self._global_budget_ledger_active() and layer_dynamic_finalizing_capacity is None:
            if bool(self.global_layer_allocator):
                # Keep the dynamic-layer path on the same single-pass allocator.
                # A token-score market signal needs top-k/entropy reductions and
                # Python scalar reads per layer, which makes prefill synchronize
                # with the GPU.  The ledger still reallocates capacity from
                # actual layer usage, but it no longer pays an extra runtime
                # scoring market before planning.
                self._last_layer_budget_curve = None
                self._last_layer_budget_signal = 1.0
                self._last_allocator_stats.update(
                    {
                        "surrogate_kv_layer_market_signal": 1.0,
                        "surrogate_kv_layer_market_add_price": 0.0,
                        "surrogate_kv_layer_market_remove_price": 0.0,
                        "surrogate_kv_layer_market_surrogate_bid": 0.0,
                        "surrogate_kv_layer_market_score_scale": 1.0,
                        "surrogate_kv_layer_market_candidate_count": 0,
                        "surrogate_kv_layer_market_budget_entries": int(
                            max(1, int(ledger_base_capacity_prompt) - int(sink_len) - int(recent_len))
                        ),
                        "surrogate_kv_layer_market_full_cost": int(max(0, int(past_len) - int(sink_len))),
                        "surrogate_kv_layer_market_used_entries": 0,
                    }
                )
            elif str(getattr(self, "mode", "")) == "surrogate_kv_dynamic_layer":
                support_signal, support_stats = self._estimate_global_layer_support_signal(
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    target_compressed_tokens=int(ledger_base_capacity_prompt),
                    sink_len=int(sink_len),
                    recent_len=int(recent_len),
                )
                market_signal, market_stats = self._estimate_global_layer_market_signal(
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    target_compressed_tokens=int(ledger_base_capacity_prompt),
                    sink_len=int(sink_len),
                    recent_len=int(recent_len),
                )
                support_signal = float(support_signal) if math.isfinite(float(support_signal)) else 1.0
                market_signal = float(market_signal) if math.isfinite(float(market_signal)) else 1.0
                combined_signal = math.sqrt(max(1e-9, support_signal) * max(1e-9, market_signal))
                self._last_layer_budget_signal = float(combined_signal)
                combined_stats = dict(support_stats or {})
                combined_stats.update(market_stats or {})
                combined_stats.update(
                    {
                        "surrogate_kv_layer_support_signal": float(support_signal),
                        "surrogate_kv_layer_surrogate_aware_signal": float(market_signal),
                        "surrogate_kv_layer_combined_signal": float(combined_signal),
                    }
                )
                self._last_allocator_stats.update(combined_stats)
            effective_capacity_prompt = self._apply_global_budget_ledger(
                q_len=q_len,
                base_capacity_prompt=ledger_base_capacity_prompt,
                min_capacity_prompt=max(1, recent_len + sink_len + 1),
            )
            ledger_planned_capacity_prompt = int(effective_capacity_prompt)
            configured_keep_ratio = min(1.0, float(effective_capacity_prompt) / max(float(q_len), 1.0))
            budget_past_total = max(1, effective_capacity_prompt - recent_len)
            budget_compressible = max(0, budget_past_total - sink_len)
            tokens_to_save = max(0, compressible_len - budget_compressible)
            adaptive_chunk_size = self._adaptive_chunk_size(
                compressible_len=compressible_len,
                budget_compressible=budget_compressible,
                tokens_to_save=tokens_to_save,
            )
            chunk_slices = [
                (compressible_start + start, compressible_start + end)
                for start, end in self._chunk_slices(compressible_len, adaptive_chunk_size)
            ]
            if budget_compressible >= compressible_len:
                self._commit_global_budget_ledger(
                    base_capacity_prompt=ledger_base_capacity_prompt,
                    planned_capacity_prompt=ledger_planned_capacity_prompt,
                    actual_capacity_prompt=q_len,
                )
                self.last_stats = self._stats(
                    full_tokens=q_len,
                    compressed_tokens=q_len,
                    recent_tokens=recent_len,
                    selected_chunks=0,
                    selected_runs=0,
                    num_chunks=len(chunk_slices),
                    chunk_size=adaptive_chunk_size,
                    sink_tokens=sink_len,
                    two_surrogate_chunks=0,
                    mode_counts={},
                    op_seconds=time.perf_counter() - update_start,
                    configured_keep_ratio=configured_keep_ratio,
                    timing_breakdown=timing_breakdown,
                )
                return key_states, value_states
        record_update_timing("post_score_ledger_stage", post_score_start)
        stage_start = time.perf_counter()
        allocator_owns_regioning = (
            self.spec.dynamic_regioning
            and self.spec.direct_strategy in {"local", "null"}
            and self.spec.selection_strategy == "dynamic"
            and self.spec.dynamic_allocator == "surrogate_kv"
        )
        if self.spec.dynamic_regioning and not allocator_owns_regioning:
            chunk_slices = self._dynamic_region_slices(
                token_scores=token_scores,
                key_states=key_states,
                compressible_start=compressible_start,
                compressible_len=compressible_len,
                max_region_len=adaptive_chunk_size,
            )
        if allocator_owns_regioning:
            # Budget-style allocators rebuild the whole middle span from
            # microchunks internally.  Running the regular dynamic planner first
            # only creates a temporary region layout that is immediately
            # collapsed back to its contiguous span.
            chunk_slices = [(compressible_start, past_len)]
            chunk_mean_scores = token_scores.new_zeros((token_scores.shape[0], 1))
            chunk_max_scores = token_scores.new_zeros((token_scores.shape[0], 1))
        else:
            chunk_mean_scores, chunk_max_scores = self._chunk_statistics_fast_mean_max(
                token_scores=token_scores,
                chunk_slices=chunk_slices,
            )
        record_update_timing("region_setup_stage_total", stage_start)
        timing_breakdown["planning"] += time.perf_counter() - stage_start
        chunk_topk_scores = chunk_max_scores

        if self.layer_scheduler == "adaptive_entropy":
            configured_keep_ratio = adaptive_entropy_keep_ratio(
                base_keep_ratio=configured_keep_ratio,
                chunk_scores=chunk_mean_scores,
                q_len=q_len,
            )
            effective_capacity_prompt = max(1, min(q_len, int(round(q_len * configured_keep_ratio))))
            budget_past_total = max(1, effective_capacity_prompt - recent_len)
            budget_compressible = max(0, budget_past_total - sink_len)
            tokens_to_save = max(0, compressible_len - budget_compressible)
            if budget_compressible >= compressible_len:
                self._commit_global_budget_ledger(
                    base_capacity_prompt=ledger_base_capacity_prompt,
                    planned_capacity_prompt=ledger_planned_capacity_prompt,
                    actual_capacity_prompt=q_len,
                )
                self.last_stats = self._stats(
                    full_tokens=q_len,
                    compressed_tokens=q_len,
                    recent_tokens=recent_len,
                    selected_chunks=0,
                    selected_runs=0,
                    num_chunks=len(chunk_slices),
                    chunk_size=adaptive_chunk_size,
                    sink_tokens=sink_len,
                    two_surrogate_chunks=0,
                    mode_counts={},
                    op_seconds=time.perf_counter() - update_start,
                    configured_keep_ratio=configured_keep_ratio,
                )
                return key_states, value_states
        length_setup_start = time.perf_counter()
        lazy_surrogate_region_tensors = (
            allocator_owns_regioning
            and self.spec.dynamic_allocator == "surrogate_kv"
            and self.layer_scheduler != "adaptive_entropy"
        )
        if lazy_surrogate_region_tensors:
            chunk_lengths = None
            surrogate_lengths = None
        else:
            chunk_lengths = torch.tensor(
                [end - start for start, end in chunk_slices],
                device=key_states.device,
                dtype=torch.long,
            )
            surrogate_lengths = torch.ones(
                (token_scores.shape[0], len(chunk_slices)),
                device=token_scores.device,
                dtype=torch.long,
            )
        record_update_timing("length_tensor_setup", length_setup_start)

        stage_start = time.perf_counter()
        chunk_proto_key_bank = chunk_proto_value_bank = None
        chunk_proto_weight_entropy = chunk_proto_weight_max = None
        chunk_key_distortion = chunk_value_distortion = None
        anchor_mask = anchor_key_bank = anchor_value_bank = None
        anchor_surrogate_key_bank = anchor_surrogate_value_bank = None
        local_key_bank = local_value_bank = local_window_lengths = None
        surkv_anchor_positions = None
        selection_scores = None
        chunk_score_distortion = None
        needs_surkv_guard = self.spec.selection_strategy == "guard" or self.spec.direct_strategy == "surkv"
        needs_surrogateability = self.spec.selection_strategy == "surrogateability"
        needs_peak_aware_selection = self.spec.selection_strategy == "peak_aware"
        needs_dynamic_selection = self.spec.selection_strategy == "dynamic"
        needs_preselection_chunk_bank = (
            needs_surkv_guard
            or needs_surrogateability
            or self.spec.direct_strategy in {"select", "anchor", "surkv"}
        )
        if needs_preselection_chunk_bank:
            (
                chunk_proto_key_bank,
                chunk_proto_value_bank,
                chunk_proto_weight_entropy,
                chunk_proto_weight_max,
                chunk_key_distortion,
                chunk_value_distortion,
            ) = self._chunk_prototype_bank_fast(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                surrogate_mode=self.spec.surrogate_mode,
                return_distortion=needs_surrogateability,
            )
        if needs_surkv_guard:
            if chunk_proto_key_bank is None or chunk_proto_value_bank is None:
                (
                    chunk_proto_key_bank,
                    chunk_proto_value_bank,
                    chunk_proto_weight_entropy,
                    chunk_proto_weight_max,
                    _,
                    _,
                ) = self._chunk_prototype_bank_fast(
                    key_states=key_states,
                    value_states=value_states,
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    surrogate_mode="mean",
                    return_distortion=False,
                )
            local_key_bank, local_value_bank, local_window_lengths = self._local_prototype_bank_fast(
                chunk_key_bank=chunk_proto_key_bank,
                chunk_value_bank=chunk_proto_value_bank,
                chunk_lengths=chunk_lengths,
                return_lengths=True,
            )
            local_key_distortion, local_value_distortion = self._local_distortion_bank_fast(
                key_states=key_states,
                value_states=value_states,
                chunk_slices=chunk_slices,
                local_key_bank=local_key_bank,
                local_value_bank=local_value_bank,
            )
            selection_scores, _ = self._guard_selection_scores(
                chunk_scores=chunk_mean_scores,
                chunk_max_scores=chunk_max_scores,
                local_key_distortion=local_key_distortion,
                local_value_distortion=local_value_distortion,
            )
        elif needs_surrogateability:
            selection_scores, risk_parts = self._surrogateability_selection_scores(
                chunk_scores=chunk_mean_scores,
                chunk_max_scores=chunk_max_scores,
                chunk_key_distortion=chunk_key_distortion,
                chunk_value_distortion=chunk_value_distortion,
            )
            if self.spec.anchor_residual and chunk_proto_key_bank is not None and chunk_proto_value_bank is not None:
                anchor_positions, residual_peakiness = self._chunk_anchor_positions_fast(
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                )
                anchor_key_bank, anchor_value_bank = self._gather_anchor_bank(
                    key_states=key_states,
                    value_states=value_states,
                    anchor_positions=anchor_positions,
                )
                residual_key_bank, residual_value_bank = self._residual_prototype_bank(
                    chunk_proto_key_bank=chunk_proto_key_bank,
                    chunk_proto_value_bank=chunk_proto_value_bank,
                    anchor_key_bank=anchor_key_bank,
                    anchor_value_bank=anchor_value_bank,
                    chunk_lengths=chunk_lengths,
                )
                anchor_mask = self._anchor_candidate_mask(
                    chunk_lengths=chunk_lengths,
                    risk_parts=risk_parts,
                    residual_peakiness=residual_peakiness,
                )
                surrogate_lengths = torch.where(
                    anchor_mask,
                    torch.full_like(surrogate_lengths, 2),
                    surrogate_lengths,
                )
                anchor_mask_view = anchor_mask[:, None, :, None]
                anchor_surrogate_key_bank = torch.where(anchor_mask_view, residual_key_bank, chunk_proto_key_bank)
                anchor_surrogate_value_bank = torch.where(anchor_mask_view, residual_value_bank, chunk_proto_value_bank)
        elif needs_peak_aware_selection:
            selection_scores = _dynamic_peak_selection_scores(chunk_mean_scores, chunk_max_scores)
        elif needs_dynamic_selection:
            if self.spec.dynamic_allocator != "surrogate_kv":
                chunk_score_distortion = self._chunk_score_distortion_fast(
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    chunk_mean_scores=chunk_mean_scores,
                    chunk_max_scores=chunk_max_scores,
                    chunk_lengths=chunk_lengths,
                )
                selection_scores = self._dynamic_region_selection_scores(
                    chunk_scores=chunk_mean_scores,
                    chunk_max_scores=chunk_max_scores,
                    chunk_lengths=chunk_lengths,
                    chunk_score_distortion=chunk_score_distortion,
                )

        # The fast pack metadata used to be built for every method, but the
        # current packing paths only need the chunk slices/lengths directly.
        # Avoiding the repeat_interleave allocation keeps TTFT warm-path closer
        # to the standard fast path for all SurKV variants.
        pack_metadata = None

        dynamic_rate_distortion_allocated = False
        if (
            self.spec.dynamic_regioning
            and self.spec.direct_strategy in {"local", "null"}
            and needs_dynamic_selection
        ):
            allocated = None
            if self.spec.dynamic_allocator == "surrogate_kv":
                alloc_call_start = time.perf_counter()
                allocated = self._dynamic_surrogate_kv_allocation(
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    target_compressed_tokens=effective_capacity_prompt,
                    sink_len=sink_len,
                    recent_len=recent_len,
                )
                record_update_timing("surrogate_allocator_call", alloc_call_start)
                if allocated is not None:
                    stats = dict(self._last_allocator_stats or {})
                    stats.update(
                        {
                            "surrogate_kv_primary_allocator": 1,
                            "surrogate_kv_posthoc_selector": 0,
                        }
                    )
                    self._last_allocator_stats = stats
            else:
                raise ValueError(f"Unsupported SurrogateKV allocator: {self.spec.dynamic_allocator}")
            if allocated is not None:
                post_alloc_start = time.perf_counter()
                (
                    chunk_slices,
                    chunk_lengths,
                    replace_mask,
                    surrogate_lengths,
                ) = allocated
                # Allocator-owned Dynamic methods already made the raw/surrogate
                # decision.  Recomputing per-chunk score stats for thousands of
                # post-allocation regions is only needed by value-light variants.
                if self.spec.dynamic_surrogate_variant in {"light", "light_soft", "peak_light", "csb_light"}:
                    chunk_mean_scores, chunk_max_scores = self._chunk_statistics_fast_mean_max(
                        token_scores=token_scores,
                        chunk_slices=chunk_slices,
                    )
                else:
                    chunk_mean_scores = token_scores.new_zeros((token_scores.shape[0], len(chunk_slices)))
                    chunk_max_scores = token_scores.new_zeros((token_scores.shape[0], len(chunk_slices)))
                chunk_topk_scores = chunk_max_scores
                dynamic_rate_distortion_allocated = True
                record_update_timing("allocated_postprocess", post_alloc_start)

        if not dynamic_rate_distortion_allocated:
            if chunk_lengths is None or surrogate_lengths is None:
                fallback_tensor_start = time.perf_counter()
                chunk_lengths = torch.tensor(
                    [end - start for start, end in chunk_slices],
                    device=key_states.device,
                    dtype=torch.long,
                )
                surrogate_lengths = torch.ones(
                    (token_scores.shape[0], len(chunk_slices)),
                    device=token_scores.device,
                    dtype=torch.long,
                )
                record_update_timing("lazy_fallback_tensor_setup", fallback_tensor_start)
            low_select_start = time.perf_counter()
            replace_mask = self._select_low_importance_chunks(
                chunk_scores=chunk_mean_scores,
                chunk_max_scores=chunk_max_scores,
                chunk_lengths=chunk_lengths,
                surrogate_lengths=surrogate_lengths,
                tokens_to_save=tokens_to_save,
                selection_scores=selection_scores,
            )
            record_update_timing("fallback_low_importance_select", low_select_start)
        should_budget_fill_dynamic = (
            self.spec.dynamic_regioning
            and self.spec.direct_strategy == "local"
            and not dynamic_rate_distortion_allocated
        )
        if should_budget_fill_dynamic and replace_mask.any():
            budget_fill_start = time.perf_counter()
            (
                chunk_slices,
                chunk_lengths,
                replace_mask,
                surrogate_lengths,
            ) = self._budget_fill_dynamic_selected_regions(
                chunk_slices=chunk_slices,
                chunk_lengths=chunk_lengths,
                replace_mask=replace_mask,
                surrogate_lengths=surrogate_lengths,
                selection_scores=selection_scores,
                token_scores=token_scores,
                target_compressed_tokens=effective_capacity_prompt,
                sink_len=sink_len,
                recent_len=recent_len,
            )
            if self.spec.dynamic_surrogate_variant in {"light", "light_soft", "peak_light", "csb_light"}:
                chunk_mean_scores, chunk_max_scores = self._chunk_statistics_fast_mean_max(
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                )
            else:
                chunk_mean_scores = token_scores.new_zeros((token_scores.shape[0], len(chunk_slices)))
                chunk_max_scores = token_scores.new_zeros((token_scores.shape[0], len(chunk_slices)))
            chunk_topk_scores = chunk_max_scores
            record_update_timing("budget_fill_dynamic", budget_fill_start)

        record_update_timing("dynamic_planning_stage_total", stage_start)
        timing_breakdown["planning"] += time.perf_counter() - stage_start
        if not replace_mask.any():
            self._commit_global_budget_ledger(
                base_capacity_prompt=ledger_base_capacity_prompt,
                planned_capacity_prompt=ledger_planned_capacity_prompt,
                actual_capacity_prompt=q_len,
            )
            merge_update_timing()
            self.last_stats = self._stats(
                full_tokens=q_len,
                compressed_tokens=q_len,
                recent_tokens=recent_len,
                selected_chunks=0,
                selected_runs=0,
                num_chunks=len(chunk_slices),
                chunk_size=adaptive_chunk_size,
                sink_tokens=sink_len,
                two_surrogate_chunks=0,
                mode_counts={},
                op_seconds=time.perf_counter() - update_start,
                configured_keep_ratio=configured_keep_ratio,
                timing_breakdown=timing_breakdown,
            )
            return key_states, value_states

        stage_start = time.perf_counter()
        runtime_global_prototypes = self._needs_runtime_global_prototypes(replace_mask=replace_mask)
        if (self.spec.direct_strategy in {"weighted", "select", "anchor"} or runtime_global_prototypes) and chunk_proto_key_bank is None:
            prototype_mode = self.spec.surrogate_mode if self.spec.direct_strategy == "weighted" else "mean"
            prototype_call_start = time.perf_counter()
            (
                chunk_proto_key_bank,
                chunk_proto_value_bank,
                chunk_proto_weight_entropy,
                chunk_proto_weight_max,
                chunk_key_distortion,
                chunk_value_distortion,
            ) = self._chunk_prototype_bank_fast(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                surrogate_mode=prototype_mode,
                return_distortion=False,
            )
            record_update_timing("runtime_chunk_prototype_bank", prototype_call_start)

        if self.spec.direct_strategy == "surkv":
            (
                surrogate_lengths,
                surkv_anchor_positions,
                anchor_key_bank,
                anchor_value_bank,
                anchor_mask,
            ) = self._surkv_anchor_plan(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                chunk_lengths=chunk_lengths,
                replace_mask=replace_mask,
                surrogate_lengths=surrogate_lengths,
                tokens_to_save=tokens_to_save,
            )
            if local_key_bank is None or local_value_bank is None or local_window_lengths is None:
                if chunk_proto_key_bank is None or chunk_proto_value_bank is None:
                    (
                        chunk_proto_key_bank,
                        chunk_proto_value_bank,
                        _,
                        _,
                        _,
                        _,
                    ) = self._chunk_prototype_bank_fast(
                        key_states=key_states,
                        value_states=value_states,
                        token_scores=token_scores,
                        chunk_slices=chunk_slices,
                        surrogate_mode="mean",
                        return_distortion=False,
                    )
                local_key_bank, local_value_bank, local_window_lengths = self._local_prototype_bank_fast(
                    chunk_key_bank=chunk_proto_key_bank,
                    chunk_value_bank=chunk_proto_value_bank,
                    chunk_lengths=chunk_lengths,
                    return_lengths=True,
                )
            if anchor_key_bank is not None and anchor_value_bank is not None and anchor_mask is not None:
                local_key_bank, local_value_bank = self._residual_local_bank_from_anchors(
                    local_key_bank=local_key_bank,
                    local_value_bank=local_value_bank,
                    local_window_lengths=local_window_lengths,
                    anchor_key_bank=anchor_key_bank,
                    anchor_value_bank=anchor_value_bank,
                    anchor_counts=anchor_mask.to(dtype=local_key_bank.dtype),
                )

        if self.spec.direct_strategy == "local":
            if chunk_proto_key_bank is None or chunk_proto_value_bank is None:
                if self.spec.dynamic_regioning:
                    prototype_call_start = time.perf_counter()
                    (
                        chunk_proto_key_bank,
                        chunk_proto_value_bank,
                        _,
                        _,
                        _,
                        _,
                    ) = self._dynamic_micro_prototype_bank(
                        key_states=key_states,
                        value_states=value_states,
                        token_scores=token_scores,
                        chunk_slices=chunk_slices,
                        surrogate_mode=self.spec.surrogate_mode,
                        peak_mode=self.spec.dynamic_surrogate_variant,
                        selected_only_mask=(replace_mask & (surrogate_lengths > 0)),
                        active_slice_indices=(
                            getattr(self, "_last_fast_pack_plan", {}) or {}
                        ).get("surrogate_chunk_indices_list"),
                    )
                    record_update_timing("dynamic_micro_prototype_bank", prototype_call_start)
                else:
                    prototype_call_start = time.perf_counter()
                    (
                        chunk_proto_key_bank,
                        chunk_proto_value_bank,
                        _,
                        _,
                        _,
                        _,
                    ) = self._chunk_prototype_bank_fast(
                        key_states=key_states,
                        value_states=value_states,
                        token_scores=token_scores,
                        chunk_slices=chunk_slices,
                        surrogate_mode=self.spec.surrogate_mode,
                        return_distortion=False,
                    )
                    record_update_timing("local_chunk_prototype_bank", prototype_call_start)
            # Each replaced region is represented by its own local prototype;
            # no cross-region mixing is used in the default path.
            local_key_bank, local_value_bank = chunk_proto_key_bank, chunk_proto_value_bank
            if self.spec.dynamic_surrogate_variant in {"light", "light_soft", "peak_light", "csb_light"}:
                prototype_call_start = time.perf_counter()
                local_value_bank = self._dynamic_light_value_bank(
                    surrogate_value_bank=local_value_bank,
                    chunk_scores=chunk_mean_scores,
                    replace_mask=replace_mask,
                    mode=self.spec.dynamic_surrogate_variant,
                )
                record_update_timing("dynamic_light_value_bank", prototype_call_start)

        global_key = global_value = None
        if runtime_global_prototypes:
            prototype_call_start = time.perf_counter()
            global_key, global_value = self._global_prototypes(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                replace_mask=replace_mask,
                fallback_start=compressible_start,
                fallback_end=past_len,
                chunk_scores=chunk_mean_scores,
                chunk_proto_key_bank=None,
                chunk_proto_value_bank=None,
            )
            record_update_timing("global_prototypes", prototype_call_start)

        record_update_timing("prototype_stage_total", stage_start)
        timing_breakdown["prototype"] += time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        compressed_keys = []
        compressed_values = []
        selected_chunks_per_batch = []
        selected_runs_per_batch = []
        two_surrogate_chunks_per_batch = []
        mode_counts_per_batch = []
        layout_meta_per_batch = []
        weighted_entropy_values = []
        weighted_max_values = []
        mapping_alpha_values = []

        for batch_idx in range(bsz):
            compress_call_start = time.perf_counter()
            if self.spec.direct_strategy == "surkv":
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                    batch_weight_stats,
                ) = self._compress_surkv_fast_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    sink_len=sink_len,
                    past_len=past_len,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                    surrogate_key_bank=local_key_bank,
                    surrogate_value_bank=local_value_bank,
                    anchor_positions_by_batch=surkv_anchor_positions,
                )
            elif self.spec.direct_strategy == "anchor":
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                    batch_weight_stats,
                ) = self._compress_anchor_fast_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    sink_len=sink_len,
                    past_len=past_len,
                    pack_metadata=pack_metadata,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                    surrogate_key_bank=anchor_surrogate_key_bank,
                    surrogate_value_bank=anchor_surrogate_value_bank,
                    anchor_mask=anchor_mask,
                    anchor_key_bank=anchor_key_bank,
                    anchor_value_bank=anchor_value_bank,
                )
            elif self.spec.direct_strategy == "select":
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                    batch_weight_stats,
                ) = self._compress_banked_fast_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    sink_len=sink_len,
                    past_len=past_len,
                    pack_metadata=pack_metadata,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                    surrogate_key_bank=chunk_proto_key_bank,
                    surrogate_value_bank=chunk_proto_value_bank,
                    mode_name="select",
                )
            elif self.spec.direct_strategy == "weighted":
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                    batch_weight_stats,
                ) = self._compress_banked_fast_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    sink_len=sink_len,
                    past_len=past_len,
                    pack_metadata=pack_metadata,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                    surrogate_key_bank=chunk_proto_key_bank,
                    surrogate_value_bank=chunk_proto_value_bank,
                    mode_name="weighted",
                    chunk_proto_weight_entropy=chunk_proto_weight_entropy,
                    chunk_proto_weight_max=chunk_proto_weight_max,
                )
            elif self.spec.direct_strategy == "local":
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                    batch_weight_stats,
                ) = self._compress_banked_fast_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    sink_len=sink_len,
                    past_len=past_len,
                    pack_metadata=pack_metadata,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                    surrogate_key_bank=local_key_bank,
                    surrogate_value_bank=local_value_bank,
                    mode_name="local",
                )
            elif self.spec.null_fastpath:
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                    batch_weight_stats,
                ) = self._compress_null_fast_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    sink_len=sink_len,
                    past_len=past_len,
                    pack_metadata=pack_metadata,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                )
            else:
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                    batch_weight_stats,
                ) = self._compress_fast_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    chunk_mean_scores=chunk_mean_scores,
                    global_key=global_key,
                    global_value=global_value,
                    sink_len=sink_len,
                    past_len=past_len,
                    pack_metadata=pack_metadata,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                )
            record_update_timing(f"compress_{self.spec.direct_strategy}_batch", compress_call_start)

            compressed_keys.append(compressed_batch_key)
            compressed_values.append(compressed_batch_value)
            selected_chunks_per_batch.append(selected_runs)
            selected_runs_per_batch.append(selected_runs)
            two_surrogate_chunks_per_batch.append(two_surrogate_chunks)
            mode_counts_per_batch.append(batch_mode_counts)
            layout_meta_per_batch.append(batch_layout_meta)
            weighted_entropy_values.extend(
                stat["weight_entropy"] for stat in batch_weight_stats if stat is not None and stat.get("weight_entropy") is not None
            )
            weighted_max_values.extend(
                stat["weight_max"] for stat in batch_weight_stats if stat is not None and stat.get("weight_max") is not None
            )
            mapping_alpha_values.extend(
                stat["mapping_alpha"] for stat in batch_weight_stats if stat is not None and stat.get("mapping_alpha") is not None
            )

        cat_start = time.perf_counter()
        compressed_key_states = torch.cat(compressed_keys, dim=0)
        compressed_value_states = torch.cat(compressed_values, dim=0)
        record_update_timing("compressed_cat", cat_start)
        record_update_timing("packing_stage_total", stage_start)
        timing_breakdown["packing"] += time.perf_counter() - stage_start
        stats_commit_start = time.perf_counter()
        dynamic_region_lengths = [int(end) - int(start) for start, end in chunk_slices] if self.spec.dynamic_regioning else []
        self._commit_global_budget_ledger(
            base_capacity_prompt=ledger_base_capacity_prompt,
            planned_capacity_prompt=ledger_planned_capacity_prompt,
            actual_capacity_prompt=int(compressed_key_states.shape[-2]),
        )
        record_update_timing("pre_stats_commit", stats_commit_start)
        merge_update_timing()
        stats_build_start = time.perf_counter()
        self.last_stats = self._stats(
            full_tokens=q_len,
            compressed_tokens=compressed_key_states.shape[-2],
            recent_tokens=recent_len,
            selected_chunks=max(selected_chunks_per_batch) if selected_chunks_per_batch else 0,
            selected_runs=max(selected_runs_per_batch) if selected_runs_per_batch else 0,
            num_chunks=len(chunk_slices),
            chunk_size=adaptive_chunk_size,
            sink_tokens=sink_len,
            two_surrogate_chunks=max(two_surrogate_chunks_per_batch) if two_surrogate_chunks_per_batch else 0,
            mode_counts=self._merge_mode_counts(mode_counts_per_batch),
            op_seconds=time.perf_counter() - update_start,
            weighted_entropy=(
                float(sum(weighted_entropy_values) / len(weighted_entropy_values))
                if weighted_entropy_values
                else None
            ),
            weighted_max=(
                float(sum(weighted_max_values) / len(weighted_max_values))
                if weighted_max_values
                else None
            ),
            mapping_alpha=(
                float(sum(mapping_alpha_values) / len(mapping_alpha_values))
                if mapping_alpha_values
                else None
            ),
            configured_keep_ratio=configured_keep_ratio,
            dynamic_region_mean_len=(
                float(sum(dynamic_region_lengths) / len(dynamic_region_lengths))
                if dynamic_region_lengths
                else None
            ),
            dynamic_region_max_len=max(dynamic_region_lengths) if dynamic_region_lengths else None,
            dynamic_region_count=len(dynamic_region_lengths) if dynamic_region_lengths else None,
            timing_breakdown=timing_breakdown,
        )
        self.last_stats["surrogate_kv_timing_update_final_stats_build_seconds"] = float(
            time.perf_counter() - stats_build_start
        )
        self.last_stats["surrogate_kv_timing_update_total_seconds"] = float(time.perf_counter() - update_start)
        self.last_layout_meta = layout_meta_per_batch
        return compressed_key_states, compressed_value_states

    def _compress_banked_fast_batch(
        self,
        *,
        batch_idx: int,
        key_states,
        value_states,
        chunk_slices,
        chunk_lengths,
        replace_mask,
        sink_len: int,
        past_len: int,
        pack_metadata,
        surrogate_lengths,
        surrogate_key_bank,
        surrogate_value_bank,
        mode_name: str,
        chunk_proto_weight_entropy=None,
        chunk_proto_weight_max=None,
    ):
        recent_key = key_states[batch_idx : batch_idx + 1, :, past_len:, :]
        recent_value = value_states[batch_idx : batch_idx + 1, :, past_len:, :]
        fast_pack_plan = getattr(self, "_last_fast_pack_plan", None)
        use_fast_pack_plan = (
            batch_idx == 0
            and isinstance(fast_pack_plan, dict)
            and tuple(chunk_slices) == fast_pack_plan.get("chunk_slices")
        )
        selected_chunk_mask = None
        output_chunk_lengths = None
        if use_fast_pack_plan:
            selected_mask_list = fast_pack_plan["selected_mask_list"]
            output_length_list = fast_pack_plan["output_length_list"]
            selected_chunk_indices_list = fast_pack_plan["selected_chunk_indices_list"]
        else:
            selected_chunk_mask = replace_mask[batch_idx]
            output_chunk_lengths = torch.where(
                selected_chunk_mask,
                surrogate_lengths,
                chunk_lengths,
            )
            selected_mask_list = [bool(v) for v in selected_chunk_mask.detach().cpu().tolist()]
            output_length_list = [int(v) for v in output_chunk_lengths.detach().cpu().tolist()]
            selected_chunk_indices_list = [idx for idx, selected in enumerate(selected_mask_list) if selected]
        total_tokens = int(sink_len) + int(sum(output_length_list)) + int(recent_key.shape[2])

        if (
            not self._save_layout_meta
            and not self._save_surrogates
            and chunk_proto_weight_entropy is None
            and chunk_proto_weight_max is None
        ):
            if use_fast_pack_plan:
                raw_spans = fast_pack_plan["raw_spans"]
            else:
                raw_spans = []
                for chunk_idx, (start, end) in enumerate(chunk_slices):
                    if not selected_mask_list[chunk_idx]:
                        raw_spans.append((int(start), int(end)))

            pieces_key = []
            pieces_value = []
            if sink_len > 0:
                pieces_key.append(key_states[batch_idx : batch_idx + 1, :, :sink_len, :])
                pieces_value.append(value_states[batch_idx : batch_idx + 1, :, :sink_len, :])

            raw_index_tensor = self._span_index_tensor(raw_spans, device=key_states.device)
            if raw_index_tensor is not None and raw_index_tensor.numel() > 0:
                pieces_key.append(key_states[batch_idx : batch_idx + 1].index_select(2, raw_index_tensor))
                pieces_value.append(value_states[batch_idx : batch_idx + 1].index_select(2, raw_index_tensor))

            selected_runs = len(selected_chunk_indices_list)
            if use_fast_pack_plan:
                selected_lengths_list = [output_length_list[idx] for idx in selected_chunk_indices_list]
                surrogate_chunk_indices_list = fast_pack_plan["surrogate_chunk_indices_list"]
                surrogate_lengths_list = fast_pack_plan["surrogate_lengths_list"]
            else:
                selected_lengths_list = [output_length_list[idx] for idx in selected_chunk_indices_list]
                surrogate_chunk_indices_list = [
                    idx for idx in selected_chunk_indices_list if int(output_length_list[idx]) > 0
                ]
                surrogate_lengths_list = [output_length_list[idx] for idx in surrogate_chunk_indices_list]
            if len(surrogate_chunk_indices_list) > 0:
                surrogate_indices = torch.tensor(
                    surrogate_chunk_indices_list,
                    device=key_states.device,
                    dtype=torch.long,
                )
                if any(length != 1 for length in surrogate_lengths_list):
                    repeat_counts = torch.tensor(
                        surrogate_lengths_list,
                        device=key_states.device,
                        dtype=torch.long,
                    )
                    surrogate_indices = surrogate_indices.repeat_interleave(repeat_counts)
                pieces_key.append(surrogate_key_bank[batch_idx : batch_idx + 1].index_select(2, surrogate_indices))
                pieces_value.append(surrogate_value_bank[batch_idx : batch_idx + 1].index_select(2, surrogate_indices))

            recent_width = recent_key.shape[2]
            if recent_width > 0:
                pieces_key.append(recent_key)
                pieces_value.append(recent_value)

            compressed_key = torch.cat(pieces_key, dim=2) if pieces_key else key_states.new_empty(
                (1, key_states.shape[1], 0, key_states.shape[-1])
            )
            compressed_value = torch.cat(pieces_value, dim=2) if pieces_value else value_states.new_empty(
                (1, value_states.shape[1], 0, value_states.shape[-1])
            )
            two_surrogate_chunks = sum(max(0, int(length) - 1) for length in selected_lengths_list)
            surrogate_run_count = sum(1 for length in selected_lengths_list if int(length) > 0)
            drop_run_count = sum(1 for length in selected_lengths_list if int(length) <= 0)
            if selected_runs > 0:
                batch_mode_counts = {}
                if surrogate_run_count > 0:
                    batch_mode_counts[mode_name] = surrogate_run_count
                if drop_run_count > 0:
                    batch_mode_counts["drop"] = drop_run_count
            else:
                batch_mode_counts = {}
            return (
                compressed_key,
                compressed_value,
                batch_mode_counts,
                two_surrogate_chunks,
                selected_runs,
                None,
                [],
            )

        compressed_key = key_states.new_empty(
            (1, key_states.shape[1], max(0, total_tokens), key_states.shape[-1])
        )
        compressed_value = value_states.new_empty(
            (1, value_states.shape[1], max(0, total_tokens), value_states.shape[-1])
        )
        cursor = 0

        def append_piece(key_piece, value_piece):
            nonlocal cursor
            width = int(key_piece.shape[2])
            if width <= 0:
                return
            compressed_key[:, :, cursor : cursor + width, :].copy_(key_piece)
            compressed_value[:, :, cursor : cursor + width, :].copy_(value_piece)
            cursor += width

        if sink_len > 0:
            append_piece(
                key_states[batch_idx : batch_idx + 1, :, :sink_len, :],
                value_states[batch_idx : batch_idx + 1, :, :sink_len, :],
            )

        selected_runs = len(selected_chunk_indices_list)
        if selected_runs > 0:
            selected_lengths_list = [output_length_list[idx] for idx in selected_chunk_indices_list]
            surrogate_run_count = sum(1 for length in selected_lengths_list if int(length) > 0)
            drop_run_count = sum(1 for length in selected_lengths_list if int(length) <= 0)
            batch_mode_counts = {}
            if surrogate_run_count > 0:
                batch_mode_counts[mode_name] = surrogate_run_count
            if drop_run_count > 0:
                batch_mode_counts["drop"] = drop_run_count
            batch_weight_stats = []
            for chunk_idx, (start, end) in enumerate(chunk_slices):
                packed_len = output_length_list[chunk_idx]
                if selected_mask_list[chunk_idx]:
                    surrogate_key = surrogate_key_bank[batch_idx : batch_idx + 1, :, chunk_idx : chunk_idx + 1, :]
                    surrogate_value = surrogate_value_bank[batch_idx : batch_idx + 1, :, chunk_idx : chunk_idx + 1, :]
                    expanded_key = surrogate_key.expand(-1, -1, packed_len, -1)
                    expanded_value = surrogate_value.expand(-1, -1, packed_len, -1)
                    append_piece(expanded_key, expanded_value)
                    self._record_saved_surrogate(
                        batch_idx=batch_idx,
                        chunk_idx=chunk_idx,
                        surrogate_key=expanded_key,
                        surrogate_value=expanded_value,
                    )
                    if chunk_proto_weight_entropy is not None and chunk_proto_weight_max is not None:
                        batch_weight_stats.append(
                            {
                                "weight_entropy": float(chunk_proto_weight_entropy[batch_idx, chunk_idx].item()),
                                "weight_max": float(chunk_proto_weight_max[batch_idx, chunk_idx].item()),
                            }
                        )
                else:
                    append_piece(
                        key_states[batch_idx : batch_idx + 1, :, start:end, :],
                        value_states[batch_idx : batch_idx + 1, :, start:end, :],
                    )
            two_surrogate_chunks = sum(max(0, int(length) - 1) for length in selected_lengths_list)
        else:
            batch_mode_counts = {}
            batch_weight_stats = []
            two_surrogate_chunks = 0
            for chunk_idx, (start, end) in enumerate(chunk_slices):
                append_piece(
                    key_states[batch_idx : batch_idx + 1, :, start:end, :],
                    value_states[batch_idx : batch_idx + 1, :, start:end, :],
                )

        recent_width = recent_key.shape[2]
        if recent_width > 0:
            append_piece(recent_key, recent_value)
        if cursor != int(total_tokens):
            compressed_key = compressed_key[:, :, :cursor, :]
            compressed_value = compressed_value[:, :, :cursor, :]
        chunk_mode_names = [mode_name if selected else None for selected in selected_mask_list]
        if selected_chunk_mask is None:
            selected_chunk_mask = torch.as_tensor(selected_mask_list, device=key_states.device, dtype=torch.bool)
        if output_chunk_lengths is None:
            output_chunk_lengths = torch.as_tensor(
                output_length_list,
                device=key_states.device,
                dtype=chunk_lengths.dtype,
            )
        batch_layout_meta = self._build_layout_meta(
            full_tokens=int(key_states.shape[2]),
            compressed_tokens=total_tokens,
            sink_len=sink_len,
            recent_len=int(recent_key.shape[2]),
            chunk_lengths=chunk_lengths,
            selected_chunk_mask=selected_chunk_mask,
            output_chunk_lengths=output_chunk_lengths,
            chunk_mode_names=chunk_mode_names,
        )
        return compressed_key, compressed_value, batch_mode_counts, two_surrogate_chunks, selected_runs, batch_layout_meta, batch_weight_stats

    def _headwise_norm_rms_prototypes_from_spans_batch(
        self,
        *,
        key_states,
        value_states,
        head_scores=None,
        head_spans: Sequence[Sequence[Tuple[int, int]]],
        base_start: int,
        base_end: int,
        micro_len: int,
    ):
        head_dim = int(key_states.shape[-1])
        key_heads = int(key_states.shape[1])
        empty_key = key_states.new_empty((1, 1, 0, head_dim))
        empty_value = value_states.new_empty((1, 1, 0, head_dim))
        if not head_spans:
            return []
        if all(not spans for spans in head_spans):
            return [(empty_key, empty_value) for _ in head_spans]

        base_start = int(base_start)
        base_end = int(base_end)
        micro_len = max(1, int(micro_len))
        span_len = max(0, int(base_end) - int(base_start))
        if span_len <= 0:
            return [(empty_key, empty_value) for _ in head_spans]

        proto_mode = str(
            os.environ.get("SURKV_HEADWISE_SURROGATE_PROTO", "peak")
            or "peak"
        ).strip().lower()
        if (
            proto_mode in {"peak", "token", "representative", "rep"}
            and isinstance(head_scores, torch.Tensor)
            and head_scores.ndim == 2
            and int(head_scores.shape[0]) >= int(key_heads)
            and int(head_scores.shape[1]) >= int(base_end)
        ):
            outputs = [(empty_key, empty_value) for _ in head_spans]
            score_source = head_scores.detach().to(device=key_states.device, dtype=torch.float32)
            flat_heads = []
            starts = []
            ends = []
            counts = [0 for _ in head_spans]
            max_width = 1
            for head_idx, spans in enumerate(head_spans):
                if not spans:
                    continue
                for start, end in spans:
                    start_i = max(int(base_start), min(int(base_end), int(start)))
                    end_i = max(start_i + 1, min(int(base_end), int(end)))
                    flat_heads.append(int(head_idx))
                    starts.append(int(start_i))
                    ends.append(int(end_i))
                    counts[int(head_idx)] += 1
                    max_width = max(int(max_width), int(end_i) - int(start_i))
            if not starts:
                return outputs
            heads_t = torch.as_tensor(flat_heads, device=key_states.device, dtype=torch.long)
            starts_t = torch.as_tensor(starts, device=key_states.device, dtype=torch.long)
            ends_t = torch.as_tensor(ends, device=key_states.device, dtype=torch.long)
            widths_t = (ends_t - starts_t).clamp_min(1)
            offsets = torch.arange(int(max_width), device=key_states.device, dtype=torch.long)
            gather_indices = starts_t.view(-1, 1) + offsets.view(1, -1)
            valid = offsets.view(1, -1) < widths_t.view(-1, 1)
            safe_indices = gather_indices.clamp(max=int(base_end) - 1)
            local_scores = score_source[heads_t.view(-1, 1), safe_indices]
            local_scores = local_scores.masked_fill(~valid, torch.finfo(local_scores.dtype).min)
            index_t = starts_t + local_scores.argmax(dim=-1).to(dtype=torch.long)
            proto_key_flat = key_states[0, heads_t, index_t, :]
            proto_value_flat = value_states[0, heads_t, index_t, :]
            cursor = 0
            for head_idx, count in enumerate(counts):
                count = int(count)
                if count <= 0:
                    continue
                next_cursor = int(cursor) + int(count)
                outputs[int(head_idx)] = (
                    proto_key_flat[int(cursor) : int(next_cursor)].reshape(1, 1, int(count), head_dim),
                    proto_value_flat[int(cursor) : int(next_cursor)].reshape(1, 1, int(count), head_dim),
                )
                cursor = int(next_cursor)
            return outputs

        regular_micro = int(span_len) // int(micro_len)
        regular_tokens = int(regular_micro) * int(micro_len)
        micro_keys = []
        micro_values = []
        micro_lengths = []
        if regular_micro > 0:
            key_regular = key_states[:, :, int(base_start) : int(base_start) + int(regular_tokens), :].reshape(
                1,
                int(key_heads),
                int(regular_micro),
                int(micro_len),
                head_dim,
            )
            value_regular = value_states[:, :, int(base_start) : int(base_start) + int(regular_tokens), :].reshape(
                1,
                int(key_heads),
                int(regular_micro),
                int(micro_len),
                head_dim,
            )
            micro_keys.append(key_regular.mean(dim=3))
            micro_values.append(value_regular.mean(dim=3))
            micro_lengths.append(
                torch.full(
                    (int(regular_micro),),
                    int(micro_len),
                    device=key_states.device,
                    dtype=torch.float32,
                )
            )
        tail_len = int(span_len) - int(regular_tokens)
        if tail_len > 0:
            tail_start = int(base_start) + int(regular_tokens)
            micro_keys.append(key_states[:, :, int(tail_start) : int(base_end), :].mean(dim=2, keepdim=True))
            micro_values.append(value_states[:, :, int(tail_start) : int(base_end), :].mean(dim=2, keepdim=True))
            micro_lengths.append(torch.full((1,), int(tail_len), device=key_states.device, dtype=torch.float32))
        if not micro_keys:
            return [(empty_key, empty_value) for _ in head_spans]

        micro_key_bank = torch.cat(micro_keys, dim=2).to(dtype=torch.float32)
        micro_value_bank = torch.cat(micro_values, dim=2).to(dtype=torch.float32)
        micro_len_bank = torch.cat(micro_lengths, dim=0).to(dtype=torch.float32)
        max_micro_idx = int(micro_key_bank.shape[2])

        weights = micro_len_bank.view(1, 1, -1, 1)
        weighted_keys = micro_key_bank * weights
        weighted_values = micro_value_bank * weights
        zero_key = weighted_keys.new_zeros((1, int(key_heads), 1, head_dim))
        zero_value = weighted_values.new_zeros((1, int(key_heads), 1, head_dim))
        key_prefix = torch.cat([zero_key, weighted_keys.cumsum(dim=2)], dim=2)
        value_prefix = torch.cat([zero_value, weighted_values.cumsum(dim=2)], dim=2)
        length_prefix = torch.cat([micro_len_bank.new_zeros((1,)), micro_len_bank.cumsum(dim=0)], dim=0)

        key_norm_prefix = None
        if self.spec.surrogate_mode in _NORM_RESTORED_KEY_MODES:
            micro_key_norm = micro_key_bank.norm(dim=-1)
            weighted_key_norm = micro_key_norm * micro_len_bank.view(1, 1, -1)
            zero_norm = weighted_key_norm.new_zeros((1, int(key_heads), 1))
            key_norm_prefix = torch.cat([zero_norm, weighted_key_norm.cumsum(dim=2)], dim=2)

        value_norm_prefix = None
        if self.spec.surrogate_mode in _RMS_RESTORED_VALUE_MODES:
            micro_value_norm = micro_value_bank.norm(dim=-1)
            weighted_value_norm = micro_value_norm * micro_len_bank.view(1, 1, -1)
            zero_norm = weighted_value_norm.new_zeros((1, int(key_heads), 1))
            value_norm_prefix = torch.cat([zero_norm, weighted_value_norm.cumsum(dim=2)], dim=2)

        outputs = []
        for head_idx, spans in enumerate(head_spans):
            if not spans:
                outputs.append((empty_key, empty_value))
                continue
            starts = []
            ends = []
            for start, end in spans:
                start_i = max(int(base_start), min(int(base_end), int(start)))
                end_i = max(start_i, min(int(base_end), int(end)))
                start_micro = max(0, min(max_micro_idx, (start_i - int(base_start)) // int(micro_len)))
                end_micro = int(math.ceil(float(end_i - int(base_start)) / float(micro_len)))
                end_micro = max(start_micro + 1, min(max_micro_idx, int(end_micro)))
                starts.append(int(start_micro))
                ends.append(int(end_micro))
            starts_t = torch.as_tensor(starts, device=key_states.device, dtype=torch.long)
            ends_t = torch.as_tensor(ends, device=key_states.device, dtype=torch.long)
            denom = (length_prefix.index_select(0, ends_t) - length_prefix.index_select(0, starts_t)).clamp_min(1e-6)

            head_key_prefix = key_prefix[:, int(head_idx) : int(head_idx) + 1]
            head_value_prefix = value_prefix[:, int(head_idx) : int(head_idx) + 1]
            proto_key = (
                head_key_prefix.index_select(2, ends_t) - head_key_prefix.index_select(2, starts_t)
            ) / denom.view(1, 1, -1, 1)
            proto_value = (
                head_value_prefix.index_select(2, ends_t) - head_value_prefix.index_select(2, starts_t)
            ) / denom.view(1, 1, -1, 1)

            if key_norm_prefix is not None:
                head_key_norm_prefix = key_norm_prefix[:, int(head_idx) : int(head_idx) + 1]
                target_key_norm = (
                    head_key_norm_prefix.index_select(2, ends_t)
                    - head_key_norm_prefix.index_select(2, starts_t)
                ) / denom.view(1, 1, -1)
                current_key_norm = proto_key.norm(dim=-1).clamp_min(1e-6)
                key_scale = _safe_key_norm_scale(
                    target_norm=target_key_norm,
                    current_norm=current_key_norm,
                )
                proto_key = proto_key * key_scale.view(1, 1, -1, 1)

            if value_norm_prefix is not None:
                head_value_norm_prefix = value_norm_prefix[:, int(head_idx) : int(head_idx) + 1]
                target_value_norm = (
                    head_value_norm_prefix.index_select(2, ends_t)
                    - head_value_norm_prefix.index_select(2, starts_t)
                ) / denom.view(1, 1, -1)
                current_value_norm = proto_value.norm(dim=-1).clamp_min(1e-6)
                proto_value = proto_value * (target_value_norm / current_value_norm).view(1, 1, -1, 1)

            outputs.append((proto_key.to(dtype=key_states.dtype), proto_value.to(dtype=value_states.dtype)))
        return outputs

    def _update_kv_headwise_ada_overlay(
        self,
        *,
        key_states,
        value_states,
        precomputed_head_scores,
        precomputed_residual_scores,
        key_head_caps,
        groups: int,
        recent_len: int,
        past_len: int,
        sink_len: int,
        configured_keep_ratio: float,
        update_start: float,
        score_seconds: float,
        exact_query_heads: bool,
        original_key_heads: int,
        original_groups: int,
        gqa_capacity_fusion: str,
    ):
        selected_support = getattr(self, "_last_ada_selected_support", None)
        if (
            int(groups) != 1
            or not isinstance(selected_support, torch.Tensor)
            or not isinstance(precomputed_head_scores, torch.Tensor)
            or precomputed_head_scores.ndim != 3
            or precomputed_head_scores.shape[0] != 1
            or int(precomputed_head_scores.shape[2]) < int(past_len)
        ):
            return None

        bsz = int(key_states.shape[0])
        key_heads = int(key_states.shape[1])
        q_len = int(key_states.shape[2])
        head_dim = int(key_states.shape[-1])
        if bsz != 1 or int(precomputed_head_scores.shape[1]) < int(key_heads):
            return None
        if (
            selected_support.ndim != 3
            or selected_support.shape[0] != 1
            or int(selected_support.shape[1]) < int(key_heads)
            or int(selected_support.shape[2]) < int(past_len)
        ):
            return None

        timing_breakdown = {
            "score": float(score_seconds),
            "planning": 0.0,
            "prototype": 0.0,
            "packing": 0.0,
        }
        plan_start = time.perf_counter()
        raw_scores = precomputed_head_scores[0, : int(key_heads), : int(past_len)].detach().to(
            device=key_states.device,
            dtype=torch.float32,
        )
        if (
            isinstance(precomputed_residual_scores, torch.Tensor)
            and precomputed_residual_scores.ndim == 3
            and precomputed_residual_scores.shape[0] == 1
            and int(precomputed_residual_scores.shape[1]) >= int(key_heads)
            and int(precomputed_residual_scores.shape[2]) >= int(past_len)
        ):
            residual_scores = precomputed_residual_scores[0, : int(key_heads), : int(past_len)].detach().to(
                device=key_states.device,
                dtype=torch.float32,
            )
        else:
            residual_scores = raw_scores

        caps = key_head_caps.to(device=key_states.device, dtype=torch.long).clamp(min=0, max=int(past_len))
        raw_mask = selected_support[0, : int(key_heads), : int(past_len)].detach().to(
            device=key_states.device,
            dtype=torch.bool,
        ).clone()
        if int(sink_len) > 0:
            raw_mask[:, : int(sink_len)] = True

        # Keep the exact Ada per-head ledger.  In the normal case Ada's support
        # already has the right count, so avoid 32 Python/GPU syncs per layer.
        raw_counts = raw_mask.sum(dim=1).to(dtype=torch.long)
        if bool(torch.any(raw_counts != caps).detach().cpu().item()):
            caps_cpu = caps.detach().to(device="cpu", dtype=torch.long).tolist()
            counts_cpu = raw_counts.detach().to(device="cpu", dtype=torch.long).tolist()
            for head_idx in range(int(key_heads)):
                cap = int(caps_cpu[int(head_idx)])
                count = int(counts_cpu[int(head_idx)])
                if count == cap:
                    continue
                head_mask = raw_mask[int(head_idx)]
                if count > cap:
                    removable = head_mask.clone()
                    if int(sink_len) > 0:
                        removable[: int(sink_len)] = False
                    remove_count = int(count) - int(cap)
                    removable_idx = torch.nonzero(removable, as_tuple=False).flatten()
                    if remove_count > 0 and removable_idx.numel() > 0:
                        remove_count = min(int(remove_count), int(removable_idx.numel()))
                        remove_scores = raw_scores[int(head_idx)].index_select(0, removable_idx)
                        remove_pos = torch.topk(-remove_scores, k=int(remove_count), largest=True, sorted=False).indices
                        head_mask[removable_idx.index_select(0, remove_pos)] = False
                else:
                    add_count = int(cap) - int(count)
                    addable = ~head_mask
                    if int(past_len) > 0:
                        addable[int(past_len) :] = False
                    add_idx = torch.nonzero(addable, as_tuple=False).flatten()
                    if add_count > 0 and add_idx.numel() > 0:
                        add_count = min(int(add_count), int(add_idx.numel()))
                        add_scores = raw_scores[int(head_idx)].index_select(0, add_idx)
                        add_pos = torch.topk(add_scores, k=int(add_count), largest=True, sorted=False).indices
                        head_mask[add_idx.index_select(0, add_pos)] = True

        raw_mask_np = raw_mask.detach().cpu().numpy().astype(np.bool_, copy=True)
        caps_np = caps.detach().cpu().numpy().astype(np.int64, copy=False)
        actions_np = np.where(raw_mask_np, 2, 0).astype(np.int8)

        token_positions = torch.arange(int(past_len), device=key_states.device)
        exchange_region = token_positions.unsqueeze(0) >= int(sink_len)
        raw_exchange_mask = raw_mask & exchange_region
        drop_exchange_mask = (~raw_mask) & exchange_region
        raw_score_pos = torch.clamp(raw_scores, min=0.0)
        residual_score_pos = torch.clamp(residual_scores, min=0.0)
        inf = torch.tensor(float("inf"), device=key_states.device, dtype=torch.float32)
        neg_inf = torch.tensor(float("-inf"), device=key_states.device, dtype=torch.float32)
        marginal_raw_t = raw_score_pos.masked_fill(~raw_exchange_mask, inf).amin(dim=1)
        drop_max_t = residual_score_pos.masked_fill(~drop_exchange_mask, neg_inf).amax(dim=1)
        active_exchange_heads_t = (drop_max_t > marginal_raw_t) & torch.isfinite(marginal_raw_t)
        active_exchange_head_idx_t = torch.nonzero(active_exchange_heads_t, as_tuple=False).flatten().to(dtype=torch.long)
        active_exchange_heads = active_exchange_head_idx_t.detach().to(device="cpu", dtype=torch.long).numpy()
        if int(active_exchange_head_idx_t.numel()) > 0:
            raw_scores_np = raw_score_pos.index_select(0, active_exchange_head_idx_t).detach().cpu().numpy()
            residual_np = residual_score_pos.index_select(0, active_exchange_head_idx_t).detach().cpu().numpy()
            marginal_raw_np = marginal_raw_t.index_select(0, active_exchange_head_idx_t).detach().cpu().numpy()
        else:
            raw_scores_np = None
            residual_np = None
            marginal_raw_np = None

        head_surrogate_spans: List[List[Tuple[int, int]]] = [[] for _ in range(int(key_heads))]
        removed_raw_by_head: List[List[int]] = [[] for _ in range(int(key_heads))]
        accepted_surrogates = np.zeros((int(key_heads),), dtype=np.int64)
        selected_gain_total = 0.0
        sold_raw_total = 0.0
        for active_row, head_idx in enumerate(active_exchange_heads.astype(np.int64).tolist()):
            cap = int(caps_np[int(head_idx)])
            if cap <= 0:
                actions_np[int(head_idx), :] = 0
                continue
            row_raw = actions_np[int(head_idx)] == 2
            if int(row_raw.sum()) <= 0:
                continue

            drop_region = actions_np[int(head_idx)] == 0
            if int(sink_len) > 0:
                drop_region[: int(sink_len)] = False
            padded = np.concatenate(
                (
                    np.asarray([False], dtype=np.bool_),
                    drop_region[: int(past_len)],
                    np.asarray([False], dtype=np.bool_),
                )
            )
            run_starts = np.flatnonzero((~padded[:-1]) & padded[1:]).astype(np.int64)
            run_ends = np.flatnonzero(padded[:-1] & (~padded[1:])).astype(np.int64)
            candidates = []
            if run_starts.size > 0:
                row_residual = residual_np[int(active_row)]
                positive = np.maximum(row_residual.astype(np.float64, copy=False), 0.0)
                signal = np.sqrt(positive)
                signal_prefix = np.concatenate(
                    (np.asarray([0.0], dtype=np.float64), np.cumsum(signal, dtype=np.float64))
                )
                energy_prefix = np.concatenate(
                    (np.asarray([0.0], dtype=np.float64), np.cumsum(positive, dtype=np.float64))
                )
                lengths = (run_ends - run_starts).astype(np.float64)
                mass = signal_prefix[run_ends] - signal_prefix[run_starts]
                energy = energy_prefix[run_ends] - energy_prefix[run_starts]
                coherent_mass = (mass * mass) / np.maximum(lengths, 1.0)
                coherent = np.maximum(0.0, np.minimum(energy, 2.0 * coherent_mass - energy))
                values = coherent / np.maximum(lengths, 1.0)
                valid = values > float(marginal_raw_np[int(active_row)])
                if bool(valid.any()):
                    valid_pos = np.flatnonzero(valid)
                    order = np.lexsort((-lengths[valid_pos], -values[valid_pos]))
                    for pos in valid_pos[order].tolist():
                        candidates.append(
                            (
                                float(values[int(pos)]),
                                int(run_ends[int(pos)] - run_starts[int(pos)]),
                                int(run_starts[int(pos)]),
                                int(run_ends[int(pos)]),
                            )
                        )
            if not candidates:
                continue

            removable = np.flatnonzero(actions_np[int(head_idx)] == 2).astype(np.int64)
            if int(sink_len) > 0:
                removable = removable[removable >= int(sink_len)]
            if removable.size <= 0:
                continue
            removable_scores = raw_scores_np[int(active_row), removable]
            removable = removable[np.lexsort((removable, removable_scores))].tolist()
            remove_cursor = 0
            for value, _length, start, end in candidates:
                while remove_cursor < len(removable) and actions_np[int(head_idx), int(removable[remove_cursor])] != 2:
                    remove_cursor += 1
                if remove_cursor >= len(removable):
                    break
                raw_idx = int(removable[remove_cursor])
                marginal_raw = float(raw_scores_np[int(active_row), int(raw_idx)])
                if float(value) <= float(marginal_raw):
                    break
                actions_np[int(head_idx), int(start) : int(end)] = 1
                actions_np[int(head_idx), int(raw_idx)] = 0
                head_surrogate_spans[int(head_idx)].append((int(start), int(end)))
                removed_raw_by_head[int(head_idx)].append(int(raw_idx))
                remove_cursor += 1
                accepted_surrogates[int(head_idx)] += 1
                selected_gain_total += float(value) - float(marginal_raw)
                sold_raw_total += float(marginal_raw)

            # Guard against roundoff/repair issues: never exceed Ada's cap.
            used = int((actions_np[int(head_idx)] == 2).sum())
            surrogate_runs = 0
            surrogate_positions = actions_np[int(head_idx)] == 1
            if bool(surrogate_positions.any()):
                padded_sur = np.concatenate(
                    (
                        np.asarray([False], dtype=np.bool_),
                        surrogate_positions,
                        np.asarray([False], dtype=np.bool_),
                    )
                )
                surrogate_runs = int(np.count_nonzero((~padded_sur[:-1]) & padded_sur[1:]))
            overflow = int(used + surrogate_runs) - int(cap)
            if overflow > 0:
                removable_raw = np.flatnonzero(actions_np[int(head_idx)] == 2).astype(np.int64)
                if int(sink_len) > 0:
                    removable_raw = removable_raw[removable_raw >= int(sink_len)]
                if removable_raw.size > 0:
                    removable_scores = raw_scores_np[int(active_row), removable_raw]
                    order = removable_raw[np.lexsort((removable_raw, removable_scores))]
                    for idx in order[: int(overflow)].tolist():
                        actions_np[int(head_idx), int(idx)] = 0
                        removed_raw_by_head[int(head_idx)].append(int(idx))

        for spans in head_surrogate_spans:
            if len(spans) > 1:
                spans.sort(key=lambda item: (int(item[0]), int(item[1])))
        child_mode_counts: List[Dict[str, int]] = []
        child_selected_runs: List[int] = []
        child_surrogate_slots: List[int] = []
        child_chunks: List[int] = []
        row_actions = actions_np[:, : int(past_len)]
        if row_actions.shape[1] > 0:
            raw_region_counts = (row_actions[:, 0] == 2).astype(np.int64)
            drop_region_counts = (row_actions[:, 0] == 0).astype(np.int64)
            if row_actions.shape[1] > 1:
                prev_actions = row_actions[:, :-1]
                next_actions = row_actions[:, 1:]
                raw_region_counts += np.count_nonzero((next_actions == 2) & (prev_actions != 2), axis=1)
                drop_region_counts += np.count_nonzero((next_actions == 0) & (prev_actions != 0), axis=1)
        else:
            raw_region_counts = np.zeros((int(key_heads),), dtype=np.int64)
            drop_region_counts = np.zeros((int(key_heads),), dtype=np.int64)
        raw_regions_per_head = raw_region_counts.astype(np.int64).tolist()
        drop_regions_per_head = drop_region_counts.astype(np.int64).tolist()
        chunk_count = int(
            max(1, int(math.ceil(max(0, int(past_len) - int(sink_len)) / float(max(1, int(self.spec.dynamic_anchor_width or 4))))))
        )
        for head_idx in range(int(key_heads)):
            spans = head_surrogate_spans[int(head_idx)]
            raw_regions = int(raw_region_counts[int(head_idx)])
            drop_regions = int(drop_region_counts[int(head_idx)])
            local_counts: Dict[str, int] = {}
            if spans:
                local_counts["local"] = int(len(spans))
            if int(drop_regions) > 0:
                local_counts["drop"] = int(drop_regions)
            child_mode_counts.append(local_counts)
            child_selected_runs.append(int(raw_regions) + int(len(spans)))
            child_surrogate_slots.append(int(len(spans)))
            child_chunks.append(int(chunk_count))

        timing_breakdown["planning"] = float(time.perf_counter() - plan_start)

        proto_start = time.perf_counter()
        surrogate_banks = self._headwise_norm_rms_prototypes_from_spans_batch(
            key_states=key_states,
            value_states=value_states,
            head_scores=raw_scores,
            head_spans=head_surrogate_spans,
            base_start=0,
            base_end=int(past_len),
            micro_len=max(1, int(self.spec.dynamic_anchor_width or 4)),
        )
        timing_breakdown["prototype"] = float(time.perf_counter() - proto_start)

        pack_start = time.perf_counter()
        raw_token_mask = raw_mask.clone()
        for head_idx, removed in enumerate(removed_raw_by_head):
            if removed:
                raw_token_mask[int(head_idx), torch.as_tensor(removed, device=key_states.device, dtype=torch.long)] = False
        flat_keys = []
        flat_values = []
        head_lens: List[int] = []
        for head_idx, bank_entry in enumerate(surrogate_banks):
            pieces_key = []
            pieces_value = []
            raw_indices = torch.nonzero(raw_token_mask[int(head_idx)], as_tuple=False).flatten().to(dtype=torch.long)
            if raw_indices.numel() > 0:
                pieces_key.append(key_states[:, int(head_idx) : int(head_idx) + 1].index_select(2, raw_indices))
                pieces_value.append(value_states[:, int(head_idx) : int(head_idx) + 1].index_select(2, raw_indices))
            proto_key, proto_value = bank_entry
            if proto_key.numel() > 0:
                pieces_key.append(proto_key)
                pieces_value.append(proto_value)
            if int(recent_len) > 0:
                pieces_key.append(key_states[:, int(head_idx) : int(head_idx) + 1, int(past_len) :, :])
                pieces_value.append(value_states[:, int(head_idx) : int(head_idx) + 1, int(past_len) :, :])
            head_key = torch.cat(pieces_key, dim=2) if pieces_key else key_states.new_empty((1, 1, 0, head_dim))
            head_value = torch.cat(pieces_value, dim=2) if pieces_value else value_states.new_empty((1, 1, 0, head_dim))
            head_lens.append(int(head_key.shape[2]))
            flat_keys.append(head_key.reshape(-1, head_dim))
            flat_values.append(head_value.reshape(-1, head_dim))
        flat_key = torch.cat(flat_keys, dim=0) if flat_keys else key_states.new_empty((0, head_dim))
        flat_value = torch.cat(flat_values, dim=0) if flat_values else value_states.new_empty((0, head_dim))
        timing_breakdown["packing"] = float(time.perf_counter() - pack_start)

        raw_tokens_t = torch.as_tensor((actions_np == 2).sum(axis=1), device=key_states.device, dtype=torch.long)
        surrogate_regions_t = torch.as_tensor([len(spans) for spans in head_surrogate_spans], device=key_states.device, dtype=torch.long)
        surrogate_tokens_t = torch.as_tensor((actions_np == 1).sum(axis=1), device=key_states.device, dtype=torch.long)
        drop_tokens_t = torch.as_tensor((actions_np == 0).sum(axis=1), device=key_states.device, dtype=torch.long)
        used_entries_t = raw_tokens_t + surrogate_regions_t
        budget_gap_t = caps.to(device=key_states.device, dtype=torch.long) - used_entries_t
        prompt_caps = caps.to(device=key_states.device, dtype=torch.long) + int(recent_len)
        head_len_tensor = torch.as_tensor(head_lens, device=key_states.device, dtype=torch.long)
        head_budget_overflow = torch.clamp(head_len_tensor - prompt_caps, min=0)
        capacity_float = caps.to(dtype=torch.float32)
        prompt_float = prompt_caps.to(dtype=torch.float32)

        self._init_headwise_flatten_metadata(
            head_lens=head_lens,
            device=key_states.device,
            key_heads=int(key_heads),
            query_heads=int(key_heads),
            num_key_value_groups=int(groups),
        )
        self._last_allocator_stats.update(
            {
                "surrogate_kv_headwise_ada_overlay": 1,
                "surrogate_kv_headwise_varlen": 1,
                "surrogate_kv_headwise_uncompressed": 0,
                "surrogate_kv_headwise_exact_query_heads": int(exact_query_heads),
                "surrogate_kv_headwise_flatten_tokens": int(flat_key.shape[0]),
                "surrogate_kv_headwise_original_key_heads": int(original_key_heads),
                "surrogate_kv_headwise_original_gqa_groups": int(original_groups),
                "surrogate_kv_headwise_key_heads": int(key_heads),
                "surrogate_kv_headwise_query_heads": int(key_heads),
                "surrogate_kv_headwise_gqa_groups": int(groups),
                "surrogate_kv_headwise_gqa_capacity_fusion_exact": int(gqa_capacity_fusion == "ada_exact_query_heads"),
                "surrogate_kv_headwise_capacity_min": int(caps.min().detach().cpu().item()),
                "surrogate_kv_headwise_capacity_max": int(caps.max().detach().cpu().item()),
                "surrogate_kv_headwise_capacity_mean": float(capacity_float.mean().detach().cpu().item()),
                "surrogate_kv_headwise_prompt_capacity_min": int(prompt_caps.min().detach().cpu().item()),
                "surrogate_kv_headwise_prompt_capacity_max": int(prompt_caps.max().detach().cpu().item()),
                "surrogate_kv_headwise_prompt_capacity_mean": float(prompt_float.mean().detach().cpu().item()),
                "surrogate_kv_headwise_budget_gap_min": int(budget_gap_t.min().detach().cpu().item()),
                "surrogate_kv_headwise_budget_gap_mean": float(budget_gap_t.to(dtype=torch.float32).mean().detach().cpu().item()),
                "surrogate_kv_headwise_budget_overflow_max": int(head_budget_overflow.max().detach().cpu().item()),
                "surrogate_kv_headwise_budget_preserved": int(int(head_budget_overflow.max().detach().cpu().item()) == 0),
                "surrogate_kv_headwise_head_len_min": int(min(head_lens) if head_lens else 0),
                "surrogate_kv_headwise_head_len_max": int(max(head_lens) if head_lens else 0),
                "surrogate_kv_headwise_head_len_mean": float(sum(head_lens) / max(1, len(head_lens))),
                "surrogate_kv_headwise_precomputed_scores": 1,
                "surrogate_kv_timing_update_score_seconds": float(score_seconds),
                "surrogate_kv_headwise_child_mean_ks_run_raw_tokens": float(raw_tokens_t.to(dtype=torch.float32).mean().detach().cpu().item()),
                "surrogate_kv_headwise_child_mean_ks_run_raw_regions": float(sum(raw_regions_per_head) / max(1, len(raw_regions_per_head))),
                "surrogate_kv_headwise_child_mean_ks_run_surrogate_regions": float(surrogate_regions_t.to(dtype=torch.float32).mean().detach().cpu().item()),
                "surrogate_kv_headwise_child_mean_ks_run_surrogate_tokens": float(surrogate_tokens_t.to(dtype=torch.float32).mean().detach().cpu().item()),
                "surrogate_kv_headwise_child_mean_ks_run_drop_tokens": float(drop_tokens_t.to(dtype=torch.float32).mean().detach().cpu().item()),
                "surrogate_kv_headwise_child_mean_ks_run_used_entries": float(used_entries_t.to(dtype=torch.float32).mean().detach().cpu().item()),
                "surrogate_kv_headwise_child_mean_ks_run_budget_gap": float(budget_gap_t.to(dtype=torch.float32).mean().detach().cpu().item()),
                "surrogate_kv_headwise_child_mean_surrogate_kv_selected_surrogates": float(surrogate_regions_t.to(dtype=torch.float32).mean().detach().cpu().item()),
                "surrogate_kv_ada_overlay_selected_surrogates": int(accepted_surrogates.sum()),
                "surrogate_kv_ada_overlay_selected_gain": float(selected_gain_total),
                "surrogate_kv_ada_overlay_sold_raw_value": float(sold_raw_total),
            }
        )
        self.last_stats = self._stats(
            full_tokens=int(q_len),
            compressed_tokens=max(head_lens) if head_lens else 0,
            recent_tokens=int(recent_len),
            selected_chunks=max(child_selected_runs) if child_selected_runs else 0,
            selected_runs=max(child_selected_runs) if child_selected_runs else 0,
            num_chunks=max(child_chunks) if child_chunks else 0,
            chunk_size=max(1, int(self.spec.dynamic_anchor_width or 4)),
            sink_tokens=int(sink_len),
            two_surrogate_chunks=max(child_surrogate_slots) if child_surrogate_slots else 0,
            mode_counts=self._merge_mode_counts(child_mode_counts),
            op_seconds=time.perf_counter() - update_start,
            configured_keep_ratio=configured_keep_ratio,
            timing_breakdown=timing_breakdown,
        )
        return flat_key, flat_value


    def update_kv_headwise(self, key_states, query_states, value_states, attention_mask=None, num_key_value_groups=1):
        update_start = time.perf_counter()
        del attention_mask
        if key_states.shape[-2] != query_states.shape[-2]:
            raise ValueError("SurKV headwise path requires key/query sequence lengths to match.")

        bsz, query_heads, q_len, head_dim = query_states.shape
        key_heads = int(key_states.shape[1])
        if int(bsz) != 1:
            raise ValueError("SurKV headwise Ada cache path currently supports batch size 1.")
        if int(query_heads) % max(1, int(key_heads)) != 0:
            raise ValueError(
                f"query heads ({query_heads}) must be divisible by key/value heads ({key_heads}) "
                "for headwise Ada packing."
            )

        groups = max(1, int(num_key_value_groups or (int(query_heads) // max(1, int(key_heads)))))
        configured_keep_ratio = min(1.0, float(self.max_capacity_prompt) / max(float(q_len), 1.0))
        if self.layer_keep_ratio is not None:
            configured_keep_ratio = min(1.0, max(1.0 / max(q_len, 1), float(self.layer_keep_ratio)))
        effective_capacity_prompt = max(1, min(int(q_len), int(round(float(q_len) * configured_keep_ratio))))
        recent_len = min(int(self.window_size), int(q_len))
        past_len = int(q_len) - int(recent_len)
        self.last_layout_meta = None
        self._last_allocator_stats = {}
        self._last_score_stats = {}
        self._last_ada_head_capacities = None
        self._last_fast_pack_plan = None
        self._pending_global_budget_ledger_stats = {}
        self._last_layer_budget_signal = 0.0
        self._last_layer_budget_curve = None
        self._headwise_cache_query_repeated = False

        def finish_uncompressed():
            lens = [int(q_len)] * int(key_states.shape[1])
            self._init_headwise_flatten_metadata(
                head_lens=lens,
                device=key_states.device,
                key_heads=int(key_states.shape[1]),
                query_heads=int(query_states.shape[1]),
                num_key_value_groups=1 if bool(getattr(self, "_headwise_cache_query_repeated", False)) else groups,
            )
            flat_key = key_states.reshape(-1, head_dim)
            flat_value = value_states.reshape(-1, head_dim)
            self._last_allocator_stats.update(
                {
                    "surrogate_kv_headwise_varlen": 1,
                    "surrogate_kv_headwise_uncompressed": 1,
                    "surrogate_kv_headwise_flatten_tokens": int(flat_key.shape[0]),
                    "surrogate_kv_headwise_exact_query_heads": int(getattr(self, "_headwise_cache_query_repeated", False)),
                    "surrogate_kv_headwise_head_len_min": int(q_len),
                    "surrogate_kv_headwise_head_len_max": int(q_len),
                    "surrogate_kv_headwise_head_len_mean": float(q_len),
                }
            )
            self.last_stats = self._stats(
                full_tokens=int(q_len),
                compressed_tokens=int(q_len),
                recent_tokens=int(recent_len),
                selected_chunks=0,
                selected_runs=0,
                num_chunks=0,
                chunk_size=0,
                sink_tokens=0,
                two_surrogate_chunks=0,
                mode_counts={},
                op_seconds=time.perf_counter() - update_start,
                configured_keep_ratio=configured_keep_ratio,
            )
            return flat_key, flat_value

        if int(q_len) <= int(effective_capacity_prompt) or int(recent_len) <= 0 or int(past_len) <= 0:
            if self._ada_exact_query_head_cache_enabled() and int(groups) > 1:
                key_states = _repeat_kv_heads(key_states, int(groups))
                value_states = _repeat_kv_heads(value_states, int(groups))
                self._headwise_cache_query_repeated = True
            return finish_uncompressed()

        sink_len = min(self._protected_sink_tokens(), int(past_len))
        score_start = time.perf_counter()
        _ = self._past_token_scores(
            key_states=key_states,
            query_states=query_states,
            value_states=value_states,
            recent_len=int(recent_len),
            past_len=int(past_len),
            head_dim=int(head_dim),
            num_key_value_groups=int(groups),
            base_capacity_prompt=int(effective_capacity_prompt),
            sink_len=int(sink_len),
            headwise_budget_only=True,
        )
        score_seconds = time.perf_counter() - score_start
        precomputed_head_scores = getattr(self, "_last_headwise_precomputed_pooled_scores", None)
        precomputed_residual_scores = getattr(self, "_last_surrogate_residual_head_token_scores", None)
        precomputed_score_stats = dict(getattr(self, "_last_score_stats", {}) or {})
        query_head_caps = getattr(self, "_last_ada_head_capacities", None)
        base_budget = max(1, min(int(past_len), int(effective_capacity_prompt) - int(recent_len)))
        if not isinstance(query_head_caps, torch.Tensor):
            query_head_caps = torch.full(
                (int(bsz), int(query_heads)),
                int(base_budget),
                device=key_states.device,
                dtype=torch.long,
            )
        query_head_caps = query_head_caps.to(device=key_states.device, dtype=torch.long).clamp(min=1, max=int(past_len))
        self._last_query_head_capacities = query_head_caps.detach()

        original_key_heads = int(key_heads)
        original_groups = int(groups)
        exact_query_heads = bool(self._ada_exact_query_head_cache_enabled() and int(groups) > 1)
        if exact_query_heads:
            key_states = _repeat_kv_heads(key_states, int(groups))
            value_states = _repeat_kv_heads(value_states, int(groups))
            key_heads = int(query_heads)
            groups = 1
            key_head_caps = query_head_caps[0, : int(query_heads)]
            gqa_capacity_fusion = "ada_exact_query_heads"
            self._headwise_cache_query_repeated = True
        elif int(query_heads) == int(key_heads):
            key_head_caps = query_head_caps[0, : int(key_heads)]
            gqa_capacity_fusion = "none"
        else:
            grouped_caps = query_head_caps[0, : int(key_heads) * int(groups)].view(int(key_heads), int(groups))
            key_head_caps = torch.round(grouped_caps.to(dtype=torch.float32).mean(dim=-1)).to(dtype=torch.long)
            gqa_capacity_fusion = "mean"

        min_past_capacity = max(1, min(int(past_len), int(sink_len) + 1))
        key_head_caps = key_head_caps.to(device=key_states.device, dtype=torch.long).clamp(
            min=int(min_past_capacity),
            max=int(past_len),
        )
        key_head_caps_cpu = [int(v) for v in key_head_caps.detach().to(device="cpu", dtype=torch.long).tolist()]
        self._last_gqa_capacity_fusion = str(gqa_capacity_fusion)

        # AdaKV already decided the per-head raw frontier in _past_token_scores().
        # The fast Ada overlay keeps that frontier as the ledger and only
        # exchanges marginal raw entries for surrogate packets when the exchange
        # improves the same per-head budget.

        head_plans: List[Dict[str, object]] = []
        head_lens: List[int] = []
        child_mode_counts: List[Dict[str, int]] = []
        child_selected_runs = []
        child_surrogate_slots = []
        child_chunks = []
        child_numeric_sum: Dict[str, float] = {}
        child_numeric_count: Dict[str, int] = {}
        timing_breakdown = {"score": float(score_seconds), "planning": 0.0, "prototype": 0.0, "packing": 0.0}

        def collect_child_numeric(name: str, value) -> None:
            if value is None:
                return
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return
            child_numeric_sum[name] = child_numeric_sum.get(name, 0.0) + numeric_value
            child_numeric_count[name] = child_numeric_count.get(name, 0) + 1

        planning_start = time.perf_counter()
        compressible_start = int(sink_len)
        compressible_len = max(0, int(past_len) - int(compressible_start))
        micro_len = max(1, int(self.spec.dynamic_anchor_width or (int(self.chunk_size) // 4)))
        atom_count_for_stats = int(math.ceil(float(compressible_len) / float(micro_len))) if compressible_len > 0 else 0
        shared_chunk_slices = [(int(compressible_start), int(past_len))] if int(compressible_len) > 0 else []
        shared_chunk_lengths = torch.as_tensor(
            [int(compressible_len)],
            device=key_states.device,
            dtype=torch.long,
        )
        score_bank_heads = int(key_heads) if bool(exact_query_heads) else int(query_heads)
        score_bank = None
        if (
            isinstance(precomputed_head_scores, torch.Tensor)
            and precomputed_head_scores.ndim == 3
            and precomputed_head_scores.shape[0] == int(bsz)
            and precomputed_head_scores.shape[1] >= int(score_bank_heads)
            and precomputed_head_scores.shape[2] == int(past_len)
        ):
            score_bank = precomputed_head_scores[:, : int(score_bank_heads), : int(past_len)].detach()
        residual_bank = None
        if (
            isinstance(precomputed_residual_scores, torch.Tensor)
            and precomputed_residual_scores.ndim == 3
            and precomputed_residual_scores.shape[0] == int(bsz)
            and precomputed_residual_scores.shape[1] >= int(score_bank_heads)
            and precomputed_residual_scores.shape[2] == int(past_len)
        ):
            residual_bank = precomputed_residual_scores[:, : int(score_bank_heads), : int(past_len)].detach()

        ada_overlay_result = None
        if _SURKV_HEADWISE_ADA_OVERLAY:
            ada_overlay_result = self._update_kv_headwise_ada_overlay(
                key_states=key_states,
                value_states=value_states,
                precomputed_head_scores=score_bank,
                precomputed_residual_scores=residual_bank,
                key_head_caps=key_head_caps,
                groups=int(groups),
                recent_len=int(recent_len),
                past_len=int(past_len),
                sink_len=int(sink_len),
                configured_keep_ratio=float(configured_keep_ratio),
                update_start=float(update_start),
                score_seconds=float(score_seconds),
                exact_query_heads=bool(exact_query_heads),
                original_key_heads=int(original_key_heads),
                original_groups=int(original_groups),
                gqa_capacity_fusion=str(gqa_capacity_fusion),
            )
        if ada_overlay_result is not None:
            return ada_overlay_result

        def make_plan_cluster():
            cluster = type(self)(
                mode=self.mode,
                window_size=self.window_size,
                max_capacity_prompt=int(effective_capacity_prompt),
                kernel_size=self.kernel_size,
                pooling=self.pooling,
                chunk_size=self.chunk_size,
                local_radius=self.local_radius,
                sink_tokens=self.sink_tokens,
                layer_keep_ratio=None,
                layer_scheduler=self.layer_scheduler,
                global_budget_ledger=False,
                global_layer_allocator=False,
                score_method=self.score_method,
                head_score_fusion="mean",
            )
            cluster.generation_horizon = int(getattr(self, "generation_horizon", 0) or 0)
            cluster._precomputed_score_stats = dict(precomputed_score_stats)
            cluster._allocator_plan_only = True
            cluster._precomputed_allocator_setup = allocator_setup
            return cluster

        plan_cluster = None
        allocator_setup = None
        if (
            bool(exact_query_heads)
            and isinstance(score_bank, torch.Tensor)
            and score_bank.shape[1] >= int(key_heads)
            and int(compressible_len) > 0
        ):
            setup_scores = score_bank[0, : int(key_heads), int(compressible_start) : int(past_len)].to(
                device=key_states.device,
                dtype=torch.float32,
            )
            regular_atoms = int(int(compressible_len) // int(micro_len))
            regular_tokens = int(regular_atoms) * int(micro_len)
            mean_parts = []
            peak_parts = []
            if int(regular_atoms) > 0:
                regular = setup_scores[:, : int(regular_tokens)].reshape(
                    int(key_heads),
                    int(regular_atoms),
                    int(micro_len),
                )
                mean_parts.append(regular.mean(dim=2))
                peak_parts.append(regular.max(dim=2).values)
            if int(regular_tokens) < int(compressible_len):
                tail = setup_scores[:, int(regular_tokens) :]
                mean_parts.append(tail.mean(dim=1, keepdim=True))
                peak_parts.append(tail.max(dim=1, keepdim=True).values)
            if mean_parts:
                atom_mean_all = torch.cat(mean_parts, dim=1)
                atom_peak_all = torch.cat(peak_parts, dim=1)
                mean_rank_all = _rank01(atom_mean_all)
                peak_rank_all = _rank01(atom_peak_all)
                atom_risk_all = torch.maximum(mean_rank_all, peak_rank_all)
                risk_arrays = (
                    torch.stack((mean_rank_all, atom_risk_all), dim=0)
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .numpy()
                )
                tail_floor = 1.0 / float(max(2, int(atom_risk_all.shape[1]) + 1))

                def setup_tail_scores(values: np.ndarray) -> np.ndarray:
                    ranks = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
                    return -np.log(np.maximum(float(tail_floor), 1.0 - ranks))

                mean_risk_all = risk_arrays[0] + 1e-6
                atom_risk_np_all = risk_arrays[1] + 1e-6
                future_risk_all = np.full_like(mean_risk_all, 1e-6)
                setup_atom_start = np.arange(
                    int(compressible_start),
                    int(past_len),
                    int(micro_len),
                    dtype=np.int64,
                )
                setup_atom_end = np.minimum(setup_atom_start + int(micro_len), int(past_len)).astype(np.int64)
                setup_atom_len_int = (setup_atom_end - setup_atom_start).astype(np.int64)
                setup_prefix_len = np.concatenate(
                    (
                        np.asarray([0], dtype=np.int64),
                        np.cumsum(setup_atom_len_int).astype(np.int64),
                    )
                )
                setup_atom_len = np.maximum(1.0, setup_atom_len_int.astype(np.float64))
                atom_indices_np = np.arange(int(atom_risk_np_all.shape[1]), dtype=np.int64)
                surrogate_signal_all = setup_tail_scores(atom_risk_np_all)
                atom_signal_all = setup_tail_scores(atom_risk_np_all)
                raw_value_all = atom_signal_all * atom_signal_all * setup_atom_len[None, :]
                raw_density_all = raw_value_all / np.maximum(setup_atom_len[None, :], 1.0)
                prefix_mass_all = np.concatenate(
                    (
                        np.zeros((int(atom_risk_np_all.shape[0]), 1), dtype=np.float64),
                        np.cumsum(surrogate_signal_all * setup_atom_len[None, :], axis=1).astype(np.float64),
                    ),
                    axis=1,
                )
                prefix_energy_all = np.concatenate(
                    (
                        np.zeros((int(atom_risk_np_all.shape[0]), 1), dtype=np.float64),
                        np.cumsum(
                            surrogate_signal_all * surrogate_signal_all * setup_atom_len[None, :],
                            axis=1,
                        ).astype(np.float64),
                    ),
                    axis=1,
                )
                raw_keep_order_all = np.argsort(-atom_risk_np_all, axis=1, kind="stable").astype(np.int64)
                raw_drop_order_all = np.argsort(atom_risk_np_all, axis=1, kind="stable").astype(np.int64)
                allocator_setup = {
                    "base_start": int(compressible_start),
                    "base_end": int(past_len),
                    "micro_len": int(micro_len),
                    "num_atoms": int(atom_risk_np_all.shape[1]),
                    "regular_atoms": int(regular_atoms),
                    "regular_tokens": int(regular_tokens),
                    "tail_floor": float(tail_floor),
                    "mean_risk": mean_risk_all,
                    "future_risk": future_risk_all,
                    "atom_risk": atom_risk_np_all,
                    "mean_signal": setup_tail_scores(mean_risk_all),
                    "future_signal": setup_tail_scores(future_risk_all),
                    "atom_signal": atom_signal_all,
                    "surrogate_signal": surrogate_signal_all,
                    "atom_start": setup_atom_start,
                    "atom_end": setup_atom_end,
                    "atom_len_int": setup_atom_len_int,
                    "atom_len": setup_atom_len,
                    "prefix_len": setup_prefix_len,
                    "full_cost": int(setup_atom_len_int.sum()),
                    "atom_indices": atom_indices_np,
                    "raw_value": raw_value_all,
                    "raw_density": raw_density_all,
                    "prefix_mass": prefix_mass_all,
                    "prefix_energy": prefix_energy_all,
                    "raw_keep_order": raw_keep_order_all,
                    "raw_drop_order": raw_drop_order_all,
                }
        plan_cluster = make_plan_cluster()

        for head_idx in range(int(key_heads)):
            head_past_capacity = int(key_head_caps_cpu[int(head_idx)])
            head_capacity_prompt = max(1, min(int(q_len), int(head_past_capacity) + int(recent_len)))
            q_start = int(head_idx * groups)
            q_end = int(q_start + groups)
            head_token_scores = None
            plan_cluster._last_surrogate_residual_token_scores = None
            if isinstance(score_bank, torch.Tensor) and score_bank.shape[1] >= int(q_end):
                grouped_head_scores = score_bank[:, int(q_start) : int(q_end), :]
                if int(q_end) - int(q_start) == 1:
                    head_token_scores = score_bank[:, int(q_start), :].detach()
                elif int(q_end) - int(q_start) > 1 and not bool(exact_query_heads):
                    head_token_scores = torch.sqrt(
                        torch.clamp(grouped_head_scores, min=0.0).square().mean(dim=1)
                    ).detach()
                else:
                    head_token_scores = grouped_head_scores.mean(dim=1).detach()
                if isinstance(residual_bank, torch.Tensor) and residual_bank.shape[1] >= int(q_end):
                    grouped_residual_scores = residual_bank[:, int(q_start) : int(q_end), :]
                    if int(q_end) - int(q_start) == 1:
                        plan_cluster._last_surrogate_residual_token_scores = residual_bank[
                            :, int(q_start), :
                        ].detach()
                    elif int(q_end) - int(q_start) > 1 and not bool(exact_query_heads):
                        plan_cluster._last_surrogate_residual_token_scores = torch.sqrt(
                            torch.clamp(grouped_residual_scores, min=0.0).square().mean(dim=1)
                        ).detach()
                    else:
                        plan_cluster._last_surrogate_residual_token_scores = grouped_residual_scores.mean(
                            dim=1
                        ).detach()

            if head_token_scores is None:
                local_score_start = time.perf_counter()
                head_token_scores = plan_cluster._past_token_scores(
                    key_states=key_states[:, int(head_idx) : int(head_idx) + 1, :, :],
                    query_states=query_states[:, int(q_start) : int(q_end), :, :],
                    value_states=value_states[:, int(head_idx) : int(head_idx) + 1, :, :],
                    recent_len=int(recent_len),
                    past_len=int(past_len),
                    head_dim=int(head_dim),
                    num_key_value_groups=int(groups),
                    base_capacity_prompt=int(head_capacity_prompt),
                    sink_len=int(sink_len),
                )
                timing_breakdown["score"] += float(time.perf_counter() - local_score_start)

            raw_spans: List[Tuple[int, int]] = []
            surrogate_spans: List[Tuple[int, int]] = []
            chunk_count = 0
            child_stats: Dict[str, object] = {}

            if int(q_len) <= int(head_capacity_prompt) or int(compressible_len) <= 0:
                raw_spans = [(int(sink_len), int(past_len))] if int(past_len) > int(sink_len) else []
                head_len = int(q_len)
                child_stats = {
                    "ks_run_raw_tokens": int(max(0, int(past_len) - int(sink_len))),
                    "ks_run_raw_regions": int(bool(raw_spans)),
                    "ks_run_surrogate_regions": 0,
                    "ks_run_surrogate_tokens": 0,
                    "ks_run_drop_tokens": 0,
                    "ks_run_drop_regions": 0,
                    "ks_run_used_entries": int(max(0, int(past_len) - int(sink_len))) + int(sink_len),
                    "ks_run_budget_gap": int(head_past_capacity) - int(past_len),
                    "surrogate_kv_selected_surrogates": 0,
                    "surrogate_kv_final_cost": int(past_len),
                    "surrogate_kv_budget_gap": int(head_past_capacity) - int(past_len),
                }
            else:
                budget_past_total = max(1, int(head_capacity_prompt) - int(recent_len))
                budget_compressible = max(0, int(budget_past_total) - int(sink_len))
                chunk_count = int(atom_count_for_stats)
                if int(budget_compressible) >= int(compressible_len):
                    raw_spans = [(int(sink_len), int(past_len))]
                    head_len = int(q_len)
                    child_stats = {
                        "ks_run_raw_tokens": int(compressible_len),
                        "ks_run_raw_regions": int(bool(raw_spans)),
                        "ks_run_surrogate_regions": 0,
                        "ks_run_surrogate_tokens": 0,
                        "ks_run_drop_tokens": 0,
                        "ks_run_drop_regions": 0,
                        "ks_run_used_entries": int(compressible_len) + int(sink_len),
                        "ks_run_budget_gap": int(head_past_capacity) - int(past_len),
                        "surrogate_kv_selected_surrogates": 0,
                        "surrogate_kv_final_cost": int(past_len),
                        "surrogate_kv_budget_gap": int(head_past_capacity) - int(past_len),
                    }
                else:
                    plan_cluster._precomputed_allocator_head = int(head_idx)
                    allocated = plan_cluster._dynamic_surrogate_kv_allocation(
                        token_scores=head_token_scores.to(dtype=torch.float32),
                        chunk_slices=shared_chunk_slices,
                        chunk_lengths=shared_chunk_lengths,
                        target_compressed_tokens=int(head_capacity_prompt),
                        sink_len=int(sink_len),
                        recent_len=int(recent_len),
                    )
                    allocated_ok = allocated is not None
                    fast_plan = dict(getattr(plan_cluster, "_last_fast_pack_plan", {}) or {})
                    child_stats = dict(getattr(plan_cluster, "_last_allocator_stats", {}) or {})
                    if not bool(allocated_ok) or not fast_plan:
                        raw_spans = [(int(sink_len), int(past_len))]
                    else:
                        raw_spans = [
                            (int(start), int(end))
                            for start, end in fast_plan.get("raw_spans", [])
                            if int(end) > int(start)
                        ]
                        planned_slices = list(fast_plan.get("chunk_slices", []) or [])
                        for surrogate_idx in fast_plan.get("surrogate_chunk_indices_list", []) or []:
                            idx = int(surrogate_idx)
                            if 0 <= idx < len(planned_slices):
                                start, end = planned_slices[idx]
                                if int(end) > int(start):
                                    surrogate_spans.append((int(start), int(end)))
                    raw_tokens = int(sum(max(0, int(end) - int(start)) for start, end in raw_spans))
                    surrogate_tokens = int(sum(max(0, int(end) - int(start)) for start, end in surrogate_spans))
                    used_entries = int(sink_len) + int(raw_tokens) + int(len(surrogate_spans))
                    head_len = int(used_entries) + int(recent_len)
                    child_stats.setdefault("ks_run_raw_tokens", int(raw_tokens))
                    child_stats.setdefault("ks_run_raw_regions", int(len(raw_spans)))
                    child_stats.setdefault("ks_run_surrogate_regions", int(len(surrogate_spans)))
                    child_stats.setdefault("ks_run_surrogate_tokens", int(surrogate_tokens))
                    child_stats.setdefault(
                        "ks_run_drop_tokens",
                        max(0, int(compressible_len) - int(raw_tokens) - int(surrogate_tokens)),
                    )
                    child_stats.setdefault("ks_run_used_entries", int(used_entries))
                    child_stats.setdefault("ks_run_budget_gap", int(head_past_capacity) - int(used_entries))

            head_lens.append(head_len)
            mode_counts = {}
            if int(child_stats.get("ks_run_surrogate_regions", 0) or 0) > 0:
                mode_counts["local"] = int(child_stats.get("ks_run_surrogate_regions", 0) or 0)
            if int(child_stats.get("ks_run_drop_tokens", 0) or 0) > 0:
                mode_counts["drop"] = int(child_stats.get("ks_run_drop_regions", 1) or 1)
            child_mode_counts.append(mode_counts)
            child_selected_runs.append(
                int(child_stats.get("ks_run_raw_regions", 0) or 0)
                + int(child_stats.get("ks_run_surrogate_regions", 0) or 0)
            )
            child_surrogate_slots.append(int(child_stats.get("ks_run_surrogate_regions", len(surrogate_spans)) or 0))
            child_chunks.append(int(chunk_count))
            for child_key in (
                "ks_run_raw_tokens",
                "ks_run_surrogate_regions",
                "ks_run_surrogate_tokens",
                "ks_run_drop_tokens",
                "ks_run_used_entries",
                "ks_run_budget_gap",
                "surrogate_kv_selected_surrogates",
                "surrogate_kv_final_cost",
                "surrogate_kv_budget_gap",
            ):
                collect_child_numeric(child_key, child_stats.get(child_key))
            for child_key, child_value in child_stats.items():
                if str(child_key).startswith("surrogate_kv_timing_alloc_"):
                    collect_child_numeric(str(child_key), child_value)
            head_plans.append({"raw_spans": raw_spans, "surrogate_spans": surrogate_spans})

        timing_breakdown["planning"] += float(time.perf_counter() - planning_start)

        proto_start = time.perf_counter()
        head_score_bank = None
        if (
            isinstance(precomputed_head_scores, torch.Tensor)
            and precomputed_head_scores.ndim == 3
            and precomputed_head_scores.shape[0] == int(bsz)
            and precomputed_head_scores.shape[1] >= int(key_heads)
            and precomputed_head_scores.shape[2] >= int(past_len)
        ):
            head_score_bank = precomputed_head_scores[0, : int(key_heads), :].detach()
        surrogate_banks = self._headwise_norm_rms_prototypes_from_spans_batch(
            key_states=key_states,
            value_states=value_states,
            head_scores=head_score_bank,
            head_spans=[plan["surrogate_spans"] for plan in head_plans],
            base_start=int(sink_len),
            base_end=int(past_len),
            micro_len=int(micro_len),
        )
        timing_breakdown["prototype"] += float(time.perf_counter() - proto_start)

        pack_start = time.perf_counter()
        flat_keys = []
        flat_values = []
        for head_idx, plan in enumerate(head_plans):
            pieces_key = []
            pieces_value = []
            if int(sink_len) > 0:
                pieces_key.append(key_states[:, int(head_idx) : int(head_idx) + 1, : int(sink_len), :])
                pieces_value.append(value_states[:, int(head_idx) : int(head_idx) + 1, : int(sink_len), :])
            raw_index_tensor = self._span_index_tensor(plan["raw_spans"], device=key_states.device)
            if raw_index_tensor is not None and raw_index_tensor.numel() > 0:
                pieces_key.append(key_states[:, int(head_idx) : int(head_idx) + 1].index_select(2, raw_index_tensor))
                pieces_value.append(value_states[:, int(head_idx) : int(head_idx) + 1].index_select(2, raw_index_tensor))
            proto_key, proto_value = (
                surrogate_banks[int(head_idx)]
                if int(head_idx) < len(surrogate_banks)
                else (
                    key_states.new_empty((1, 1, 0, head_dim)),
                    value_states.new_empty((1, 1, 0, head_dim)),
                )
            )
            if proto_key.numel() > 0:
                pieces_key.append(proto_key)
                pieces_value.append(proto_value)
            if int(recent_len) > 0:
                pieces_key.append(key_states[:, int(head_idx) : int(head_idx) + 1, int(past_len) :, :])
                pieces_value.append(value_states[:, int(head_idx) : int(head_idx) + 1, int(past_len) :, :])
            head_key = torch.cat(pieces_key, dim=2) if pieces_key else key_states.new_empty((1, 1, 0, head_dim))
            head_value = torch.cat(pieces_value, dim=2) if pieces_value else value_states.new_empty((1, 1, 0, head_dim))
            head_lens[int(head_idx)] = int(head_key.shape[2])
            flat_keys.append(head_key.reshape(-1, head_dim))
            flat_values.append(head_value.reshape(-1, head_dim))
        timing_breakdown["packing"] += float(time.perf_counter() - pack_start)

        flat_key = torch.cat(flat_keys, dim=0) if flat_keys else key_states.new_empty((0, head_dim))
        flat_value = torch.cat(flat_values, dim=0) if flat_values else value_states.new_empty((0, head_dim))
        self._init_headwise_flatten_metadata(
            head_lens=head_lens,
            device=key_states.device,
            key_heads=int(key_heads),
            query_heads=int(query_heads),
            num_key_value_groups=int(groups),
        )

        capacity_float = key_head_caps.to(dtype=torch.float32)
        prompt_caps = key_head_caps.to(dtype=torch.long) + int(recent_len)
        prompt_float = prompt_caps.to(dtype=torch.float32)
        head_len_tensor = torch.as_tensor(head_lens, device=key_states.device, dtype=torch.long)
        head_budget_gap = prompt_caps - head_len_tensor if head_lens else prompt_caps.new_empty((0,))
        head_budget_overflow = torch.clamp(-head_budget_gap, min=0) if head_lens else prompt_caps.new_empty((0,))
        self._last_allocator_stats.update(
            {
                "surrogate_kv_headwise_varlen": 1,
                "surrogate_kv_headwise_uncompressed": 0,
                "surrogate_kv_headwise_exact_query_heads": int(exact_query_heads),
                "surrogate_kv_headwise_flatten_tokens": int(flat_key.shape[0]),
                "surrogate_kv_headwise_original_key_heads": int(original_key_heads),
                "surrogate_kv_headwise_original_gqa_groups": int(original_groups),
                "surrogate_kv_headwise_key_heads": int(key_heads),
                "surrogate_kv_headwise_query_heads": int(query_heads),
                "surrogate_kv_headwise_gqa_groups": int(groups),
                "surrogate_kv_headwise_gqa_capacity_fusion_sum": int(gqa_capacity_fusion == "sum"),
                "surrogate_kv_headwise_gqa_capacity_fusion_mean": int(gqa_capacity_fusion == "mean"),
                "surrogate_kv_headwise_gqa_capacity_fusion_exact": int(gqa_capacity_fusion == "ada_exact_query_heads"),
                "surrogate_kv_headwise_capacity_min": int(key_head_caps.min().detach().cpu().item()),
                "surrogate_kv_headwise_capacity_max": int(key_head_caps.max().detach().cpu().item()),
                "surrogate_kv_headwise_capacity_mean": float(capacity_float.mean().detach().cpu().item()),
                "surrogate_kv_headwise_prompt_capacity_min": int(prompt_caps.min().detach().cpu().item()),
                "surrogate_kv_headwise_prompt_capacity_max": int(prompt_caps.max().detach().cpu().item()),
                "surrogate_kv_headwise_prompt_capacity_mean": float(prompt_float.mean().detach().cpu().item()),
                "surrogate_kv_headwise_budget_gap_min": int(head_budget_gap.min().detach().cpu().item()) if head_lens else 0,
                "surrogate_kv_headwise_budget_gap_mean": (
                    float(head_budget_gap.to(dtype=torch.float32).mean().detach().cpu().item()) if head_lens else 0.0
                ),
                "surrogate_kv_headwise_budget_overflow_max": (
                    int(head_budget_overflow.max().detach().cpu().item()) if head_lens else 0
                ),
                "surrogate_kv_headwise_budget_preserved": (
                    int(int(head_budget_overflow.max().detach().cpu().item()) == 0) if head_lens else 1
                ),
                "surrogate_kv_headwise_head_len_min": int(min(head_lens) if head_lens else 0),
                "surrogate_kv_headwise_head_len_max": int(max(head_lens) if head_lens else 0),
                "surrogate_kv_headwise_head_len_mean": float(sum(head_lens) / max(1, len(head_lens))),
                "surrogate_kv_headwise_precomputed_scores": int(isinstance(precomputed_head_scores, torch.Tensor)),
                "surrogate_kv_timing_update_score_seconds": float(score_seconds),
            }
        )
        for child_key, child_sum in child_numeric_sum.items():
            child_count = max(1, int(child_numeric_count.get(child_key, 1)))
            self._last_allocator_stats[f"surrogate_kv_headwise_child_mean_{child_key}"] = float(child_sum) / float(child_count)

        self.last_stats = self._stats(
            full_tokens=int(q_len),
            compressed_tokens=max(head_lens) if head_lens else 0,
            recent_tokens=int(recent_len),
            selected_chunks=max(child_selected_runs) if child_selected_runs else 0,
            selected_runs=max(child_selected_runs) if child_selected_runs else 0,
            num_chunks=max(child_chunks) if child_chunks else 0,
            chunk_size=int(self.chunk_size),
            sink_tokens=int(sink_len),
            two_surrogate_chunks=max(child_surrogate_slots) if child_surrogate_slots else 0,
            mode_counts=self._merge_mode_counts(child_mode_counts),
            op_seconds=time.perf_counter() - update_start,
            configured_keep_ratio=configured_keep_ratio,
            timing_breakdown=timing_breakdown,
        )
        return flat_key, flat_value

    def _init_headwise_flatten_metadata(
        self,
        *,
        head_lens: Sequence[int],
        device,
        key_heads: int,
        query_heads: int,
        num_key_value_groups: int,
    ) -> None:
        del query_heads
        metadata_heads = int(key_heads)
        self.gqa_support = int(num_key_value_groups) > 1
        self.num_key_value_groups = max(1, int(num_key_value_groups))
        self.head_lens = torch.tensor([int(v) for v in head_lens], dtype=torch.int32, device=device)
        self.klen_sum = int(sum(int(v) for v in head_lens))
        self.max_seqlen_k = int(max([int(v) for v in head_lens] or [0]))
        self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
        self.cu_klen = self.cu_headlens - self.head_lens
        self.cu_klen = torch.cat(
            [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=device)],
            dim=0,
        )
        self.layer_qlens = torch.ones(metadata_heads, dtype=torch.int32, device=device)
        self.qlen_sum = int(metadata_heads)
        self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
        self.cu_qlen = torch.cat(
            [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=device)],
            dim=0,
        )
        self.cu_offset = torch.arange(0, metadata_heads + 1, dtype=torch.int32, device=device)
        self.cu_head_offset = torch.arange(1, metadata_heads + 1, dtype=torch.int32, device=device)

    @staticmethod
    def _span_index_tensor(spans: Sequence[Tuple[int, int]], *, device):
        compact_spans = [(int(start), int(end)) for start, end in spans if int(end) > int(start)]
        if not compact_spans:
            return None
        if len(compact_spans) == 1:
            start, end = compact_spans[0]
            return torch.arange(start, end, device=device, dtype=torch.long)

        starts_list = [start for start, _ in compact_spans]
        lengths_list = [end - start for start, end in compact_spans]
        total = sum(lengths_list)
        if total <= 0:
            return None
        starts = torch.tensor(starts_list, device=device, dtype=torch.long)
        lengths = torch.tensor(lengths_list, device=device, dtype=torch.long)
        flat_offsets = torch.arange(total, device=device, dtype=torch.long)
        span_ends = torch.cumsum(lengths, dim=0)
        span_ids = torch.searchsorted(span_ends, flat_offsets, right=True)
        span_bases = torch.cat([lengths.new_zeros(1), span_ends[:-1]], dim=0)
        return starts.index_select(0, span_ids) + flat_offsets - span_bases.index_select(0, span_ids)

    def _stats(
        self,
        *,
        full_tokens: int,
        compressed_tokens: int,
        recent_tokens: int,
        selected_chunks: int,
        selected_runs: int,
        num_chunks: int,
        chunk_size: int,
        sink_tokens: int,
        two_surrogate_chunks: int,
        mode_counts,
        op_seconds: float = 0.0,
        weighted_entropy: float | None = None,
        weighted_max: float | None = None,
        mapping_alpha: float | None = None,
        configured_keep_ratio: float | None = None,
        dynamic_region_mean_len: float | None = None,
        dynamic_region_max_len: int | None = None,
        dynamic_region_count: int | None = None,
        timing_breakdown: Dict[str, float] | None = None,
    ):
        if self.spec.null_fastpath:
            surrogate_slots = 0
        elif mode_counts and int(mode_counts.get("drop", 0) or 0) > 0:
            surrogate_slots = sum(
                int(count)
                for mode_name, count in mode_counts.items()
                if mode_name != "drop"
            ) + int(two_surrogate_chunks)
        else:
            surrogate_slots = max(0, int(selected_runs) + int(two_surrogate_chunks))
        kept_tokens = max(0, int(compressed_tokens) - int(recent_tokens) - int(sink_tokens) - surrogate_slots)
        kept_chunks = max(0, int(num_chunks) - int(selected_runs))
        if configured_keep_ratio is None:
            configured_keep_ratio = self.layer_keep_ratio
        if configured_keep_ratio is None:
            configured_keep_ratio = min(1.0, float(self.max_capacity_prompt) / max(float(full_tokens), 1.0))
        stats = {
            "full_tokens": full_tokens,
            "compressed_tokens": compressed_tokens,
            "recent_tokens": recent_tokens,
            "selected_chunks": selected_chunks,
            "selected_runs": selected_runs,
            "surrogate_slots": surrogate_slots,
            "kept_tokens": kept_tokens,
            "kept_chunks": kept_chunks,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "sink_tokens": sink_tokens,
            "two_surrogate_chunks": two_surrogate_chunks,
            "configured_keep_ratio": float(configured_keep_ratio),
            "avg_weight_entropy": weighted_entropy,
            "avg_weight_max": weighted_max,
            "avg_mapping_alpha": mapping_alpha,
            "dynamic_region_mean_len": dynamic_region_mean_len,
            "dynamic_region_max_len": dynamic_region_max_len,
            "dynamic_region_count": dynamic_region_count,
            "mode_counts": mode_counts,
            "op_seconds": float(op_seconds),
        }
        if getattr(self, "layer_idx", None) is not None:
            try:
                stats["surrogate_kv_layer_idx"] = int(self.layer_idx)
            except (TypeError, ValueError):
                pass
        if timing_breakdown:
            for name in ("score", "planning", "prototype", "packing"):
                stats[f"timing_{name}_seconds"] = float(timing_breakdown.get(name, 0.0) or 0.0)
        score_stats = getattr(self, "_last_score_stats", None)
        if score_stats:
            stats.update(score_stats)
        allocator_stats = getattr(self, "_last_allocator_stats", None)
        if allocator_stats:
            stats.update(allocator_stats)
        return stats

    def _merge_mode_counts(self, mode_counts_per_batch):
        merged = {}
        for counts in mode_counts_per_batch:
            for mode_name, value in counts.items():
                merged[mode_name] = merged.get(mode_name, 0) + value
        return merged

    def _chunk_slices(self, length: int, chunk_size: int) -> List[Tuple[int, int]]:
        cache_key = (int(length), int(chunk_size))
        cached = _CHUNK_SLICE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        chunk_slices = [(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]
        return _cache_put(_CHUNK_SLICE_CACHE, cache_key, chunk_slices)

    def _is_regular_chunk_layout(self, chunk_slices: Sequence[Tuple[int, int]]) -> bool:
        if len(chunk_slices) <= 1:
            return True
        base_start = int(chunk_slices[0][0])
        chunk_size = int(chunk_slices[0][1] - chunk_slices[0][0])
        if chunk_size <= 0:
            return False
        for idx, (start, end) in enumerate(chunk_slices):
            start = int(start)
            end = int(end)
            if start != base_start + idx * chunk_size:
                return False
            length = end - start
            if idx < len(chunk_slices) - 1 and length != chunk_size:
                return False
            if idx == len(chunk_slices) - 1 and not (0 < length <= chunk_size):
                return False
        return True

    def _chunk_statistics_generic_mean_max(self, *, token_scores, chunk_slices: Sequence[Tuple[int, int]]):
        if not chunk_slices:
            empty = token_scores.new_empty((token_scores.shape[0], 0))
            return empty, empty
        means = []
        maxes = []
        for start, end in chunk_slices:
            chunk = token_scores[:, int(start) : int(end)]
            means.append(chunk.mean(dim=-1))
            maxes.append(chunk.max(dim=-1).values)
        return torch.stack(means, dim=-1), torch.stack(maxes, dim=-1)

    def _contiguous_chunk_span(self, chunk_slices: Sequence[Tuple[int, int]]) -> Tuple[int, int] | None:
        if not chunk_slices:
            return None
        prev_end = int(chunk_slices[0][0])
        for start, end in chunk_slices:
            start = int(start)
            end = int(end)
            if end <= start or start != prev_end:
                return None
            prev_end = end
        return int(chunk_slices[0][0]), int(chunk_slices[-1][1])

    def _chunk_statistics_irregular_prefix_mean_max(self, *, token_scores, chunk_slices: Sequence[Tuple[int, int]]):
        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return self._chunk_statistics_generic_mean_max(
                token_scores=token_scores,
                chunk_slices=chunk_slices,
            )

        base_start, base_end = span
        device = token_scores.device
        starts = torch.tensor([int(start) - base_start for start, _ in chunk_slices], device=device, dtype=torch.long)
        ends = torch.tensor([int(end) - base_start for _, end in chunk_slices], device=device, dtype=torch.long)
        lengths = (ends - starts).clamp_min(1).to(dtype=torch.float32)

        # Dynamic regions are irregular, so reshape-based chunking is unavailable.
        # Prefix sums and scatter-reduce keep this path vectorized instead of
        # slicing every region in Python.
        scores = token_scores[:, base_start:base_end].to(dtype=torch.float32)
        zero = scores.new_zeros((scores.shape[0], 1))
        prefix = torch.cat([zero, scores.cumsum(dim=1)], dim=1)
        means = (prefix.index_select(1, ends) - prefix.index_select(1, starts)) / lengths.view(1, -1)

        token_chunk_ids = torch.arange(len(chunk_slices), device=device, dtype=torch.long).repeat_interleave(
            (ends - starts).clamp_min(1)
        )
        maxes = scores.new_full((scores.shape[0], len(chunk_slices)), torch.finfo(scores.dtype).min)
        maxes.scatter_reduce_(
            1,
            token_chunk_ids.view(1, -1).expand(scores.shape[0], -1),
            scores,
            reduce="amax",
            include_self=False,
        )
        return means, maxes

    def _dynamic_region_slices(
        self,
        *,
        token_scores,
        key_states,
        compressible_start: int,
        compressible_len: int,
        max_region_len: int,
    ) -> List[Tuple[int, int]]:
        """Build bounded dynamic regions with score-valley boundaries.

        Dynamic may move boundaries, but very large regions collapsed too many
        retrieval landmarks into a single KV entry.  Use a modest 2x fixed-chunk
        budget by default: enough room to avoid wasting surrogate slots on
        tiny safe spans, while still preventing the old 128-token collapse mode.
        """
        compressible_start = int(compressible_start)
        compressible_len = int(compressible_len)
        if compressible_len <= 0:
            return []

        target_region_len = max(1, min(int(max_region_len), compressible_len))
        dynamic_region_cap = max(target_region_len, int(self.chunk_size) * 2)
        configured_micro_len = int(self.spec.dynamic_anchor_width or 8)
        micro_len = max(1, min(configured_micro_len, target_region_len, compressible_len))
        if compressible_len <= micro_len:
            return [(compressible_start, compressible_start + compressible_len)]

        micro_slices = [(start, min(start + micro_len, compressible_len)) for start in range(0, compressible_len, micro_len)]
        num_micro = len(micro_slices)
        if num_micro <= 1:
            return [(compressible_start, compressible_start + compressible_len)]

        target_regions = max(1, min(num_micro, math.ceil(compressible_len / max(1, target_region_len))))
        if target_regions >= num_micro:
            return [(compressible_start + start, compressible_start + end) for start, end in micro_slices]

        starts = [int(start) for start, _ in micro_slices]
        ends = [int(end) for _, end in micro_slices]
        regular_micro = 0
        for idx, (start, end) in enumerate(micro_slices):
            if int(start) != idx * int(micro_len):
                break
            if int(end) - int(start) != int(micro_len):
                break
            regular_micro += 1

        # Compute microchunk risk on-device, then copy only the micro-risk
        # vector to Python for boundary picking.  The older path copied every
        # token score to CPU and did the reductions in Python, which made the
        # Dynamic hot path pay a fixed TTFT tax on every layer.
        scores = token_scores[:, compressible_start : compressible_start + compressible_len].detach().to(dtype=torch.float32)
        mean_parts = []
        if regular_micro > 0:
            regular_tokens = regular_micro * int(micro_len)
            regular_scores = scores[:, :regular_tokens].reshape(scores.shape[0], regular_micro, int(micro_len))
            mean_parts.append(regular_scores.mean(dim=(0, 2)))

        for start, end in micro_slices[regular_micro:]:
            segment = scores[:, int(start) : int(end)]
            mean_parts.append(segment.mean(dim=(0, 1), keepdim=False).view(1))

        means_t = torch.cat(mean_parts, dim=0)
        # The best low/mid-prune Dynamic variant used simple score valleys for
        # boundaries.  Later max-rank peak/variance boundaries made the planner
        # too eager to reshape otherwise stable spans and hurt 65% Qasper.
        micro_risk = _rank01(means_t.view(1, -1))[0]
        micro_risk_list = [float(v) for v in micro_risk.to(device="cpu").tolist()]
        regions: List[Tuple[int, int]] = []
        target_micro = max(1, math.ceil(float(target_region_len) / float(micro_len)))
        min_micro = max(1, target_micro // 2)
        # A selected Dynamic region is represented by one surrogate.  Keep the
        # cap loose enough to merge boring spans, but bounded enough that one
        # surrogate cannot swallow a large retrieval neighborhood.
        max_micro = max(target_micro, math.ceil(float(dynamic_region_cap) / float(micro_len)))

        # Boundary risk at i means cutting between micro i and i+1.  Pick a
        # local low-risk valley around the target chunk size.  This preserves
        # the low-prune behavior that worked, while still allowing boundaries
        # to move away from fixed 32-token grid cuts.
        boundary_cost = []
        for idx in range(num_micro - 1):
            left = float(micro_risk_list[idx])
            right = float(micro_risk_list[idx + 1])
            boundary_cost.append(min(left, right) + 0.15 * abs(left - right))

        cursor = 0
        while cursor < num_micro:
            remaining = num_micro - cursor
            if remaining <= max_micro:
                end_micro = num_micro
            else:
                left = cursor + min_micro
                right = min(cursor + max_micro, num_micro - 1)
                ideal = cursor + target_micro
                if left >= right:
                    end_micro = min(cursor + target_micro, num_micro)
                else:
                    best_boundary = left
                    best_score = float("inf")
                    for boundary in range(left, right + 1):
                        cut_idx = boundary - 1
                        length_penalty = 0.015 * abs(boundary - ideal) / max(float(target_micro), 1.0)
                        score = float(boundary_cost[cut_idx]) + length_penalty
                        if score < best_score:
                            best_score = score
                            best_boundary = boundary
                    end_micro = best_boundary

            start = compressible_start + int(starts[cursor])
            end = compressible_start + int(ends[end_micro - 1])
            if end > start:
                regions.append((start, end))
            cursor = end_micro

        return regions

    def _dynamic_surrogate_kv_allocation(
        self,
        *,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        chunk_lengths,
        target_compressed_tokens: int,
        sink_len: int,
        recent_len: int,
        predictive: bool = False,
    ):
        """SurrogateKV allocator.

        Single-ledger frontier exchange.  The allocator builds one raw
        frontier, generates K-D-K residual packets from that frontier, and
        admits each packet only when the same budget ledger can pay for the
        surrogate slot and immediately buy back whole raw atoms with any freed
        entries.  There is no terminal repair/fill pass: budget accounting is
        part of admission.
        """
        del chunk_lengths
        profile_timing = bool(_SURKV_PROFILE_TIMING)
        profile_t0 = time.perf_counter()
        profile_last = profile_t0
        profile_times: Dict[str, float] = {}

        def profile_mark(name: str) -> None:
            nonlocal profile_last
            if not profile_timing:
                return
            now = time.perf_counter()
            profile_times[name] = profile_times.get(name, 0.0) + float(now - profile_last)
            profile_last = now

        def profile_export() -> Dict[str, float]:
            if not profile_timing:
                return {}
            now = time.perf_counter()
            payload = {
                f"surrogate_kv_timing_alloc_{name}_seconds": float(value)
                for name, value in profile_times.items()
            }
            payload["surrogate_kv_timing_alloc_total_seconds"] = float(now - profile_t0)
            payload["surrogate_kv_timing_alloc_unmarked_seconds"] = max(
                0.0,
                float(now - profile_t0) - float(sum(profile_times.values())),
            )
            return payload

        if not chunk_slices or token_scores.shape[0] != 1:
            return None

        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return None
        base_start, base_end = span
        if int(base_end) <= int(base_start):
            return None

        micro_len = max(1, int(self.spec.dynamic_anchor_width or (int(self.chunk_size) // 4)))
        setup = getattr(self, "_precomputed_allocator_setup", None)
        setup_head = int(getattr(self, "_precomputed_allocator_head", -1))
        precomputed_layout = (
            not bool(predictive)
            and isinstance(setup, dict)
            and int(setup.get("base_start", -1)) == int(base_start)
            and int(setup.get("base_end", -1)) == int(base_end)
            and int(setup.get("micro_len", -1)) == int(micro_len)
            and all(
                key in setup
                for key in (
                    "atom_start",
                    "atom_end",
                    "atom_len_int",
                    "prefix_len",
                    "num_atoms",
                    "regular_atoms",
                    "regular_tokens",
                    "tail_floor",
                    "full_cost",
                )
            )
            and int(setup_head) >= 0
        )
        if bool(precomputed_layout):
            atom_start_arr = np.asarray(setup["atom_start"], dtype=np.int64)
            atom_end_arr = np.asarray(setup["atom_end"], dtype=np.int64)
            atom_len_int_arr = np.asarray(setup["atom_len_int"], dtype=np.int64)
            prefix_len = np.asarray(setup["prefix_len"], dtype=np.int64)
            regular_atoms = int(setup["regular_atoms"])
            regular_tokens = int(setup["regular_tokens"])
            num_atoms = int(setup["num_atoms"])
            full_cost = int(setup["full_cost"])
            tail_floor = float(setup["tail_floor"])
        else:
            atom_start_arr = np.arange(int(base_start), int(base_end), int(micro_len), dtype=np.int64)
            if atom_start_arr.size <= 0:
                return None
            atom_end_arr = np.minimum(atom_start_arr + int(micro_len), int(base_end)).astype(np.int64)
            atom_len_int_arr = (atom_end_arr - atom_start_arr).astype(np.int64)
            prefix_len = np.concatenate(([0], np.cumsum(atom_len_int_arr))).astype(np.int64)
            regular_atoms = int((int(base_end) - int(base_start)) // int(micro_len))
            regular_tokens = int(regular_atoms) * int(micro_len)
            num_atoms = int(atom_start_arr.size)
            if num_atoms <= 0:
                return None
            full_cost = int(atom_len_int_arr.sum())
            tail_floor = 1.0 / float(max(2, num_atoms + 1))
        if num_atoms <= 0:
            return None
        tail_price_scale = -math.log(float(tail_floor))
        use_precomputed_setup = bool(precomputed_layout)

        if bool(use_precomputed_setup):
            mean_risk_arr = np.asarray(setup["mean_risk"][int(setup_head)], dtype=np.float64)
            future_risk_arr = np.asarray(setup["future_risk"][int(setup_head)], dtype=np.float64)
            atom_risk_arr = np.asarray(setup["atom_risk"][int(setup_head)], dtype=np.float64)
            mean_signal_arr = np.asarray(setup["mean_signal"][int(setup_head)], dtype=np.float64)
            future_signal_arr = np.asarray(setup["future_signal"][int(setup_head)], dtype=np.float64)
            atom_signal_arr = np.asarray(setup["atom_signal"][int(setup_head)], dtype=np.float64)
            surrogate_signal_arr = np.asarray(setup["surrogate_signal"][int(setup_head)], dtype=np.float64)
        else:
            scores = token_scores[:, int(base_start) : int(base_end)].detach().to(dtype=torch.float32)

            def atom_mean_peak_parts(score_tensor):
                local_mean_parts = []
                local_peak_parts = []
                if regular_atoms > 0:
                    regular = score_tensor[0, :regular_tokens].reshape(int(regular_atoms), int(micro_len))
                    local_mean_parts.append(regular.mean(dim=1))
                    local_peak_parts.append(regular.max(dim=1).values)
                if regular_atoms < int(atom_start_arr.size):
                    segment = score_tensor[0, int(regular_atoms) * int(micro_len) :]
                    local_mean_parts.append(segment.mean().view(1))
                    local_peak_parts.append(segment.max().view(1))
                return local_mean_parts, local_peak_parts

            mean_parts, peak_parts = atom_mean_peak_parts(scores)
            if not mean_parts:
                return None

            atom_mean = torch.cat(mean_parts, dim=0).view(1, -1)
            atom_peak = torch.cat(peak_parts, dim=0).view(1, -1)
            surrogate_atom_mean = atom_mean
            surrogate_atom_peak = atom_peak
            mean_rank_t = _rank01(atom_mean)[0]
            peak_rank_t = _rank01(atom_peak)[0]
            surrogate_mean_rank_t = _rank01(surrogate_atom_mean)[0]
            surrogate_peak_rank_t = _rank01(surrogate_atom_peak)[0]
            surrogate_risk_t = torch.maximum(surrogate_mean_rank_t, surrogate_peak_rank_t)
            current_risk_t = torch.maximum(mean_rank_t, peak_rank_t)
            if bool(predictive):
                atom_spread = torch.clamp(atom_peak - atom_mean, min=0.0)
                spread_rank_t = _rank01(atom_spread)[0]
                if atom_mean.shape[1] > 1:
                    left_mean = torch.cat((atom_mean[:, :1], atom_mean[:, :-1]), dim=1)
                    right_mean = torch.cat((atom_mean[:, 1:], atom_mean[:, -1:]), dim=1)
                    neighbor_floor = torch.maximum(left_mean, right_mean)
                    local_contrast = torch.clamp(atom_peak - neighbor_floor, min=0.0)
                    contrast_rank_t = _rank01(local_contrast)[0]
                else:
                    contrast_rank_t = spread_rank_t
                future_rank_t = torch.minimum(spread_rank_t, contrast_rank_t)
                # Future-looking spikes should protect raw atoms first.  Letting
                # them inflate residual surrogate value makes long, incoherent
                # packets look cheap at tight budgets.
                atom_risk_t = torch.maximum(current_risk_t, future_rank_t)
            else:
                future_rank_t = torch.zeros_like(current_risk_t)
                atom_risk_t = current_risk_t
            if bool(predictive):
                risk_arrays = (
                    torch.stack((mean_rank_t, future_rank_t, atom_risk_t), dim=0)
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .numpy()
                )
                mean_risk_arr = risk_arrays[0] + 1e-6
                future_risk_arr = risk_arrays[1] + 1e-6
                atom_risk_arr = risk_arrays[2] + 1e-6
            else:
                risk_arrays = (
                    torch.stack((mean_rank_t, atom_risk_t), dim=0)
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .numpy()
                )
                mean_risk_arr = risk_arrays[0] + 1e-6
                future_risk_arr = np.full_like(mean_risk_arr, 1e-6)
                atom_risk_arr = risk_arrays[1] + 1e-6

            def tail_scores(values: Sequence[float]) -> np.ndarray:
                ranks = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
                return -np.log(np.maximum(float(tail_floor), 1.0 - ranks))

            mean_signal_arr = tail_scores(mean_risk_arr)
            future_signal_arr = tail_scores(future_risk_arr)
            atom_signal_arr = tail_scores(atom_risk_arr)
            surrogate_risk_arr = (
                surrogate_risk_t.detach().to(device="cpu", dtype=torch.float64).numpy() + 1e-6
            )
            surrogate_signal_arr = tail_scores(surrogate_risk_arr)

        device = token_scores.device
        atom_indices_arr = np.arange(num_atoms, dtype=np.int64)

        budget_entries = int(target_compressed_tokens) - int(sink_len) - int(recent_len)
        budget_entries = max(1, int(budget_entries))

        # Predictive evidence participates in raw frontier protection above.
        # Residual packets stay mean-evidence based so a future spike does not
        # turn a long mixed span into one over-valued mean surrogate.
        precomputed_ledger = bool(
            use_precomputed_setup
            and all(
                key in setup
                for key in (
                    "atom_len",
                    "atom_indices",
                    "raw_value",
                    "raw_density",
                    "prefix_mass",
                    "prefix_energy",
                    "raw_keep_order",
                    "raw_drop_order",
                )
            )
        )
        if bool(precomputed_ledger):
            atom_len_arr = np.asarray(setup["atom_len"], dtype=np.float64)
            atom_indices_arr = np.asarray(setup["atom_indices"], dtype=np.int64)
            raw_value_arr = np.asarray(setup["raw_value"][int(setup_head)], dtype=np.float64)
            raw_density_arr = np.asarray(setup["raw_density"][int(setup_head)], dtype=np.float64)
            prefix_mass = np.asarray(setup["prefix_mass"][int(setup_head)], dtype=np.float64)
            prefix_energy = np.asarray(setup["prefix_energy"][int(setup_head)], dtype=np.float64)
            raw_keep_order = np.asarray(setup["raw_keep_order"][int(setup_head)], dtype=np.int64)
            raw_drop_order = np.asarray(setup["raw_drop_order"][int(setup_head)], dtype=np.int64)
        else:
            atom_len_arr = np.maximum(1.0, atom_len_int_arr.astype(np.float64))
            raw_value_arr = atom_signal_arr * atom_signal_arr * atom_len_arr
            raw_density_arr = raw_value_arr / np.maximum(atom_len_arr, 1.0)
            prefix_mass = np.concatenate(([0.0], np.cumsum(surrogate_signal_arr * atom_len_arr))).astype(np.float64)
            prefix_energy = np.concatenate(
                ([0.0], np.cumsum(surrogate_signal_arr * surrogate_signal_arr * atom_len_arr))
            ).astype(np.float64)
            raw_keep_order = np.lexsort((atom_indices_arr, -atom_risk_arr))
            raw_drop_order = np.lexsort((atom_indices_arr, atom_risk_arr))
        profile_mark("setup_rank_prefix")

        raw_keep_order_list = raw_keep_order.astype(np.int64, copy=False).tolist()
        raw_drop_order_list = raw_drop_order.astype(np.int64, copy=False).tolist()
        common_atom_len = max(1, int(micro_len))
        non_common_atom_indices = np.flatnonzero(atom_len_int_arr != int(common_atom_len)).astype(np.int64)
        can_fast_best_effort_fill = bool(non_common_atom_indices.size <= 1)

        def action_runs(actions_ref: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            actions_arr = np.asarray(actions_ref, dtype=np.int8)
            if actions_arr.size <= 0:
                empty_i = np.empty((0,), dtype=np.int64)
                empty_a = np.empty((0,), dtype=np.int8)
                return empty_i, empty_i, empty_a, empty_i
            changes = np.flatnonzero(actions_arr[1:] != actions_arr[:-1]).astype(np.int64) + 1
            starts = np.concatenate((np.asarray([0], dtype=np.int64), changes))
            ends = np.concatenate((changes, np.asarray([actions_arr.size], dtype=np.int64)))
            run_actions = actions_arr[starts]
            run_lens = prefix_len[ends] - prefix_len[starts]
            return starts, ends, run_actions, run_lens

        def collect_stats(actions_ref: Sequence[int]) -> Tuple[Dict[str, float], List[int]]:
            _starts, _ends, run_actions, run_lens = action_runs(actions_ref)
            raw_mask = run_actions == 2
            surrogate_mask = run_actions == 1
            drop_mask = run_actions == 0
            raw_tokens = int(run_lens[raw_mask].sum()) if raw_mask.any() else 0
            raw_regions = int(raw_mask.sum())
            surrogate_tokens = int(run_lens[surrogate_mask].sum()) if surrogate_mask.any() else 0
            surrogate_regions = int(surrogate_mask.sum())
            drop_tokens = int(run_lens[drop_mask].sum()) if drop_mask.any() else 0
            drop_regions = int(drop_mask.sum())
            surrogate_lens = run_lens[surrogate_mask].astype(np.int64).tolist()
            used_entries = int(raw_tokens) + int(surrogate_regions)
            return (
                {
                    "ks_run_raw_tokens": int(raw_tokens),
                    "ks_run_raw_regions": int(raw_regions),
                    "ks_run_surrogate_regions": int(surrogate_regions),
                    "ks_run_surrogate_tokens": int(surrogate_tokens),
                    "ks_run_drop_tokens": int(drop_tokens),
                    "ks_run_drop_regions": int(drop_regions),
                    "ks_run_budget_entries": int(budget_entries),
                    "ks_run_full_cost": int(full_cost),
                    "ks_run_used_entries": int(used_entries),
                    "ks_run_budget_gap": int(budget_entries) - int(used_entries),
                    "ks_run_surrogate_mean_len": (
                        float(sum(surrogate_lens) / len(surrogate_lens)) if surrogate_lens else 0.0
                    ),
                    "ks_run_surrogate_max_len": max(surrogate_lens) if surrogate_lens else 0,
                },
                surrogate_lens,
            )

        def materialize_actions(actions_ref: Sequence[int], *, tensors: bool = True):
            starts, ends, run_actions, _run_lens = action_runs(actions_ref)
            new_slices = [
                (int(atom_start_arr[int(start_idx)]), int(atom_end_arr[int(end_idx) - 1]))
                for start_idx, end_idx in zip(starts.tolist(), ends.tolist())
            ]
            new_selected = run_actions != 2
            new_surrogate_lengths = np.where(run_actions == 0, 0, 1).astype(np.int64)
            new_chunk_lengths_arr = (
                atom_end_arr[ends.astype(np.int64) - 1] - atom_start_arr[starts.astype(np.int64)]
            ).astype(np.int64, copy=False)
            output_lengths_arr = np.where(
                new_selected,
                new_surrogate_lengths,
                new_chunk_lengths_arr,
            ).astype(np.int64, copy=False)
            selected_indices = np.flatnonzero(new_selected).astype(np.int64)
            surrogate_indices = selected_indices[new_surrogate_lengths[selected_indices] > 0]
            self._last_fast_pack_plan = {
                "chunk_slices": tuple(new_slices),
                "selected_mask_list": [bool(v) for v in new_selected.tolist()],
                "output_length_list": [int(v) for v in output_lengths_arr.tolist()],
                "raw_spans": [
                    (int(new_slices[idx][0]), int(new_slices[idx][1]))
                    for idx in range(len(new_slices))
                    if not bool(new_selected[idx])
                ],
                "selected_chunk_indices_list": [int(v) for v in selected_indices.tolist()],
                "surrogate_chunk_indices_list": [int(v) for v in surrogate_indices.tolist()],
                "surrogate_lengths_list": [
                    int(new_surrogate_lengths[int(idx)])
                    for idx in surrogate_indices.tolist()
                ],
            }
            if not bool(tensors):
                return True
            new_chunk_lengths = torch.as_tensor(new_chunk_lengths_arr, device=device, dtype=torch.long)
            new_replace_mask = torch.as_tensor(new_selected[None, :], device=device, dtype=torch.bool)
            new_surrogate_lengths_tensor = torch.as_tensor(
                new_surrogate_lengths[None, :],
                device=device,
                dtype=torch.long,
            )
            return new_slices, new_chunk_lengths, new_replace_mask, new_surrogate_lengths_tensor

        profile_mark("define_base_helpers")

        if budget_entries >= full_cost:
            actions_full = np.full((num_atoms,), 2, dtype=np.int8)
            stats, _ = collect_stats(actions_full)
            profile_mark("full_budget_stats")
            stats.update(
                {
                    "surrogate_kv_allocator": "online_frontier_exchange",
                    "surrogate_kv_candidate_seeds": 0,
                    "surrogate_kv_candidate_surrogates": 0,
                    "surrogate_kv_stack_merges": 0,
                    "surrogate_kv_selected_surrogates": 0,
                    "surrogate_kv_online_buyback_atoms": 0,
                    "surrogate_kv_online_buyback_tokens": 0,
                    "surrogate_kv_online_buyback_value": 0.0,
                    "surrogate_kv_budget_gap": int(budget_entries) - int(full_cost),
                    "surrogate_kv_tail_price_scale": float(tail_price_scale),
                    "surrogate_kv_predictive": int(bool(predictive)),
                }
            )
            stats.update(profile_export())
            self._last_allocator_stats = stats
            if bool(getattr(self, "_allocator_plan_only", False)):
                materialize_start = time.perf_counter()
                materialized = materialize_actions(actions_full, tensors=False)
                stats["surrogate_kv_timing_alloc_materialize_seconds"] = float(time.perf_counter() - materialize_start)
            else:
                materialize_start = time.perf_counter()
                materialized = materialize_actions(actions_full)
                stats["surrogate_kv_timing_alloc_materialize_seconds"] = float(time.perf_counter() - materialize_start)
            stats["surrogate_kv_timing_alloc_total_seconds"] = float(time.perf_counter() - profile_t0)
            self._last_allocator_stats = stats
            return materialized

        actions = np.full((num_atoms,), 2, dtype=np.int8)
        drop_required = max(0, int(full_cost) - int(budget_entries))
        if int(drop_required) > 0:
            drop_lengths = atom_len_int_arr[raw_drop_order]
            drop_cumsum = np.cumsum(drop_lengths).astype(np.int64)
            drop_count = int(np.searchsorted(drop_cumsum, int(drop_required), side="left") + 1)
            drop_count = max(0, min(int(drop_count), int(raw_drop_order.size)))
            if int(drop_count) > 0:
                dropped_atoms = raw_drop_order[:drop_count]
                actions[dropped_atoms] = 0
                current_cost = int(full_cost) - int(drop_cumsum[int(drop_count) - 1])
            else:
                current_cost = int(full_cost)
        else:
            current_cost = int(full_cost)

        kept_density = raw_density_arr[actions == 2]
        dropped_density = raw_density_arr[actions == 0]
        raw_slot_price = float(dropped_density.max()) if dropped_density.size else 0.0
        frontier_price = float(kept_density.min()) if kept_density.size else raw_slot_price
        region_open_cost = float(max(raw_slot_price, frontier_price))
        try:
            generation_horizon = int(getattr(self, "generation_horizon", 0) or 0)
        except (TypeError, ValueError):
            generation_horizon = 0
        horizon_risk = max(0.0, min(1.0, (float(generation_horizon) / 512.0) ** 2))
        horizon_surrogate_tax = 1.0 + float(horizon_risk) * float(
            os.environ.get("SURKV_HORIZON_SURROGATE_TAX", "1.5")
        )
        region_open_cost *= float(horizon_surrogate_tax)
        surrogate_open_price = float(raw_slot_price) * float(max(1, int(micro_len)))
        initial_cost = int(current_cost)
        initial_actions = actions.copy()
        initial_buyback_order = raw_keep_order[initial_actions[raw_keep_order] == 0].astype(np.int64, copy=False)
        if initial_buyback_order.size:
            initial_buyback_prefix_len = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(atom_len_int_arr[initial_buyback_order]).astype(np.int64),
                )
            )
            initial_buyback_prefix_value = np.concatenate(
                (
                    np.asarray([0.0], dtype=np.float64),
                    np.cumsum(raw_value_arr[initial_buyback_order]).astype(np.float64),
                )
            )
        else:
            initial_buyback_prefix_len = np.asarray([0], dtype=np.int64)
            initial_buyback_prefix_value = np.asarray([0.0], dtype=np.float64)
        initial_buyback_order_list = initial_buyback_order.astype(np.int64, copy=False).tolist()
        static_raw_mask = (initial_actions == 2).astype(np.float64)
        static_drop_mask = (initial_actions == 0).astype(np.float64)
        prefix_static_raw_tokens = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum((atom_len_int_arr * (initial_actions == 2)).astype(np.int64)),
            )
        )
        prefix_static_raw_count = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum((initial_actions == 2).astype(np.int64)),
            )
        )
        prefix_static_raw_value = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(raw_value_arr * static_raw_mask).astype(np.float64),
            )
        )
        prefix_static_drop_len = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(atom_len_arr * static_drop_mask).astype(np.float64),
            )
        )
        prefix_static_drop_mass = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(surrogate_signal_arr * atom_len_arr * static_drop_mask).astype(np.float64),
            )
        )
        prefix_static_drop_energy = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(
                    surrogate_signal_arr * surrogate_signal_arr * atom_len_arr * static_drop_mask
                ).astype(np.float64),
            )
        )
        static_raw_atoms = np.flatnonzero(initial_actions == 2).astype(np.int64)
        static_raw_value = raw_value_arr[static_raw_atoms] if static_raw_atoms.size else np.empty((0,), dtype=np.float64)
        static_raw_contrib_slope = (
            2.0 * surrogate_signal_arr[static_raw_atoms] * atom_len_arr[static_raw_atoms]
            if static_raw_atoms.size
            else np.empty((0,), dtype=np.float64)
        )
        static_raw_contrib_base = (
            surrogate_signal_arr[static_raw_atoms]
            * surrogate_signal_arr[static_raw_atoms]
            * atom_len_arr[static_raw_atoms]
            if static_raw_atoms.size
            else np.empty((0,), dtype=np.float64)
        )
        profile_mark("initial_frontier")

        def coherent_residual_value_from_mask(
            local_mean: np.ndarray,
            local_len: np.ndarray,
            mask: np.ndarray,
        ) -> float:
            if not bool(np.any(mask)):
                return 0.0
            masked_mean = local_mean[mask]
            masked_len = local_len[mask]
            total_len = float(masked_len.sum())
            if total_len <= 0.0:
                return 0.0
            mass = float((masked_mean * masked_len).sum())
            energy = float((masked_mean * masked_mean * masked_len).sum())
            coherent_mass = float(mass * mass / max(1.0, total_len))
            return float(max(0.0, min(float(energy), float(2.0 * coherent_mass - energy))))

        def coherent_residual_value_from_prefix(start_idx: int, end_idx: int) -> float:
            token_len = int(prefix_len[int(end_idx)] - prefix_len[int(start_idx)])
            if token_len <= 0:
                return 0.0
            mass = float(prefix_mass[int(end_idx)] - prefix_mass[int(start_idx)])
            energy = float(prefix_energy[int(end_idx)] - prefix_energy[int(start_idx)])
            coherent_mass = float(mass * mass / max(1.0, float(token_len)))
            return float(max(0.0, min(float(energy), float(2.0 * coherent_mass - energy))))

        def estimate_initial_buyback_value(slot_count: int) -> float:
            slots = max(0, int(slot_count))
            if slots <= 0 or initial_buyback_prefix_len.size <= 1:
                return 0.0
            pos = int(np.searchsorted(initial_buyback_prefix_len, int(slots), side="right") - 1)
            pos = max(0, min(pos, int(initial_buyback_prefix_value.size) - 1))
            return float(initial_buyback_prefix_value[pos])

        def packet_gain(
            *,
            value: float,
            sold_loss: float,
            sold_deficit: float,
            budget_delta: int,
            static_buyback: bool,
        ) -> float:
            _ = static_buyback
            _ = sold_deficit
            positive_cost = max(0, int(budget_delta))
            freed_slots = max(0, -int(budget_delta))
            buyback_credit = estimate_initial_buyback_value(int(freed_slots))
            return (
                float(value)
                - float(sold_loss)
                - float(sold_deficit)
                - float(region_open_cost)
                - float(raw_slot_price) * float(positive_cost)
                + float(buyback_credit)
            )

        static_packet_cache: Dict[Tuple[int, int], Tuple[float, float, float, int, int] | None] = {}

        def score_static_packet(start_idx: int, end_idx: int):
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            cache_key = (start_idx, end_idx)
            if cache_key in static_packet_cache:
                return static_packet_cache[cache_key]
            if start_idx >= end_idx:
                static_packet_cache[cache_key] = None
                return None
            token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
            if token_len <= 0:
                static_packet_cache[cache_key] = None
                return None
            mass = float(prefix_mass[end_idx] - prefix_mass[start_idx])
            mu = mass / max(1.0, float(token_len))
            drop_len = float(prefix_static_drop_len[end_idx] - prefix_static_drop_len[start_idx])
            residual_value = 0.0
            if drop_len > 0.0:
                drop_mass = float(prefix_static_drop_mass[end_idx] - prefix_static_drop_mass[start_idx])
                drop_energy = float(prefix_static_drop_energy[end_idx] - prefix_static_drop_energy[start_idx])
                residual_contrib = float(2.0 * float(mu) * float(drop_mass) - float(drop_energy))
                coherent_mass = float(drop_mass * drop_mass / max(1.0, float(drop_len)))
                coherent_value = max(0.0, min(float(drop_energy), float(2.0 * coherent_mass - drop_energy)))
                residual_value = min(float(coherent_value), max(0.0, float(residual_contrib)))
            raw_left = int(prefix_static_raw_count[int(start_idx)])
            raw_right = int(prefix_static_raw_count[int(end_idx)])
            raw_count = int(raw_right) - int(raw_left)
            sold_loss = float(prefix_static_raw_value[end_idx] - prefix_static_raw_value[start_idx])
            sold_tokens = int(prefix_static_raw_tokens[end_idx] - prefix_static_raw_tokens[start_idx])
            sold_recovery = 0.0
            if raw_count > 0:
                if int(raw_count) <= 16:
                    recovery_acc = 0.0
                    for raw_pos in range(int(raw_left), int(raw_right)):
                        raw_value_scalar = float(static_raw_value[int(raw_pos)])
                        raw_contrib_scalar = (
                            float(static_raw_contrib_slope[int(raw_pos)]) * float(mu)
                            - float(static_raw_contrib_base[int(raw_pos)])
                        )
                        recovery_acc += min(float(raw_value_scalar), max(0.0, float(raw_contrib_scalar)))
                    sold_recovery = float(recovery_acc)
                else:
                    raw_value = static_raw_value[raw_left:raw_right]
                    raw_contrib = (
                        static_raw_contrib_slope[raw_left:raw_right] * float(mu)
                        - static_raw_contrib_base[raw_left:raw_right]
                    )
                    sold_recovery = float(
                        np.minimum(raw_value, np.maximum(0.0, raw_contrib)).sum()
                    )
            sold_deficit = max(0.0, float(sold_loss) - float(sold_recovery))
            value = min(
                float(residual_value) + float(sold_recovery),
                coherent_residual_value_from_prefix(int(start_idx), int(end_idx)),
            )
            budget_delta = int(1 - int(sold_tokens))
            gain = packet_gain(
                value=float(value),
                sold_loss=float(sold_loss),
                sold_deficit=float(sold_deficit),
                budget_delta=int(budget_delta),
                static_buyback=True,
            )
            result = (
                float(gain),
                float(value),
                float(sold_loss),
                int(budget_delta),
                int(token_len),
            )
            static_packet_cache[cache_key] = result
            return result

        # Candidate tuple:
        # gain, value, sold_loss, budget_delta, token_len, start, end, seed_count
        Candidate = Tuple[float, float, float, int, int, int, int, int]
        starts, ends, run_actions, _run_lens = action_runs(initial_actions)
        seeds: List[Candidate] = []
        candidate_count = 0
        shadow_candidate_count = 0

        def append_static_candidate(
            out: List[Candidate],
            start_idx: int,
            end_idx: int,
            seed_count: int,
        ) -> None:
            metrics = score_static_packet(int(start_idx), int(end_idx))
            if metrics is None:
                return
            gain, value, sold_loss, budget_delta, token_len = metrics
            out.append(
                (
                    float(gain),
                    float(value),
                    float(sold_loss),
                    int(budget_delta),
                    int(token_len),
                    int(start_idx),
                    int(end_idx),
                    int(seed_count),
                )
            )

        profile_mark("define_candidate_helpers")

        def append_drop_run_candidates_from_actions(actions_for_runs: np.ndarray, *, shadow: bool) -> int:
            local_starts, local_ends, local_actions, _local_lens = action_runs(actions_for_runs)
            added = 0
            for run_idx, action in enumerate(local_actions.tolist()):
                if int(action) != 0:
                    continue
                if int(run_idx) <= 0 or int(run_idx) + 1 >= int(local_actions.size):
                    continue
                if int(local_actions[int(run_idx) - 1]) != 2 or int(local_actions[int(run_idx) + 1]) != 2:
                    continue
                start_idx = int(local_starts[int(run_idx)])
                end_idx = int(local_ends[int(run_idx)])
                append_static_candidate(seeds, start_idx, end_idx, 1)
                added += 1
            return int(added)

        candidate_count += append_drop_run_candidates_from_actions(initial_actions, shadow=False)

        # Mid budgets can lose the broad packets that were valuable at tighter
        # budgets because the current raw frontier breaks one long residual run
        # into many short ones.  Add one nested "price octave" frontier as extra
        # packet candidates; the normal current-ledger scoring below still
        # decides whether the packet is worth buying under the actual budget.
        nested_frontier_price = float(region_open_cost) * 2.0
        if math.isfinite(nested_frontier_price) and nested_frontier_price > 0.0:
            shadow_actions = np.where(raw_density_arr >= float(nested_frontier_price), 2, 0).astype(np.int8)
            if bool(np.any(shadow_actions == 2)) and bool(np.any(shadow_actions == 0)):
                before_count = int(len(seeds))
                candidate_count += append_drop_run_candidates_from_actions(shadow_actions, shadow=True)
                shadow_candidate_count = int(len(seeds)) - int(before_count)
        profile_mark("seed_candidates")

        candidates: List[Candidate] = list(seeds)
        stack: List[Candidate] = []
        stack_merges = 0
        pareto_merge_candidates = 0
        for seed in sorted(seeds, key=lambda item: (item[5], item[6])):
            stack.append(seed)
            while len(stack) >= 2:
                left = stack[-2]
                right = stack[-1]
                left_start = int(left[5])
                left_end = int(left[6])
                right_start = int(right[5])
                right_end = int(right[6])
                if int(left_end) > int(right_start):
                    break
                if int(left_end) == int(right_start):
                    break
                gap_actions = initial_actions[int(left_end) : int(right_start)]
                if gap_actions.size <= 0 or bool(np.any(gap_actions != 2)):
                    break
                metrics = score_static_packet(int(left_start), int(right_end))
                if metrics is None:
                    break
                gain, value, sold_loss, budget_delta, token_len = metrics
                split_gain = float(left[0]) + float(right[0])
                split_delta = int(left[3]) + int(right[3])
                if float(gain) <= 0.0 or float(gain) <= float(split_gain):
                    break
                merged: Candidate = (
                    float(gain),
                    float(value),
                    float(sold_loss),
                    int(budget_delta),
                    int(token_len),
                    int(left_start),
                    int(right_end),
                    int(left[7]) + int(right[7]),
                )
                stack.pop()
                stack.pop()
                stack.append(merged)
                candidates.append(merged)
                stack_merges += 1
        profile_mark("merge_candidates")

        online_buyback_atoms = 0
        online_buyback_tokens = 0
        online_buyback_value = 0.0
        current_drop_tokens = int(atom_len_int_arr[actions == 0].sum())
        buy_cursor = 0

        def buy_raw_until_full() -> None:
            nonlocal buy_cursor
            nonlocal current_cost
            nonlocal online_buyback_atoms, online_buyback_tokens, online_buyback_value
            nonlocal current_drop_tokens
            while int(current_cost) < int(budget_entries) and int(buy_cursor) < len(initial_buyback_order_list):
                atom_idx = int(initial_buyback_order_list[int(buy_cursor)])
                if int(current_cost) >= int(budget_entries):
                    break
                if int(actions[int(atom_idx)]) != 0:
                    buy_cursor += 1
                    continue
                atom_len = int(atom_len_int_arr[int(atom_idx)])
                if int(current_cost) + int(atom_len) > int(budget_entries):
                    break
                actions[int(atom_idx)] = 2
                current_cost += int(atom_len)
                current_drop_tokens -= int(atom_len)
                online_buyback_atoms += 1
                online_buyback_tokens += int(atom_len)
                online_buyback_value += float(raw_value_arr[int(atom_idx)])
                buy_cursor += 1

        def buy_raw_best_effort() -> None:
            nonlocal current_cost
            nonlocal online_buyback_atoms, online_buyback_tokens, online_buyback_value
            nonlocal current_drop_tokens
            if int(current_cost) >= int(budget_entries):
                return
            if bool(can_fast_best_effort_fill):
                remaining = int(budget_entries) - int(current_cost)
                if remaining <= 0:
                    return
                eligible_order = raw_keep_order[actions[raw_keep_order] == 0]
                if eligible_order.size <= 0:
                    return

                selected_parts: List[np.ndarray] = []
                eligible_lens = atom_len_int_arr[eligible_order]
                special_positions = np.flatnonzero(eligible_lens != int(common_atom_len)).astype(np.int64)

                def take_common_segment(segment: np.ndarray) -> None:
                    nonlocal remaining
                    if segment.size <= 0 or remaining < int(common_atom_len):
                        return
                    take_count = min(int(segment.size), int(remaining) // int(common_atom_len))
                    if take_count <= 0:
                        return
                    selected_parts.append(segment[:take_count])
                    remaining -= int(take_count) * int(common_atom_len)

                if special_positions.size == 0:
                    take_common_segment(eligible_order)
                else:
                    special_pos = int(special_positions[0])
                    take_common_segment(eligible_order[:special_pos])
                    special_atom = int(eligible_order[special_pos])
                    special_len = int(atom_len_int_arr[special_atom])
                    if special_len > 0 and special_len <= int(remaining):
                        selected_parts.append(eligible_order[special_pos : special_pos + 1])
                        remaining -= int(special_len)
                    take_common_segment(eligible_order[special_pos + 1 :])

                if selected_parts:
                    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
                    if selected.size > 0:
                        selected_tokens = int(atom_len_int_arr[selected].sum())
                        selected_value = float(raw_value_arr[selected].sum())
                        actions[selected] = 2
                        current_cost += int(selected_tokens)
                        current_drop_tokens -= int(selected_tokens)
                        online_buyback_atoms += int(selected.size)
                        online_buyback_tokens += int(selected_tokens)
                        online_buyback_value += float(selected_value)
                return
            for atom_idx in raw_keep_order_list:
                atom_idx = int(atom_idx)
                if int(actions[atom_idx]) != 0:
                    continue
                atom_len = int(atom_len_int_arr[atom_idx])
                if atom_len <= 0 or int(current_cost) + int(atom_len) > int(budget_entries):
                    continue
                actions[atom_idx] = 2
                current_cost += int(atom_len)
                current_drop_tokens -= int(atom_len)
                online_buyback_atoms += 1
                online_buyback_tokens += int(atom_len)
                online_buyback_value += float(raw_value_arr[atom_idx])
                if int(current_cost) >= int(budget_entries):
                    break

        def estimate_current_buyback_value(slot_count: int) -> float:
            slots = max(0, int(slot_count))
            if slots <= 0:
                return 0.0
            used = 0
            value = 0.0
            for atom_idx in raw_keep_order_list:
                atom_idx = int(atom_idx)
                if int(actions[atom_idx]) != 0:
                    continue
                atom_len = int(atom_len_int_arr[atom_idx])
                if used + int(atom_len) > slots:
                    continue
                used += int(atom_len)
                value += float(raw_value_arr[atom_idx])
                if used >= slots:
                    break
            return float(value)

        def estimate_cursor_buyback(slot_count: int) -> Tuple[int, float]:
            slots = max(0, int(slot_count))
            if slots <= 0 or int(buy_cursor) >= int(initial_buyback_prefix_len.size) - 1:
                return 0, 0.0
            base_len = int(initial_buyback_prefix_len[int(buy_cursor)])
            target_len = int(base_len) + int(slots)
            pos = int(np.searchsorted(initial_buyback_prefix_len, int(target_len), side="right") - 1)
            pos = max(int(buy_cursor), min(pos, int(initial_buyback_prefix_len.size) - 1))
            tokens = int(initial_buyback_prefix_len[int(pos)] - initial_buyback_prefix_len[int(buy_cursor)])
            value = float(initial_buyback_prefix_value[int(pos)] - initial_buyback_prefix_value[int(buy_cursor)])
            return int(tokens), float(value)

        def estimate_current_buyback_after_span(
            start_idx: int,
            end_idx: int,
            slot_count: int,
        ) -> Tuple[int, float]:
            slots = max(0, int(slot_count))
            if slots <= 0:
                return 0, 0.0
            used = 0
            value = 0.0
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            for buy_pos in range(int(buy_cursor), len(initial_buyback_order_list)):
                atom_idx = int(initial_buyback_order_list[int(buy_pos)])
                if start_idx <= atom_idx < end_idx:
                    continue
                if int(actions[atom_idx]) != 0:
                    continue
                atom_len = int(atom_len_int_arr[atom_idx])
                if used + int(atom_len) > slots:
                    continue
                used += int(atom_len)
                value += float(raw_value_arr[atom_idx])
                if used >= slots:
                    break
            return int(used), float(value)

        score_current_packet_calls = 0

        def score_current_packet(start_idx: int, end_idx: int):
            nonlocal score_current_packet_calls
            score_current_packet_calls += 1
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            if start_idx >= end_idx:
                return None
            if bool(np.any(actions[start_idx:end_idx] == 1)):
                return None
            token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
            if token_len <= 0:
                return None
            mass = float(prefix_mass[end_idx] - prefix_mass[start_idx])
            mu = mass / max(1.0, float(token_len))
            local_actions = actions[start_idx:end_idx]
            local_mean = surrogate_signal_arr[start_idx:end_idx]
            local_len = atom_len_arr[start_idx:end_idx]
            local_contrib = (2.0 * local_mean * float(mu) - local_mean * local_mean) * local_len
            residual_mask = local_actions == 0
            drop_tokens_inside = int(local_len[residual_mask].sum()) if bool(np.any(residual_mask)) else 0
            residual_contrib = float(local_contrib[residual_mask].sum()) if bool(np.any(residual_mask)) else 0.0
            residual_value = min(
                coherent_residual_value_from_mask(local_mean, local_len, residual_mask),
                max(0.0, float(residual_contrib)),
            )
            raw_inside = np.flatnonzero(local_actions == 2).astype(np.int64) + int(start_idx)
            sold_loss = float(raw_value_arr[raw_inside].sum()) if raw_inside.size else 0.0
            sold_tokens = int(atom_len_int_arr[raw_inside].sum()) if raw_inside.size else 0
            sold_recovery = 0.0
            if raw_inside.size:
                raw_rel = raw_inside - int(start_idx)
                sold_recovery = float(
                    np.minimum(
                        raw_value_arr[raw_inside],
                        np.maximum(0.0, local_contrib[raw_rel]),
                    ).sum()
                )
            sold_deficit = max(0.0, float(sold_loss) - float(sold_recovery))
            value = min(
                float(residual_value) + float(sold_recovery),
                coherent_residual_value_from_prefix(int(start_idx), int(end_idx)),
            )
            left_sur = int(start_idx) > 0 and int(actions[int(start_idx) - 1]) == 1
            right_sur = int(end_idx) < int(num_atoms) and int(actions[int(end_idx)]) == 1
            surrogate_region_delta = int(1 - int(bool(left_sur)) - int(bool(right_sur)))
            budget_delta = int(surrogate_region_delta - int(sold_tokens))
            proposed_cost = int(current_cost) + int(budget_delta)
            needed_payment = max(0, int(proposed_cost) - int(budget_entries))
            payer_atoms: List[int] = []
            payer_tokens = 0
            payer_loss = 0.0
            if int(needed_payment) > 0:
                for atom_idx in raw_drop_order_list:
                    atom_idx = int(atom_idx)
                    if int(actions[atom_idx]) != 2:
                        continue
                    if int(start_idx) <= int(atom_idx) < int(end_idx):
                        continue
                    atom_len = int(atom_len_int_arr[atom_idx])
                    if atom_len <= 0:
                        continue
                    payer_atoms.append(int(atom_idx))
                    payer_tokens += int(atom_len)
                    payer_loss += float(raw_value_arr[atom_idx])
                    if int(payer_tokens) >= int(needed_payment):
                        break
            if int(payer_tokens) < int(needed_payment):
                return (
                    float("-inf"),
                    float(value),
                    float(sold_loss),
                    int(budget_delta),
                    int(token_len),
                    int(raw_inside.size),
                    int(sold_tokens),
                    int(budget_entries),
                    int(drop_tokens_inside),
                    payer_atoms,
                    int(payer_tokens),
                    float(payer_loss),
                    0,
                    0.0,
                )
            after_payment_cost = int(proposed_cost) - int(payer_tokens)
            available_slots = int(budget_entries) - int(after_payment_cost)
            buyback_unit = max(1, int(atom_len_int_arr.min()))
            buyback_slots = int(buyback_unit) * int(max(0, int(available_slots)) // int(buyback_unit))
            buyback_tokens, buyback_value = estimate_cursor_buyback(int(buyback_slots))
            post_gap = int(budget_entries) - int(after_payment_cost) - int(buyback_tokens)
            slack_cost = 0.0
            if int(payer_tokens) <= 0 and int(budget_delta) > 0:
                slack_cost = float(raw_slot_price) * float(int(budget_delta))
            gain = (
                float(value)
                - float(sold_loss)
                - float(sold_deficit)
                - float(payer_loss)
                + float(buyback_value)
                - float(region_open_cost) * float(max(0, int(surrogate_region_delta)))
                - float(slack_cost)
            )
            return (
                float(gain),
                float(value),
                float(sold_loss),
                int(budget_delta),
                int(token_len),
                int(raw_inside.size),
                int(sold_tokens),
                int(post_gap),
                int(drop_tokens_inside),
                payer_atoms,
                int(payer_tokens),
                float(payer_loss),
                int(buyback_tokens),
                float(buyback_value),
            )

        profile_mark("define_online_helpers")

        # Fill the frontier immediately; later buyback is done immediately
        # after each accepted packet, using the same ledger.
        buy_raw_until_full()
        initial_frontier_filled_cost = int(current_cost)
        profile_mark("frontier_fill")

        selected_surrogates = 0
        selected_gain = 0.0
        selected_value = 0.0
        selected_sold_raw_atoms = 0
        selected_sold_raw_tokens = 0
        selected_sold_raw_value = 0.0
        selected_payer_atoms = 0
        selected_payer_tokens = 0
        selected_payer_value = 0.0
        rejected_overlap = 0
        rejected_budget = 0
        rejected_value = 0
        generated_candidate_count = int(len(candidates))
        market_candidate_limit = max(16, int(math.ceil(0.5 * math.sqrt(max(1, int(num_atoms))))))
        if int(len(candidates)) > int(market_candidate_limit):
            gain_quota = max(1, int(market_candidate_limit) // 2)
            density_quota = max(1, int(market_candidate_limit) - int(gain_quota))
            gain_ranked = sorted(
                candidates,
                key=lambda item: (float(item[0]), float(item[1]), int(item[4])),
                reverse=True,
            )
            density_ranked = sorted(
                candidates,
                key=lambda item: (
                    float(item[0]) / max(1.0, float(item[4])),
                    float(item[0]),
                    float(item[1]),
                ),
                reverse=True,
            )
            kept_candidates: List[Candidate] = []
            kept_keys = set()

            def keep_market_candidate(item: Candidate) -> None:
                key = (int(item[5]), int(item[6]))
                if key in kept_keys:
                    return
                kept_keys.add(key)
                kept_candidates.append(item)

            for item in gain_ranked[: int(gain_quota)]:
                keep_market_candidate(item)
            for item in density_ranked[: int(density_quota)]:
                keep_market_candidate(item)
            if len(kept_candidates) < int(market_candidate_limit):
                for item in gain_ranked:
                    keep_market_candidate(item)
                    if len(kept_candidates) >= int(market_candidate_limit):
                        break
            candidates = kept_candidates[: int(market_candidate_limit)]
        profile_mark("market_prune")

        candidates_sorted = sorted(
            candidates,
            key=lambda item: (float(item[0]), float(item[1]), int(item[4])),
            reverse=True,
        )
        for candidate in candidates_sorted:
            _gain, _value, _sold_loss, _budget_delta, _token_len, start_idx, end_idx, _seed_count = candidate
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            if start_idx >= end_idx:
                continue
            if bool(np.any(actions[start_idx:end_idx] == 1)):
                rejected_overlap += 1
                continue
            metrics = score_current_packet(start_idx, end_idx)
            if metrics is None:
                rejected_overlap += 1
                continue
            (
                gain,
                value,
                sold_loss,
                budget_delta,
                _token_len_current,
                sold_raw_atoms,
                sold_tokens,
                post_gap,
                drop_tokens_inside,
                payer_atoms,
                payer_tokens,
                payer_loss,
                buyback_tokens,
                buyback_value,
            ) = metrics
            if float(gain) <= 0.0:
                rejected_value += 1
                continue
            if payer_atoms:
                payer_arr = np.asarray(payer_atoms, dtype=np.int64)
                actions[payer_arr] = 0
                current_drop_tokens += int(payer_tokens)
            actions[start_idx:end_idx] = 1
            current_drop_tokens -= int(drop_tokens_inside)
            current_cost += int(budget_delta) - int(payer_tokens)
            selected_surrogates += 1
            selected_gain += float(gain)
            selected_value += float(value)
            selected_sold_raw_atoms += int(sold_raw_atoms) + int(len(payer_atoms))
            selected_sold_raw_tokens += int(sold_tokens) + int(payer_tokens)
            selected_sold_raw_value += float(sold_loss) + float(payer_loss)
            selected_payer_atoms += int(len(payer_atoms))
            selected_payer_tokens += int(payer_tokens)
            selected_payer_value += float(payer_loss)
            buy_raw_until_full()
            buy_raw_best_effort()
        profile_mark("select_candidates")

        final_raw_fill_atoms = 0
        final_raw_fill_tokens = 0
        final_raw_fill_value = 0.0
        while True:
            final_probe_stats, _ = collect_stats(actions)
            current_cost = int(final_probe_stats["ks_run_used_entries"])
            if int(current_cost) >= int(budget_entries):
                break
            before_cost = int(current_cost)
            before_atoms = int(online_buyback_atoms)
            before_tokens = int(online_buyback_tokens)
            before_value = float(online_buyback_value)
            buy_raw_best_effort()
            final_raw_fill_atoms += int(online_buyback_atoms) - int(before_atoms)
            final_raw_fill_tokens += int(online_buyback_tokens) - int(before_tokens)
            final_raw_fill_value += float(online_buyback_value) - float(before_value)
            if int(current_cost) <= int(before_cost):
                break
        profile_mark("final_fill")

        stats, surrogate_lens = collect_stats(actions)
        profile_mark("collect_stats")
        stats.update(
            {
                "surrogate_kv_allocator": "online_frontier_exchange",
                "surrogate_kv_candidate_seeds": int(len(seeds)),
                "surrogate_kv_shadow_candidate_seeds": int(shadow_candidate_count),
                "surrogate_kv_candidate_surrogates": int(len(candidates)),
                "surrogate_kv_candidate_generated_total": int(generated_candidate_count),
                "surrogate_kv_market_candidate_limit": int(market_candidate_limit),
                "surrogate_kv_candidate_generated": int(candidate_count),
                "surrogate_kv_stack_merges": int(stack_merges),
                "surrogate_kv_pareto_merge_candidates": int(pareto_merge_candidates),
                "surrogate_kv_selected_surrogates": int(selected_surrogates),
                "surrogate_kv_selected_gain": float(selected_gain),
                "surrogate_kv_selected_value": float(selected_value),
                "surrogate_kv_sold_raw_atoms": int(selected_sold_raw_atoms),
                "surrogate_kv_sold_raw_tokens": int(selected_sold_raw_tokens),
                "surrogate_kv_sold_raw_value": float(selected_sold_raw_value),
                "surrogate_kv_payer_raw_atoms": int(selected_payer_atoms),
                "surrogate_kv_payer_raw_tokens": int(selected_payer_tokens),
                "surrogate_kv_payer_raw_value": float(selected_payer_value),
                "surrogate_kv_online_buyback_atoms": int(online_buyback_atoms),
                "surrogate_kv_online_buyback_tokens": int(online_buyback_tokens),
                "surrogate_kv_online_buyback_value": float(online_buyback_value),
                "surrogate_kv_final_raw_fill_atoms": int(final_raw_fill_atoms),
                "surrogate_kv_final_raw_fill_tokens": int(final_raw_fill_tokens),
                "surrogate_kv_final_raw_fill_value": float(final_raw_fill_value),
                "surrogate_kv_initial_cost": int(initial_cost),
                "surrogate_kv_initial_frontier_filled_cost": int(initial_frontier_filled_cost),
                "surrogate_kv_final_cost": int(stats["ks_run_used_entries"]),
                "surrogate_kv_budget_gap": int(stats["ks_run_budget_gap"]),
                "surrogate_kv_region_open_cost": float(region_open_cost),
                "surrogate_kv_generation_horizon": int(generation_horizon),
                "surrogate_kv_horizon_surrogate_tax": float(horizon_surrogate_tax),
                "surrogate_kv_exact_exchange_open_price": float(surrogate_open_price),
                "surrogate_kv_raw_slot_price": float(raw_slot_price),
                "surrogate_kv_exact_exchange": 1,
                "surrogate_kv_post_gap_priced": 1,
                "surrogate_kv_hard_gap_reject": 0,
                "surrogate_kv_tail_price_scale": float(tail_price_scale),
                "surrogate_kv_predictive": int(bool(predictive)),
                "surrogate_kv_future_signal_mean": float(future_signal_arr.mean()) if future_signal_arr.size else 0.0,
                "surrogate_kv_rejected_overlap": int(rejected_overlap),
                "surrogate_kv_rejected_budget": int(rejected_budget),
                "surrogate_kv_rejected_value": int(rejected_value),
                "surrogate_kv_score_current_packet_calls": int(score_current_packet_calls),
                "surrogate_kv_micro_len": int(micro_len),
                "surrogate_kv_no_terminal_repair": 1,
                "ks_run_candidate_seeds": int(len(seeds)),
                "ks_run_shadow_candidate_seeds": int(shadow_candidate_count),
                "ks_run_candidate_surrogates": int(len(candidates)),
                "ks_run_merge_accepts": int(stack_merges),
                "ks_run_selected_surrogates": int(selected_surrogates),
                "ks_run_sold_raw_atoms": int(selected_sold_raw_atoms),
                "ks_run_sold_raw_tokens": int(selected_sold_raw_tokens),
                "ks_run_sold_raw_value": float(selected_sold_raw_value),
                "ks_run_buyback_raw_atoms": int(online_buyback_atoms),
                "ks_run_buyback_raw_tokens": int(online_buyback_tokens),
                "ks_run_buyback_raw_value": float(online_buyback_value),
                "ks_run_region_open_cost": float(region_open_cost),
                "ks_run_raw_slot_price": float(raw_slot_price),
                "ks_run_tail_price_scale": float(tail_price_scale),
                "ks_run_selected_value": float(selected_value),
                "ks_run_selected_gain": float(selected_gain),
                "ks_run_rejected_budget": int(rejected_budget),
                "ks_run_rejected_value": int(rejected_value),
                "ks_run_merge_terminal_price": 0,
                "ks_run_merge_terminal_keep_frac": float(budget_entries) / max(1.0, float(full_cost)),
            }
        )
        profile_mark("stats_update")
        stats.update(profile_export())
        self._last_allocator_stats = stats
        if bool(getattr(self, "_allocator_plan_only", False)):
            materialize_start = time.perf_counter()
            materialized = materialize_actions(actions, tensors=False)
            stats["surrogate_kv_timing_alloc_materialize_seconds"] = float(time.perf_counter() - materialize_start)
        else:
            materialize_start = time.perf_counter()
            materialized = materialize_actions(actions)
            stats["surrogate_kv_timing_alloc_materialize_seconds"] = float(time.perf_counter() - materialize_start)
        stats["surrogate_kv_timing_alloc_total_seconds"] = float(time.perf_counter() - profile_t0)
        self._last_allocator_stats = stats
        return materialized

    def _past_token_scores(
        self,
        *,
        key_states,
        query_states,
        value_states,
        recent_len: int,
        past_len: int,
        head_dim: int,
        num_key_value_groups: int,
        base_capacity_prompt: int,
        sink_len: int,
        headwise_budget_only: bool = False,
    ):
        self._last_surrogate_residual_token_scores = None
        self._last_pooled_head_token_scores = None
        self._last_headwise_precomputed_pooled_scores = None
        self._last_surrogate_residual_head_token_scores = None
        if query_states.shape[1] == key_states.shape[1]:
            attn_weights = torch.matmul(query_states[..., -recent_len:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        else:
            grouped_queries = query_states[:, :, -recent_len:, :].reshape(
                query_states.shape[0],
                key_states.shape[1],
                num_key_value_groups,
                recent_len,
                head_dim,
            )
            attn_weights = torch.einsum(
                "bngrd,bndt->bngrt",
                grouped_queries,
                key_states.transpose(2, 3),
            ) / math.sqrt(head_dim)
        mask_key = (_device_key(attn_weights.device), int(recent_len), str(attn_weights.dtype))
        recent_mask = _RECENT_MASK_CACHE.get(mask_key)
        if recent_mask is None:
            recent_mask = torch.full(
                (recent_len, recent_len),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )
            mask_cond = torch.arange(recent_len, device=attn_weights.device)
            recent_mask.masked_fill_(mask_cond < (mask_cond + 1).view(recent_len, 1), 0)
            recent_mask = _cache_put(_RECENT_MASK_CACHE, mask_key, recent_mask, max_size=64)
        if attn_weights.dim() == 4:
            attn_weights[:, :, -recent_len:, -recent_len:] += recent_mask[None, None, :, :]
        else:
            attn_weights[..., -recent_len:] += recent_mask[None, None, None, :, :]
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        score_method = str(getattr(self, "score_method", "attention") or "attention").replace("-", "_").lower()
        if score_method in {"attention", "attn", "snap", "snapkv"}:
            if attn_probs.dim() == 4:
                head_token_scores = attn_probs[:, :, -recent_len:, :past_len].sum(dim=-2).to(dtype=torch.float32)
            else:
                head_token_scores = attn_probs[..., :past_len].sum(dim=-2).reshape(
                    query_states.shape[0],
                    query_states.shape[1],
                    past_len,
                ).to(dtype=torch.float32)
            score_stats = {
                "surrogate_kv_score_attention": 1,
            }
        else:
            raise ValueError(f"Unsupported SurKV score method: {self.score_method}")

        pooled_scores = self._pool_head_token_scores(head_token_scores)
        self._last_pooled_head_token_scores = pooled_scores.detach()
        self._last_headwise_precomputed_pooled_scores = pooled_scores.detach()
        if _env_flag("SURKV_SCORE_DIAGNOSTICS", False):
            pooled_positive = pooled_scores.detach().to(dtype=torch.float32)
            pooled_positive = torch.clamp(pooled_positive - pooled_positive.amin(dim=-1, keepdim=True), min=0.0)
            pooled_mass = pooled_positive.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            pooled_prob = pooled_positive / pooled_mass
            pooled_entropy = -(pooled_prob * torch.log(pooled_prob.clamp_min(1e-9))).sum(dim=-1)
            pooled_entropy_norm = pooled_entropy / math.log(max(2, int(past_len)))
            topk = max(1, min(int(past_len), int(max(8, round(float(past_len) * 0.02)))))
            pooled_top_mass = torch.topk(pooled_prob, k=int(topk), dim=-1).values.sum(dim=-1)
            score_stats["surrogate_kv_score_entropy_norm"] = float(
                pooled_entropy_norm.mean().detach().cpu().item()
            )
            score_stats["surrogate_kv_score_top2pct_mass"] = float(
                pooled_top_mass.mean().detach().cpu().item()
            )
        score_method = str(getattr(self, "score_method", "attention") or "attention").replace("-", "_").lower()
        fusion = str(getattr(self, "head_score_fusion", "mean") or "mean").replace("-", "_").lower()
        if bool(headwise_budget_only) and fusion in {"ada_shared", "ada", "adakv_shared", "adakv"} and score_method in {
            "attention",
            "attn",
            "snap",
            "snapkv",
        }:
            token_scores, fusion_stats = self._ada_shared_head_caps_only(
                pooled_scores,
                base_capacity_prompt=base_capacity_prompt,
                recent_len=recent_len,
                past_len=past_len,
            )
        else:
            token_scores, fusion_stats = self._fuse_head_token_scores(
                pooled_scores,
                base_capacity_prompt=base_capacity_prompt,
                recent_len=recent_len,
                past_len=past_len,
                sink_len=sink_len,
            )
        score_stats.update(fusion_stats)
        self._last_surrogate_residual_head_token_scores = pooled_scores.detach()
        self._last_surrogate_residual_token_scores = token_scores.detach()
        self._last_score_stats = dict(score_stats)
        self._last_allocator_stats.update(score_stats)

        score_for_signal = pooled_scores
        if (
            self._global_budget_ledger_active()
            and not bool(self.global_layer_allocator)
            and str(getattr(self, "mode", "")) != "surrogate_kv_dynamic_layer"
        ):
            pooled_float = score_for_signal.detach().to(dtype=torch.float32)
            head_peak = pooled_float.amax(dim=-1).mean()
            head_mean = pooled_float.mean().clamp_min(1e-9)
            sharpness = (head_peak / head_mean).clamp_min(1.0)
            signal = head_peak * (1.0 + 0.1 * torch.log1p(sharpness))
            self._last_layer_budget_signal = float(signal.detach().cpu().item())

        return token_scores

    def _pool_head_token_scores(self, head_token_scores):
        if self.pooling == "avgpool":
            return F.avg_pool1d(
                head_token_scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        if self.pooling == "maxpool":
            return F.max_pool1d(
                head_token_scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        raise ValueError(f"Unsupported pooling method: {self.pooling}")

    def _fuse_head_token_scores(
        self,
        head_token_scores,
        *,
        base_capacity_prompt: int,
        recent_len: int,
        past_len: int,
        sink_len: int,
    ):
        del sink_len
        fusion = str(getattr(self, "head_score_fusion", "mean") or "mean").replace("-", "_").lower()
        stats = {
            "surrogate_kv_score_head_count": int(head_token_scores.shape[1]),
            "surrogate_kv_score_head_fusion_mean": 0,
            "surrogate_kv_score_head_fusion_max": 0,
            "surrogate_kv_score_head_fusion_ada_shared": 0,
        }
        if fusion in {"mean", "avg", "average"}:
            stats["surrogate_kv_score_head_fusion_mean"] = 1
            self._last_ada_selected_support = None
            return head_token_scores.mean(dim=1), stats
        if fusion == "max":
            stats["surrogate_kv_score_head_fusion_max"] = 1
            self._last_ada_selected_support = None
            return head_token_scores.max(dim=1).values, stats
        if fusion in {"ada_shared", "ada", "adakv_shared", "adakv"}:
            fused, ada_stats = self._ada_shared_fused_scores(
                head_token_scores,
                base_capacity_prompt=base_capacity_prompt,
                recent_len=recent_len,
                past_len=past_len,
            )
            stats["surrogate_kv_score_head_fusion_ada_shared"] = 1
            stats.update(ada_stats)
            return fused, stats
        raise ValueError(f"Unsupported SurKV head score fusion: {self.head_score_fusion}")

    def _ada_shared_head_caps_only(
        self,
        head_token_scores,
        *,
        base_capacity_prompt: int,
        recent_len: int,
        past_len: int,
    ):
        bsz, heads, tokens = head_token_scores.shape
        stats = {
            "surrogate_kv_score_head_count": int(heads),
            "surrogate_kv_score_head_fusion_mean": 0,
            "surrogate_kv_score_head_fusion_max": 0,
            "surrogate_kv_score_head_fusion_ada_shared": 1,
        }
        if heads <= 0 or tokens <= 0:
            self._last_ada_head_capacities = None
            self._last_ada_selected_support = None
            stats.update(
                {
                    "surrogate_kv_ada_shared_budget": 0,
                    "surrogate_kv_ada_shared_floor_ratio": float(getattr(self, "ada_floor_ratio", 0.2)),
                }
            )
            return head_token_scores.mean(dim=1), stats

        base_budget = max(1, min(int(tokens), int(base_capacity_prompt) - int(recent_len)))
        floor_ratio = max(0.0, min(0.95, float(getattr(self, "ada_floor_ratio", 0.2))))
        floor_capacity = max(0, min(int(tokens), int(float(base_budget) * floor_ratio)))

        raw_scores = torch.nan_to_num(head_token_scores.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        top_k = max(1, min(int(tokens), int(base_budget)))
        top_mass = torch.topk(raw_scores, k=int(top_k), dim=-1, largest=True, sorted=False).values.sum(
            dim=-1,
            keepdim=True,
        )
        total_mass = raw_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        allocation_scores = raw_scores * (top_mass / total_mass)
        flat_scores = allocation_scores.reshape(bsz, heads * tokens)
        flat_k = max(1, min(int(flat_scores.shape[-1]), int(heads) * int(base_budget)))
        flat_top = torch.topk(flat_scores, k=flat_k, dim=-1).indices
        head_ids = flat_top // int(tokens)
        head_adaptive_capacity = torch.zeros((bsz, heads), device=head_token_scores.device, dtype=torch.long)
        head_adaptive_capacity.scatter_add_(1, head_ids, torch.ones_like(head_ids, dtype=torch.long))
        head_adaptive_capacity = torch.round(
            head_adaptive_capacity.to(dtype=torch.float32) * (1.0 - floor_ratio) + float(floor_capacity)
        ).to(dtype=torch.long)
        head_adaptive_capacity = torch.clamp(head_adaptive_capacity, min=0, max=int(tokens))

        max_capacity = int(head_adaptive_capacity.max().detach().cpu().item()) if head_adaptive_capacity.numel() else 0
        selected = torch.zeros((bsz, heads, tokens), device=head_token_scores.device, dtype=torch.bool)
        if int(max_capacity) > 0:
            top_indices = torch.topk(
                raw_scores,
                k=int(max_capacity),
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            rank_values = torch.arange(int(max_capacity), device=head_token_scores.device, dtype=torch.long).view(
                1,
                1,
                int(max_capacity),
            )
            active = rank_values < head_adaptive_capacity.unsqueeze(-1)
            selected.scatter_(-1, top_indices, active)
        support = selected.to(dtype=torch.float32).mean(dim=1)
        fused = support + raw_scores.mean(dim=1) * 1e-3
        self._last_ada_head_capacities = head_adaptive_capacity.detach()
        self._last_ada_selected_support = selected.detach()

        stats.update(
            {
                "surrogate_kv_ada_shared_budget": int(base_budget),
                "surrogate_kv_ada_shared_floor_ratio": float(floor_ratio),
                "surrogate_kv_ada_shared_normalize": 1,
                "surrogate_kv_ada_shared_support_weight": 1.0,
                "surrogate_kv_ada_shared_headwise_caps_only": 1,
            }
        )
        if bool(_SURKV_DIAGNOSTIC_STATS):
            capacity_float = head_adaptive_capacity.detach().to(dtype=torch.float32)
            stats.update(
                {
                    "surrogate_kv_ada_shared_head_budget_min": int(
                        head_adaptive_capacity.min().detach().cpu().item()
                    ),
                    "surrogate_kv_ada_shared_head_budget_max": int(
                        head_adaptive_capacity.max().detach().cpu().item()
                    ),
                    "surrogate_kv_ada_shared_head_budget_mean": float(capacity_float.mean().cpu().item()),
                    "surrogate_kv_ada_shared_selected_token_fraction": float(
                        support.gt(0).to(dtype=torch.float32).mean().detach().cpu().item()
                    ),
                }
            )
        return fused, stats

    def _ada_shared_fused_scores(
        self,
        head_token_scores,
        *,
        base_capacity_prompt: int,
        recent_len: int,
        past_len: int,
    ):
        bsz, heads, tokens = head_token_scores.shape
        if heads <= 0 or tokens <= 0:
            self._last_ada_head_capacities = None
            self._last_ada_selected_support = None
            return head_token_scores.mean(dim=1), {
                "surrogate_kv_ada_shared_budget": 0,
                "surrogate_kv_ada_shared_floor_ratio": float(getattr(self, "ada_floor_ratio", 0.2)),
            }

        base_budget = max(1, min(int(tokens), int(base_capacity_prompt) - int(recent_len)))
        floor_ratio = max(0.0, min(0.95, float(getattr(self, "ada_floor_ratio", 0.2))))
        floor_capacity = max(0, min(int(tokens), int(float(base_budget) * floor_ratio)))

        raw_scores = torch.nan_to_num(head_token_scores.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        top_k = max(1, min(int(tokens), int(base_budget)))
        top_mass = torch.topk(raw_scores, k=int(top_k), dim=-1, largest=True, sorted=False).values.sum(
            dim=-1,
            keepdim=True,
        )
        total_mass = raw_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        allocation_scores = raw_scores * (top_mass / total_mass)
        normalized = _rank01(raw_scores.reshape(bsz * heads, tokens)).view(bsz, heads, tokens)
        flat_scores = allocation_scores.reshape(bsz, heads * tokens)
        flat_k = max(1, min(int(flat_scores.shape[-1]), int(heads) * int(base_budget)))
        flat_top = torch.topk(flat_scores, k=flat_k, dim=-1).indices
        head_ids = flat_top // int(tokens)
        head_adaptive_capacity = torch.zeros((bsz, heads), device=head_token_scores.device, dtype=torch.long)
        head_adaptive_capacity.scatter_add_(1, head_ids, torch.ones_like(head_ids, dtype=torch.long))
        head_adaptive_capacity = torch.round(
            head_adaptive_capacity.to(dtype=torch.float32) * (1.0 - floor_ratio) + float(floor_capacity)
        ).to(dtype=torch.long)
        head_adaptive_capacity = torch.clamp(head_adaptive_capacity, min=0, max=int(tokens))

        max_capacity = int(head_adaptive_capacity.max().detach().cpu().item()) if head_adaptive_capacity.numel() else 0
        selected = torch.zeros((bsz, heads, tokens), device=head_token_scores.device, dtype=torch.bool)
        if int(max_capacity) > 0:
            top_indices = torch.topk(
                raw_scores,
                k=int(max_capacity),
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            rank_values = torch.arange(int(max_capacity), device=head_token_scores.device, dtype=torch.long).view(
                1,
                1,
                int(max_capacity),
            )
            active = rank_values < head_adaptive_capacity.unsqueeze(-1)
            selected.scatter_(-1, top_indices, active)
        selected_float = selected.to(dtype=torch.float32)
        support = selected_float.mean(dim=1)
        selected_rank_mass = (normalized * selected_float).mean(dim=1)
        support_weight = 1.0
        fused = selected_rank_mass + support + normalized.mean(dim=1) * 1e-3
        self._last_ada_head_capacities = head_adaptive_capacity.detach()
        self._last_ada_selected_support = selected.detach()

        stats = {
            "surrogate_kv_ada_shared_budget": int(base_budget),
            "surrogate_kv_ada_shared_floor_ratio": float(floor_ratio),
            "surrogate_kv_ada_shared_normalize": 1,
            "surrogate_kv_ada_shared_support_weight": float(support_weight),
        }
        if bool(_SURKV_DIAGNOSTIC_STATS):
            capacity_float = head_adaptive_capacity.detach().to(dtype=torch.float32)
            stats.update(
                {
                    "surrogate_kv_ada_shared_head_budget_min": int(
                        head_adaptive_capacity.min().detach().cpu().item()
                    ),
                    "surrogate_kv_ada_shared_head_budget_max": int(
                        head_adaptive_capacity.max().detach().cpu().item()
                    ),
                    "surrogate_kv_ada_shared_head_budget_mean": float(capacity_float.mean().cpu().item()),
                    "surrogate_kv_ada_shared_selected_token_fraction": float(
                        support.gt(0).to(dtype=torch.float32).mean().detach().cpu().item()
                    ),
                }
            )
        return fused, stats

    def _chunk_statistics_fast_mean_max(self, *, token_scores, chunk_slices: Sequence[Tuple[int, int]]):
        if not chunk_slices:
            empty = token_scores.new_empty((token_scores.shape[0], 0))
            return empty, empty
        if not self._is_regular_chunk_layout(chunk_slices):
            return self._chunk_statistics_irregular_prefix_mean_max(
                token_scores=token_scores,
                chunk_slices=chunk_slices,
            )

        base_start = chunk_slices[0][0]
        chunk_size = chunk_slices[0][1] - chunk_slices[0][0]
        if chunk_size <= 0:
            empty = token_scores.new_empty((token_scores.shape[0], 0))
            return empty, empty

        tail_len = chunk_slices[-1][1] - chunk_slices[-1][0]
        regular_chunks = len(chunk_slices) if tail_len == chunk_size else len(chunk_slices) - 1
        chunk_means = []
        chunk_maxes = []

        if regular_chunks > 0:
            regular_tokens = regular_chunks * chunk_size
            regular = token_scores[:, base_start : base_start + regular_tokens].reshape(
                token_scores.shape[0],
                regular_chunks,
                chunk_size,
            )
            chunk_means.append(regular.mean(dim=-1))
            chunk_maxes.append(regular.max(dim=-1).values)

        if tail_len != chunk_size:
            tail_start = base_start + regular_chunks * chunk_size
            tail = token_scores[:, tail_start : tail_start + tail_len]
            chunk_means.append(tail.mean(dim=-1, keepdim=True))
            chunk_maxes.append(tail.max(dim=-1, keepdim=True).values)

        return torch.cat(chunk_means, dim=-1), torch.cat(chunk_maxes, dim=-1)

    def _dynamic_micro_prototype_bank(
        self,
        *,
        key_states,
        value_states,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        surrogate_mode: str,
        peak_mode: str = "",
        selected_only_mask=None,
        active_slice_indices: Optional[Sequence[int]] = None,
    ):
        """Build Dynamic local surrogates from fixed micro-prototypes.

        Dynamic regions are variable-size, but their boundaries are aligned to
        the eight-token microgrid.  Instead of gathering every token in every
        irregular region, summarize the microgrid once and combine those
        micro-prototypes with prefix sums.  This keeps the Dynamic assembly path
        closer to the fixed-chunk path while preserving variable regioning.
        """
        if not chunk_slices:
            bsz, heads, _, head_dim = key_states.shape
            empty_bank = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_score = key_states.new_empty((bsz, 0), dtype=torch.float32)
            return empty_bank, empty_bank.clone(), None, None, empty_score, empty_score.clone()

        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return self._chunk_prototype_bank_fast(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                surrogate_mode=surrogate_mode,
                return_distortion=False,
            )

        base_start, base_end = span
        span_len = int(base_end) - int(base_start)
        if span_len <= 0:
            bsz, heads, _, head_dim = key_states.shape
            empty_bank = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_score = key_states.new_empty((bsz, 0), dtype=torch.float32)
            return empty_bank, empty_bank.clone(), None, None, empty_score, empty_score.clone()

        device = key_states.device
        bsz, heads, _, head_dim = key_states.shape
        full_region_count = len(chunk_slices)
        active_region_indices = None
        active_slice_indices_list = None
        if active_slice_indices is not None and full_region_count > 0:
            active_slice_indices_list = [
                int(idx)
                for idx in active_slice_indices
                if 0 <= int(idx) < int(full_region_count)
            ]
            if not active_slice_indices_list:
                empty_key = key_states.new_zeros((bsz, heads, full_region_count, head_dim))
                empty_value = value_states.new_zeros((bsz, heads, full_region_count, head_dim))
                return empty_key, empty_value, None, None, None, None
            if len(active_slice_indices_list) != int(full_region_count):
                active_region_indices = torch.as_tensor(
                    active_slice_indices_list,
                    device=device,
                    dtype=torch.long,
                )
        elif selected_only_mask is not None and full_region_count > 0:
            selected_any = selected_only_mask.detach().to(device=device, dtype=torch.bool)
            if selected_any.dim() == 2:
                selected_any = selected_any.any(dim=0)
            elif selected_any.dim() != 1:
                selected_any = None
            if selected_any is not None and selected_any.numel() == full_region_count:
                if not bool(selected_any.any().item()):
                    empty_key = key_states.new_zeros((bsz, heads, full_region_count, head_dim))
                    empty_value = value_states.new_zeros((bsz, heads, full_region_count, head_dim))
                    return empty_key, empty_value, None, None, None, None
                if not bool(selected_any.all().item()):
                    active_region_indices = torch.nonzero(selected_any, as_tuple=False).flatten()
                    active_slice_indices_list = [int(idx) for idx in active_region_indices.detach().cpu().tolist()]

        prototype_token_indices = (getattr(self, "_last_fast_pack_plan", {}) or {}).get("prototype_token_indices")
        if (
            isinstance(prototype_token_indices, dict)
            and active_slice_indices_list is not None
            and active_slice_indices_list
        ):
            active_keys = []
            active_values = []
            for region_idx in active_slice_indices_list:
                token_ids = prototype_token_indices.get(int(region_idx))
                if not token_ids:
                    start, end = chunk_slices[int(region_idx)]
                    token_ids = list(range(int(start), int(end)))
                token_index = torch.as_tensor(token_ids, device=device, dtype=torch.long)
                if token_index.numel() <= 0:
                    active_keys.append(key_states.new_zeros((bsz, heads, head_dim), dtype=torch.float32))
                    active_values.append(value_states.new_zeros((bsz, heads, head_dim), dtype=torch.float32))
                    continue
                key_chunk = key_states.index_select(2, token_index).to(dtype=torch.float32)
                value_chunk = value_states.index_select(2, token_index).to(dtype=torch.float32)
                score_mass = torch.clamp(
                    token_scores.index_select(1, token_index).to(dtype=torch.float32),
                    min=1e-6,
                )
                score_weights = score_mass.view(bsz, 1, -1, 1)
                denom = score_mass.sum(dim=1).clamp_min(1e-6).view(bsz, 1, 1)
                if surrogate_mode in _KEY_WEIGHTED_SURROGATE_MODES:
                    key_proto = (key_chunk * score_weights).sum(dim=2) / denom
                elif surrogate_mode in _NORM_RESTORED_KEY_MODES:
                    key_proto = _restore_mean_key_norm(key_chunk.mean(dim=2), key_chunk, token_dim=2)
                elif surrogate_mode in _PIVOT_KEY_MODES:
                    pivot_rel = score_mass.argmax(dim=1)
                    pivot_index = pivot_rel.view(bsz, 1, 1, 1).expand(bsz, heads, 1, head_dim)
                    key_proto = key_chunk.gather(dim=2, index=pivot_index).squeeze(2)
                else:
                    key_proto = key_chunk.mean(dim=2)
                if surrogate_mode in _VALUE_WEIGHTED_SURROGATE_MODES:
                    value_proto = (value_chunk * score_weights).sum(dim=2) / denom
                else:
                    value_proto = value_chunk.mean(dim=2)
                if surrogate_mode in _RMS_RESTORED_VALUE_MODES:
                    value_proto = _restore_rms_value_norm(value_proto, value_chunk, token_dim=2)
                active_keys.append(key_proto)
                active_values.append(value_proto)
            proto_key_bank = torch.stack(active_keys, dim=2).to(dtype=key_states.dtype)
            proto_value_bank = torch.stack(active_values, dim=2).to(dtype=value_states.dtype)
            proto_key_bank, proto_value_bank = self._scatter_active_prototype_bank(
                proto_key_bank=proto_key_bank,
                proto_value_bank=proto_value_bank,
                active_region_indices=active_region_indices,
                full_region_count=full_region_count,
            )
            return (
                proto_key_bank,
                proto_value_bank,
                None,
                None,
                None,
                None,
            )

        # Raw rescue allocators may introduce 4-token raw boundaries while all
        # surrogate spans remain on the 8-token grid.  Prototype construction
        # only needs selected/surrogate regions, so choose the microgrid from
        # those spans rather than from raw boundaries that never need a
        # synthetic prototype.
        boundary_slices = (
            [chunk_slices[idx] for idx in active_slice_indices_list]
            if active_slice_indices_list is not None
            else chunk_slices
        )
        direct_selected_modes = {
            "mean",
            "weighted_mean",
            "asym_key_weighted",
            "asym_value_weighted",
            "norm_value_weighted",
            "pivot_value_weighted",
            "norm_rms_mean",
            "value_sqrt_weighted_rms",
        }
        # In the headwise path, even a small selected set is multiplied by
        # key/value heads and layers.  Prefer the prefix/vectorized path by
        # default; keep a tunable direct path for tiny synthetic smoke cases.
        direct_selected_limit = max(0, int(os.environ.get("SURKV_DYNAMIC_DIRECT_PROTO_LIMIT", "0")))
        if (
            active_region_indices is not None
            and active_slice_indices is not None
            and len(active_slice_indices) <= int(direct_selected_limit)
            and surrogate_mode in direct_selected_modes
            and str(peak_mode or "").lower() in {"", "predictive_k_token"}
        ):
            active_token_work = sum(max(0, int(end) - int(start)) for start, end in boundary_slices)
            if 0 < active_token_work < span_len:
                direct_peak_mode = str(peak_mode or "").lower()
                active_keys = []
                active_values = []
                for start, end in boundary_slices:
                    start_i = int(start)
                    end_i = int(end)
                    if end_i <= start_i:
                        active_keys.append(key_states.new_zeros((bsz, heads, head_dim), dtype=torch.float32))
                        active_values.append(value_states.new_zeros((bsz, heads, head_dim), dtype=torch.float32))
                    else:
                        key_chunk = key_states[:, :, start_i:end_i, :].to(dtype=torch.float32)
                        value_chunk = value_states[:, :, start_i:end_i, :].to(dtype=torch.float32)
                        score_mass = None
                        score_weights = None
                        denom = None
                        if surrogate_mode in _SCORE_WEIGHTED_SURROGATE_MODES or direct_peak_mode == "predictive_k_token":
                            score_mass = torch.clamp(
                                token_scores[:, start_i:end_i].to(dtype=torch.float32),
                                min=1e-6,
                            )
                            if surrogate_mode in _LIGHT_VALUE_WEIGHT_MODES:
                                score_mass = torch.sqrt(score_mass)
                        if surrogate_mode in _SCORE_WEIGHTED_SURROGATE_MODES and score_mass is not None:
                            denom = score_mass.sum(dim=1).clamp_min(1e-6).view(bsz, 1, 1)
                            score_weights = score_mass.view(bsz, 1, end_i - start_i, 1)
                        if surrogate_mode in _KEY_WEIGHTED_SURROGATE_MODES and score_weights is not None:
                            key_proto = (key_chunk * score_weights).sum(dim=2) / denom
                        elif surrogate_mode in _NORM_RESTORED_KEY_MODES:
                            key_proto = _restore_mean_key_norm(key_chunk.mean(dim=2), key_chunk, token_dim=2)
                        elif surrogate_mode in _PIVOT_KEY_MODES and score_mass is not None:
                            pivot_rel = score_mass.argmax(dim=1)
                            pivot_index = pivot_rel.view(bsz, 1, 1, 1).expand(bsz, heads, 1, head_dim)
                            key_proto = key_chunk.gather(dim=2, index=pivot_index).squeeze(2)
                        else:
                            key_proto = key_chunk.mean(dim=2)
                        if direct_peak_mode == "predictive_k_token" and score_mass is not None:
                            peak_rel = score_mass.argmax(dim=1)
                            peak_index = peak_rel.view(bsz, 1, 1, 1).expand(bsz, heads, 1, head_dim)
                            peak_key = key_chunk.gather(dim=2, index=peak_index).squeeze(2)
                            peak_denom = score_mass.sum(dim=1).clamp_min(1e-6)
                            peak_score = score_mass.gather(1, peak_rel.view(bsz, 1)).squeeze(1)
                            peak_mix = (peak_score / peak_denom).clamp(0.0, 1.0).view(bsz, 1, 1)
                            key_proto = (1.0 - peak_mix) * key_proto + peak_mix * peak_key
                        active_keys.append(key_proto)
                        if surrogate_mode in _VALUE_WEIGHTED_SURROGATE_MODES and score_weights is not None:
                            value_proto = (value_chunk * score_weights).sum(dim=2) / denom
                        else:
                            value_proto = value_chunk.mean(dim=2)
                        if surrogate_mode in _RMS_RESTORED_VALUE_MODES:
                            value_proto = _restore_rms_value_norm(value_proto, value_chunk, token_dim=2)
                        active_values.append(value_proto)
                proto_key_bank = torch.stack(active_keys, dim=2).to(dtype=key_states.dtype)
                proto_value_bank = torch.stack(active_values, dim=2).to(dtype=value_states.dtype)
                proto_key_bank, proto_value_bank = self._scatter_active_prototype_bank(
                    proto_key_bank=proto_key_bank,
                    proto_value_bank=proto_value_bank,
                    active_region_indices=active_region_indices,
                    full_region_count=full_region_count,
                )
                return (
                    proto_key_bank,
                    proto_value_bank,
                    None,
                    None,
                    None,
                    None,
                )
        relative_boundaries = [0]
        for start, end in boundary_slices:
            relative_boundaries.append(int(start) - int(base_start))
            relative_boundaries.append(int(end) - int(base_start))

        def aligned_or_tail(offset: int, unit: int) -> bool:
            return int(offset) == int(span_len) or int(offset) % int(unit) == 0

        if all(aligned_or_tail(offset, 8) for offset in relative_boundaries):
            micro_len = 8
        elif all(aligned_or_tail(offset, 4) for offset in relative_boundaries):
            micro_len = 4
        else:
            return self._chunk_prototype_bank_fast(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                surrogate_mode=surrogate_mode,
                return_distortion=False,
            )

        boundary_offsets = list(range(0, span_len, micro_len))
        boundary_offsets.append(span_len)
        boundary_to_micro = {int(offset): idx for idx, offset in enumerate(boundary_offsets)}
        region_source_slices = boundary_slices if active_slice_indices_list is not None else chunk_slices
        try:
            region_starts = [boundary_to_micro[int(start) - int(base_start)] for start, _ in region_source_slices]
            region_ends = [boundary_to_micro[int(end) - int(base_start)] for _, end in region_source_slices]
        except KeyError:
            return self._chunk_prototype_bank_fast(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                surrogate_mode=surrogate_mode,
                return_distortion=False,
            )

        regular_micro = span_len // micro_len
        tail_len = span_len - regular_micro * micro_len
        micro_keys = []
        micro_values = []
        micro_score_sums = []
        micro_lengths = []
        needs_score_weights = surrogate_mode in _SCORE_WEIGHTED_SURROGATE_MODES

        if regular_micro > 0:
            regular_tokens = regular_micro * micro_len
            key_regular = key_states[:, :, base_start : base_start + regular_tokens, :].reshape(
                bsz,
                heads,
                regular_micro,
                micro_len,
                head_dim,
            )
            value_regular = value_states[:, :, base_start : base_start + regular_tokens, :].reshape(
                bsz,
                heads,
                regular_micro,
                micro_len,
                head_dim,
            )
            micro_keys.append(key_regular.mean(dim=3))
            micro_values.append(value_regular.mean(dim=3))
            if needs_score_weights:
                score_regular = token_scores[:, base_start : base_start + regular_tokens].reshape(
                    token_scores.shape[0],
                    regular_micro,
                    micro_len,
                )
                micro_score_sums.append(score_regular.to(dtype=torch.float32).sum(dim=-1))
            micro_lengths.append(torch.full((regular_micro,), micro_len, device=device, dtype=torch.float32))

        if tail_len > 0:
            tail_start = base_start + regular_micro * micro_len
            micro_keys.append(key_states[:, :, tail_start:base_end, :].mean(dim=2, keepdim=True))
            micro_values.append(value_states[:, :, tail_start:base_end, :].mean(dim=2, keepdim=True))
            if needs_score_weights:
                micro_score_sums.append(
                    token_scores[:, tail_start:base_end].to(dtype=torch.float32).sum(dim=-1, keepdim=True)
                )
            micro_lengths.append(torch.full((1,), tail_len, device=device, dtype=torch.float32))

        micro_key_bank = torch.cat(micro_keys, dim=2)
        micro_value_bank = torch.cat(micro_values, dim=2)
        micro_len_bank = torch.cat(micro_lengths, dim=0)
        peak_mode = str(peak_mode or "").lower()

        length_weights = micro_len_bank.view(1, -1).expand(bsz, -1)
        if needs_score_weights:
            micro_mass = torch.cat(micro_score_sums, dim=1)
            score_weights = torch.clamp(micro_mass, min=1e-6)
            if surrogate_mode in _LIGHT_VALUE_WEIGHT_MODES:
                score_weights = torch.sqrt(score_weights)
        else:
            score_weights = length_weights

        if surrogate_mode in _KEY_WEIGHTED_SURROGATE_MODES or (
            peak_mode in {"predictive_k", "predictive_k_token"} and needs_score_weights
        ):
            key_weights = score_weights
        else:
            key_weights = length_weights

        if surrogate_mode in _VALUE_WEIGHTED_SURROGATE_MODES:
            value_weights = score_weights
        else:
            value_weights = length_weights

        # Evidence weights are used only for peak/CSB diagnostics.  Prefer the
        # score mass whenever it is available, but keep mean-mode unchanged.
        weights = score_weights if needs_score_weights else length_weights

        starts = torch.tensor(region_starts, device=device, dtype=torch.long)
        ends = torch.tensor(region_ends, device=device, dtype=torch.long)
        region_width_values = [
            max(1, int(end) - int(start))
            for start, end in zip(region_starts, region_ends)
        ]
        region_widths = torch.tensor(region_width_values, device=device, dtype=torch.long)
        max_region_width = max(region_width_values) if region_width_values else 0

        key_weights_f = key_weights.to(dtype=torch.float32)
        value_weights_f = value_weights.to(dtype=torch.float32)
        weighted_keys = micro_key_bank.to(dtype=torch.float32) * key_weights_f.view(bsz, 1, -1, 1)
        weighted_values = micro_value_bank.to(dtype=torch.float32) * value_weights_f.view(bsz, 1, -1, 1)
        zero_key = weighted_keys.new_zeros((bsz, heads, 1, head_dim))
        zero_value = weighted_values.new_zeros((bsz, heads, 1, head_dim))
        key_prefix = torch.cat([zero_key, weighted_keys.cumsum(dim=2)], dim=2)
        value_prefix = torch.cat([zero_value, weighted_values.cumsum(dim=2)], dim=2)
        key_weight_prefix = torch.cat([key_weights_f.new_zeros((bsz, 1)), key_weights_f.cumsum(dim=1)], dim=1)
        value_weight_prefix = torch.cat([value_weights_f.new_zeros((bsz, 1)), value_weights_f.cumsum(dim=1)], dim=1)
        length_weight_prefix = torch.cat(
            [length_weights.new_zeros((bsz, 1)), length_weights.to(dtype=torch.float32).cumsum(dim=1)],
            dim=1,
        )
        evidence_weights_f = weights.to(dtype=torch.float32)
        weight_prefix = torch.cat(
            [evidence_weights_f.new_zeros((bsz, 1)), evidence_weights_f.cumsum(dim=1)],
            dim=1,
        )

        key_denom = (key_weight_prefix.index_select(1, ends) - key_weight_prefix.index_select(1, starts)).clamp_min(1e-6)
        value_denom = (
            value_weight_prefix.index_select(1, ends) - value_weight_prefix.index_select(1, starts)
        ).clamp_min(1e-6)
        denom = (weight_prefix.index_select(1, ends) - weight_prefix.index_select(1, starts)).clamp_min(1e-6)
        proto_key_bank = (key_prefix.index_select(2, ends) - key_prefix.index_select(2, starts)) / key_denom.view(
            bsz,
            1,
            -1,
            1,
        )
        proto_value_bank = (
            value_prefix.index_select(2, ends) - value_prefix.index_select(2, starts)
        ) / value_denom.view(bsz, 1, -1, 1)

        if surrogate_mode in _NORM_RESTORED_KEY_MODES and len(region_starts) > 0:
            micro_key_norm = micro_key_bank.to(dtype=torch.float32).norm(dim=-1)
            weighted_key_norm = micro_key_norm * length_weights.to(dtype=torch.float32).view(bsz, 1, -1)
            zero_norm = weighted_key_norm.new_zeros((bsz, heads, 1))
            key_norm_prefix = torch.cat([zero_norm, weighted_key_norm.cumsum(dim=2)], dim=2)
            target_key_norm = (
                key_norm_prefix.index_select(2, ends) - key_norm_prefix.index_select(2, starts)
            ) / key_denom.view(bsz, 1, -1)
            current_key_norm = proto_key_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
            key_scale = _safe_key_norm_scale(
                target_norm=target_key_norm,
                current_norm=current_key_norm,
            )
            proto_key_bank = proto_key_bank * key_scale.view(bsz, heads, -1, 1)

        if surrogate_mode in _PIVOT_KEY_MODES and len(region_starts) > 0 and max_region_width > 0:
            offsets = torch.arange(max_region_width, device=device, dtype=torch.long)
            micro_indices = starts.view(-1, 1) + offsets.view(1, -1)
            valid = offsets.view(1, -1) < region_widths.view(-1, 1)
            safe_indices = micro_indices.clamp(max=micro_key_bank.shape[2] - 1)
            gathered_weights = score_weights.index_select(1, safe_indices.reshape(-1)).view(
                bsz,
                len(region_starts),
                max_region_width,
            )
            gathered_weights = gathered_weights.masked_fill(
                ~valid.view(1, len(region_starts), max_region_width),
                torch.finfo(gathered_weights.dtype).min,
            )
            pivot_rel = gathered_weights.argmax(dim=-1)
            pivot_micro_indices = starts.view(1, -1) + pivot_rel
            pivot_gather_index = pivot_micro_indices[:, None, :, None].expand(bsz, heads, len(region_starts), head_dim)
            proto_key_bank = micro_key_bank.gather(dim=2, index=pivot_gather_index).to(dtype=torch.float32)

        if surrogate_mode in _RMS_RESTORED_VALUE_MODES and len(region_starts) > 0:
            micro_value_norm_sq = micro_value_bank.to(dtype=torch.float32).square().sum(dim=-1)
            weighted_value_norm_sq = micro_value_norm_sq * length_weights.to(dtype=torch.float32).view(bsz, 1, -1)
            zero_value_norm = weighted_value_norm_sq.new_zeros((bsz, heads, 1))
            value_norm_prefix = torch.cat([zero_value_norm, weighted_value_norm_sq.cumsum(dim=2)], dim=2)
            value_length_denom = (
                length_weight_prefix.index_select(1, ends) - length_weight_prefix.index_select(1, starts)
            ).clamp_min(1e-6)
            target_value_norm = (
                (value_norm_prefix.index_select(2, ends) - value_norm_prefix.index_select(2, starts))
                / value_length_denom.view(bsz, 1, -1)
            ).clamp_min(1e-12).sqrt()
            current_value_norm = proto_value_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
            proto_value_bank = proto_value_bank * (target_value_norm / current_value_norm).view(bsz, heads, -1, 1)

        if (
            peak_mode
            in {
                "peak",
                "peak_v",
                "peak_soft",
                "peak_strong",
                "peak_light",
                "predictive_k",
                "predictive_k_token",
                "csb_peakv",
            }
            and len(region_starts) > 0
        ):
            if max_region_width > 0:
                offsets = torch.arange(max_region_width, device=device, dtype=torch.long)
                micro_indices = starts.view(-1, 1) + offsets.view(1, -1)
                valid = offsets.view(1, -1) < region_widths.view(-1, 1)
                safe_indices = micro_indices.clamp(max=micro_key_bank.shape[2] - 1)
                gathered_weights = weights.index_select(1, safe_indices.reshape(-1)).view(
                    bsz,
                    len(region_starts),
                    max_region_width,
                )
                gathered_weights = gathered_weights.masked_fill(
                    ~valid.view(1, len(region_starts), max_region_width),
                    torch.finfo(gathered_weights.dtype).min,
                )
                peak_rel = gathered_weights.argmax(dim=-1)
                peak_micro_indices = starts.view(1, -1) + peak_rel
                peak_gather_index = peak_micro_indices[:, None, :, None].expand(bsz, heads, len(region_starts), head_dim)
                peak_key_bank = micro_key_bank.gather(dim=2, index=peak_gather_index).to(dtype=torch.float32)
                peak_value_bank = micro_value_bank.gather(dim=2, index=peak_gather_index).to(dtype=torch.float32)
                peak_weight = weights.gather(dim=1, index=peak_micro_indices).to(dtype=torch.float32)
                peak_share = (peak_weight / denom).clamp(0.0, 1.0)
                if peak_mode == "peak_soft":
                    peak_share = 0.5 * peak_share
                elif peak_mode == "peak_strong":
                    peak_share = torch.sqrt(peak_share.clamp_min(0.0))
                peak_mix = peak_share.view(bsz, 1, len(region_starts), 1)
                if peak_mode not in {"peak_v", "csb_peakv"}:
                    proto_key_bank = (1.0 - peak_mix) * proto_key_bank + peak_mix * peak_key_bank
                if peak_mode not in {"predictive_k", "predictive_k_token"}:
                    proto_value_bank = (1.0 - peak_mix) * proto_value_bank + peak_mix * peak_value_bank

        if peak_mode in {"csb_k", "csb_v", "csb_kv", "csb_peakv", "csb_light"} and len(region_starts) > 0:
            # Calibrated Surrogate Bank: keep the synthetic entry independent,
            # but match the norm envelope of the evicted span.  This preserves
            # attention/output scale without modifying any retained raw KV.
            if peak_mode in {"csb_k", "csb_kv", "csb_peakv", "csb_light"}:
                micro_key_norm = micro_key_bank.to(dtype=torch.float32).norm(dim=-1)
                weighted_key_norm = micro_key_norm * weights.to(dtype=torch.float32).view(bsz, 1, -1)
                zero_norm = weighted_key_norm.new_zeros((bsz, heads, 1))
                key_norm_prefix = torch.cat([zero_norm, weighted_key_norm.cumsum(dim=2)], dim=2)
                target_key_norm = (
                    key_norm_prefix.index_select(2, ends) - key_norm_prefix.index_select(2, starts)
                ) / denom.view(bsz, 1, -1)
                current_key_norm = proto_key_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
                key_scale = _safe_key_norm_scale(
                    target_norm=target_key_norm,
                    current_norm=current_key_norm,
                )
                proto_key_bank = proto_key_bank * key_scale.view(bsz, heads, -1, 1)

            if peak_mode in {"csb_v", "csb_kv", "csb_peakv", "csb_light"}:
                micro_value_norm = micro_value_bank.to(dtype=torch.float32).norm(dim=-1)
                weighted_value_norm = micro_value_norm * weights.to(dtype=torch.float32).view(bsz, 1, -1)
                zero_norm = weighted_value_norm.new_zeros((bsz, heads, 1))
                value_norm_prefix = torch.cat([zero_norm, weighted_value_norm.cumsum(dim=2)], dim=2)
                target_value_norm = (
                    value_norm_prefix.index_select(2, ends) - value_norm_prefix.index_select(2, starts)
                ) / denom.view(bsz, 1, -1)
                current_value_norm = proto_value_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
                proto_value_bank = proto_value_bank * (target_value_norm / current_value_norm).view(bsz, heads, -1, 1)

        proto_key_bank, proto_value_bank = self._scatter_active_prototype_bank(
            proto_key_bank=proto_key_bank.to(dtype=key_states.dtype),
            proto_value_bank=proto_value_bank.to(dtype=value_states.dtype),
            active_region_indices=active_region_indices,
            full_region_count=full_region_count,
        )
        return (
            proto_key_bank,
            proto_value_bank,
            None,
            None,
            None,
            None,
        )

    def _scatter_active_prototype_bank(
        self,
        *,
        proto_key_bank,
        proto_value_bank,
        active_region_indices,
        full_region_count: int,
    ):
        if active_region_indices is None:
            return proto_key_bank, proto_value_bank
        bsz, heads, _, head_dim = proto_key_bank.shape
        full_key = proto_key_bank.new_zeros((bsz, heads, full_region_count, head_dim))
        full_value = proto_value_bank.new_zeros((bsz, heads, full_region_count, head_dim))
        full_key.index_copy_(2, active_region_indices.to(device=proto_key_bank.device), proto_key_bank)
        full_value.index_copy_(2, active_region_indices.to(device=proto_value_bank.device), proto_value_bank)
        return full_key, full_value

    def _dynamic_light_value_bank(self, *, surrogate_value_bank, chunk_scores, replace_mask, mode: str):
        if surrogate_value_bank is None or chunk_scores is None or replace_mask is None:
            return surrogate_value_bank
        if surrogate_value_bank.shape[2] != chunk_scores.shape[1]:
            return surrogate_value_bank
        risk_confidence = _rank01(chunk_scores.detach().to(dtype=torch.float32))
        if str(mode or "").lower() in {"light_soft", "peak_light"}:
            risk_confidence = 0.5 + 0.5 * risk_confidence
        scale = torch.where(
            replace_mask,
            risk_confidence,
            torch.ones_like(risk_confidence),
        )
        return surrogate_value_bank * scale.to(device=surrogate_value_bank.device, dtype=surrogate_value_bank.dtype).view(
            scale.shape[0],
            1,
            scale.shape[1],
            1,
        )

    def _chunk_prototype_bank_fast(
        self,
        *,
        key_states,
        value_states,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        surrogate_mode: str,
        return_distortion: bool = False,
    ):
        if not chunk_slices:
            bsz, heads, _, head_dim = key_states.shape
            empty_bank = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_score = key_states.new_empty((bsz, 0), dtype=torch.float32)
            return empty_bank, empty_bank.clone(), None, None, empty_score, empty_score.clone()
        if not self._is_regular_chunk_layout(chunk_slices):
            span = self._contiguous_chunk_span(chunk_slices)
            if span is not None:
                max_len = max(max(1, int(end) - int(start)) for start, end in chunk_slices)
                if 0 < max_len <= 128 and surrogate_mode in {
                    "mean",
                    "weighted_mean",
                    "asym_key_weighted",
                    "asym_value_weighted",
                    "norm_value_weighted",
                    "pivot_value_weighted",
                    "norm_rms_mean",
                    "value_sqrt_weighted_rms",
                }:
                    return self._chunk_prototype_bank_irregular_padded(
                        key_states=key_states,
                        value_states=value_states,
                        token_scores=token_scores,
                        chunk_slices=chunk_slices,
                        surrogate_mode=surrogate_mode,
                        return_distortion=return_distortion,
                    )
            if surrogate_mode == "mean" and not return_distortion:
                return self._chunk_prototype_bank_irregular_mean_prefix(
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                )
            return self._chunk_prototype_bank_generic(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                surrogate_mode=surrogate_mode,
                return_distortion=return_distortion,
            )

        base_start = chunk_slices[0][0]
        chunk_size = chunk_slices[0][1] - chunk_slices[0][0]
        tail_len = chunk_slices[-1][1] - chunk_slices[-1][0]
        regular_chunks = len(chunk_slices) if tail_len == chunk_size else len(chunk_slices) - 1
        proto_keys = []
        proto_values = []
        entropy_parts = []
        max_parts = []
        key_distortion_parts = []
        value_distortion_parts = []

        def build_chunk_proto(chunk_keys, chunk_values, chunk_scores):
            if surrogate_mode in _SCORE_WEIGHTED_SURROGATE_MODES:
                weights = torch.clamp(chunk_scores.to(dtype=torch.float32), min=1e-6)
                if surrogate_mode in _LIGHT_VALUE_WEIGHT_MODES:
                    weights = torch.sqrt(weights)
                weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
                weight_view = weights[:, None, :, :, None].to(dtype=chunk_keys.dtype)
                if surrogate_mode in _KEY_WEIGHTED_SURROGATE_MODES:
                    key_proto = (chunk_keys * weight_view).sum(dim=3)
                elif surrogate_mode in _NORM_RESTORED_KEY_MODES:
                    key_proto = _restore_mean_key_norm(chunk_keys.mean(dim=3), chunk_keys, token_dim=3)
                elif surrogate_mode in _PIVOT_KEY_MODES:
                    pivot_rel = weights.argmax(dim=-1)
                    pivot_index = pivot_rel[:, None, :, None, None].expand(
                        chunk_keys.shape[0],
                        chunk_keys.shape[1],
                        chunk_keys.shape[2],
                        1,
                        chunk_keys.shape[-1],
                    )
                    key_proto = chunk_keys.gather(dim=3, index=pivot_index).squeeze(3)
                else:
                    key_proto = chunk_keys.mean(dim=3)
                if surrogate_mode in _VALUE_WEIGHTED_SURROGATE_MODES:
                    value_proto = (chunk_values * weight_view).sum(dim=3)
                else:
                    value_proto = chunk_values.mean(dim=3)
                if chunk_scores.shape[-1] <= 1:
                    entropy = weights.new_zeros(weights.shape[:-1])
                else:
                    entropy = -(weights * torch.log(torch.clamp(weights, min=1e-12))).sum(dim=-1)
                    entropy = entropy / math.log(chunk_scores.shape[-1])
                max_weight = weights.max(dim=-1).values
            else:
                if surrogate_mode in _NORM_RESTORED_KEY_MODES:
                    key_proto = _restore_mean_key_norm(chunk_keys.mean(dim=3), chunk_keys, token_dim=3)
                else:
                    key_proto = chunk_keys.mean(dim=3)
                value_proto = chunk_values.mean(dim=3)
                entropy = None
                max_weight = None

            if surrogate_mode in _RMS_RESTORED_VALUE_MODES:
                value_proto = _restore_rms_value_norm(value_proto, chunk_values, token_dim=3)

            if not return_distortion:
                return key_proto, value_proto, entropy, max_weight, None, None

            key_diff = chunk_keys.to(dtype=torch.float32) - key_proto.to(dtype=torch.float32).unsqueeze(3)
            value_diff = chunk_values.to(dtype=torch.float32) - value_proto.to(dtype=torch.float32).unsqueeze(3)
            key_distortion = key_diff.square().sum(dim=-1).mean(dim=3).mean(dim=1)
            value_distortion = value_diff.square().sum(dim=-1).mean(dim=3).mean(dim=1)
            return key_proto, value_proto, entropy, max_weight, key_distortion, value_distortion

        if regular_chunks > 0:
            regular_tokens = regular_chunks * chunk_size
            regular_keys = key_states[:, :, base_start : base_start + regular_tokens, :].reshape(
                key_states.shape[0],
                key_states.shape[1],
                regular_chunks,
                chunk_size,
                key_states.shape[-1],
            )
            regular_values = value_states[:, :, base_start : base_start + regular_tokens, :].reshape(
                value_states.shape[0],
                value_states.shape[1],
                regular_chunks,
                chunk_size,
                value_states.shape[-1],
            )
            regular_scores = token_scores[:, base_start : base_start + regular_tokens].reshape(
                token_scores.shape[0],
                regular_chunks,
                chunk_size,
            )
            key_proto, value_proto, entropy, max_weight, key_distortion, value_distortion = build_chunk_proto(
                regular_keys,
                regular_values,
                regular_scores,
            )
            proto_keys.append(key_proto)
            proto_values.append(value_proto)
            if entropy is not None:
                entropy_parts.append(entropy)
                max_parts.append(max_weight)
            if key_distortion is not None:
                key_distortion_parts.append(key_distortion)
                value_distortion_parts.append(value_distortion)

        if tail_len != chunk_size:
            tail_start = base_start + regular_chunks * chunk_size
            tail_keys = key_states[:, :, tail_start : tail_start + tail_len, :].unsqueeze(2)
            tail_values = value_states[:, :, tail_start : tail_start + tail_len, :].unsqueeze(2)
            tail_scores = token_scores[:, tail_start : tail_start + tail_len].unsqueeze(1)
            key_proto, value_proto, entropy, max_weight, key_distortion, value_distortion = build_chunk_proto(
                tail_keys,
                tail_values,
                tail_scores,
            )
            proto_keys.append(key_proto)
            proto_values.append(value_proto)
            if entropy is not None:
                entropy_parts.append(entropy)
                max_parts.append(max_weight)
            if key_distortion is not None:
                key_distortion_parts.append(key_distortion)
                value_distortion_parts.append(value_distortion)

        proto_key_bank = torch.cat(proto_keys, dim=2)
        proto_value_bank = torch.cat(proto_values, dim=2)
        entropy_bank = torch.cat(entropy_parts, dim=1) if entropy_parts else None
        max_bank = torch.cat(max_parts, dim=1) if max_parts else None
        key_distortion_bank = torch.cat(key_distortion_parts, dim=1) if key_distortion_parts else None
        value_distortion_bank = torch.cat(value_distortion_parts, dim=1) if value_distortion_parts else None
        return proto_key_bank, proto_value_bank, entropy_bank, max_bank, key_distortion_bank, value_distortion_bank

    def _chunk_prototype_bank_irregular_mean_prefix(
        self,
        *,
        key_states,
        value_states,
        chunk_slices: Sequence[Tuple[int, int]],
    ):
        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return self._chunk_prototype_bank_generic(
                key_states=key_states,
                value_states=value_states,
                token_scores=key_states.new_zeros((key_states.shape[0], key_states.shape[2])),
                chunk_slices=chunk_slices,
                surrogate_mode="mean",
                return_distortion=False,
            )

        base_start, base_end = span
        device = key_states.device
        start_offsets = [int(start) - base_start for start, _ in chunk_slices]
        length_values = [max(1, int(end) - int(start)) for start, end in chunk_slices]
        starts = torch.tensor(start_offsets, device=device, dtype=torch.long)
        lengths_long = torch.tensor(length_values, device=device, dtype=torch.long)
        max_len = max(length_values) if length_values else 0

        if 0 < max_len <= 128:
            # Dynamic regions are explicitly bounded to fixed chunk bandwidth.
            # For this case, padded gather avoids full-span fp32 prefix sums and
            # keeps TTFT closer to the regular fixed-chunk path.
            offsets = torch.arange(max_len, device=device, dtype=torch.long)
            rel_indices = starts.view(-1, 1) + offsets.view(1, -1)
            valid = offsets.view(1, -1) < lengths_long.view(-1, 1)
            abs_indices = (rel_indices + int(base_start)).clamp(max=int(base_end) - 1).reshape(-1)

            bsz, heads, _, head_dim = key_states.shape
            num_chunks = len(chunk_slices)
            key_gather = key_states.index_select(2, abs_indices).view(bsz, heads, num_chunks, max_len, head_dim)
            value_gather = value_states.index_select(2, abs_indices).view(bsz, heads, num_chunks, max_len, head_dim)
            mask = valid.to(dtype=key_states.dtype).view(1, 1, num_chunks, max_len, 1)
            denom = lengths_long.to(dtype=key_states.dtype).view(1, 1, num_chunks, 1).clamp_min(1)
            proto_key_bank = (key_gather * mask).sum(dim=3) / denom
            proto_value_bank = (value_gather * mask).sum(dim=3) / denom
        else:
            starts = torch.tensor(start_offsets, device=device, dtype=torch.long)
            ends = torch.tensor([int(end) - base_start for _, end in chunk_slices], device=device, dtype=torch.long)
            lengths = (ends - starts).clamp_min(1).to(dtype=torch.float32).view(1, 1, -1, 1)

            # Fallback for large irregular spans where padded gather would
            # duplicate too many tokens.
            key_span = key_states[:, :, base_start:base_end, :].to(dtype=torch.float32)
            value_span = value_states[:, :, base_start:base_end, :].to(dtype=torch.float32)
            zero_key = key_span.new_zeros((key_span.shape[0], key_span.shape[1], 1, key_span.shape[-1]))
            zero_value = value_span.new_zeros((value_span.shape[0], value_span.shape[1], 1, value_span.shape[-1]))
            key_prefix = torch.cat([zero_key, key_span.cumsum(dim=2)], dim=2)
            value_prefix = torch.cat([zero_value, value_span.cumsum(dim=2)], dim=2)
            proto_key_bank = (key_prefix.index_select(2, ends) - key_prefix.index_select(2, starts)) / lengths
            proto_value_bank = (value_prefix.index_select(2, ends) - value_prefix.index_select(2, starts)) / lengths
        return (
            proto_key_bank.to(dtype=key_states.dtype),
            proto_value_bank.to(dtype=value_states.dtype),
            None,
            None,
            None,
            None,
        )

    def _chunk_prototype_bank_irregular_padded(
        self,
        *,
        key_states,
        value_states,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        surrogate_mode: str,
        return_distortion: bool = False,
    ):
        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return self._chunk_prototype_bank_generic(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                surrogate_mode=surrogate_mode,
                return_distortion=return_distortion,
            )

        base_start, base_end = span
        device = key_states.device
        lengths_long = torch.tensor(
            [max(1, int(end) - int(start)) for start, end in chunk_slices],
            device=device,
            dtype=torch.long,
        )
        starts = torch.tensor([int(start) - base_start for start, _ in chunk_slices], device=device, dtype=torch.long)
        max_len = int(lengths_long.max().item()) if lengths_long.numel() > 0 else 0
        if max_len <= 0:
            bsz, heads, _, head_dim = key_states.shape
            empty_bank = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_score = key_states.new_empty((bsz, 0), dtype=torch.float32)
            return empty_bank, empty_bank.clone(), None, None, empty_score, empty_score.clone()

        offsets = torch.arange(max_len, device=device, dtype=torch.long)
        rel_indices = starts.view(-1, 1) + offsets.view(1, -1)
        valid = offsets.view(1, -1) < lengths_long.view(-1, 1)
        abs_indices = (rel_indices + int(base_start)).clamp(max=int(base_end) - 1).reshape(-1)

        bsz, heads, _, head_dim = key_states.shape
        num_chunks = len(chunk_slices)
        key_gather = key_states.index_select(2, abs_indices).view(bsz, heads, num_chunks, max_len, head_dim)
        value_gather = value_states.index_select(2, abs_indices).view(bsz, heads, num_chunks, max_len, head_dim)
        valid_float = valid.to(device=device, dtype=torch.float32)

        entropy_bank = max_bank = None
        if surrogate_mode in _SCORE_WEIGHTED_SURROGATE_MODES:
            score_gather = token_scores.index_select(1, abs_indices).view(bsz, num_chunks, max_len).to(dtype=torch.float32)
            weights = torch.clamp(score_gather, min=1e-6) * valid_float.view(1, num_chunks, max_len)
            if surrogate_mode in _LIGHT_VALUE_WEIGHT_MODES:
                weights = torch.sqrt(weights)
            weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
            weight_view = weights.to(dtype=key_states.dtype).view(bsz, 1, num_chunks, max_len, 1)
            mask = valid.to(dtype=key_states.dtype).view(1, 1, num_chunks, max_len, 1)
            denom = lengths_long.to(dtype=key_states.dtype).view(1, 1, num_chunks, 1).clamp_min(1)
            if surrogate_mode in _KEY_WEIGHTED_SURROGATE_MODES:
                proto_key_bank = (key_gather * weight_view).sum(dim=3)
            elif surrogate_mode in _NORM_RESTORED_KEY_MODES:
                proto_key_bank = (key_gather * mask).sum(dim=3) / denom
                target_key_norm = (
                    key_gather.to(dtype=torch.float32).norm(dim=-1) * valid_float.view(1, 1, num_chunks, max_len)
                ).sum(dim=3) / lengths_long.to(device=device, dtype=torch.float32).view(1, 1, num_chunks).clamp_min(1)
                current_key_norm = proto_key_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
                key_scale = _safe_key_norm_scale(
                    target_norm=target_key_norm,
                    current_norm=current_key_norm,
                )
                proto_key_bank = proto_key_bank * key_scale.view(bsz, heads, num_chunks, 1)
            elif surrogate_mode in _PIVOT_KEY_MODES:
                pivot_rel = weights.argmax(dim=-1)
                pivot_index = pivot_rel[:, None, :, None, None].expand(bsz, heads, num_chunks, 1, head_dim)
                proto_key_bank = key_gather.gather(dim=3, index=pivot_index).squeeze(3)
            else:
                proto_key_bank = (key_gather * mask).sum(dim=3) / denom
            if surrogate_mode in _VALUE_WEIGHTED_SURROGATE_MODES:
                proto_value_bank = (value_gather * weight_view).sum(dim=3)
            else:
                proto_value_bank = (value_gather * mask).sum(dim=3) / denom

            safe_weights = torch.clamp(weights, min=1e-12)
            entropy_bank = -(weights * torch.log(safe_weights)).sum(dim=-1)
            log_denom = torch.log(lengths_long.to(device=device, dtype=torch.float32).clamp_min(2)).view(1, num_chunks)
            entropy_bank = entropy_bank / log_denom
            max_bank = weights.max(dim=-1).values
        else:
            mask = valid.to(dtype=key_states.dtype).view(1, 1, num_chunks, max_len, 1)
            denom = lengths_long.to(dtype=key_states.dtype).view(1, 1, num_chunks, 1).clamp_min(1)
            proto_key_bank = (key_gather * mask).sum(dim=3) / denom
            proto_value_bank = (value_gather * mask).sum(dim=3) / denom
            if surrogate_mode in _NORM_RESTORED_KEY_MODES:
                target_key_norm = (
                    key_gather.to(dtype=torch.float32).norm(dim=-1) * valid_float.view(1, 1, num_chunks, max_len)
                ).sum(dim=3) / lengths_long.to(device=device, dtype=torch.float32).view(1, 1, num_chunks).clamp_min(1)
                current_key_norm = proto_key_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
                key_scale = _safe_key_norm_scale(
                    target_norm=target_key_norm,
                    current_norm=current_key_norm,
                )
                proto_key_bank = proto_key_bank * key_scale.view(bsz, heads, num_chunks, 1)

        if surrogate_mode in _RMS_RESTORED_VALUE_MODES:
            value_norm_sq = value_gather.to(dtype=torch.float32).square().sum(dim=-1)
            target_value_norm = (
                (value_norm_sq * valid_float.view(1, 1, num_chunks, max_len)).sum(dim=3)
                / lengths_long.to(device=device, dtype=torch.float32).view(1, 1, num_chunks).clamp_min(1)
            ).clamp_min(1e-12).sqrt()
            current_value_norm = proto_value_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
            proto_value_bank = proto_value_bank * (target_value_norm / current_value_norm).view(bsz, heads, num_chunks, 1)

        key_distortion_bank = value_distortion_bank = None
        if return_distortion:
            mask = valid_float.view(1, 1, num_chunks, max_len)
            denom = lengths_long.to(device=device, dtype=torch.float32).view(1, 1, num_chunks).clamp_min(1)
            key_diff = key_gather.to(dtype=torch.float32) - proto_key_bank.to(dtype=torch.float32).unsqueeze(3)
            value_diff = value_gather.to(dtype=torch.float32) - proto_value_bank.to(dtype=torch.float32).unsqueeze(3)
            key_distortion_bank = (key_diff.square().sum(dim=-1) * mask).sum(dim=3) / denom
            value_distortion_bank = (value_diff.square().sum(dim=-1) * mask).sum(dim=3) / denom
            key_distortion_bank = key_distortion_bank.mean(dim=1)
            value_distortion_bank = value_distortion_bank.mean(dim=1)

        return (
            proto_key_bank.to(dtype=key_states.dtype),
            proto_value_bank.to(dtype=value_states.dtype),
            entropy_bank,
            max_bank,
            key_distortion_bank,
            value_distortion_bank,
        )

    def _chunk_prototype_bank_generic(
        self,
        *,
        key_states,
        value_states,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        surrogate_mode: str,
        return_distortion: bool = False,
    ):
        if not chunk_slices:
            bsz, heads, _, head_dim = key_states.shape
            empty_bank = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_score = key_states.new_empty((bsz, 0), dtype=torch.float32)
            return empty_bank, empty_bank.clone(), None, None, empty_score, empty_score.clone()

        proto_keys = []
        proto_values = []
        entropy_parts = []
        max_parts = []
        key_distortion_parts = []
        value_distortion_parts = []

        for start, end in chunk_slices:
            start = int(start)
            end = int(end)
            chunk_keys = key_states[:, :, start:end, :].unsqueeze(2)
            chunk_values = value_states[:, :, start:end, :].unsqueeze(2)
            chunk_scores = token_scores[:, start:end].unsqueeze(1)

            if surrogate_mode in _SCORE_WEIGHTED_SURROGATE_MODES:
                weights = torch.clamp(chunk_scores.to(dtype=torch.float32), min=1e-6)
                if surrogate_mode in _LIGHT_VALUE_WEIGHT_MODES:
                    weights = torch.sqrt(weights)
                weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
                weight_view = weights[:, None, :, :, None].to(dtype=chunk_keys.dtype)
                if surrogate_mode in _KEY_WEIGHTED_SURROGATE_MODES:
                    key_proto = (chunk_keys * weight_view).sum(dim=3)
                elif surrogate_mode in _NORM_RESTORED_KEY_MODES:
                    key_proto = _restore_mean_key_norm(chunk_keys.mean(dim=3), chunk_keys, token_dim=3)
                elif surrogate_mode in _PIVOT_KEY_MODES:
                    pivot_rel = weights.argmax(dim=-1)
                    pivot_index = pivot_rel[:, None, :, None, None].expand(
                        chunk_keys.shape[0],
                        chunk_keys.shape[1],
                        chunk_keys.shape[2],
                        1,
                        chunk_keys.shape[-1],
                    )
                    key_proto = chunk_keys.gather(dim=3, index=pivot_index).squeeze(3)
                else:
                    key_proto = chunk_keys.mean(dim=3)
                if surrogate_mode in _VALUE_WEIGHTED_SURROGATE_MODES:
                    value_proto = (chunk_values * weight_view).sum(dim=3)
                else:
                    value_proto = chunk_values.mean(dim=3)
                if chunk_scores.shape[-1] <= 1:
                    entropy = weights.new_zeros(weights.shape[:-1])
                else:
                    entropy = -(weights * torch.log(torch.clamp(weights, min=1e-12))).sum(dim=-1)
                    entropy = entropy / math.log(chunk_scores.shape[-1])
                max_weight = weights.max(dim=-1).values
            else:
                if surrogate_mode in _NORM_RESTORED_KEY_MODES:
                    key_proto = _restore_mean_key_norm(chunk_keys.mean(dim=3), chunk_keys, token_dim=3)
                else:
                    key_proto = chunk_keys.mean(dim=3)
                value_proto = chunk_values.mean(dim=3)
                entropy = None
                max_weight = None

            if surrogate_mode in _RMS_RESTORED_VALUE_MODES:
                value_proto = _restore_rms_value_norm(value_proto, chunk_values, token_dim=3)

            proto_keys.append(key_proto)
            proto_values.append(value_proto)
            if entropy is not None:
                entropy_parts.append(entropy)
                max_parts.append(max_weight)

            if return_distortion:
                key_diff = chunk_keys.to(dtype=torch.float32) - key_proto.to(dtype=torch.float32).unsqueeze(3)
                value_diff = chunk_values.to(dtype=torch.float32) - value_proto.to(dtype=torch.float32).unsqueeze(3)
                key_distortion_parts.append(key_diff.square().sum(dim=-1).mean(dim=3).mean(dim=1))
                value_distortion_parts.append(value_diff.square().sum(dim=-1).mean(dim=3).mean(dim=1))

        proto_key_bank = torch.cat(proto_keys, dim=2)
        proto_value_bank = torch.cat(proto_values, dim=2)
        entropy_bank = torch.cat(entropy_parts, dim=1) if entropy_parts else None
        max_bank = torch.cat(max_parts, dim=1) if max_parts else None
        key_distortion_bank = torch.cat(key_distortion_parts, dim=1) if key_distortion_parts else None
        value_distortion_bank = torch.cat(value_distortion_parts, dim=1) if value_distortion_parts else None
        return proto_key_bank, proto_value_bank, entropy_bank, max_bank, key_distortion_bank, value_distortion_bank

    def _build_layout_meta(
        self,
        *,
        full_tokens: int,
        compressed_tokens: int,
        sink_len: int,
        recent_len: int,
        chunk_lengths,
        selected_chunk_mask,
        output_chunk_lengths,
        chunk_mode_names,
    ):
        if not self._save_layout_meta:
            return None

        sink_len = int(sink_len)
        recent_len = int(recent_len)
        compressed_tokens = int(compressed_tokens)
        chunk_length_list = [int(v) for v in chunk_lengths.tolist()]
        output_length_list = [int(v) for v in output_chunk_lengths.tolist()]
        selected_mask_list = [bool(v) for v in selected_chunk_mask.tolist()]
        cursor = sink_len
        kept_ranges = []
        surrogate_ranges = []
        chunk_entries = []
        for chunk_idx, (orig_len, packed_len, selected, mode_name) in enumerate(
            zip(chunk_length_list, output_length_list, selected_mask_list, chunk_mode_names)
        ):
            span_start = cursor
            span_end = span_start + packed_len
            if selected:
                surrogate_ranges.append([int(span_start), int(span_end)])
            else:
                kept_ranges.append([int(span_start), int(span_end)])
            cursor = span_end
            chunk_entries.append(
                {
                    "chunk_index": int(chunk_idx),
                    "original_length": int(orig_len),
                    "packed_length": int(packed_len),
                    "selected": bool(selected),
                    "mode": mode_name,
                    "packed_span": [int(span_start), int(span_end)],
                }
            )
        chunk_region_start = sink_len
        chunk_region_end = compressed_tokens - recent_len
        return {
            "layout_version": 2,
            "layout": "sink|chunks|recent",
            "full_tokens": int(full_tokens),
            "compressed_tokens": int(compressed_tokens),
            "sink_range": [0, sink_len],
            "chunk_region_range": [chunk_region_start, chunk_region_end],
            "kept_range": [
                kept_ranges[0][0] if kept_ranges else chunk_region_start,
                kept_ranges[-1][1] if kept_ranges else chunk_region_start,
            ],
            "surrogate_range": [
                surrogate_ranges[0][0] if surrogate_ranges else chunk_region_start,
                surrogate_ranges[-1][1] if surrogate_ranges else chunk_region_start,
            ],
            "kept_ranges": kept_ranges,
            "surrogate_ranges": surrogate_ranges,
            "recent_range": [compressed_tokens - recent_len, compressed_tokens],
            "chunk_lengths": chunk_length_list,
            "output_chunk_lengths": output_length_list,
            "selected_chunk_mask": selected_mask_list,
            "chunks": chunk_entries,
        }

    def _adaptive_chunk_size(self, *, compressible_len: int, budget_compressible: int, tokens_to_save: int) -> int:
        del tokens_to_save
        base_chunk_size = max(1, int(self.chunk_size))
        adaptive_chunk_size = max(base_chunk_size, math.ceil(compressible_len / max(1, budget_compressible)))
        return min(adaptive_chunk_size, compressible_len)

    def _needs_runtime_global_prototypes(self, *, replace_mask) -> bool:
        return self.spec.direct_strategy == "global" and bool(replace_mask.any().item())

    def _global_prototypes(
        self,
        *,
        key_states,
        value_states,
        token_scores,
        chunk_slices,
        replace_mask,
        fallback_start: int,
        fallback_end: int,
        chunk_scores,
        chunk_proto_key_bank,
        chunk_proto_value_bank,
    ):
        bsz = key_states.shape[0]
        num_heads = key_states.shape[1]
        head_dim = key_states.shape[-1]
        global_keys = []
        global_values = []
        if chunk_slices:
            chunk_base_start = chunk_slices[0][0]
            chunk_lengths = torch.tensor(
                [end - start for start, end in chunk_slices],
                device=replace_mask.device,
                dtype=torch.long,
            )
        else:
            chunk_base_start = fallback_start
            chunk_lengths = torch.empty((0,), device=replace_mask.device, dtype=torch.long)

        for batch_idx in range(bsz):
            if chunk_proto_key_bank is not None and chunk_proto_value_bank is not None and chunk_slices:
                if replace_mask[batch_idx].any():
                    selected_chunk_indices = torch.nonzero(replace_mask[batch_idx], as_tuple=False).flatten()
                    selected_keys = chunk_proto_key_bank[batch_idx : batch_idx + 1].index_select(2, selected_chunk_indices)
                    selected_values = chunk_proto_value_bank[batch_idx : batch_idx + 1].index_select(2, selected_chunk_indices)
                    selected_scores = chunk_scores[batch_idx].index_select(0, selected_chunk_indices)
                    global_key, global_value = self._prototype(selected_keys, selected_values, selected_scores)
                else:
                    fallback_keys = chunk_proto_key_bank[batch_idx : batch_idx + 1]
                    fallback_values = chunk_proto_value_bank[batch_idx : batch_idx + 1]
                    fallback_scores = chunk_scores[batch_idx]
                    global_key, global_value = self._prototype(fallback_keys, fallback_values, fallback_scores)
            elif chunk_lengths.numel() > 0 and replace_mask[batch_idx].any():
                selected_token_mask = replace_mask[batch_idx].repeat_interleave(chunk_lengths)
                selected_positions = torch.nonzero(selected_token_mask, as_tuple=False).flatten()
                selected_positions = selected_positions + chunk_base_start
                selected_keys = key_states[batch_idx : batch_idx + 1].index_select(2, selected_positions)
                selected_values = value_states[batch_idx : batch_idx + 1].index_select(2, selected_positions)
                selected_scores = token_scores[batch_idx].index_select(0, selected_positions)
                global_key, global_value = self._prototype(selected_keys, selected_values, selected_scores)
            else:
                fallback_keys = key_states[batch_idx : batch_idx + 1, :, fallback_start:fallback_end, :]
                fallback_values = value_states[batch_idx : batch_idx + 1, :, fallback_start:fallback_end, :]
                fallback_scores = token_scores[batch_idx, fallback_start:fallback_end]
                global_key, global_value = self._prototype(fallback_keys, fallback_values, fallback_scores)

            global_keys.append(global_key)
            global_values.append(global_value)

        if not global_keys:
            device = key_states.device
            dtype = key_states.dtype
            empty_key = torch.zeros((bsz, num_heads, 1, head_dim), device=device, dtype=dtype)
            empty_value = torch.zeros((bsz, num_heads, 1, head_dim), device=device, dtype=dtype)
            return empty_key, empty_value
        return torch.cat(global_keys, dim=0), torch.cat(global_values, dim=0)

    def _prototype(self, key_tensor, value_tensor, token_scores):
        key_proto, value_proto, _ = self._prototype_with_stats(key_tensor, value_tensor, token_scores)
        return key_proto, value_proto

    def _prototype_with_stats(self, key_tensor, value_tensor, token_scores):
        return prototype_pair(
            key_tensor,
            value_tensor,
            token_scores,
            surrogate_mode=self.spec.surrogate_mode,
        )

    def _protected_sink_tokens(self) -> int:
        if self.spec.protected_sink:
            return max(0, int(self.sink_tokens))
        return 0

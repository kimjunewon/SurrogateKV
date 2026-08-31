# Runtime integration adapted in part from SnapKV (Apache-2.0),
# KVCache-Factory, and AdaKV (MIT), with SurrogateKV modifications.
# See NOTICE and THIRD_PARTY_LICENSES.md.

from __future__ import annotations

import math
import os
import time
from typing import ClassVar, Dict, Sequence, Tuple

import torch

from .registry import MODE_TO_SPEC, MethodSpec
from .runtime.cache_pipeline import (
    CachePackingMixin,
    CacheScoringMixin,
    CacheStateMixin,
)
from .runtime.common import (
    _SURKV_HEAD_SCORE_FUSION,
    _SURKV_PROFILE_TIMING,
    _SURKV_SCORE_METHOD,
    _env_flag,
    _rank01,
)
from .runtime.headwise_runtime import HeadwiseRuntimeMixin
from .runtime.layer_budget import LayerBudgetMixin
from .runtime.prototype_bank import PrototypeBankMixin
from .runtime.region_allocator import allocate_surrogate_regions
from .schedule import adaptive_entropy_keep_ratio


class SurKVCluster(
    LayerBudgetMixin,
    CacheStateMixin,
    CachePackingMixin,
    CacheScoringMixin,
    PrototypeBankMixin,
    HeadwiseRuntimeMixin,
):
    _GLOBAL_BUDGET_LEDGER: ClassVar[Dict[str, object]] = {
        "enabled": False,
    }
    _GLOBAL_LAYER_DYNAMIC_STATE: ClassVar[Dict[str, object]] = {
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


    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        if self.mode == "surrogate_kv_ada" and not bool(getattr(self, "_allocator_plan_only", False)):
            raise RuntimeError(
                "SurrogateKV-Ada requires update_kv_headwise() so that Ada-KV's per-head "
                "capacities and selections are preserved."
            )
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

        sink_len = 0
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
                # Avoid per-layer top-k reductions and host reads here. The
                # ledger still reallocates capacity from actual layer usage.
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
        # The public SurrogateKV variants use one allocator-owned middle span.
        # The allocator rebuilds raw/surrogate/drop regions from micro-atoms.
        chunk_slices = [(compressible_start, past_len)]
        chunk_mean_scores = token_scores.new_zeros((token_scores.shape[0], 1))
        record_update_timing("region_setup_stage_total", stage_start)
        timing_breakdown["planning"] += time.perf_counter() - stage_start
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
        lazy_surrogate_region_tensors = self.layer_scheduler != "adaptive_entropy"
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
        local_key_bank = local_value_bank = None

        alloc_call_start = time.perf_counter()
        allocated = self._allocate_surrogate_regions(
            token_scores=token_scores,
            chunk_slices=chunk_slices,
            chunk_lengths=chunk_lengths,
            target_compressed_tokens=effective_capacity_prompt,
            sink_len=sink_len,
            recent_len=recent_len,
        )
        record_update_timing("surrogate_allocator_call", alloc_call_start)
        if allocated is None:
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

        post_alloc_start = time.perf_counter()
        chunk_slices, chunk_lengths, replace_mask, surrogate_lengths = allocated
        chunk_mean_scores = token_scores.new_zeros((token_scores.shape[0], len(chunk_slices)))
        stats = dict(self._last_allocator_stats or {})
        stats.update(
            {
                "surrogate_kv_primary_allocator": 1,
                "surrogate_kv_posthoc_selector": 0,
            }
        )
        self._last_allocator_stats = stats
        record_update_timing("allocated_postprocess", post_alloc_start)

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
        if chunk_proto_key_bank is None or chunk_proto_value_bank is None:
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
        # Each replaced region is represented by its own local prototype.
        local_key_bank, local_value_bank = chunk_proto_key_bank, chunk_proto_value_bank

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

        for batch_idx in range(bsz):
            compress_call_start = time.perf_counter()
            (
                compressed_batch_key,
                compressed_batch_value,
                batch_mode_counts,
                two_surrogate_chunks,
                selected_runs,
                batch_layout_meta,
            ) = self._compress_banked_fast_batch(
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
                mode_name="local",
            )
            record_update_timing("compress_local_batch", compress_call_start)

            compressed_keys.append(compressed_batch_key)
            compressed_values.append(compressed_batch_value)
            selected_chunks_per_batch.append(selected_runs)
            selected_runs_per_batch.append(selected_runs)
            two_surrogate_chunks_per_batch.append(two_surrogate_chunks)
            mode_counts_per_batch.append(batch_mode_counts)
            layout_meta_per_batch.append(batch_layout_meta)

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


    def _allocate_surrogate_regions(
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
        return allocate_surrogate_regions(
            self,
            token_scores=token_scores,
            chunk_slices=chunk_slices,
            chunk_lengths=chunk_lengths,
            target_compressed_tokens=target_compressed_tokens,
            sink_len=sink_len,
            recent_len=recent_len,
            predictive=predictive,
            _rank01_fn=_rank01,
            _profile_timing=_SURKV_PROFILE_TIMING,
        )

from __future__ import annotations

import time

import numpy as np
import torch

from .allocation import SurrogateAllocationMixin
from .packing import SurrogatePackingMixin
from .prototypes import SurrogatePrototypeMixin
from .registry import MODE_TO_SPEC, MethodSpec
from .schedule import adaptive_entropy_keep_ratio


class SurKVCluster(
    SurrogateAllocationMixin,
    SurrogatePackingMixin,
    SurrogatePrototypeMixin,
):
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
        sink_policy: str = "static",
        sink_preference: int = 0,
        layer_keep_ratio: float | None = None,
        layer_scheduler: str = "uniform",
    ) -> None:
        self.last_stats = {}
        self.last_layout_meta = None
        self._zero_pair_cache = {}
        self._save_layout_meta = False
        self._save_surrogates = False
        self._last_surrogates = {}
        self._last_allocator_stats = {}
        self._last_sink_allocator_stats = {}
        self._last_fast_pack_plan = None
        self._set_config(
            mode=mode,
            window_size=window_size,
            max_capacity_prompt=max_capacity_prompt,
            kernel_size=kernel_size,
            pooling=pooling,
            chunk_size=chunk_size,
            local_radius=local_radius,
            sink_tokens=sink_tokens,
            sink_policy=sink_policy,
            sink_preference=sink_preference,
            layer_keep_ratio=layer_keep_ratio,
            layer_scheduler=layer_scheduler,
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
        sink_policy: str = "static",
        sink_preference: int = 0,
        layer_keep_ratio: float | None = None,
        layer_scheduler: str = "uniform",
    ) -> None:
        self._last_fast_pack_plan = None
        self._last_allocator_stats = {}
        self._last_sink_allocator_stats = {}
        self._set_config(
            mode=mode,
            window_size=window_size,
            max_capacity_prompt=max_capacity_prompt,
            kernel_size=kernel_size,
            pooling=pooling,
            chunk_size=chunk_size,
            local_radius=local_radius,
            sink_tokens=sink_tokens,
            sink_policy=sink_policy,
            sink_preference=sink_preference,
            layer_keep_ratio=layer_keep_ratio,
            layer_scheduler=layer_scheduler,
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
        sink_policy: str,
        sink_preference: int,
        layer_keep_ratio: float | None,
        layer_scheduler: str,
    ) -> None:
        self.mode = mode
        self.spec: MethodSpec = MODE_TO_SPEC[mode]
        self.window_size = int(window_size)
        self.max_capacity_prompt = int(max_capacity_prompt)
        self.kernel_size = int(kernel_size)
        self.pooling = str(pooling)
        self.chunk_size = int(chunk_size)
        self.local_radius = int(local_radius)
        self.sink_tokens = max(0, int(sink_tokens))
        self.sink_policy = self._normalize_sink_policy(sink_policy)
        self.sink_preference = max(-1, min(1, int(sink_preference or 0)))
        self.layer_keep_ratio = None if layer_keep_ratio is None else min(1.0, max(0.0, float(layer_keep_ratio)))
        self.layer_scheduler = str(layer_scheduler).strip().lower()

    @staticmethod
    def _normalize_sink_policy(policy: str | None) -> str:
        normalized = str(policy or "static").strip().lower().replace("-", "_")
        aliases = {
            "default": "static",
            "protected": "static",
            "none": "off",
            "false": "off",
            "0": "off",
            "always": "on",
            "true": "on",
            "1": "on",
            "auto": "dynamic",
            "dynamic_saliency": "dynamic",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"static", "off", "on", "dynamic"}:
            raise ValueError(f"Unsupported SurrogateKV sink policy: {policy}")
        return normalized

    def enable_layout_meta(self, enable: bool = True) -> None:
        self._save_layout_meta = bool(enable)
        if not enable:
            self.last_layout_meta = None

    def enable_surrogate_saving(self, enable: bool = True) -> None:
        self._save_surrogates = bool(enable)
        if not enable:
            self._last_surrogates.clear()

    def get_last_surrogates(self):
        if not self._last_surrogates:
            return {}
        return {
            key: (
                np.asarray(value, dtype=np.float32)
                if isinstance(value, np.ndarray)
                else value.detach().cpu().numpy().astype(np.float32)
            )
            for key, value in self._last_surrogates.items()
        }

    def _record_saved_surrogate(self, *, batch_idx: int, chunk_idx: int, surrogate_key, surrogate_value) -> None:
        if not self._save_surrogates:
            return
        self._last_surrogates[f"surrogate_k_b{batch_idx}_c{chunk_idx}"] = (
            surrogate_key.detach().cpu().numpy().astype(np.float32)
        )
        self._last_surrogates[f"surrogate_v_b{batch_idx}_c{chunk_idx}"] = (
            surrogate_value.detach().cpu().numpy().astype(np.float32)
        )

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        update_start = time.perf_counter()
        del attention_mask
        assert key_states.shape[-2] == query_states.shape[-2]
        self.last_layout_meta = None
        self._last_allocator_stats = {}
        self._last_sink_allocator_stats = {}
        self._last_fast_pack_plan = None

        bsz, _, q_len, head_dim = query_states.shape
        timing_breakdown = {"score": 0.0, "planning": 0.0, "prototype": 0.0, "packing": 0.0}
        configured_keep_ratio = min(1.0, float(self.max_capacity_prompt) / max(float(q_len), 1.0))
        if self.layer_keep_ratio is not None:
            configured_keep_ratio = min(1.0, max(1.0 / max(q_len, 1), float(self.layer_keep_ratio)))
        effective_capacity_prompt = max(1, min(q_len, int(round(q_len * configured_keep_ratio))))
        recent_len = min(self.window_size, q_len)

        def finish_passthrough(*, sink_len: int = 0, num_chunks: int = 0, chunk_size: int = 0):
            self.last_stats = self._stats(
                full_tokens=q_len,
                compressed_tokens=q_len,
                recent_tokens=recent_len,
                selected_chunks=0,
                selected_runs=0,
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                sink_tokens=sink_len,
                two_surrogate_chunks=0,
                mode_counts={},
                op_seconds=time.perf_counter() - update_start,
                configured_keep_ratio=configured_keep_ratio,
                timing_breakdown=timing_breakdown,
            )
            return key_states, value_states

        if q_len <= effective_capacity_prompt or recent_len <= 0:
            return finish_passthrough()

        past_len = q_len - recent_len
        if past_len <= 0:
            return finish_passthrough()

        sink_len = min(self._protected_sink_tokens(), past_len)
        compressible_start = sink_len
        compressible_len = past_len - compressible_start
        if compressible_len <= 0:
            return finish_passthrough(sink_len=sink_len)

        budget_past_total = max(1, effective_capacity_prompt - recent_len)
        budget_compressible = max(0, budget_past_total - sink_len)
        tokens_to_save = max(0, compressible_len - budget_compressible)
        adaptive_chunk_size = self._adaptive_chunk_size(
            compressible_len=compressible_len,
            budget_compressible=budget_compressible,
            tokens_to_save=tokens_to_save,
        )
        regular_chunk_slices = [
            (compressible_start + start, compressible_start + end)
            for start, end in self._chunk_slices(compressible_len, adaptive_chunk_size)
        ]

        if budget_compressible >= compressible_len:
            return finish_passthrough(
                sink_len=sink_len,
                num_chunks=len(regular_chunk_slices),
                chunk_size=adaptive_chunk_size,
            )

        stage_start = time.perf_counter()
        token_scores = self._past_token_scores(
            key_states=key_states,
            query_states=query_states,
            recent_len=recent_len,
            past_len=past_len,
            head_dim=head_dim,
            num_key_value_groups=num_key_value_groups,
        )
        timing_breakdown["score"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        method_kind = self.spec.kind
        if method_kind == "surrogate":
            chunk_slices = [(compressible_start, past_len)]
            chunk_mean_scores = token_scores.new_zeros((token_scores.shape[0], 1))
            chunk_max_scores = token_scores.new_zeros((token_scores.shape[0], 1))
        elif method_kind == "drop":
            chunk_slices = regular_chunk_slices
            chunk_mean_scores, chunk_max_scores = self._chunk_statistics_fast_mean_max(
                token_scores=token_scores,
                chunk_slices=chunk_slices,
            )
        else:
            raise ValueError(f"Unsupported SurKV method kind: {method_kind}")

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
                timing_breakdown["planning"] += time.perf_counter() - stage_start
                return finish_passthrough(
                    sink_len=sink_len,
                    num_chunks=len(chunk_slices),
                    chunk_size=adaptive_chunk_size,
                )

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

        if method_kind == "surrogate":
            if self.sink_policy == "dynamic":
                chosen_plan = self._plan_cache_with_dynamic_sink(
                    value_states=value_states,
                    token_scores=token_scores,
                    past_len=past_len,
                    recent_len=recent_len,
                    effective_capacity_prompt=effective_capacity_prompt,
                )
                if chosen_plan is None:
                    timing_breakdown["planning"] += time.perf_counter() - stage_start
                    return finish_passthrough(sink_len=0, num_chunks=len(chunk_slices), chunk_size=adaptive_chunk_size)
                sink_len = int(chosen_plan["sink_len"])
                compressible_start = int(chosen_plan["compressible_start"])
                compressible_len = int(chosen_plan["compressible_len"])
                budget_past_total = int(chosen_plan["budget_past_total"])
                budget_compressible = int(chosen_plan["budget_compressible"])
                tokens_to_save = int(chosen_plan["tokens_to_save"])
                adaptive_chunk_size = int(chosen_plan["adaptive_chunk_size"])
                regular_chunk_slices = chosen_plan["regular_chunk_slices"]
                chunk_slices = chosen_plan["chunk_slices"]
                chunk_lengths = chosen_plan["chunk_lengths"]
                replace_mask = chosen_plan["replace_mask"]
                surrogate_lengths = chosen_plan["surrogate_lengths"]
                chunk_mean_scores = chosen_plan["chunk_mean_scores"]
                chunk_max_scores = chosen_plan["chunk_max_scores"]
                stats = dict(chosen_plan.get("allocator_stats") or {})
                stats.update(
                    {
                        "surrogate_kv_dfx_backend": 0,
                        "surrogate_kv_bounded_frontier_market": 1,
                        "surrogate_kv_posthoc_selector": 0,
                    }
                )
                self._last_allocator_stats = stats
                self._last_fast_pack_plan = chosen_plan.get("fast_pack_plan")
            else:
                allocated = self._allocate_surrogate_by_spec(
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    target_compressed_tokens=effective_capacity_prompt,
                    sink_len=sink_len,
                    recent_len=recent_len,
                    merge_first=False,
                    current_frontier_accept=False,
                    admission_shadow_price=True,
                    frontier_region_price=True,
                    completion_order=False,
                    bounded_market=True,
                    budget_complete_peel=True,
                )
                if allocated is None:
                    chunk_slices = regular_chunk_slices
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
                    chunk_mean_scores, chunk_max_scores = self._chunk_statistics_fast_mean_max(
                        token_scores=token_scores,
                        chunk_slices=chunk_slices,
                    )
                    replace_mask = self._select_low_importance_chunks(
                        chunk_scores=chunk_mean_scores,
                        chunk_max_scores=chunk_max_scores,
                        chunk_lengths=chunk_lengths,
                        surrogate_lengths=surrogate_lengths,
                        tokens_to_save=tokens_to_save,
                    )
                    self._last_allocator_stats = {"surrogate_kv_allocator_fallback": 1}
                else:
                    chunk_slices, chunk_lengths, replace_mask, surrogate_lengths = allocated
                    chunk_mean_scores = token_scores.new_zeros((token_scores.shape[0], len(chunk_slices)))
                    chunk_max_scores = token_scores.new_zeros((token_scores.shape[0], len(chunk_slices)))
                    stats = dict(self._last_allocator_stats or {})
                    stats.update(
                        {
                            "surrogate_kv_dfx_backend": 0,
                            "surrogate_kv_bounded_frontier_market": 1,
                            "surrogate_kv_posthoc_selector": 0,
                        }
                    )
                    self._last_allocator_stats = stats
        else:
            replace_mask = self._select_low_importance_chunks(
                chunk_scores=chunk_mean_scores,
                chunk_max_scores=chunk_max_scores,
                chunk_lengths=chunk_lengths,
                surrogate_lengths=surrogate_lengths,
                tokens_to_save=tokens_to_save,
            )
        timing_breakdown["planning"] += time.perf_counter() - stage_start

        if not replace_mask.any():
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
        surrogate_key_bank = surrogate_value_bank = None
        surrogate_bank_indices = None
        if method_kind == "surrogate":
            selected_surrogate_mask = replace_mask & (surrogate_lengths > 0)
            needs_surrogate_bank = bool(selected_surrogate_mask.any().item())
            if needs_surrogate_bank or self._save_layout_meta or self._save_surrogates:
                compact_output = not self._save_layout_meta and not self._save_surrogates
                (
                    surrogate_key_bank,
                    surrogate_value_bank,
                    _entropy,
                    _max_weight,
                    _key_distortion,
                    _value_distortion,
                    surrogate_bank_indices,
                ) = self._dynamic_micro_prototype_bank(
                    key_states=key_states,
                    value_states=value_states,
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    surrogate_mode=self.spec.surrogate_mode,
                    selected_only_mask=selected_surrogate_mask,
                    compact_output=compact_output,
                )
        timing_breakdown["prototype"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        compressed_keys = []
        compressed_values = []
        selected_runs_per_batch = []
        two_surrogate_chunks_per_batch = []
        mode_counts_per_batch = []
        layout_meta_per_batch = []

        for batch_idx in range(bsz):
            if method_kind == "surrogate":
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                ) = self._compress_surrogate_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    sink_len=sink_len,
                    past_len=past_len,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                    surrogate_key_bank=surrogate_key_bank,
                    surrogate_value_bank=surrogate_value_bank,
                    surrogate_bank_indices=surrogate_bank_indices,
                )
            else:
                (
                    compressed_batch_key,
                    compressed_batch_value,
                    batch_mode_counts,
                    two_surrogate_chunks,
                    selected_runs,
                    batch_layout_meta,
                ) = self._compress_drop_batch(
                    batch_idx=batch_idx,
                    key_states=key_states,
                    value_states=value_states,
                    chunk_slices=chunk_slices,
                    chunk_lengths=chunk_lengths,
                    replace_mask=replace_mask,
                    sink_len=sink_len,
                    past_len=past_len,
                    surrogate_lengths=surrogate_lengths[batch_idx],
                )

            compressed_keys.append(compressed_batch_key)
            compressed_values.append(compressed_batch_value)
            selected_runs_per_batch.append(selected_runs)
            two_surrogate_chunks_per_batch.append(two_surrogate_chunks)
            mode_counts_per_batch.append(batch_mode_counts)
            layout_meta_per_batch.append(batch_layout_meta)

        compressed_key_states = torch.cat(compressed_keys, dim=0)
        compressed_value_states = torch.cat(compressed_values, dim=0)
        timing_breakdown["packing"] += time.perf_counter() - stage_start
        dynamic_region_lengths = (
            [int(end) - int(start) for start, end in chunk_slices]
            if self.spec.dynamic_regioning
            else []
        )
        max_selected_runs = max(selected_runs_per_batch) if selected_runs_per_batch else 0
        max_two_surrogate_chunks = max(two_surrogate_chunks_per_batch) if two_surrogate_chunks_per_batch else 0
        self.last_stats = self._stats(
            full_tokens=q_len,
            compressed_tokens=compressed_key_states.shape[-2],
            recent_tokens=recent_len,
            selected_chunks=max_selected_runs,
            selected_runs=max_selected_runs,
            num_chunks=len(chunk_slices),
            chunk_size=adaptive_chunk_size,
            sink_tokens=sink_len,
            two_surrogate_chunks=max_two_surrogate_chunks,
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
        self.last_layout_meta = layout_meta_per_batch
        return compressed_key_states, compressed_value_states

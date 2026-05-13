from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .methods.base import AllocationPlan, MethodSpec, SurrogateContext
from .methods.registry import MODE_TO_SPEC
from .methods.utils.schedule import adaptive_entropy_keep_ratio


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


class SurKVCluster:
    """Small runtime shell shared by all SurKV methods.

    Method-specific behavior lives under ``surkv.methods``.  ``core.py`` owns
    only the attention-score pass, planner invocation, packing, and accounting.
    Future methods plug in through ``MethodSpec.plan_chunks`` and
    ``MethodSpec.build_surrogates`` without re-growing this file.
    """

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
    ) -> None:
        self.last_stats = {}
        self._save_surrogates = False
        self._last_surrogates = {}
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
    ) -> None:
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
    ) -> None:
        if mode not in MODE_TO_SPEC:
            supported = ", ".join(sorted(MODE_TO_SPEC))
            raise ValueError(f"Unsupported SurKV mode {mode!r}. Supported modes: {supported}")
        self.mode = mode
        self.spec: MethodSpec = MODE_TO_SPEC[mode]
        self.window_size = int(window_size)
        self.max_capacity_prompt = int(max_capacity_prompt)
        self.kernel_size = int(kernel_size)
        self.pooling = str(pooling)
        self.chunk_size = int(chunk_size)
        self.local_radius = int(local_radius)
        self.sink_tokens = int(sink_tokens)
        self.layer_keep_ratio = None if layer_keep_ratio is None else min(1.0, max(0.0, float(layer_keep_ratio)))
        self.layer_scheduler = str(layer_scheduler or "uniform").strip().lower()

    def enable_surrogate_saving(self, enable: bool = True):
        self._save_surrogates = bool(enable)
        if not enable:
            self._last_surrogates.clear()

    def get_last_surrogates(self):
        if not self._last_surrogates:
            return {}
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

        self._last_surrogates.clear()

        bsz, _, q_len, head_dim = query_states.shape
        timing_breakdown = {"score": 0.0, "planning": 0.0, "prototype": 0.0, "packing": 0.0}
        configured_keep_ratio = self._configured_keep_ratio(q_len)
        effective_capacity_prompt = max(1, min(q_len, int(round(q_len * configured_keep_ratio))))
        recent_len = min(max(0, self.window_size), q_len)

        if q_len <= effective_capacity_prompt or recent_len <= 0:
            return self._return_full(
                key_states,
                value_states,
                update_start=update_start,
                q_len=q_len,
                recent_len=recent_len,
                configured_keep_ratio=configured_keep_ratio,
            )

        past_len = q_len - recent_len
        sink_len = min(self._protected_sink_tokens(), past_len)
        compressible_start = sink_len
        compressible_len = past_len - compressible_start
        if compressible_len <= 0:
            return self._return_full(
                key_states,
                value_states,
                update_start=update_start,
                q_len=q_len,
                recent_len=recent_len,
                sink_len=sink_len,
                configured_keep_ratio=configured_keep_ratio,
            )

        budget_past_total = max(1, effective_capacity_prompt - recent_len)
        budget_compressible = max(0, budget_past_total - sink_len)
        tokens_to_save = max(0, compressible_len - budget_compressible)
        adaptive_chunk_size = self._adaptive_chunk_size(
            compressible_len=compressible_len,
            budget_compressible=budget_compressible,
        )
        chunk_slices = [
            (compressible_start + start, compressible_start + end)
            for start, end in self._chunk_slices(compressible_len, adaptive_chunk_size)
        ]

        if budget_compressible >= compressible_len or not chunk_slices:
            return self._return_full(
                key_states,
                value_states,
                update_start=update_start,
                q_len=q_len,
                recent_len=recent_len,
                sink_len=sink_len,
                num_chunks=len(chunk_slices),
                chunk_size=adaptive_chunk_size,
                configured_keep_ratio=configured_keep_ratio,
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
        chunk_scores = self._chunk_mean_scores(token_scores=token_scores, chunk_slices=chunk_slices)
        if self.layer_scheduler == "adaptive_entropy":
            configured_keep_ratio = adaptive_entropy_keep_ratio(
                base_keep_ratio=configured_keep_ratio,
                chunk_scores=chunk_scores,
                q_len=q_len,
            )
            effective_capacity_prompt = max(1, min(q_len, int(round(q_len * configured_keep_ratio))))
            budget_past_total = max(1, effective_capacity_prompt - recent_len)
            budget_compressible = max(0, budget_past_total - sink_len)
            tokens_to_save = max(0, compressible_len - budget_compressible)
            if budget_compressible >= compressible_len:
                return self._return_full(
                    key_states,
                    value_states,
                    update_start=update_start,
                    q_len=q_len,
                    recent_len=recent_len,
                    sink_len=sink_len,
                    num_chunks=len(chunk_slices),
                    chunk_size=adaptive_chunk_size,
                    configured_keep_ratio=configured_keep_ratio,
                    timing_breakdown=timing_breakdown,
                )

        chunk_lengths = torch.tensor(
            [end - start for start, end in chunk_slices],
            device=key_states.device,
            dtype=torch.long,
        )
        surrogate_lengths = torch.ones(
            (bsz, len(chunk_slices)),
            device=key_states.device,
            dtype=torch.long,
        )

        plan = self.spec.plan_chunks(
            SurrogateContext(
                key_states=key_states,
                value_states=value_states,
                token_scores=token_scores,
                chunk_scores=chunk_scores,
                chunk_slices=tuple(chunk_slices),
                chunk_lengths=chunk_lengths,
                replace_mask=torch.zeros_like(surrogate_lengths, dtype=torch.bool),
                surrogate_lengths=surrogate_lengths,
                past_len=past_len,
                sink_len=sink_len,
                recent_len=recent_len,
                local_radius=self.local_radius,
                tokens_to_save=tokens_to_save,
                budget_compressible=budget_compressible,
                chunk_size=adaptive_chunk_size,
            )
        )
        if not isinstance(plan, AllocationPlan):
            raise TypeError(f"{self.spec.name} planner returned {type(plan).__name__}, expected AllocationPlan")
        chunk_slices = tuple(plan.chunk_slices)
        chunk_lengths = plan.chunk_lengths
        replace_mask = plan.replace_mask
        surrogate_lengths = plan.surrogate_lengths
        allocator_stats = dict(plan.allocator_stats or {})
        timing_breakdown["planning"] += time.perf_counter() - stage_start

        if not replace_mask.any():
            return self._return_full(
                key_states,
                value_states,
                update_start=update_start,
                q_len=q_len,
                recent_len=recent_len,
                sink_len=sink_len,
                num_chunks=len(chunk_slices),
                chunk_size=adaptive_chunk_size,
                configured_keep_ratio=configured_keep_ratio,
                timing_breakdown=timing_breakdown,
            )

        stage_start = time.perf_counter()
        method_context = SurrogateContext(
            key_states=key_states,
            value_states=value_states,
            token_scores=token_scores,
            chunk_scores=chunk_scores,
            chunk_slices=tuple(chunk_slices),
            chunk_lengths=chunk_lengths,
            replace_mask=replace_mask,
            surrogate_lengths=surrogate_lengths,
            past_len=past_len,
            sink_len=sink_len,
            recent_len=recent_len,
            local_radius=self.local_radius,
            tokens_to_save=tokens_to_save,
            budget_compressible=budget_compressible,
            chunk_size=adaptive_chunk_size,
        )
        surrogate_key_bank, surrogate_value_bank = self.spec.build_surrogates(method_context)
        timing_breakdown["prototype"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        compressed_keys = []
        compressed_values = []
        selected_runs_per_batch = []
        two_surrogate_chunks_per_batch = []
        surrogate_slots_per_batch = []
        mode_counts_per_batch = []

        for batch_idx in range(bsz):
            (
                compressed_batch_key,
                compressed_batch_value,
                batch_mode_counts,
                two_surrogate_chunks,
                selected_runs,
                surrogate_slots,
            ) = self._pack_batch(
                batch_idx=batch_idx,
                key_states=key_states,
                value_states=value_states,
                chunk_slices=chunk_slices,
                chunk_lengths=chunk_lengths,
                replace_mask=replace_mask,
                surrogate_lengths=surrogate_lengths[batch_idx],
                surrogate_key_bank=surrogate_key_bank,
                surrogate_value_bank=surrogate_value_bank,
                sink_len=sink_len,
                past_len=past_len,
                mode_name=self.spec.mode,
            )
            compressed_keys.append(compressed_batch_key)
            compressed_values.append(compressed_batch_value)
            selected_runs_per_batch.append(selected_runs)
            two_surrogate_chunks_per_batch.append(two_surrogate_chunks)
            surrogate_slots_per_batch.append(surrogate_slots)
            mode_counts_per_batch.append(batch_mode_counts)

        compressed_key_states = torch.cat(compressed_keys, dim=0)
        compressed_value_states = torch.cat(compressed_values, dim=0)
        timing_breakdown["packing"] += time.perf_counter() - stage_start

        self.last_stats = self._stats(
            full_tokens=q_len,
            compressed_tokens=compressed_key_states.shape[-2],
            recent_tokens=recent_len,
            selected_chunks=max(selected_runs_per_batch) if selected_runs_per_batch else 0,
            selected_runs=max(selected_runs_per_batch) if selected_runs_per_batch else 0,
            num_chunks=len(chunk_slices),
            chunk_size=adaptive_chunk_size,
            sink_tokens=sink_len,
            two_surrogate_chunks=max(two_surrogate_chunks_per_batch) if two_surrogate_chunks_per_batch else 0,
            surrogate_slots=max(surrogate_slots_per_batch) if surrogate_slots_per_batch else None,
            mode_counts=self._merge_mode_counts(mode_counts_per_batch),
            op_seconds=time.perf_counter() - update_start,
            configured_keep_ratio=configured_keep_ratio,
            timing_breakdown=timing_breakdown,
            allocator_stats=allocator_stats,
        )
        return compressed_key_states, compressed_value_states

    def _return_full(
        self,
        key_states,
        value_states,
        *,
        update_start: float,
        q_len: int,
        recent_len: int,
        sink_len: int = 0,
        num_chunks: int = 0,
        chunk_size: int = 0,
        configured_keep_ratio: float | None = None,
        timing_breakdown: Dict[str, float] | None = None,
    ):
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

    def _pack_batch(
        self,
        *,
        batch_idx: int,
        key_states,
        value_states,
        chunk_slices: Sequence[Tuple[int, int]],
        chunk_lengths,
        replace_mask,
        surrogate_lengths,
        surrogate_key_bank,
        surrogate_value_bank,
        sink_len: int,
        past_len: int,
        mode_name: str,
    ):
        recent_key = key_states[batch_idx : batch_idx + 1, :, past_len:, :]
        recent_value = value_states[batch_idx : batch_idx + 1, :, past_len:, :]
        selected_chunk_mask = replace_mask[batch_idx]
        output_chunk_lengths = torch.where(selected_chunk_mask, surrogate_lengths, chunk_lengths)
        selected_mask_list = [bool(v) for v in selected_chunk_mask.detach().cpu().tolist()]
        output_length_list = [int(v) for v in output_chunk_lengths.detach().cpu().tolist()]
        selected_chunk_indices = [idx for idx, selected in enumerate(selected_mask_list) if selected]

        key_pieces = []
        value_pieces = []
        if sink_len > 0:
            key_pieces.append(key_states[batch_idx : batch_idx + 1, :, :sink_len, :])
            value_pieces.append(value_states[batch_idx : batch_idx + 1, :, :sink_len, :])

        for chunk_idx, (start, end) in enumerate(chunk_slices):
            packed_len = output_length_list[chunk_idx]
            if selected_mask_list[chunk_idx]:
                if packed_len <= 0:
                    continue
                surrogate_key = surrogate_key_bank[batch_idx : batch_idx + 1, :, chunk_idx : chunk_idx + 1, :]
                surrogate_value = surrogate_value_bank[batch_idx : batch_idx + 1, :, chunk_idx : chunk_idx + 1, :]
                expanded_key = surrogate_key.expand(-1, -1, packed_len, -1)
                expanded_value = surrogate_value.expand(-1, -1, packed_len, -1)
                key_pieces.append(expanded_key)
                value_pieces.append(expanded_value)
                self._record_saved_surrogate(
                    batch_idx=batch_idx,
                    chunk_idx=chunk_idx,
                    surrogate_key=expanded_key,
                    surrogate_value=expanded_value,
                )
            else:
                key_pieces.append(key_states[batch_idx : batch_idx + 1, :, start:end, :])
                value_pieces.append(value_states[batch_idx : batch_idx + 1, :, start:end, :])

        if recent_key.shape[2] > 0:
            key_pieces.append(recent_key)
            value_pieces.append(recent_value)

        compressed_key = torch.cat(key_pieces, dim=2) if key_pieces else key_states.new_empty(
            (1, key_states.shape[1], 0, key_states.shape[-1])
        )
        compressed_value = torch.cat(value_pieces, dim=2) if value_pieces else value_states.new_empty(
            (1, value_states.shape[1], 0, value_states.shape[-1])
        )
        selected_lengths = [output_length_list[idx] for idx in selected_chunk_indices]
        two_surrogate_chunks = sum(max(0, int(length) - 1) for length in selected_lengths)
        surrogate_slots = sum(max(0, int(length)) for length in selected_lengths)
        selected_runs = len(selected_chunk_indices)
        batch_mode_counts = {}
        for selected, packed_len in zip(selected_mask_list, output_length_list):
            if not selected:
                continue
            selected_mode = "drop" if int(packed_len) <= 0 else mode_name
            batch_mode_counts[selected_mode] = batch_mode_counts.get(selected_mode, 0) + 1
        return (
            compressed_key,
            compressed_value,
            batch_mode_counts,
            two_surrogate_chunks,
            selected_runs,
            surrogate_slots,
        )

    def _configured_keep_ratio(self, q_len: int) -> float:
        keep_ratio = min(1.0, float(self.max_capacity_prompt) / max(float(q_len), 1.0))
        if self.layer_keep_ratio is not None:
            keep_ratio = min(1.0, max(1.0 / max(q_len, 1), float(self.layer_keep_ratio)))
        return keep_ratio

    def _past_token_scores(
        self,
        *,
        key_states,
        query_states,
        recent_len: int,
        past_len: int,
        head_dim: int,
        num_key_value_groups: int,
    ):
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
            attn_weights = torch.einsum("bngrd,bndt->bngrt", grouped_queries, key_states.transpose(2, 3)) / math.sqrt(head_dim)

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
        if attn_probs.dim() == 4:
            past_scores = attn_probs[:, :, -recent_len:, :past_len].sum(dim=-2)
        else:
            past_scores = attn_probs[..., :past_len].sum(dim=-2).reshape(
                query_states.shape[0],
                query_states.shape[1],
                past_len,
            )

        if self.pooling == "avgpool":
            pooled_scores = F.avg_pool1d(past_scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        elif self.pooling == "maxpool":
            pooled_scores = F.max_pool1d(past_scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        return pooled_scores.mean(dim=1)

    def _chunk_mean_scores(self, *, token_scores, chunk_slices: Sequence[Tuple[int, int]]):
        if not chunk_slices:
            return token_scores.new_empty((token_scores.shape[0], 0))
        if not self._uses_regular_spans(chunk_slices):
            return self._chunk_mean_scores_from_spans(token_scores=token_scores, chunk_slices=chunk_slices)

        base_start = int(chunk_slices[0][0])
        chunk_size = int(chunk_slices[0][1] - chunk_slices[0][0])
        if chunk_size <= 0:
            return token_scores.new_empty((token_scores.shape[0], 0))

        tail_len = int(chunk_slices[-1][1] - chunk_slices[-1][0])
        regular_chunks = len(chunk_slices) if tail_len == chunk_size else len(chunk_slices) - 1
        chunk_means = []

        if regular_chunks > 0:
            regular_tokens = regular_chunks * chunk_size
            regular = token_scores[:, base_start : base_start + regular_tokens].reshape(
                token_scores.shape[0],
                regular_chunks,
                chunk_size,
            )
            chunk_means.append(regular.mean(dim=-1))

        if tail_len != chunk_size:
            tail_start = base_start + regular_chunks * chunk_size
            tail = token_scores[:, tail_start : tail_start + tail_len]
            chunk_means.append(tail.mean(dim=-1, keepdim=True))

        return torch.cat(chunk_means, dim=-1)

    @staticmethod
    def _chunk_mean_scores_from_spans(*, token_scores, chunk_slices: Sequence[Tuple[int, int]]):
        means = []
        for start, end in chunk_slices:
            chunk = token_scores[:, int(start) : int(end)]
            means.append(chunk.mean(dim=-1))
        return torch.stack(means, dim=-1)

    @staticmethod
    def _uses_regular_spans(chunk_slices: Sequence[Tuple[int, int]]) -> bool:
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
        surrogate_slots: int | None = None,
        op_seconds: float = 0.0,
        configured_keep_ratio: float | None = None,
        timing_breakdown: Dict[str, float] | None = None,
        allocator_stats: dict[str, object] | None = None,
    ):
        if self.spec.null_fastpath:
            surrogate_slots = 0
        elif surrogate_slots is None:
            surrogate_slots = max(0, int(selected_runs) + int(two_surrogate_chunks))
        kept_tokens = max(0, int(compressed_tokens) - int(recent_tokens) - int(sink_tokens) - surrogate_slots)
        kept_chunks = max(0, int(num_chunks) - int(selected_runs))
        if configured_keep_ratio is None:
            configured_keep_ratio = self._configured_keep_ratio(full_tokens)
        stats = {
            "full_tokens": int(full_tokens),
            "compressed_tokens": int(compressed_tokens),
            "recent_tokens": int(recent_tokens),
            "selected_chunks": int(selected_chunks),
            "selected_runs": int(selected_runs),
            "surrogate_slots": int(surrogate_slots),
            "kept_tokens": int(kept_tokens),
            "kept_chunks": int(kept_chunks),
            "num_chunks": int(num_chunks),
            "chunk_size": int(chunk_size),
            "sink_tokens": int(sink_tokens),
            "two_surrogate_chunks": int(two_surrogate_chunks),
            "configured_keep_ratio": float(configured_keep_ratio),
            "avg_weight_entropy": None,
            "avg_weight_max": None,
            "avg_mapping_alpha": None,
            "region_mean_len": None,
            "region_max_len": None,
            "region_count": None,
            "mode_counts": mode_counts,
            "op_seconds": float(op_seconds),
        }
        if timing_breakdown:
            for name in ("score", "planning", "prototype", "packing"):
                stats[f"timing_{name}_seconds"] = float(timing_breakdown.get(name, 0.0) or 0.0)
        if allocator_stats:
            stats["allocator_stats"] = dict(allocator_stats)
            for key, value in allocator_stats.items():
                stats[key] = value
        return stats

    @staticmethod
    def _merge_mode_counts(mode_counts_per_batch):
        merged = {}
        for counts in mode_counts_per_batch:
            for mode_name, value in counts.items():
                merged[mode_name] = merged.get(mode_name, 0) + int(value)
        return merged

    def _chunk_slices(self, length: int, chunk_size: int) -> List[Tuple[int, int]]:
        cache_key = (int(length), int(chunk_size))
        cached = _CHUNK_SLICE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        chunk_slices = [(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]
        return _cache_put(_CHUNK_SLICE_CACHE, cache_key, chunk_slices)

    def _adaptive_chunk_size(self, *, compressible_len: int, budget_compressible: int) -> int:
        base_chunk_size = max(1, int(self.chunk_size))
        adaptive_chunk_size = max(base_chunk_size, math.ceil(compressible_len / max(1, budget_compressible)))
        return min(adaptive_chunk_size, compressible_len)

    def _protected_sink_tokens(self) -> int:
        if self.spec.protected_sink:
            return max(0, int(self.sink_tokens))
        return 0

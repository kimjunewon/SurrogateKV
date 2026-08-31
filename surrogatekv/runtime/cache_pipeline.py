from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    _CHUNK_SLICE_CACHE,
    _RECENT_MASK_CACHE,
    _SURKV_DIAGNOSTIC_STATS,
    _cache_put,
    _device_key,
    _env_flag,
    _rank01,
)


class CacheStateMixin:
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


class CachePackingMixin:
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
        surrogate_lengths,
        surrogate_key_bank,
        surrogate_value_bank,
        mode_name: str,
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

        if not self._save_layout_meta and not self._save_surrogates:
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
                else:
                    append_piece(
                        key_states[batch_idx : batch_idx + 1, :, start:end, :],
                        value_states[batch_idx : batch_idx + 1, :, start:end, :],
                    )
            two_surrogate_chunks = sum(max(0, int(length) - 1) for length in selected_lengths_list)
        else:
            batch_mode_counts = {}
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
        return compressed_key, compressed_value, batch_mode_counts, two_surrogate_chunks, selected_runs, batch_layout_meta

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
        if mode_counts and int(mode_counts.get("drop", 0) or 0) > 0:
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


class CacheScoringMixin:
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
        score_method = str(getattr(self, "score_method", "attention") or "attention").replace("-", "_").lower()
        if score_method in {"attention", "attn", "snap", "snapkv"}:
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

from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Tuple

import torch

from .common import (
    _KEY_WEIGHTED_SURROGATE_MODES,
    _LIGHT_VALUE_WEIGHT_MODES,
    _NORM_RESTORED_KEY_MODES,
    _PIVOT_KEY_MODES,
    _RMS_RESTORED_VALUE_MODES,
    _SCORE_WEIGHTED_SURROGATE_MODES,
    _VALUE_WEIGHTED_SURROGATE_MODES,
    _restore_mean_key_norm,
    _restore_rms_value_norm,
    _safe_key_norm_scale,
)
class MicroPrototypeBankMixin:
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


class PrototypeBankMixin(MicroPrototypeBankMixin):


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

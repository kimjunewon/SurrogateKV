from __future__ import annotations

from .tensor_utils import *  # noqa: F401,F403


class SurrogatePrototypeMixin:
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
            past_scores = attn_probs[:, :, -recent_len:, :past_len].sum(dim=-2)
        else:
            past_scores = attn_probs[..., :past_len].sum(dim=-2).reshape(
                query_states.shape[0],
                query_states.shape[1],
                past_len,
            )

        if self.pooling == "avgpool":
            pooled_scores = F.avg_pool1d(
                past_scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        elif self.pooling == "maxpool":
            pooled_scores = F.max_pool1d(
                past_scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        return pooled_scores.mean(dim=1)

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
        selected_only_mask=None,
        compact_output: bool = False,
    ):
        if surrogate_mode != "norm_rms_mean":
            raise ValueError(f"Unsupported SurrogateKV prototype mode: {surrogate_mode}")
        if not chunk_slices:
            bsz, heads, _, head_dim = key_states.shape
            empty_key = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_value = value_states.new_empty((bsz, heads, 0, head_dim))
            return empty_key, empty_value, None, None, None, None, None

        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return (
                *self._chunk_prototype_bank_fast(
                    key_states=key_states,
                    value_states=value_states,
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    surrogate_mode=surrogate_mode,
                ),
                None,
            )

        base_start, base_end = span
        span_len = int(base_end) - int(base_start)
        bsz, heads, _, head_dim = key_states.shape
        if span_len <= 0:
            empty_key = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_value = value_states.new_empty((bsz, heads, 0, head_dim))
            return empty_key, empty_value, None, None, None, None, None

        device = key_states.device
        full_region_count = len(chunk_slices)
        active_region_indices = None
        active_slice_indices = None
        if selected_only_mask is not None and full_region_count > 0:
            selected_any = selected_only_mask.detach().to(device=device, dtype=torch.bool)
            if selected_any.dim() == 2:
                selected_any = selected_any.any(dim=0)
            elif selected_any.dim() != 1:
                selected_any = None
            if selected_any is not None and selected_any.numel() == full_region_count:
                if not bool(selected_any.any().item()):
                    if compact_output:
                        empty_key = key_states.new_empty((bsz, heads, 0, head_dim))
                        empty_value = value_states.new_empty((bsz, heads, 0, head_dim))
                        return empty_key, empty_value, None, None, None, None, []
                    empty_key = key_states.new_zeros((bsz, heads, full_region_count, head_dim))
                    empty_value = value_states.new_zeros((bsz, heads, full_region_count, head_dim))
                    return empty_key, empty_value, None, None, None, None, None
                if not bool(selected_any.all().item()):
                    active_region_indices = torch.nonzero(selected_any, as_tuple=False).flatten()
                    active_slice_indices = [int(idx) for idx in active_region_indices.detach().cpu().tolist()]

        region_slices = [chunk_slices[idx] for idx in active_slice_indices] if active_slice_indices is not None else chunk_slices

        if active_slice_indices is not None and len(active_slice_indices) <= 16:
            token_work = sum(max(0, int(end) - int(start)) for start, end in region_slices)
            if 0 < token_work < span_len:
                key_parts = []
                value_parts = []
                for start, end in region_slices:
                    start_i = int(start)
                    end_i = int(end)
                    if end_i <= start_i:
                        key_parts.append(key_states.new_zeros((bsz, heads, head_dim), dtype=torch.float32))
                        value_parts.append(value_states.new_zeros((bsz, heads, head_dim), dtype=torch.float32))
                        continue
                    key_chunk = key_states[:, :, start_i:end_i, :].to(dtype=torch.float32)
                    value_chunk = value_states[:, :, start_i:end_i, :].to(dtype=torch.float32)
                    key_parts.append(_restore_mean_key_norm(key_chunk.mean(dim=2), key_chunk, token_dim=2))
                    value_parts.append(_restore_rms_value_norm(value_chunk.mean(dim=2), value_chunk, token_dim=2))
                proto_key_bank = torch.stack(key_parts, dim=2).to(dtype=key_states.dtype)
                proto_value_bank = torch.stack(value_parts, dim=2).to(dtype=value_states.dtype)
                if compact_output:
                    return proto_key_bank, proto_value_bank, None, None, None, None, active_slice_indices
                proto_key_bank, proto_value_bank = self._scatter_active_prototype_bank(
                    proto_key_bank=proto_key_bank,
                    proto_value_bank=proto_value_bank,
                    active_region_indices=active_region_indices,
                    full_region_count=full_region_count,
                )
                return proto_key_bank, proto_value_bank, None, None, None, None, None

        relative_boundaries = [0]
        for start, end in region_slices:
            relative_boundaries.append(int(start) - int(base_start))
            relative_boundaries.append(int(end) - int(base_start))

        def aligned_or_tail(offset: int, unit: int) -> bool:
            return int(offset) == int(span_len) or int(offset) % int(unit) == 0

        if all(aligned_or_tail(offset, 8) for offset in relative_boundaries):
            micro_len = 8
        elif all(aligned_or_tail(offset, 4) for offset in relative_boundaries):
            micro_len = 4
        else:
            return (
                *self._chunk_prototype_bank_fast(
                    key_states=key_states,
                    value_states=value_states,
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    surrogate_mode=surrogate_mode,
                ),
                None,
            )

        boundary_offsets = list(range(0, span_len, micro_len))
        boundary_offsets.append(span_len)
        boundary_to_micro = {int(offset): idx for idx, offset in enumerate(boundary_offsets)}
        try:
            region_starts = [boundary_to_micro[int(start) - int(base_start)] for start, _ in region_slices]
            region_ends = [boundary_to_micro[int(end) - int(base_start)] for _, end in region_slices]
        except KeyError:
            return (
                *self._chunk_prototype_bank_fast(
                    key_states=key_states,
                    value_states=value_states,
                    token_scores=token_scores,
                    chunk_slices=chunk_slices,
                    surrogate_mode=surrogate_mode,
                ),
                None,
            )

        regular_micro = span_len // micro_len
        tail_len = span_len - regular_micro * micro_len
        micro_keys = []
        micro_values = []
        micro_lengths = []

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
            micro_lengths.append(torch.full((regular_micro,), micro_len, device=device, dtype=torch.float32))

        if tail_len > 0:
            tail_start = base_start + regular_micro * micro_len
            micro_keys.append(key_states[:, :, tail_start:base_end, :].mean(dim=2, keepdim=True))
            micro_values.append(value_states[:, :, tail_start:base_end, :].mean(dim=2, keepdim=True))
            micro_lengths.append(torch.full((1,), tail_len, device=device, dtype=torch.float32))

        micro_key_bank = torch.cat(micro_keys, dim=2)
        micro_value_bank = torch.cat(micro_values, dim=2)
        micro_len_bank = torch.cat(micro_lengths, dim=0)
        length_weights = micro_len_bank.view(1, -1).expand(bsz, -1)

        starts = torch.tensor(region_starts, device=device, dtype=torch.long)
        ends = torch.tensor(region_ends, device=device, dtype=torch.long)
        length_weights_f = length_weights.to(dtype=torch.float32)
        weighted_keys = micro_key_bank.to(dtype=torch.float32) * length_weights_f.view(bsz, 1, -1, 1)
        weighted_values = micro_value_bank.to(dtype=torch.float32) * length_weights_f.view(bsz, 1, -1, 1)
        zero_key = weighted_keys.new_zeros((bsz, heads, 1, head_dim))
        zero_value = weighted_values.new_zeros((bsz, heads, 1, head_dim))
        key_prefix = torch.cat([zero_key, weighted_keys.cumsum(dim=2)], dim=2)
        value_prefix = torch.cat([zero_value, weighted_values.cumsum(dim=2)], dim=2)
        length_prefix = torch.cat([length_weights_f.new_zeros((bsz, 1)), length_weights_f.cumsum(dim=1)], dim=1)
        denom = (length_prefix.index_select(1, ends) - length_prefix.index_select(1, starts)).clamp_min(1e-6)

        proto_key_bank = (key_prefix.index_select(2, ends) - key_prefix.index_select(2, starts)) / denom.view(bsz, 1, -1, 1)
        proto_value_bank = (value_prefix.index_select(2, ends) - value_prefix.index_select(2, starts)) / denom.view(bsz, 1, -1, 1)

        micro_key_norm = micro_key_bank.to(dtype=torch.float32).norm(dim=-1)
        weighted_key_norm = micro_key_norm * length_weights_f.view(bsz, 1, -1)
        zero_norm = weighted_key_norm.new_zeros((bsz, heads, 1))
        key_norm_prefix = torch.cat([zero_norm, weighted_key_norm.cumsum(dim=2)], dim=2)
        target_key_norm = (key_norm_prefix.index_select(2, ends) - key_norm_prefix.index_select(2, starts)) / denom.view(bsz, 1, -1)
        current_key_norm = proto_key_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
        proto_key_bank = proto_key_bank * (target_key_norm / current_key_norm).view(bsz, heads, -1, 1)

        micro_value_norm_sq = micro_value_bank.to(dtype=torch.float32).square().sum(dim=-1)
        weighted_value_norm_sq = micro_value_norm_sq * length_weights_f.view(bsz, 1, -1)
        zero_value_norm = weighted_value_norm_sq.new_zeros((bsz, heads, 1))
        value_norm_prefix = torch.cat([zero_value_norm, weighted_value_norm_sq.cumsum(dim=2)], dim=2)
        target_value_norm = ((value_norm_prefix.index_select(2, ends) - value_norm_prefix.index_select(2, starts)) / denom.view(bsz, 1, -1)).clamp_min(1e-12).sqrt()
        current_value_norm = proto_value_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
        proto_value_bank = proto_value_bank * (target_value_norm / current_value_norm).view(bsz, heads, -1, 1)

        proto_key_bank = proto_key_bank.to(dtype=key_states.dtype)
        proto_value_bank = proto_value_bank.to(dtype=value_states.dtype)
        if compact_output:
            return proto_key_bank, proto_value_bank, None, None, None, None, active_slice_indices

        proto_key_bank, proto_value_bank = self._scatter_active_prototype_bank(
            proto_key_bank=proto_key_bank,
            proto_value_bank=proto_value_bank,
            active_region_indices=active_region_indices,
            full_region_count=full_region_count,
        )
        return proto_key_bank, proto_value_bank, None, None, None, None, None

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

    def _chunk_prototype_bank_fast(
        self,
        *,
        key_states,
        value_states,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        surrogate_mode: str,
    ):
        if surrogate_mode != "norm_rms_mean":
            raise ValueError(f"Unsupported SurrogateKV prototype mode: {surrogate_mode}")
        if not chunk_slices:
            bsz, heads, _, head_dim = key_states.shape
            empty_key = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_value = value_states.new_empty((bsz, heads, 0, head_dim))
            return empty_key, empty_value, None, None, None, None
        if not self._is_regular_chunk_layout(chunk_slices):
            span = self._contiguous_chunk_span(chunk_slices)
            if span is not None:
                max_len = max(max(1, int(end) - int(start)) for start, end in chunk_slices)
                if 0 < max_len <= 128:
                    return self._chunk_prototype_bank_irregular_padded(
                        key_states=key_states,
                        value_states=value_states,
                        chunk_slices=chunk_slices,
                    )
            return self._chunk_prototype_bank_generic(
                key_states=key_states,
                value_states=value_states,
                chunk_slices=chunk_slices,
            )

        base_start = chunk_slices[0][0]
        chunk_size = chunk_slices[0][1] - chunk_slices[0][0]
        tail_len = chunk_slices[-1][1] - chunk_slices[-1][0]
        regular_chunks = len(chunk_slices) if tail_len == chunk_size else len(chunk_slices) - 1
        proto_keys = []
        proto_values = []

        def build_chunk_proto(chunk_keys, chunk_values):
            key_proto = _restore_mean_key_norm(chunk_keys.mean(dim=3), chunk_keys, token_dim=3)
            value_proto = _restore_rms_value_norm(chunk_values.mean(dim=3), chunk_values, token_dim=3)
            return key_proto, value_proto

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
            key_proto, value_proto = build_chunk_proto(regular_keys, regular_values)
            proto_keys.append(key_proto)
            proto_values.append(value_proto)

        if tail_len != chunk_size:
            tail_start = base_start + regular_chunks * chunk_size
            tail_keys = key_states[:, :, tail_start : tail_start + tail_len, :].unsqueeze(2)
            tail_values = value_states[:, :, tail_start : tail_start + tail_len, :].unsqueeze(2)
            key_proto, value_proto = build_chunk_proto(tail_keys, tail_values)
            proto_keys.append(key_proto)
            proto_values.append(value_proto)

        return torch.cat(proto_keys, dim=2), torch.cat(proto_values, dim=2), None, None, None, None

    def _chunk_prototype_bank_irregular_padded(
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
                chunk_slices=chunk_slices,
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
            empty_key = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_value = value_states.new_empty((bsz, heads, 0, head_dim))
            return empty_key, empty_value, None, None, None, None

        offsets = torch.arange(max_len, device=device, dtype=torch.long)
        rel_indices = starts.view(-1, 1) + offsets.view(1, -1)
        valid = offsets.view(1, -1) < lengths_long.view(-1, 1)
        abs_indices = (rel_indices + int(base_start)).clamp(max=int(base_end) - 1).reshape(-1)

        bsz, heads, _, head_dim = key_states.shape
        num_chunks = len(chunk_slices)
        key_gather = key_states.index_select(2, abs_indices).view(bsz, heads, num_chunks, max_len, head_dim)
        value_gather = value_states.index_select(2, abs_indices).view(bsz, heads, num_chunks, max_len, head_dim)
        valid_float = valid.to(device=device, dtype=torch.float32)
        mask = valid.to(dtype=key_states.dtype).view(1, 1, num_chunks, max_len, 1)
        denom = lengths_long.to(dtype=key_states.dtype).view(1, 1, num_chunks, 1).clamp_min(1)
        proto_key_bank = (key_gather * mask).sum(dim=3) / denom
        proto_value_bank = (value_gather * mask).sum(dim=3) / denom

        target_key_norm = (
            key_gather.to(dtype=torch.float32).norm(dim=-1) * valid_float.view(1, 1, num_chunks, max_len)
        ).sum(dim=3) / lengths_long.to(device=device, dtype=torch.float32).view(1, 1, num_chunks).clamp_min(1)
        current_key_norm = proto_key_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
        proto_key_bank = proto_key_bank * (target_key_norm / current_key_norm).view(bsz, heads, num_chunks, 1)

        value_norm_sq = value_gather.to(dtype=torch.float32).square().sum(dim=-1)
        target_value_norm = (
            (value_norm_sq * valid_float.view(1, 1, num_chunks, max_len)).sum(dim=3)
            / lengths_long.to(device=device, dtype=torch.float32).view(1, 1, num_chunks).clamp_min(1)
        ).clamp_min(1e-12).sqrt()
        current_value_norm = proto_value_bank.to(dtype=torch.float32).norm(dim=-1).clamp_min(1e-6)
        proto_value_bank = proto_value_bank * (target_value_norm / current_value_norm).view(bsz, heads, num_chunks, 1)
        return proto_key_bank.to(dtype=key_states.dtype), proto_value_bank.to(dtype=value_states.dtype), None, None, None, None

    def _chunk_prototype_bank_generic(
        self,
        *,
        key_states,
        value_states,
        chunk_slices: Sequence[Tuple[int, int]],
    ):
        if not chunk_slices:
            bsz, heads, _, head_dim = key_states.shape
            empty_key = key_states.new_empty((bsz, heads, 0, head_dim))
            empty_value = value_states.new_empty((bsz, heads, 0, head_dim))
            return empty_key, empty_value, None, None, None, None

        proto_keys = []
        proto_values = []
        for start, end in chunk_slices:
            start_i = int(start)
            end_i = int(end)
            chunk_keys = key_states[:, :, start_i:end_i, :].unsqueeze(2)
            chunk_values = value_states[:, :, start_i:end_i, :].unsqueeze(2)
            proto_keys.append(_restore_mean_key_norm(chunk_keys.mean(dim=3), chunk_keys, token_dim=3))
            proto_values.append(_restore_rms_value_norm(chunk_values.mean(dim=3), chunk_values, token_dim=3))
        return torch.cat(proto_keys, dim=2), torch.cat(proto_values, dim=2), None, None, None, None

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

    def _select_low_importance_chunks(
        self,
        *,
        chunk_scores,
        chunk_max_scores,
        chunk_lengths,
        surrogate_lengths,
        tokens_to_save: int,
        selection_scores=None,
    ):
        del chunk_max_scores
        if selection_scores is None:
            selection_scores = chunk_scores
        return select_chunks_fast(
            chunk_scores=selection_scores,
            chunk_lengths=chunk_lengths,
            surrogate_lengths=surrogate_lengths,
            tokens_to_save=tokens_to_save,
            exclusion_radius=0,
        )

    def _protected_sink_tokens(self) -> int:
        policy = getattr(self, "sink_policy", "static")
        if policy == "off":
            return 0
        if policy == "on":
            return max(0, int(self.sink_tokens))
        if policy == "dynamic":
            return 0
        if self.spec.protected_sink:
            return max(0, int(self.sink_tokens))
        return 0

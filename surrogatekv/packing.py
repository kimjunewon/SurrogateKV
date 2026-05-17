from __future__ import annotations

from .tensor_utils import *  # noqa: F401,F403


class SurrogatePackingMixin:
    def _zero_surrogate_pair(self, chunk_key, chunk_value):
        key_cache_key = (
            chunk_key.device,
            chunk_key.dtype,
            chunk_key.shape[1],
            chunk_key.shape[-1],
        )
        value_cache_key = (
            chunk_value.device,
            chunk_value.dtype,
            chunk_value.shape[1],
            chunk_value.shape[-1],
        )

        surrogate_key = self._zero_pair_cache.get(key_cache_key)
        if surrogate_key is None:
            surrogate_key = torch.zeros(
                (1, chunk_key.shape[1], 1, chunk_key.shape[-1]),
                device=chunk_key.device,
                dtype=chunk_key.dtype,
            )
            self._zero_pair_cache[key_cache_key] = surrogate_key

        surrogate_value = self._zero_pair_cache.get(value_cache_key)
        if surrogate_value is None:
            surrogate_value = torch.zeros(
                (1, chunk_value.shape[1], 1, chunk_value.shape[-1]),
                device=chunk_value.device,
                dtype=chunk_value.dtype,
            )
            self._zero_pair_cache[value_cache_key] = surrogate_value

        return surrogate_key, surrogate_value

    def _compress_drop_batch(
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
    ):
        recent_key = key_states[batch_idx : batch_idx + 1, :, past_len:, :]
        recent_value = value_states[batch_idx : batch_idx + 1, :, past_len:, :]
        selected_chunk_mask = replace_mask[batch_idx]
        if surrogate_lengths.dim() > 1:
            surrogate_lengths = surrogate_lengths[batch_idx]
        output_chunk_lengths = torch.where(selected_chunk_mask, surrogate_lengths, chunk_lengths)
        selected_mask_list = [bool(v) for v in selected_chunk_mask.detach().cpu().tolist()]
        output_length_list = [int(v) for v in output_chunk_lengths.detach().cpu().tolist()]
        selected_chunk_indices = [idx for idx, selected in enumerate(selected_mask_list) if selected]
        total_tokens = int(sink_len) + int(sum(output_length_list)) + int(recent_key.shape[2])
        key_pieces = []
        value_pieces = []

        if sink_len > 0:
            key_pieces.append(key_states[batch_idx : batch_idx + 1, :, :sink_len, :])
            value_pieces.append(value_states[batch_idx : batch_idx + 1, :, :sink_len, :])

        selected_runs = len(selected_chunk_indices)
        selected_lengths = [output_length_list[idx] for idx in selected_chunk_indices]
        zero_key = zero_value = None
        for chunk_idx, (start, end) in enumerate(chunk_slices):
            packed_len = output_length_list[chunk_idx]
            if selected_mask_list[chunk_idx]:
                if packed_len <= 0:
                    continue
                if zero_key is None or zero_value is None:
                    zero_key, zero_value = self._zero_surrogate_pair(
                        key_states[batch_idx : batch_idx + 1, :, :1, :],
                        value_states[batch_idx : batch_idx + 1, :, :1, :],
                    )
                expanded_key = zero_key.expand(-1, -1, packed_len, -1)
                expanded_value = zero_value.expand(-1, -1, packed_len, -1)
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
        two_surrogate_chunks = sum(max(0, int(length) - 1) for length in selected_lengths)
        mode_counts = {"drop": selected_runs} if selected_runs > 0 else {}
        layout_meta = self._build_layout_meta(
            full_tokens=int(key_states.shape[2]),
            compressed_tokens=total_tokens,
            sink_len=sink_len,
            recent_len=int(recent_key.shape[2]),
            chunk_lengths=chunk_lengths,
            selected_chunk_mask=selected_chunk_mask,
            output_chunk_lengths=output_chunk_lengths,
            chunk_mode_names=["drop" if selected else None for selected in selected_mask_list],
        )
        return compressed_key, compressed_value, mode_counts, two_surrogate_chunks, selected_runs, layout_meta

    def _compress_surrogate_batch(
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
        surrogate_bank_indices=None,
    ):
        recent_key = key_states[batch_idx : batch_idx + 1, :, past_len:, :]
        recent_value = value_states[batch_idx : batch_idx + 1, :, past_len:, :]
        fast_pack_plan = getattr(self, "_last_fast_pack_plan", None)
        use_fast_pack_plan = (
            batch_idx == 0
            and isinstance(fast_pack_plan, dict)
            and tuple(chunk_slices) == fast_pack_plan.get("chunk_slices")
        )

        if use_fast_pack_plan:
            selected_mask_list = fast_pack_plan["selected_mask_list"]
            output_length_list = fast_pack_plan["output_length_list"]
            selected_chunk_indices = fast_pack_plan["selected_chunk_indices_list"]
            selected_chunk_mask = torch.as_tensor(selected_mask_list, device=key_states.device, dtype=torch.bool)
            output_chunk_lengths = torch.as_tensor(output_length_list, device=key_states.device, dtype=chunk_lengths.dtype)
        else:
            selected_chunk_mask = replace_mask[batch_idx]
            output_chunk_lengths = torch.where(selected_chunk_mask, surrogate_lengths, chunk_lengths)
            selected_mask_list = [bool(v) for v in selected_chunk_mask.detach().cpu().tolist()]
            output_length_list = [int(v) for v in output_chunk_lengths.detach().cpu().tolist()]
            selected_chunk_indices = [idx for idx, selected in enumerate(selected_mask_list) if selected]

        total_tokens = int(sink_len) + int(sum(output_length_list)) + int(recent_key.shape[2])
        bank_index_map = None
        if surrogate_bank_indices is not None:
            if hasattr(surrogate_bank_indices, "detach"):
                bank_indices_list = [int(v) for v in surrogate_bank_indices.detach().cpu().tolist()]
            else:
                bank_indices_list = [int(v) for v in surrogate_bank_indices]
            bank_index_map = {chunk_idx: bank_idx for bank_idx, chunk_idx in enumerate(bank_indices_list)}

        selected_lengths = [output_length_list[idx] for idx in selected_chunk_indices]
        surrogate_runs = sum(1 for length in selected_lengths if int(length) > 0)
        drop_runs = sum(1 for length in selected_lengths if int(length) <= 0)
        mode_counts = {}
        if surrogate_runs > 0:
            mode_counts["surrogate"] = surrogate_runs
        if drop_runs > 0:
            mode_counts["drop"] = drop_runs
        two_surrogate_chunks = sum(max(0, int(length) - 1) for length in selected_lengths)

        if not self._save_layout_meta and not self._save_surrogates:
            if use_fast_pack_plan:
                raw_spans = fast_pack_plan["raw_spans"]
                surrogate_chunk_indices = fast_pack_plan["surrogate_chunk_indices_list"]
                surrogate_lengths_list = fast_pack_plan["surrogate_lengths_list"]
            else:
                raw_spans = [
                    (int(start), int(end))
                    for chunk_idx, (start, end) in enumerate(chunk_slices)
                    if not selected_mask_list[chunk_idx]
                ]
                surrogate_chunk_indices = [idx for idx in selected_chunk_indices if int(output_length_list[idx]) > 0]
                surrogate_lengths_list = [output_length_list[idx] for idx in surrogate_chunk_indices]

            key_pieces = []
            value_pieces = []
            if sink_len > 0:
                key_pieces.append(key_states[batch_idx : batch_idx + 1, :, :sink_len, :])
                value_pieces.append(value_states[batch_idx : batch_idx + 1, :, :sink_len, :])

            raw_index_tensor = self._span_index_tensor(raw_spans, device=key_states.device)
            if raw_index_tensor is not None and raw_index_tensor.numel() > 0:
                key_pieces.append(key_states[batch_idx : batch_idx + 1].index_select(2, raw_index_tensor))
                value_pieces.append(value_states[batch_idx : batch_idx + 1].index_select(2, raw_index_tensor))

            if surrogate_chunk_indices:
                if bank_index_map is not None:
                    surrogate_bank_indices_list = [bank_index_map[int(idx)] for idx in surrogate_chunk_indices]
                else:
                    surrogate_bank_indices_list = surrogate_chunk_indices
                bank_indices = torch.tensor(surrogate_bank_indices_list, device=key_states.device, dtype=torch.long)
                if any(length != 1 for length in surrogate_lengths_list):
                    repeat_counts = torch.tensor(surrogate_lengths_list, device=key_states.device, dtype=torch.long)
                    bank_indices = bank_indices.repeat_interleave(repeat_counts)
                key_pieces.append(surrogate_key_bank[batch_idx : batch_idx + 1].index_select(2, bank_indices))
                value_pieces.append(surrogate_value_bank[batch_idx : batch_idx + 1].index_select(2, bank_indices))

            if recent_key.shape[2] > 0:
                key_pieces.append(recent_key)
                value_pieces.append(recent_value)
            compressed_key = torch.cat(key_pieces, dim=2) if key_pieces else key_states.new_empty(
                (1, key_states.shape[1], 0, key_states.shape[-1])
            )
            compressed_value = torch.cat(value_pieces, dim=2) if value_pieces else value_states.new_empty(
                (1, value_states.shape[1], 0, value_states.shape[-1])
            )
            return compressed_key, compressed_value, mode_counts, two_surrogate_chunks, len(selected_chunk_indices), None

        compressed_key = key_states.new_empty((1, key_states.shape[1], max(0, total_tokens), key_states.shape[-1]))
        compressed_value = value_states.new_empty((1, value_states.shape[1], max(0, total_tokens), value_states.shape[-1]))
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

        for chunk_idx, (start, end) in enumerate(chunk_slices):
            packed_len = output_length_list[chunk_idx]
            if selected_mask_list[chunk_idx]:
                if packed_len <= 0:
                    continue
                bank_chunk_idx = bank_index_map[int(chunk_idx)] if bank_index_map is not None else chunk_idx
                surrogate_key = surrogate_key_bank[batch_idx : batch_idx + 1, :, bank_chunk_idx : bank_chunk_idx + 1, :]
                surrogate_value = surrogate_value_bank[batch_idx : batch_idx + 1, :, bank_chunk_idx : bank_chunk_idx + 1, :]
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

        if recent_key.shape[2] > 0:
            append_piece(recent_key, recent_value)
        if cursor != int(total_tokens):
            compressed_key = compressed_key[:, :, :cursor, :]
            compressed_value = compressed_value[:, :, :cursor, :]
        layout_meta = self._build_layout_meta(
            full_tokens=int(key_states.shape[2]),
            compressed_tokens=total_tokens,
            sink_len=sink_len,
            recent_len=int(recent_key.shape[2]),
            chunk_lengths=chunk_lengths,
            selected_chunk_mask=selected_chunk_mask,
            output_chunk_lengths=output_chunk_lengths,
            chunk_mode_names=["surrogate" if selected and output_length_list[idx] > 0 else "drop" if selected else None for idx, selected in enumerate(selected_mask_list)],
        )
        return compressed_key, compressed_value, mode_counts, two_surrogate_chunks, len(selected_chunk_indices), layout_meta

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
        configured_keep_ratio: float | None = None,
        dynamic_region_mean_len: float | None = None,
        dynamic_region_max_len: int | None = None,
        dynamic_region_count: int | None = None,
        timing_breakdown: Dict[str, float] | None = None,
    ):
        if self.spec.kind == "drop":
            surrogate_slots = 0
        elif mode_counts and int(mode_counts.get("drop", 0) or 0) > 0:
            surrogate_slots = sum(int(count) for mode_name, count in mode_counts.items() if mode_name != "drop")
            surrogate_slots += int(two_surrogate_chunks)
        else:
            surrogate_slots = max(0, int(selected_runs) + int(two_surrogate_chunks))
        kept_tokens = max(0, int(compressed_tokens) - int(recent_tokens) - int(sink_tokens) - int(surrogate_slots))
        kept_chunks = max(0, int(num_chunks) - int(selected_runs))
        if configured_keep_ratio is None:
            configured_keep_ratio = self.layer_keep_ratio
        if configured_keep_ratio is None:
            configured_keep_ratio = min(1.0, float(self.max_capacity_prompt) / max(float(full_tokens), 1.0))
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
            "dynamic_region_mean_len": dynamic_region_mean_len,
            "dynamic_region_max_len": dynamic_region_max_len,
            "dynamic_region_count": dynamic_region_count,
            "mode_counts": mode_counts,
            "op_seconds": float(op_seconds),
        }
        if timing_breakdown:
            for name in ("score", "planning", "prototype", "packing"):
                stats[f"timing_{name}_seconds"] = float(timing_breakdown.get(name, 0.0) or 0.0)
        sink_allocator_stats = getattr(self, "_last_sink_allocator_stats", None)
        if sink_allocator_stats:
            stats.update(sink_allocator_stats)
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

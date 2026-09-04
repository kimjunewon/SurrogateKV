# Portions adapted and modified from AdaKV (MIT) and SnapKV (Apache-2.0).
# See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .common import (
    _NORM_RESTORED_KEY_MODES,
    _RMS_RESTORED_VALUE_MODES,
    _SURKV_HEADWISE_ADA_OVERLAY,
    _rank01,
    _repeat_kv_heads,
    _safe_key_norm_scale,
)


class HeadwiseAdaOverlayMixin:
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
        raw_only_control = str(os.environ.get("SURKV_HEADWISE_RAW_ONLY", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if (
            int(groups) != 1
            or not isinstance(selected_support, torch.Tensor)
            or not isinstance(precomputed_head_scores, torch.Tensor)
            or precomputed_head_scores.ndim != 3
            or precomputed_head_scores.shape[0] != 1
            or int(precomputed_head_scores.shape[2]) < int(past_len)
        ):
            if raw_only_control:
                support_shape = tuple(selected_support.shape) if isinstance(selected_support, torch.Tensor) else None
                score_shape = (
                    tuple(precomputed_head_scores.shape)
                    if isinstance(precomputed_head_scores, torch.Tensor)
                    else None
                )
                print(
                    "[SurKV][raw-only] Ada overlay unavailable: "
                    f"groups={groups} support_shape={support_shape} score_shape={score_shape} past_len={past_len}",
                    flush=True,
                )
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
        if raw_only_control:
            active_exchange_heads_t = torch.zeros_like(active_exchange_heads_t)
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
                "surrogate_kv_headwise_raw_only_control": int(raw_only_control),
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
                "surrogate_kv_headwise_budget_gap_mean": float(
                    budget_gap_t.to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "surrogate_kv_headwise_budget_overflow_max": int(head_budget_overflow.max().detach().cpu().item()),
                "surrogate_kv_headwise_budget_preserved": int(int(head_budget_overflow.max().detach().cpu().item()) == 0),
                "surrogate_kv_headwise_head_len_min": int(min(head_lens) if head_lens else 0),
                "surrogate_kv_headwise_head_len_max": int(max(head_lens) if head_lens else 0),
                "surrogate_kv_headwise_head_len_mean": float(sum(head_lens) / max(1, len(head_lens))),
                "surrogate_kv_headwise_precomputed_scores": 1,
                "surrogate_kv_timing_update_score_seconds": float(score_seconds),
                "surrogate_kv_headwise_child_mean_ks_run_raw_tokens": float(
                    raw_tokens_t.to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "surrogate_kv_headwise_child_mean_ks_run_raw_regions": float(
                    sum(raw_regions_per_head) / max(1, len(raw_regions_per_head))
                ),
                "surrogate_kv_headwise_child_mean_ks_run_surrogate_regions": float(
                    surrogate_regions_t.to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "surrogate_kv_headwise_child_mean_ks_run_surrogate_tokens": float(
                    surrogate_tokens_t.to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "surrogate_kv_headwise_child_mean_ks_run_drop_tokens": float(
                    drop_tokens_t.to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "surrogate_kv_headwise_child_mean_ks_run_used_entries": float(
                    used_entries_t.to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "surrogate_kv_headwise_child_mean_ks_run_budget_gap": float(
                    budget_gap_t.to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "surrogate_kv_headwise_child_mean_surrogate_kv_selected_surrogates": float(
                    surrogate_regions_t.to(dtype=torch.float32).mean().detach().cpu().item()
                ),
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


class HeadwisePrototypeBankMixin:
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


class HeadwiseRuntimeMixin(HeadwisePrototypeBankMixin, HeadwiseAdaOverlayMixin):


    def update_kv_headwise(self, key_states, query_states, value_states, attention_mask=None, num_key_value_groups=1):
        update_start = time.perf_counter()
        del attention_mask
        if any(states.ndim != 4 for states in (key_states, query_states, value_states)):
            raise ValueError("SurKV headwise inputs must use [batch, heads, sequence, head_dim] layout.")
        if key_states.shape != value_states.shape:
            raise ValueError("SurKV headwise key and value states must have matching shapes.")
        if key_states.shape[0] != query_states.shape[0] or key_states.shape[-1] != query_states.shape[-1]:
            raise ValueError("SurKV headwise key/query batch and head dimensions must match.")
        if key_states.shape[-2] != query_states.shape[-2]:
            raise ValueError("SurKV headwise key/query sequence lengths must match.")

        bsz, query_heads, q_len, head_dim = query_states.shape
        key_heads = int(key_states.shape[1])
        if int(bsz) != 1:
            raise ValueError("SurKV headwise Ada cache path currently supports batch size 1.")
        if int(query_heads) % max(1, int(key_heads)) != 0:
            raise ValueError(
                f"query heads ({query_heads}) must be divisible by key/value heads ({key_heads}) "
                "for headwise Ada packing."
            )

        expected_groups = int(query_heads) // max(1, int(key_heads))
        groups = max(1, int(num_key_value_groups or expected_groups))
        if groups != expected_groups:
            raise ValueError(
                f"num_key_value_groups must be {expected_groups} for {query_heads} query heads "
                f"and {key_heads} key/value heads; received {groups}."
            )
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

        sink_len = 0
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
                    allocated = plan_cluster._allocate_surrogate_regions(
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

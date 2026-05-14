from __future__ import annotations

from .tensor_utils import *  # noqa: F401,F403


class SurrogateAllocationMixin:
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

    def _allocate_surrogate_frontier(
        self,
        *,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        chunk_lengths,
        target_compressed_tokens: int,
        sink_len: int,
        recent_len: int,
        merge_first: bool = False,
        current_frontier_accept: bool = False,
        admission_shadow_price: bool = False,
        value_frontier: bool = False,
        frontier_region_price: bool = False,
        completion_order: bool = False,
        bounded_market: bool = False,
        reserve_coherent_price: bool = False,
        budget_complete_peel: bool = False,
    ):
        """Allocate raw/surrogate/drop regions with a raw-first frontier market.

        This allocator deliberately does not reuse the KSRMerge/Dom pricing
        branches.  It starts from a fixed-atom raw frontier, proposes surrogate
        candidates only from frontier residual runs, and admits a surrogate only
        when its current after-state beats the raw atoms it displaces.
        """
        del chunk_lengths
        if not chunk_slices or token_scores.shape[0] != 1:
            return None

        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return None
        base_start, base_end = span
        if int(base_end) <= int(base_start):
            return None

        micro_len = max(1, int(self.spec.dynamic_anchor_width or (int(self.chunk_size) // 4)))
        atom_start_arr = np.arange(int(base_start), int(base_end), int(micro_len), dtype=np.int64)
        if atom_start_arr.size <= 0:
            return None
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

        atom_mean = torch.cat(mean_parts, dim=0).view(1, -1)
        atom_peak = torch.cat(peak_parts, dim=0).view(1, -1)
        mean_rank_t = _rank01(atom_mean)[0]
        peak_rank_t = _rank01(atom_peak)[0]
        atom_risk_t = torch.maximum(mean_rank_t, peak_rank_t)
        mean_risk_arr = mean_rank_t.detach().cpu().numpy().astype(np.float64) + 1e-6
        atom_risk_arr = atom_risk_t.detach().cpu().numpy().astype(np.float64) + 1e-6

        device = token_scores.device
        num_atoms = int(atom_start_arr.size)
        if num_atoms <= 0:
            return None

        budget_entries = int(target_compressed_tokens) - int(sink_len) - int(recent_len)
        budget_entries = max(1, int(budget_entries))
        full_cost = int(atom_len_int_arr.sum())

        tail_floor = 1.0 / float(max(2, num_atoms + 1))

        def tail_scores(values: Sequence[float]) -> np.ndarray:
            ranks = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
            return -np.log(np.maximum(float(tail_floor), 1.0 - ranks))

        mean_signal_arr = tail_scores(mean_risk_arr)
        atom_signal_arr = tail_scores(atom_risk_arr)
        atom_len_arr = np.maximum(1.0, atom_len_int_arr.astype(np.float64))
        raw_value_arr = atom_signal_arr * atom_signal_arr * atom_len_arr
        raw_density_arr = raw_value_arr / np.maximum(atom_len_arr, 1.0)
        prefix_len = np.concatenate(([0], np.cumsum(atom_len_int_arr))).astype(np.int64)

        atom_indices_arr = np.arange(num_atoms, dtype=np.int64)
        if bool(value_frontier):
            raw_keep_order = np.lexsort((atom_indices_arr, -raw_density_arr, -raw_value_arr))
            raw_drop_order = np.lexsort((atom_indices_arr, raw_density_arr, raw_value_arr))
        else:
            raw_keep_order = np.lexsort((atom_indices_arr, -atom_risk_arr))
            raw_drop_order = np.lexsort((atom_indices_arr, atom_risk_arr))
        raw_keep_order_list = [int(v) for v in raw_keep_order.tolist()]
        raw_drop_order_list = [int(v) for v in raw_drop_order.tolist()]

        def action_runs(actions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            actions_arr = np.asarray(actions, dtype=np.int8)
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

        def collect_stats(actions: Sequence[int]) -> Tuple[Dict[str, float], List[int]]:
            _starts, _ends, run_actions, run_lens = action_runs(actions)
            raw_mask = run_actions == 2
            surrogate_mask = run_actions == 1
            drop_mask = run_actions == 0
            raw_tokens = int(run_lens[raw_mask].sum()) if raw_mask.any() else 0
            surrogate_tokens = int(run_lens[surrogate_mask].sum()) if surrogate_mask.any() else 0
            drop_tokens = int(run_lens[drop_mask].sum()) if drop_mask.any() else 0
            surrogate_lens = run_lens[surrogate_mask].astype(np.int64).tolist()
            used_entries = int(raw_tokens) + int(surrogate_mask.sum())
            return (
                {
                    "ks_run_raw_tokens": int(raw_tokens),
                    "ks_run_raw_regions": int(raw_mask.sum()),
                    "ks_run_surrogate_regions": int(surrogate_mask.sum()),
                    "ks_run_surrogate_tokens": int(surrogate_tokens),
                    "ks_run_drop_tokens": int(drop_tokens),
                    "ks_run_drop_regions": int(drop_mask.sum()),
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

        def action_used_entries(actions: Sequence[int]) -> int:
            stats, _surrogate_lens = collect_stats(actions)
            return int(stats["ks_run_used_entries"])

        def materialize_actions(actions: Sequence[int]):
            starts, ends, run_actions, _run_lens = action_runs(actions)
            new_slices = tuple(
                (int(atom_start_arr[int(start_idx)]), int(atom_end_arr[int(end_idx) - 1]))
                for start_idx, end_idx in zip(starts.tolist(), ends.tolist())
            )
            new_selected = run_actions != 2
            new_surrogate_lengths = np.where(run_actions == 0, 0, 1).astype(np.int64)
            new_chunk_lengths_arr = (
                atom_end_arr[ends.astype(np.int64) - 1] - atom_start_arr[starts.astype(np.int64)]
            ).astype(np.int64, copy=False)
            new_chunk_lengths = torch.as_tensor(new_chunk_lengths_arr, device=device, dtype=torch.long)
            new_replace_mask = torch.as_tensor(new_selected[None, :], device=device, dtype=torch.bool)
            new_surrogate_lengths_tensor = torch.as_tensor(
                new_surrogate_lengths[None, :],
                device=device,
                dtype=torch.long,
            )
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
            return new_slices, new_chunk_lengths, new_replace_mask, new_surrogate_lengths_tensor

        if budget_entries >= full_cost:
            actions = np.full((num_atoms,), 2, dtype=np.int8)
            stats, _ = collect_stats(actions)
            stats.update(
                {
                    "ks_run_candidate_seeds": 0,
                    "ks_run_candidate_surrogates": 0,
                    "ks_run_merge_accepts": 0,
                    "ks_run_pareto_merge_candidates": 0,
                    "ks_run_selected_surrogates": 0,
                    "ks_run_selected_value": 0.0,
                    "ks_run_selected_gain": 0.0,
                    "ks_run_sold_raw_atoms": 0,
                    "ks_run_sold_raw_tokens": 0,
                    "ks_run_sold_raw_value": 0.0,
                    "ks_run_buyback_raw_atoms": 0,
                    "ks_run_buyback_raw_tokens": 0,
                    "ks_run_buyback_raw_value": 0.0,
                    "ks_run_region_open_cost": 0.0,
                    "ks_run_raw_slot_price": 0.0,
                    "ks_run_initial_surrogate_slack": 0,
                    "ks_run_unused_surrogate_slack": 0,
                    "ks_run_rejected_budget": 0,
                    "ks_run_rejected_value": 0,
                    "ks_run_rejected_anchor": 0,
                    "ks_run_rejected_sold_k": 0,
                    "ks_run_candidate_rejected_sold_k": 0,
                    "ks_run_candidate_budget_impossible": 0,
                    "ks_run_sold_k_deficit_value": 0.0,
                    "ks_run_candidate_sold_k_deficit_value": 0.0,
                    "ks_run_dfx_enabled": 1,
                    "ks_run_dfx_merge_first": int(bool(merge_first)),
                    "ks_run_dfx_current_frontier_accept": int(bool(current_frontier_accept)),
                    "ks_run_dfx_admission_shadow_price": int(bool(admission_shadow_price)),
                    "ks_run_dfx_value_frontier": int(bool(value_frontier)),
                    "ks_run_dfx_frontier_region_price": int(bool(frontier_region_price)),
                    "ks_run_dfx_completion_order": int(bool(completion_order)),
                    "ks_run_dfx_bounded_market": int(bool(bounded_market)),
                    "ks_run_dfx_reserve_coherent_price": int(bool(reserve_coherent_price)),
                    "ks_run_budget_complete_peel": int(bool(budget_complete_peel)),
                    "ks_run_surrogate_peel_fill_atoms": 0,
                    "ks_run_surrogate_peel_fill_tokens": 0,
                    "ks_run_surrogate_peel_fill_value": 0.0,
                    "ks_run_surrogate_peel_fill_delta": 0,
                    "ks_run_raw_value": float(raw_value_arr.sum()),
                    "ks_run_surrogate_projection_value": 0.0,
                    "ks_run_objective_value": float(raw_value_arr.sum()),
                }
            )
            self._last_allocator_stats = stats
            return materialize_actions(actions)

        actions = np.full((num_atoms,), 2, dtype=np.int8)
        current_cost = int(full_cost)
        for atom_idx in raw_drop_order_list:
            if current_cost <= budget_entries:
                break
            if int(actions[atom_idx]) != 2:
                continue
            actions[atom_idx] = 0
            current_cost -= int(atom_len_int_arr[atom_idx])
        initial_actions = actions.copy()
        initial_slack = max(0, int(budget_entries) - int(current_cost))
        initial_raw_mask = initial_actions == 2
        initial_drop_mask = initial_actions == 0
        prefix_initial_raw_len = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum(atom_len_int_arr * initial_raw_mask.astype(np.int64), dtype=np.int64),
            )
        )
        prefix_initial_raw_value = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(raw_value_arr * initial_raw_mask.astype(np.float64), dtype=np.float64),
            )
        )
        prefix_initial_drop_len = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(atom_len_arr * initial_drop_mask.astype(np.float64), dtype=np.float64),
            )
        )
        prefix_initial_drop_mass = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(mean_signal_arr * atom_len_arr * initial_drop_mask.astype(np.float64), dtype=np.float64),
            )
        )
        prefix_initial_drop_energy = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(
                    mean_signal_arr * mean_signal_arr * atom_len_arr * initial_drop_mask.astype(np.float64),
                    dtype=np.float64,
                ),
            )
        )
        initial_raw_float_mask = initial_raw_mask.astype(np.float64)
        initial_raw_value_arr = raw_value_arr * initial_raw_float_mask
        initial_raw_recovery_slope_arr = 2.0 * mean_signal_arr * atom_len_arr * initial_raw_float_mask
        initial_raw_recovery_intercept_arr = (
            mean_signal_arr * mean_signal_arr * atom_len_arr * initial_raw_float_mask
        )
        buyback_order_list = [idx for idx in raw_keep_order_list if int(initial_actions[idx]) == 0]
        victim_order_list = [idx for idx in raw_drop_order_list if int(initial_actions[idx]) == 2]
        buyback_order_arr = np.asarray(buyback_order_list, dtype=np.int64)
        buyback_order_len_arr = np.asarray(
            [int(atom_len_int_arr[idx]) for idx in buyback_order_list],
            dtype=np.int64,
        )
        buyback_order_value_arr = np.asarray(
            [float(raw_value_arr[idx]) for idx in buyback_order_list],
            dtype=np.float64,
        )
        buyback_uniform_lengths = bool(
            buyback_order_len_arr.size <= 1
            or np.all(buyback_order_len_arr == buyback_order_len_arr[0])
        )
        buyback_prefix_tokens = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(buyback_order_len_arr, dtype=np.int64))
        )
        buyback_prefix_value = np.concatenate(
            (np.asarray([0.0], dtype=np.float64), np.cumsum(buyback_order_value_arr, dtype=np.float64))
        )
        max_buyback_lookup_slack = max(0, int(budget_entries))
        if max_buyback_lookup_slack > 0 and buyback_prefix_tokens.size > 1:
            buyback_lookup_slacks = np.arange(int(max_buyback_lookup_slack) + 1, dtype=np.int64)
            buyback_lookup_indices = np.searchsorted(
                buyback_prefix_tokens,
                buyback_lookup_slacks,
                side="right",
            ).astype(np.int64) - 1
            buyback_lookup_indices = np.clip(buyback_lookup_indices, 0, int(buyback_prefix_tokens.size) - 1)
            buyback_value_lookup = buyback_prefix_value[buyback_lookup_indices]
            buyback_token_lookup = buyback_prefix_tokens[buyback_lookup_indices]
        else:
            buyback_value_lookup = np.asarray([0.0], dtype=np.float64)
            buyback_token_lookup = np.asarray([0], dtype=np.int64)

        def raw_buyback_prefix_value(slack: int) -> float:
            if int(slack) <= 0 or buyback_prefix_tokens.size <= 1:
                return 0.0
            if bool(buyback_uniform_lengths) and int(slack) <= int(max_buyback_lookup_slack):
                return float(buyback_value_lookup[int(slack)])
            remaining = int(slack)
            value = 0.0
            for atom_idx in buyback_order_list:
                atom_len = int(atom_len_int_arr[atom_idx])
                if atom_len > remaining:
                    continue
                value += float(raw_value_arr[atom_idx])
                remaining -= int(atom_len)
                if remaining <= 0:
                    break
            return float(value)

        def raw_buyback_prefix_tokens(slack: int) -> int:
            if int(slack) <= 0 or buyback_prefix_tokens.size <= 1:
                return 0
            if bool(buyback_uniform_lengths) and int(slack) <= int(max_buyback_lookup_slack):
                return int(buyback_token_lookup[int(slack)])
            remaining = int(slack)
            tokens = 0
            for atom_idx in buyback_order_list:
                atom_len = int(atom_len_int_arr[atom_idx])
                if atom_len > remaining:
                    continue
                tokens += int(atom_len)
                remaining -= int(atom_len)
                if remaining <= 0:
                    break
            return int(tokens)

        frontier_density_arr = raw_density_arr[actions == 2]
        kept_frontier_price = float(frontier_density_arr.min()) if frontier_density_arr.size else 0.0
        dropped_density_arr = raw_density_arr[actions == 0]
        raw_slot_price = float(dropped_density_arr.max()) if dropped_density_arr.size else 0.0
        region_open_cost = (
            max(float(kept_frontier_price), float(raw_slot_price))
            if bool(frontier_region_price)
            else float(kept_frontier_price)
        )
        seed_frontier_pressure = float(
            np.mean(raw_density_arr <= float(region_open_cost))
        ) if raw_density_arr.size else 1.0

        Candidate = Tuple[int, int, int, int, int]

        def residual_projection_value(local_mean: np.ndarray, local_len: np.ndarray, mask: np.ndarray) -> float:
            if not bool(np.any(mask)):
                return 0.0
            masked_mean = local_mean[mask]
            masked_len = local_len[mask]
            total_len = float(masked_len.sum())
            if total_len <= 0.0:
                return 0.0
            mass = float((masked_mean * masked_len).sum())
            energy = float((masked_mean * masked_mean * masked_len).sum())
            projection = float(mass * mass / max(1.0, total_len))
            return max(0.0, min(float(energy), float(2.0 * projection - energy)))

        mean_mass_prefix = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(mean_signal_arr * atom_len_arr, dtype=np.float64),
            )
        )
        mean_energy_prefix = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(mean_signal_arr * mean_signal_arr * atom_len_arr, dtype=np.float64),
            )
        )

        def interval_projection_value(start_idx: int, end_idx: int) -> float:
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
            if token_len <= 0:
                return 0.0
            mass = float(mean_mass_prefix[end_idx] - mean_mass_prefix[start_idx])
            energy = float(mean_energy_prefix[end_idx] - mean_energy_prefix[start_idx])
            projection = float(mass * mass / max(1.0, float(token_len)))
            return max(0.0, min(float(energy), float(2.0 * projection - energy)))

        def initial_raw_indices_for_span(start_idx: int, end_idx: int) -> np.ndarray:
            raw_local = np.flatnonzero(initial_raw_mask[int(start_idx) : int(end_idx)]).astype(np.int64)
            if raw_local.size <= 0:
                return np.empty((0,), dtype=np.int64)
            return raw_local + int(start_idx)

        def initial_candidate_metrics(cand: Candidate):
            start_idx, end_idx, left_anchor, right_anchor, _seed_count = cand
            if (
                start_idx < 0
                or end_idx > num_atoms
                or start_idx >= end_idx
                or left_anchor < 0
                or right_anchor >= num_atoms
                or initial_actions[left_anchor] != 2
                or initial_actions[right_anchor] != 2
            ):
                return None
            token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
            if token_len <= 0:
                return None

            mass = float(mean_mass_prefix[end_idx] - mean_mass_prefix[start_idx])
            energy = float(mean_energy_prefix[end_idx] - mean_energy_prefix[start_idx])
            token_len_f = float(token_len)
            mu = mass / (token_len_f if token_len_f > 1.0 else 1.0)

            residual_len = float(prefix_initial_drop_len[end_idx] - prefix_initial_drop_len[start_idx])
            residual_mass = float(prefix_initial_drop_mass[end_idx] - prefix_initial_drop_mass[start_idx])
            residual_energy = float(prefix_initial_drop_energy[end_idx] - prefix_initial_drop_energy[start_idx])
            if residual_len <= 0.0:
                residual_projection = 0.0
            else:
                residual_base_projection = residual_mass * residual_mass / (
                    residual_len if residual_len > 1.0 else 1.0
                )
                residual_projection = 2.0 * residual_base_projection - residual_energy
                if residual_projection <= 0.0:
                    residual_projection = 0.0
                elif residual_projection > residual_energy:
                    residual_projection = residual_energy
            residual_contrib = 2.0 * mu * residual_mass - residual_energy
            if residual_contrib <= 0.0:
                residual_value = 0.0
            elif residual_contrib < residual_projection:
                residual_value = residual_contrib
            else:
                residual_value = residual_projection

            sold_loss = float(prefix_initial_raw_value[end_idx] - prefix_initial_raw_value[start_idx])
            sold_tokens = int(prefix_initial_raw_len[end_idx] - prefix_initial_raw_len[start_idx])
            sold_recovery = 0.0
            if int(sold_tokens) > 0:
                local_contrib = (
                    initial_raw_recovery_slope_arr[start_idx:end_idx] * float(mu)
                    - initial_raw_recovery_intercept_arr[start_idx:end_idx]
                )
                sold_recovery = float(
                    np.minimum(
                        initial_raw_value_arr[start_idx:end_idx],
                        np.maximum(0.0, local_contrib),
                    ).sum()
                )

            whole_base_projection = mass * mass / (token_len_f if token_len_f > 1.0 else 1.0)
            whole_projection_value = 2.0 * whole_base_projection - energy
            if whole_projection_value <= 0.0:
                whole_projection_value = 0.0
            elif whole_projection_value > energy:
                whole_projection_value = energy
            value = residual_value + sold_recovery
            if value > whole_projection_value:
                value = whole_projection_value
            coherent_credit = 0.0
            raw_reserve_debt = 0.0
            if bool(reserve_coherent_price):
                if residual_len > 0.0 and residual_energy > 0.0:
                    residual_coherence = float(residual_projection) / max(float(residual_energy), 1e-9)
                    coherent_credit = float(residual_projection) * float(residual_coherence)
            # The full raw index list is only needed for accepted-candidate
            # accounting.  Avoid building it for every scored market packet.
            return (
                value,
                sold_loss,
                sold_tokens,
                token_len,
                coherent_credit,
                raw_reserve_debt,
            )

        def candidate_metrics(actions_ref: np.ndarray, cand: Candidate):
            if actions_ref is initial_actions:
                metrics = initial_candidate_metrics(cand)
                if metrics is None:
                    return None
                value, sold_loss, sold_tokens, token_len, _coherent_credit, _raw_reserve_debt = metrics
                return value, sold_loss, sold_tokens, np.empty((0,), dtype=np.int64), token_len
            start_idx, end_idx, left_anchor, right_anchor, _seed_count = [int(v) for v in cand]
            if (
                start_idx < 0
                or end_idx > num_atoms
                or start_idx >= end_idx
                or left_anchor < 0
                or right_anchor >= num_atoms
                or int(actions_ref[left_anchor]) != 2
                or int(actions_ref[right_anchor]) != 2
                or bool(np.any(actions_ref[start_idx:end_idx] == 1))
            ):
                return None
            token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
            if token_len <= 0:
                return None
            local_actions = actions_ref[start_idx:end_idx]
            local_mean = mean_signal_arr[start_idx:end_idx]
            local_len = atom_len_arr[start_idx:end_idx]
            mass = float((local_mean * local_len).sum())
            mu = mass / max(1.0, float(token_len))
            local_contrib = (2.0 * local_mean * float(mu) - local_mean * local_mean) * local_len
            residual_mask = local_actions == 0
            residual_contrib = float(local_contrib[residual_mask].sum()) if bool(np.any(residual_mask)) else 0.0
            residual_value = min(
                residual_projection_value(local_mean, local_len, residual_mask),
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
            # A single surrogate cannot independently recover the dropped
            # residual and every downgraded raw atom.  Cap the packet value by
            # the projection energy of the whole represented interval so raw
            # recovery is only credited when the merged span is coherent.
            whole_projection_value = residual_projection_value(
                local_mean,
                local_len,
                np.ones_like(local_actions, dtype=bool),
            )
            value = min(float(residual_value) + float(sold_recovery), float(whole_projection_value))
            return float(value), float(sold_loss), int(sold_tokens), raw_inside, int(token_len)

        def raw_buyback_for_slack(
            actions_ref: np.ndarray,
            slack: int,
            exclude: np.ndarray,
            span_start: Optional[int] = None,
            span_end: Optional[int] = None,
            collect_atoms: bool = True,
        ) -> Tuple[np.ndarray, float, int]:
            if int(slack) <= 0:
                return np.empty((0,), dtype=np.int64), 0.0, 0
            if (not bool(collect_atoms)) and actions_ref is initial_actions:
                if buyback_order_arr.size <= 0:
                    return np.empty((0,), dtype=np.int64), 0.0, 0
                blocked_start = -1 if span_start is None else int(span_start)
                blocked_end = -1 if span_end is None else int(span_end)
                if not bool(np.any(exclude)) and blocked_start < 0:
                    return np.empty((0,), dtype=np.int64), raw_buyback_prefix_value(int(slack)), raw_buyback_prefix_tokens(int(slack))
                if bool(buyback_uniform_lengths):
                    eligible = ~exclude[buyback_order_arr]
                    if blocked_start >= 0:
                        eligible &= (buyback_order_arr < int(blocked_start)) | (buyback_order_arr >= int(blocked_end))
                    if not bool(np.any(eligible)):
                        return np.empty((0,), dtype=np.int64), 0.0, 0
                    eligible_lens = buyback_order_len_arr[eligible]
                    eligible_values = buyback_order_value_arr[eligible]
                    token_prefix = np.cumsum(eligible_lens, dtype=np.int64)
                    keep_count = int(np.searchsorted(token_prefix, int(slack), side="right"))
                    if keep_count <= 0:
                        return np.empty((0,), dtype=np.int64), 0.0, 0
                    value = float(eligible_values[:keep_count].sum())
                    tokens = int(token_prefix[int(keep_count) - 1])
                    return np.empty((0,), dtype=np.int64), float(value), int(tokens)
            remaining = int(slack)
            atoms: List[int] = []
            value = 0.0
            tokens = 0
            blocked_start = -1 if span_start is None else int(span_start)
            blocked_end = -1 if span_end is None else int(span_end)
            for atom_idx in buyback_order_list:
                if (
                    bool(exclude[atom_idx])
                    or (blocked_start >= 0 and blocked_start <= int(atom_idx) < blocked_end)
                    or int(actions_ref[atom_idx]) != 0
                ):
                    continue
                atom_len = int(atom_len_int_arr[atom_idx])
                if atom_len > remaining:
                    continue
                if bool(collect_atoms):
                    atoms.append(atom_idx)
                value += float(raw_value_arr[atom_idx])
                tokens += int(atom_len)
                remaining -= int(atom_len)
                if remaining <= 0:
                    break
            atom_arr = np.asarray(atoms, dtype=np.int64) if bool(collect_atoms) else np.empty((0,), dtype=np.int64)
            return atom_arr, float(value), int(tokens)

        def raw_victims_for_need(
            actions_ref: np.ndarray,
            needed: int,
            protected: np.ndarray,
        ):
            if int(needed) <= 0:
                return np.empty((0,), dtype=np.int64), 0.0, 0
            remaining = int(needed)
            victims: List[int] = []
            value = 0.0
            tokens = 0
            for atom_idx in victim_order_list:
                if bool(protected[atom_idx]) or int(actions_ref[atom_idx]) != 2:
                    continue
                atom_len = int(atom_len_int_arr[atom_idx])
                victims.append(atom_idx)
                value += float(raw_value_arr[atom_idx])
                tokens += int(atom_len)
                remaining -= int(atom_len)
                if remaining <= 0:
                    break
            if remaining > 0:
                return None
            return np.asarray(victims, dtype=np.int64), float(value), int(tokens)

        def exchange_for_candidate(
            actions_ref: np.ndarray,
            cand: Candidate,
            locked_raw: np.ndarray,
            used_entries: int,
        ):
            metrics = candidate_metrics(actions_ref, cand)
            if metrics is None:
                return None
            value, sold_loss, sold_tokens, raw_inside, token_len = metrics
            start_idx, end_idx, left_anchor, right_anchor, _seed_count = [int(v) for v in cand]
            if raw_inside.size and bool(np.any(locked_raw[raw_inside])):
                return None
            slack = int(budget_entries) - int(used_entries)
            budget_delta = int(1 - int(sold_tokens))
            needed = max(0, int(budget_delta) - int(slack))
            protected = locked_raw.copy()
            protected[start_idx:end_idx] = True
            protected[int(left_anchor)] = True
            protected[int(right_anchor)] = True
            victim_result = raw_victims_for_need(actions_ref, int(needed), protected)
            if victim_result is None:
                return None
            victim_atoms, victim_loss, victim_tokens = victim_result
            after_slack = int(slack) - int(budget_delta) + int(victim_tokens)
            if after_slack < 0:
                return None
            exclude = np.zeros((num_atoms,), dtype=bool)
            exclude[start_idx:end_idx] = True
            if victim_atoms.size:
                exclude[victim_atoms] = True
            buyback_atoms, buyback_value, buyback_tokens = raw_buyback_for_slack(actions_ref, int(after_slack), exclude)
            gain = float(value) - float(sold_loss) - float(victim_loss) + float(buyback_value)
            return {
                "gain": float(gain),
                "value": float(value),
                "sold_loss": float(sold_loss),
                "sold_tokens": int(sold_tokens),
                "raw_inside": raw_inside,
                "victim_atoms": victim_atoms,
                "victim_loss": float(victim_loss),
                "victim_tokens": int(victim_tokens),
                "buyback_atoms": buyback_atoms,
                "buyback_value": float(buyback_value),
                "buyback_tokens": int(buyback_tokens),
                "budget_delta": int(budget_delta),
                "token_len": int(token_len),
            }

        initial_record_cache: Dict[Candidate, Optional[Tuple[float, int, float, float, int, int, float, float]]] = {}

        def initial_record(cand: Candidate):
            try:
                return initial_record_cache[cand]
            except KeyError:
                pass
            metrics = initial_candidate_metrics(cand)
            if metrics is None:
                initial_record_cache[cand] = None
                return None
            value, sold_loss, sold_tokens, token_len, coherent_credit, raw_reserve_debt = metrics
            effective_value = float(value)
            effective_loss = float(sold_loss)
            budget_delta = int(1 - int(sold_tokens))
            after_slack = int(initial_slack) - int(budget_delta)
            if after_slack < 0:
                gain = -float("inf")
            else:
                gain = (
                    float(effective_value)
                    - float(effective_loss)
                    - float(region_open_cost)
                    + raw_buyback_prefix_value(int(after_slack))
                    - float(base_fill_prefix_value)
                )
            record = (
                float(gain),
                int(budget_delta),
                float(value),
                float(sold_loss),
                int(sold_tokens),
                int(token_len),
                float(coherent_credit),
                float(raw_reserve_debt),
            )
            initial_record_cache[cand] = record
            return record

        run_starts, run_ends, run_actions, _run_lens = action_runs(initial_actions)
        seeds: List[Candidate] = []
        for run_start, run_end, run_action in zip(run_starts.tolist(), run_ends.tolist(), run_actions.tolist()):
            if int(run_action) != 0:
                continue
            run_start = int(run_start)
            run_end = int(run_end)
            left_anchor = int(run_start) - 1
            right_anchor = int(run_end)
            if (
                left_anchor < 0
                or right_anchor >= num_atoms
                or int(initial_actions[left_anchor]) != 2
                or int(initial_actions[right_anchor]) != 2
            ):
                continue
            seeds.append((int(run_start), int(run_end), int(left_anchor), int(right_anchor), 1))

        candidate_seeds = int(len(seeds))
        seed_prefilter_limit = 0
        seed_prefilter_dropped = 0
        if bool(bounded_market) and int(len(seeds)) > 0:
            seed_base_limit = max(16, int(math.ceil(math.sqrt(max(1, int(num_atoms))))))
            seed_prefilter_limit = max(
                int(seed_base_limit),
                int(math.ceil(float(len(seeds)) * max(0.0, min(1.0, float(seed_frontier_pressure))))),
            )
            if int(len(seeds)) > int(seed_prefilter_limit):
                seed_prefilter_score_cache: Dict[Candidate, Tuple[float, float, int]] = {}

                def seed_prefilter_score(cand: Candidate) -> Tuple[float, float, int]:
                    cached = seed_prefilter_score_cache.get(cand)
                    if cached is not None:
                        return cached
                    start_idx, end_idx, _left_anchor, _right_anchor, _seed_count = [int(v) for v in cand]
                    token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
                    value = float(interval_projection_value(start_idx, end_idx))
                    gain = float(value) - float(region_open_cost) - float(raw_slot_price)
                    density = float(gain) / max(1.0, float(token_len))
                    score = (float(gain), float(density), int(token_len))
                    seed_prefilter_score_cache[cand] = score
                    return score

                gain_quota = max(1, int(seed_prefilter_limit) * 2 // 3)
                density_quota = max(1, int(seed_prefilter_limit) - int(gain_quota))
                seed_prefilter_records = [(cand, seed_prefilter_score(cand)) for cand in seeds]
                gain_ranked_seeds = [
                    cand
                    for cand, _score in sorted(
                        seed_prefilter_records,
                        key=lambda item: (item[1][0], item[1][2]),
                        reverse=True,
                    )
                ]
                density_ranked_seeds = [
                    cand
                    for cand, _score in sorted(
                        seed_prefilter_records,
                        key=lambda item: (item[1][1], item[1][0]),
                        reverse=True,
                    )
                ]
                kept_seeds: List[Candidate] = []
                kept_seed_keys = set()

                def keep_seed(cand: Candidate) -> None:
                    key = (int(cand[0]), int(cand[1]))
                    if key in kept_seed_keys:
                        return
                    kept_seed_keys.add(key)
                    kept_seeds.append(cand)

                for cand in gain_ranked_seeds[: int(gain_quota)]:
                    keep_seed(cand)
                for cand in density_ranked_seeds[: int(density_quota)]:
                    keep_seed(cand)
                if len(kept_seeds) < int(seed_prefilter_limit):
                    for cand in gain_ranked_seeds:
                        keep_seed(cand)
                        if len(kept_seeds) >= int(seed_prefilter_limit):
                            break
                seed_prefilter_dropped = int(len(seeds)) - int(len(kept_seeds))
                seeds = sorted(
                    kept_seeds[: int(seed_prefilter_limit)],
                    key=lambda cand: (int(cand[0]), int(cand[1])),
                )

        stack: List[Candidate] = []
        pareto_candidates: List[Candidate] = []
        merge_accepts = 0

        base_fill_prefix_value = raw_buyback_prefix_value(int(initial_slack))
        packet_gain_cache: Dict[Tuple[Candidate, ...], float] = {}

        def packet_gain_from_records(records: Sequence[Tuple[float, int, float, float, int, int, float, float]]) -> float:
            total_delta = 0
            total_value = 0.0
            total_sold_loss = 0.0
            for record in records:
                _gain, budget_delta, value, sold_loss, _sold_tokens, _token_len, coherent_credit, raw_reserve_debt = record
                total_delta += int(budget_delta)
                total_value += float(value)
                total_sold_loss += float(sold_loss)
            after_slack = int(initial_slack) - int(total_delta)
            if after_slack < 0:
                return -float("inf")
            return (
                float(total_value)
                - float(total_sold_loss)
                - float(region_open_cost) * float(len(records))
                + raw_buyback_prefix_value(int(after_slack))
                - float(base_fill_prefix_value)
            )

        def packet_initial_gain(cands: Sequence[Candidate]) -> float:
            cache_key = tuple(cands)
            cached = packet_gain_cache.get(cache_key)
            if cached is not None:
                return float(cached)
            records = []
            for cand in cands:
                record = initial_record(cand)
                if record is None:
                    packet_gain_cache[cache_key] = -float("inf")
                    return -float("inf")
                records.append(record)
            gain = packet_gain_from_records(records)
            packet_gain_cache[cache_key] = float(gain)
            return float(gain)

        def initial_gain(cand: Candidate) -> float:
            record = initial_record(cand)
            if record is None:
                return -float("inf")
            return float(record[0])

        for seed in seeds:
            stack.append(seed)
            while len(stack) >= 2:
                left = stack[-2]
                right = stack[-1]
                if int(left[1]) > int(right[0]):
                    break
                merged: Candidate = (int(left[0]), int(right[1]), int(left[2]), int(right[3]), int(left[4]) + int(right[4]))
                left_record = initial_record(left)
                right_record = initial_record(right)
                merged_record = initial_record(merged)
                if left_record is None or right_record is None:
                    split_gain = -float("inf")
                    split_delta = 1
                else:
                    split_gain = float(packet_gain_from_records((left_record, right_record)))
                    split_delta = int(left_record[1]) + int(right_record[1])
                if merged_record is None:
                    merged_gain = -float("inf")
                    merged_delta = 1
                else:
                    merged_gain = float(merged_record[0])
                    merged_delta = int(merged_record[1])
                prefer_merged = (
                    bool(merge_first)
                    and float(merged_gain) > 0.0
                    and int(merged_delta) < int(split_delta)
                )
                if bool(prefer_merged) or float(merged_gain) > float(split_gain):
                    stack.pop()
                    stack.pop()
                    stack.append(merged)
                    merge_accepts += 1
                    continue
                if float(merged_gain) > 0.0 and int(merged_delta) < int(split_delta):
                    pareto_candidates.append(merged)
                break

        all_candidates: List[Candidate] = []
        seen_candidates = set()
        for cand in stack + pareto_candidates:
            key = (int(cand[0]), int(cand[1]))
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            if float(initial_gain(cand)) > 0.0:
                all_candidates.append(cand)

        candidate_market_limit = 0
        candidate_generated_total = int(len(all_candidates))
        market_gain_quota = 0
        market_density_quota = 0
        market_coverage_quota = 0
        market_self_funding_quota = 0
        market_seed_quota = 0
        market_coverage_candidates = 0
        market_self_funding_candidates = 0
        market_seed_candidates = 0

        def candidate_token_len(cand: Candidate) -> int:
            return int(prefix_len[int(cand[1])] - prefix_len[int(cand[0])])

        candidate_market_record_cache: Dict[Candidate, Tuple[float, int, float, float, int, int, float, float]] = {}

        def candidate_market_record(cand: Candidate) -> Tuple[float, int, float, float, int, int, float, float]:
            cached = candidate_market_record_cache.get(cand)
            if cached is not None:
                return cached
            record = initial_record(cand)
            if record is None:
                record = (-float("inf"), 0, -float("inf"), -float("inf"), 1, int(cand[4]), 0.0, -float("inf"))
            else:
                gain, budget_delta, value, _sold_loss, _sold_tokens, token_len, coherent_credit, _raw_reserve_debt = record
                token_len = max(1, int(token_len))
                density = float(gain) / float(token_len)
                # Keep broad coherent packets visible to the bounded market.
                # Admission still uses the raw-frontier after-state gain, so
                # this only chooses which candidates get a chance to compete.
                effective_coverage_value = float(value) + (float(coherent_credit) if bool(reserve_coherent_price) else 0.0)
                coverage = float(effective_coverage_value) * math.log1p(float(token_len))
                record = (
                    float(gain),
                    int(token_len),
                    float(density),
                    float(coverage),
                    int(budget_delta),
                    int(cand[4]),
                    float(value),
                    float(coverage),
                )
            candidate_market_record_cache[cand] = record
            return record

        if bool(bounded_market):
            # Keep the market sublinear, but not so narrow that high-prune
            # coverage packets are truncated before admission can compare them.
            candidate_market_limit = max(16, int(math.ceil(math.sqrt(max(1, int(num_atoms))))))
            if int(len(all_candidates)) > int(candidate_market_limit):
                market_gain_quota = max(1, int(candidate_market_limit) * 35 // 100)
                market_density_quota = max(1, int(candidate_market_limit) * 15 // 100)
                market_self_funding_quota = max(1, int(candidate_market_limit) * 20 // 100)
                market_seed_quota = max(1, int(candidate_market_limit) * 15 // 100)
                market_coverage_quota = max(
                    1,
                    int(candidate_market_limit)
                    - int(market_gain_quota)
                    - int(market_density_quota)
                    - int(market_self_funding_quota)
                    - int(market_seed_quota),
                )
                candidate_records = [(cand, candidate_market_record(cand)) for cand in all_candidates]
                candidate_record_by_cand = {cand: record for cand, record in candidate_records}

                gain_ranked = sorted(
                    candidate_records,
                    key=lambda item: (
                        item[1][0],
                        item[1][1],
                    ),
                    reverse=True,
                )
                gain_ranked = [cand for cand, _record in gain_ranked]
                density_ranked = sorted(
                    candidate_records,
                    key=lambda item: (
                        item[1][2],
                        item[1][0],
                    ),
                    reverse=True,
                )
                density_ranked = [cand for cand, _record in density_ranked]
                self_funding_ranked = sorted(
                    candidate_records,
                    key=lambda item: (
                        -min(0, item[1][4]),
                        item[1][0],
                        item[1][3],
                    ),
                    reverse=True,
                )
                self_funding_ranked = [cand for cand, _record in self_funding_ranked]
                seed_ranked = sorted(
                    candidate_records,
                    key=lambda item: (
                        int(item[1][5]) == 1,
                        item[1][0],
                        item[1][2],
                    ),
                    reverse=True,
                )
                seed_ranked = [cand for cand, _record in seed_ranked]
                coverage_ranked = sorted(
                    candidate_records,
                    key=lambda item: (
                        item[1][3],
                        item[1][1],
                        item[1][0],
                    ),
                    reverse=True,
                )
                coverage_ranked = [cand for cand, _record in coverage_ranked]
                kept_candidates: List[Candidate] = []
                kept_candidate_keys = set()
                claimed_atoms = np.zeros((num_atoms,), dtype=bool)

                def keep_candidate(cand: Candidate, *, min_novel_fraction: float = 0.0) -> bool:
                    key = (int(cand[0]), int(cand[1]))
                    if key in kept_candidate_keys:
                        return False
                    if float(min_novel_fraction) > 0.0:
                        span_claimed = claimed_atoms[int(cand[0]) : int(cand[1])]
                        if span_claimed.size:
                            novel = int(span_claimed.size) - int(span_claimed.sum())
                            if float(novel) / max(1.0, float(span_claimed.size)) < float(min_novel_fraction):
                                return False
                    kept_candidate_keys.add(key)
                    kept_candidates.append(cand)
                    claimed_atoms[int(cand[0]) : int(cand[1])] = True
                    return True

                for cand in gain_ranked[: int(market_gain_quota)]:
                    keep_candidate(cand)
                before_self_funding = int(len(kept_candidates))
                for cand in self_funding_ranked:
                    if candidate_record_by_cand[cand][4] > 0:
                        break
                    keep_candidate(cand, min_novel_fraction=0.15)
                    if len(kept_candidates) - before_self_funding >= int(market_self_funding_quota):
                        break
                market_self_funding_candidates = max(0, int(len(kept_candidates)) - int(before_self_funding))
                before_seed = int(len(kept_candidates))
                for cand in seed_ranked:
                    if int(candidate_record_by_cand[cand][5]) != 1:
                        break
                    keep_candidate(cand, min_novel_fraction=0.10)
                    if len(kept_candidates) - before_seed >= int(market_seed_quota):
                        break
                market_seed_candidates = max(0, int(len(kept_candidates)) - int(before_seed))
                for cand in density_ranked:
                    keep_candidate(cand, min_novel_fraction=0.10)
                    if len(kept_candidates) >= (
                        int(market_gain_quota)
                        + int(market_self_funding_candidates)
                        + int(market_seed_candidates)
                        + int(market_density_quota)
                    ):
                        break
                before_coverage = int(len(kept_candidates))
                for cand in coverage_ranked:
                    keep_candidate(cand, min_novel_fraction=0.35)
                    if len(kept_candidates) >= int(candidate_market_limit):
                        break
                    if len(kept_candidates) - before_coverage >= int(market_coverage_quota):
                        break
                market_coverage_candidates = max(0, int(len(kept_candidates)) - int(before_coverage))
                if len(kept_candidates) < int(candidate_market_limit):
                    for cand in coverage_ranked:
                        keep_candidate(cand, min_novel_fraction=0.15)
                        if len(kept_candidates) >= int(candidate_market_limit):
                            break
                if len(kept_candidates) < int(candidate_market_limit):
                    for cand in gain_ranked:
                        keep_candidate(cand)
                        if len(kept_candidates) >= int(candidate_market_limit):
                            break
                all_candidates = kept_candidates[: int(candidate_market_limit)]

        candidate_queue: List[Tuple[float, int, Candidate]] = []
        for idx, cand in enumerate(all_candidates):
            gain = float(initial_gain(cand))
            order_gain = float(gain)
            if bool(completion_order):
                order_gain += float(region_open_cost) * float(max(0, int(cand[4]) - 1))
            candidate_queue.append((-float(order_gain), int(idx), cand))
        candidate_queue.sort()

        selected_exclude = np.zeros((num_atoms,), dtype=bool)
        no_locked_raw = np.zeros((num_atoms,), dtype=bool)
        no_buyback_exclude = np.zeros((num_atoms,), dtype=bool)
        current_slack = int(initial_slack)
        current_fill_value = raw_buyback_prefix_value(int(current_slack))
        selected_surrogates = 0
        selected_value = 0.0
        selected_gain = 0.0
        selected_coherent_credit = 0.0
        selected_raw_reserve_debt = 0.0
        sold_raw_atoms = 0
        sold_raw_tokens = 0
        sold_raw_value = 0.0
        victim_raw_atoms = 0
        victim_raw_tokens = 0
        victim_raw_value = 0.0
        buyback_raw_atoms = 0
        buyback_raw_tokens = 0
        buyback_raw_value = 0.0
        rejected_budget = 0
        rejected_value = 0
        rejected_anchor = 0
        rejected_sold_k = 0

        for _neg_gain, _idx, cand in candidate_queue:
            start_idx, end_idx, left_anchor, right_anchor, _seed_count = [int(v) for v in cand]
            if (
                int(left_anchor) < 0
                or int(right_anchor) >= num_atoms
                or int(actions[left_anchor]) != 2
                or int(actions[right_anchor]) != 2
                or bool(np.any(actions[start_idx:end_idx] == 1))
            ):
                rejected_anchor += 1
                continue

            if bool(current_frontier_accept):
                exchange = exchange_for_candidate(
                    actions,
                    cand,
                    no_locked_raw,
                    int(current_cost),
                )
                if exchange is None:
                    rejected_budget += 1
                    continue
                marginal_gain = (
                    float(exchange["gain"])
                    - float(region_open_cost)
                    - float(current_fill_value)
                )
                if float(marginal_gain) <= 0.0:
                    rejected_value += 1
                    continue

                victim_atoms_arr = exchange["victim_atoms"]
                if victim_atoms_arr.size:
                    actions[victim_atoms_arr] = 0
                    victim_raw_atoms += int(victim_atoms_arr.size)
                    victim_raw_tokens += int(exchange["victim_tokens"])
                    victim_raw_value += float(exchange["victim_loss"])

                raw_inside = exchange["raw_inside"]
                actions[start_idx:end_idx] = 1
                current_cost += int(exchange["budget_delta"]) - int(exchange["victim_tokens"])
                current_slack = max(0, int(budget_entries) - int(current_cost))
                _current_fill_atoms, current_fill_value, _current_fill_tokens = raw_buyback_for_slack(
                    actions,
                    int(current_slack),
                    no_buyback_exclude,
                    collect_atoms=False,
                )

                selected_surrogates += 1
                selected_value += float(exchange["value"])
                selected_gain += float(marginal_gain)
                sold_raw_atoms += int(raw_inside.size)
                sold_raw_tokens += int(exchange["sold_tokens"])
                sold_raw_value += float(exchange["sold_loss"])
                continue

            record = initial_record(cand)
            if record is None:
                rejected_budget += 1
                continue
            (
                _initial_gain_value,
                budget_delta,
                value,
                sold_loss,
                sold_tokens,
                _token_len,
                coherent_credit,
                raw_reserve_debt,
            ) = record
            effective_value = float(value)
            effective_loss = float(sold_loss)
            raw_inside = np.empty((0,), dtype=np.int64)
            next_slack = int(current_slack) - int(budget_delta)
            if int(next_slack) < 0:
                rejected_budget += 1
                continue

            _next_fill_atoms, next_fill_value, _next_fill_tokens = raw_buyback_for_slack(
                initial_actions,
                int(next_slack),
                selected_exclude,
                int(start_idx),
                int(end_idx),
                collect_atoms=False,
            )
            if bool(admission_shadow_price) and int(budget_delta) > 0:
                actual_fill_loss = max(0.0, float(current_fill_value) - float(next_fill_value))
                shadow_fill_loss = float(raw_slot_price) * float(budget_delta)
                opportunity_cost = max(float(actual_fill_loss), float(shadow_fill_loss))
                marginal_gain = (
                    float(effective_value)
                    - float(effective_loss)
                    - float(region_open_cost)
                    - float(opportunity_cost)
                )
            else:
                marginal_gain = (
                    float(effective_value)
                    - float(effective_loss)
                    - float(region_open_cost)
                    + float(next_fill_value)
                    - float(current_fill_value)
                )
            if float(marginal_gain) <= 0.0:
                rejected_value += 1
                continue

            actions[start_idx:end_idx] = 1
            selected_exclude[int(start_idx) : int(end_idx)] = True
            current_slack = int(next_slack)
            current_fill_value = float(next_fill_value)
            if raw_inside.size <= 0 and int(sold_tokens) > 0:
                raw_inside = initial_raw_indices_for_span(int(start_idx), int(end_idx))

            selected_surrogates += 1
            selected_value += float(value)
            selected_gain += float(marginal_gain)
            selected_coherent_credit += float(coherent_credit)
            selected_raw_reserve_debt += float(raw_reserve_debt)
            sold_raw_atoms += int(raw_inside.size)
            sold_raw_tokens += int(sold_tokens)
            sold_raw_value += float(sold_loss)
            current_cost += int(budget_delta)

        def fill_raw_frontier(actions_ref: np.ndarray, cost: int) -> Tuple[int, int, float, int]:
            fill_atoms = 0
            fill_tokens = 0
            fill_value = 0.0
            current = int(cost)
            fill_order = raw_keep_order_list if bool(budget_complete_peel) else buyback_order_list
            for atom_idx in fill_order:
                if int(actions_ref[atom_idx]) != 0:
                    continue
                atom_len = int(atom_len_int_arr[atom_idx])
                if current + int(atom_len) > int(budget_entries):
                    continue
                actions_ref[atom_idx] = 2
                current += int(atom_len)
                fill_atoms += 1
                fill_tokens += int(atom_len)
                fill_value += float(raw_value_arr[atom_idx])
                if current >= int(budget_entries):
                    break
            return int(fill_atoms), int(fill_tokens), float(fill_value), int(current)

        def peel_surrogate_frontier(actions_ref: np.ndarray, cost: int) -> Tuple[int, int, float, int, int]:
            if not bool(budget_complete_peel):
                return 0, 0, 0.0, 0, int(cost)
            current = int(action_used_entries(actions_ref))
            if current >= int(budget_entries):
                return 0, 0, 0.0, 0, int(current)

            starts, ends, run_actions, _run_lens = action_runs(actions_ref)
            states: List[List[int]] = []
            heap: List[Tuple[float, float, int, int, int, int, int]] = []

            def push_boundary(run_id: int, side: int) -> None:
                left, right, version = states[int(run_id)]
                if int(left) >= int(right):
                    return
                atom_idx = int(left) if int(side) == 0 else int(right) - 1
                if int(side) == 1 and int(atom_idx) == int(left):
                    return
                run_atoms = int(right) - int(left)
                atom_len = int(atom_len_int_arr[atom_idx])
                delta = int(atom_len) - 1 if int(run_atoms) <= 1 else int(atom_len)
                if int(delta) < 0:
                    delta = 0
                value = float(raw_value_arr[atom_idx])
                density = float(value) / float(max(1, int(delta)))
                heapq.heappush(
                    heap,
                    (-float(density), -float(value), int(delta), int(run_id), int(side), int(atom_idx), int(version)),
                )

            for run_start, run_end, run_action in zip(starts.tolist(), ends.tolist(), run_actions.tolist()):
                if int(run_action) != 1:
                    continue
                run_id = int(len(states))
                states.append([int(run_start), int(run_end), 0])
                push_boundary(run_id, 0)
                push_boundary(run_id, 1)

            peel_atoms = 0
            peel_tokens = 0
            peel_value = 0.0
            peel_delta = 0
            while heap and int(current) < int(budget_entries):
                _neg_density, _neg_value, _delta_hint, run_id, side, atom_idx, version = heapq.heappop(heap)
                if int(run_id) >= len(states):
                    continue
                left, right, current_version = states[int(run_id)]
                if int(version) != int(current_version) or int(left) >= int(right):
                    continue
                expected_atom = int(left) if int(side) == 0 else int(right) - 1
                if int(atom_idx) != int(expected_atom) or int(actions_ref[atom_idx]) != 1:
                    continue
                run_atoms = int(right) - int(left)
                atom_len = int(atom_len_int_arr[atom_idx])
                delta = int(atom_len) - 1 if int(run_atoms) <= 1 else int(atom_len)
                if int(delta) < 0:
                    delta = 0
                if int(current) + int(delta) > int(budget_entries):
                    continue

                actions_ref[atom_idx] = 2
                current += int(delta)
                peel_atoms += 1
                peel_tokens += int(atom_len)
                peel_value += float(raw_value_arr[atom_idx])
                peel_delta += int(delta)

                if int(side) == 0:
                    left += 1
                else:
                    right -= 1
                states[int(run_id)] = [int(left), int(right), int(current_version) + 1]
                push_boundary(run_id, 0)
                push_boundary(run_id, 1)

            return int(peel_atoms), int(peel_tokens), float(peel_value), int(peel_delta), int(current)

        final_fill_atoms, final_fill_tokens, final_fill_value, current_cost = fill_raw_frontier(actions, current_cost)
        buyback_raw_atoms += int(final_fill_atoms)
        buyback_raw_tokens += int(final_fill_tokens)
        buyback_raw_value += float(final_fill_value)
        (
            surrogate_peel_fill_atoms,
            surrogate_peel_fill_tokens,
            surrogate_peel_fill_value,
            surrogate_peel_fill_delta,
            current_cost,
        ) = peel_surrogate_frontier(actions, current_cost)
        buyback_raw_atoms += int(surrogate_peel_fill_atoms)
        buyback_raw_tokens += int(surrogate_peel_fill_tokens)
        buyback_raw_value += float(surrogate_peel_fill_value)

        final_raw_value = float(raw_value_arr[actions == 2].sum()) if bool(np.any(actions == 2)) else 0.0
        final_surrogate_projection_value = 0.0
        final_starts, final_ends, final_run_actions, _final_run_lens = action_runs(actions)
        for run_start, run_end, run_action in zip(
            final_starts.tolist(),
            final_ends.tolist(),
            final_run_actions.tolist(),
        ):
            if int(run_action) == 1:
                final_surrogate_projection_value += float(interval_projection_value(int(run_start), int(run_end)))
        final_objective_value = float(final_raw_value) + float(final_surrogate_projection_value)

        stats, _surrogate_lens = collect_stats(actions)
        stats.update(
            {
                "ks_run_candidate_seeds": int(candidate_seeds),
                "ks_run_candidate_surrogates": int(len(all_candidates)),
                "ks_run_candidate_generated_total": int(candidate_generated_total),
                "ks_run_candidate_market_limit": int(candidate_market_limit),
                "ks_run_candidate_market_gain_quota": int(market_gain_quota),
                "ks_run_candidate_market_density_quota": int(market_density_quota),
                "ks_run_candidate_market_coverage_quota": int(market_coverage_quota),
                "ks_run_candidate_market_self_funding_quota": int(market_self_funding_quota),
                "ks_run_candidate_market_seed_quota": int(market_seed_quota),
                "ks_run_candidate_market_coverage_candidates": int(market_coverage_candidates),
                "ks_run_candidate_market_self_funding_candidates": int(market_self_funding_candidates),
                "ks_run_candidate_market_seed_candidates": int(market_seed_candidates),
                "ks_run_seed_prefilter_limit": int(seed_prefilter_limit),
                "ks_run_seed_prefilter_dropped": int(seed_prefilter_dropped),
                "ks_run_seed_frontier_pressure": float(seed_frontier_pressure),
                "ks_run_merge_accepts": int(merge_accepts),
                "ks_run_pareto_merge_candidates": int(len(pareto_candidates)),
                "ks_run_selected_surrogates": int(selected_surrogates),
                "ks_run_selected_value": float(selected_value),
                "ks_run_selected_gain": float(selected_gain),
                "ks_run_dfx_coherent_credit": float(selected_coherent_credit),
                "ks_run_dfx_raw_reserve_debt": float(selected_raw_reserve_debt),
                "ks_run_sold_raw_atoms": int(sold_raw_atoms),
                "ks_run_sold_raw_tokens": int(sold_raw_tokens),
                "ks_run_sold_raw_value": float(sold_raw_value),
                "ks_run_buyback_raw_atoms": int(buyback_raw_atoms),
                "ks_run_buyback_raw_tokens": int(buyback_raw_tokens),
                "ks_run_buyback_raw_value": float(buyback_raw_value),
                "ks_run_frontier_fill_atoms": int(final_fill_atoms),
                "ks_run_frontier_fill_tokens": int(final_fill_tokens),
                "ks_run_frontier_fill_value": float(final_fill_value),
                "ks_run_surrogate_peel_fill_atoms": int(surrogate_peel_fill_atoms),
                "ks_run_surrogate_peel_fill_tokens": int(surrogate_peel_fill_tokens),
                "ks_run_surrogate_peel_fill_value": float(surrogate_peel_fill_value),
                "ks_run_surrogate_peel_fill_delta": int(surrogate_peel_fill_delta),
                "ks_run_raw_value": float(final_raw_value),
                "ks_run_surrogate_projection_value": float(final_surrogate_projection_value),
                "ks_run_objective_value": float(final_objective_value),
                "ks_run_region_open_cost": float(region_open_cost),
                "ks_run_dfx_kept_frontier_price": float(kept_frontier_price),
                "ks_run_dfx_region_cost_total": float(region_open_cost) * float(selected_surrogates),
                "ks_run_raw_slot_price": float(raw_slot_price),
                "ks_run_initial_surrogate_slack": int(initial_slack),
                "ks_run_unused_surrogate_slack": int(max(0, int(budget_entries) - int(current_cost))),
                "ks_run_anchor_violations": 0,
                "ks_run_rejected_budget": int(rejected_budget),
                "ks_run_rejected_value": int(rejected_value),
                "ks_run_rejected_anchor": int(rejected_anchor),
                "ks_run_rejected_sold_k": int(rejected_sold_k),
                "ks_run_candidate_rejected_sold_k": 0,
                "ks_run_candidate_budget_impossible": 0,
                "ks_run_sold_k_deficit_value": 0.0,
                "ks_run_candidate_sold_k_deficit_value": 0.0,
                "ks_run_dfx_enabled": 1,
                "ks_run_dfx_merge_first": int(bool(merge_first)),
                "ks_run_dfx_current_frontier_accept": int(bool(current_frontier_accept)),
                "ks_run_dfx_admission_shadow_price": int(bool(admission_shadow_price)),
                "ks_run_dfx_value_frontier": int(bool(value_frontier)),
                "ks_run_dfx_frontier_region_price": int(bool(frontier_region_price)),
                "ks_run_dfx_completion_order": int(bool(completion_order)),
                "ks_run_dfx_bounded_market": int(bool(bounded_market)),
                "ks_run_dfx_reserve_coherent_price": int(bool(reserve_coherent_price)),
                "ks_run_budget_complete_peel": int(bool(budget_complete_peel)),
                "ks_run_dfx_victim_raw_atoms": int(victim_raw_atoms),
                "ks_run_dfx_victim_raw_tokens": int(victim_raw_tokens),
                "ks_run_dfx_victim_raw_value": float(victim_raw_value),
            }
        )
        self._last_allocator_stats = stats

        return materialize_actions(actions)

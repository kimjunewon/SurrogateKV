from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .common import _rank01

_SURKV_PROFILE_TIMING = str(os.environ.get("SURKV_PROFILE_TIMING", "")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _build_allocator_timing_stats(profile_times: Dict[str, float], profile_t0: float, *, enabled: bool) -> Dict[str, float]:
    if not enabled:
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


def _encode_action_runs(
    actions_ref: Sequence[int],
    prefix_len: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _summarize_region_actions(
    actions_ref: Sequence[int],
    *,
    prefix_len: np.ndarray,
    budget_entries: int,
    full_cost: int,
) -> Tuple[Dict[str, float], List[int]]:
    _starts, _ends, run_actions, run_lens = _encode_action_runs(actions_ref, prefix_len)
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


def _materialize_region_actions(
    owner,
    actions_ref: Sequence[int],
    *,
    atom_start_arr: np.ndarray,
    atom_end_arr: np.ndarray,
    prefix_len: np.ndarray,
    device,
    tensors: bool = True,
):
    starts, ends, run_actions, _run_lens = _encode_action_runs(actions_ref, prefix_len)
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
    owner._last_fast_pack_plan = {
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


def _coherent_residual_from_mask(
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


def _coherent_residual_from_prefix(
    prefix_len: np.ndarray,
    prefix_mass: np.ndarray,
    prefix_energy: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    token_len = int(prefix_len[int(end_idx)] - prefix_len[int(start_idx)])
    if token_len <= 0:
        return 0.0
    mass = float(prefix_mass[int(end_idx)] - prefix_mass[int(start_idx)])
    energy = float(prefix_energy[int(end_idx)] - prefix_energy[int(start_idx)])
    coherent_mass = float(mass * mass / max(1.0, float(token_len)))
    return float(max(0.0, min(float(energy), float(2.0 * coherent_mass - energy))))


def _estimate_raw_buyback_credit(
    initial_buyback_prefix_len: np.ndarray,
    initial_buyback_prefix_value: np.ndarray,
    slot_count: int,
) -> float:
    slots = max(0, int(slot_count))
    if slots <= 0 or initial_buyback_prefix_len.size <= 1:
        return 0.0
    pos = int(np.searchsorted(initial_buyback_prefix_len, int(slots), side="right") - 1)
    pos = max(0, min(pos, int(initial_buyback_prefix_value.size) - 1))
    return float(initial_buyback_prefix_value[pos])


def _score_surrogate_packet_gain(
    *,
    value: float,
    sold_loss: float,
    sold_deficit: float,
    budget_delta: int,
    region_open_cost: float,
    raw_slot_price: float,
    initial_buyback_prefix_len: np.ndarray,
    initial_buyback_prefix_value: np.ndarray,
) -> float:
    positive_cost = max(0, int(budget_delta))
    freed_slots = max(0, -int(budget_delta))
    buyback_credit = _estimate_raw_buyback_credit(
        initial_buyback_prefix_len,
        initial_buyback_prefix_value,
        int(freed_slots),
    )
    return (
        float(value)
        - float(sold_loss)
        - float(sold_deficit)
        - float(region_open_cost)
        - float(raw_slot_price) * float(positive_cost)
        + float(buyback_credit)
    )


def _build_full_budget_stats(*, budget_entries: int, full_cost: int, tail_price_scale: float, predictive: bool):
    return {
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


def _append_surrogate_candidate(out: List[tuple], start_idx: int, end_idx: int, seed_count: int, score_candidate_packet) -> None:
    metrics = score_candidate_packet(int(start_idx), int(end_idx))
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


def _append_drop_run_surrogate_candidates(actions_for_runs: np.ndarray, *, action_runs_fn, append_candidate_fn) -> int:
    local_starts, local_ends, local_actions, _local_lens = action_runs_fn(actions_for_runs)
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
        append_candidate_fn(start_idx, end_idx, 1)
        added += 1
    return int(added)


def _finalize_allocation_plan(owner, stats: Dict[str, object], materialize_plan_fn, actions, profile_t0: float):
    if bool(getattr(owner, "_allocator_plan_only", False)):
        materialize_start = time.perf_counter()
        materialized = materialize_plan_fn(actions, tensors=False)
    else:
        materialize_start = time.perf_counter()
        materialized = materialize_plan_fn(actions)
    stats["surrogate_kv_timing_alloc_materialize_seconds"] = float(time.perf_counter() - materialize_start)
    stats["surrogate_kv_timing_alloc_total_seconds"] = float(time.perf_counter() - profile_t0)
    owner._last_allocator_stats = stats
    return materialized


def _build_final_allocation_stats(values: Dict[str, object]) -> Dict[str, object]:
    stats = values["stats"]
    future_signal_arr = values["future_signal_arr"]
    return {
        "surrogate_kv_allocator": "online_frontier_exchange",
        "surrogate_kv_candidate_seeds": int(len(values["seeds"])),
        "surrogate_kv_shadow_candidate_seeds": int(values["shadow_candidate_count"]),
        "surrogate_kv_candidate_surrogates": int(len(values["candidates"])),
        "surrogate_kv_candidate_generated_total": int(values["generated_candidate_count"]),
        "surrogate_kv_market_candidate_limit": int(values["market_candidate_limit"]),
        "surrogate_kv_candidate_generated": int(values["candidate_count"]),
        "surrogate_kv_stack_merges": int(values["stack_merges"]),
        "surrogate_kv_pareto_merge_candidates": int(values["pareto_merge_candidates"]),
        "surrogate_kv_selected_surrogates": int(values["selected_surrogates"]),
        "surrogate_kv_selected_gain": float(values["selected_gain"]),
        "surrogate_kv_selected_value": float(values["selected_value"]),
        "surrogate_kv_sold_raw_atoms": int(values["selected_sold_raw_atoms"]),
        "surrogate_kv_sold_raw_tokens": int(values["selected_sold_raw_tokens"]),
        "surrogate_kv_sold_raw_value": float(values["selected_sold_raw_value"]),
        "surrogate_kv_payer_raw_atoms": int(values["selected_payer_atoms"]),
        "surrogate_kv_payer_raw_tokens": int(values["selected_payer_tokens"]),
        "surrogate_kv_payer_raw_value": float(values["selected_payer_value"]),
        "surrogate_kv_online_buyback_atoms": int(values["online_buyback_atoms"]),
        "surrogate_kv_online_buyback_tokens": int(values["online_buyback_tokens"]),
        "surrogate_kv_online_buyback_value": float(values["online_buyback_value"]),
        "surrogate_kv_final_raw_fill_atoms": int(values["final_raw_fill_atoms"]),
        "surrogate_kv_final_raw_fill_tokens": int(values["final_raw_fill_tokens"]),
        "surrogate_kv_final_raw_fill_value": float(values["final_raw_fill_value"]),
        "surrogate_kv_initial_cost": int(values["initial_cost"]),
        "surrogate_kv_initial_frontier_filled_cost": int(values["initial_frontier_filled_cost"]),
        "surrogate_kv_final_cost": int(stats["ks_run_used_entries"]),
        "surrogate_kv_budget_gap": int(stats["ks_run_budget_gap"]),
        "surrogate_kv_region_open_cost": float(values["region_open_cost"]),
        "surrogate_kv_generation_horizon": int(values["generation_horizon"]),
        "surrogate_kv_horizon_surrogate_tax": float(values["horizon_surrogate_tax"]),
        "surrogate_kv_exact_exchange_open_price": float(values["surrogate_open_price"]),
        "surrogate_kv_raw_slot_price": float(values["raw_slot_price"]),
        "surrogate_kv_exact_exchange": 1,
        "surrogate_kv_post_gap_priced": 1,
        "surrogate_kv_hard_gap_reject": 0,
        "surrogate_kv_tail_price_scale": float(values["tail_price_scale"]),
        "surrogate_kv_predictive": int(bool(values["predictive"])),
        "surrogate_kv_future_signal_mean": float(future_signal_arr.mean()) if future_signal_arr.size else 0.0,
        "surrogate_kv_rejected_overlap": int(values["rejected_overlap"]),
        "surrogate_kv_rejected_budget": int(values["rejected_budget"]),
        "surrogate_kv_rejected_value": int(values["rejected_value"]),
        "surrogate_kv_score_current_packet_calls": int(values["score_current_packet_calls"]),
        "surrogate_kv_micro_len": int(values["micro_len"]),
        "surrogate_kv_no_terminal_repair": 1,
        "ks_run_candidate_seeds": int(len(values["seeds"])),
        "ks_run_shadow_candidate_seeds": int(values["shadow_candidate_count"]),
        "ks_run_candidate_surrogates": int(len(values["candidates"])),
        "ks_run_merge_accepts": int(values["stack_merges"]),
        "ks_run_selected_surrogates": int(values["selected_surrogates"]),
        "ks_run_sold_raw_atoms": int(values["selected_sold_raw_atoms"]),
        "ks_run_sold_raw_tokens": int(values["selected_sold_raw_tokens"]),
        "ks_run_sold_raw_value": float(values["selected_sold_raw_value"]),
        "ks_run_buyback_raw_atoms": int(values["online_buyback_atoms"]),
        "ks_run_buyback_raw_tokens": int(values["online_buyback_tokens"]),
        "ks_run_buyback_raw_value": float(values["online_buyback_value"]),
        "ks_run_region_open_cost": float(values["region_open_cost"]),
        "ks_run_raw_slot_price": float(values["raw_slot_price"]),
        "ks_run_tail_price_scale": float(values["tail_price_scale"]),
        "ks_run_selected_value": float(values["selected_value"]),
        "ks_run_selected_gain": float(values["selected_gain"]),
        "ks_run_rejected_budget": int(values["rejected_budget"]),
        "ks_run_rejected_value": int(values["rejected_value"]),
        "ks_run_merge_terminal_price": 0,
        "ks_run_merge_terminal_keep_frac": float(values["budget_entries"]) / max(1.0, float(values["full_cost"])),
    }

def allocate_surrogate_regions(
    self,
    *,
    token_scores,
    chunk_slices: Sequence[Tuple[int, int]],
    chunk_lengths,
    target_compressed_tokens: int,
    sink_len: int,
    recent_len: int,
    predictive: bool = False,
    _rank01_fn=None,
    _profile_timing: bool | None = None,
):
    """SurrogateKV allocator.

    Single-ledger frontier exchange. The allocator builds one raw frontier,
    generates K-D-K residual packets from that frontier, and admits each
    packet only when the same budget ledger can pay for the surrogate slot.
    Freed slots and any residual whole-atom slack are refilled with raw atoms
    under that ledger.
    """
    del chunk_lengths
    rank01 = _rank01 if _rank01_fn is None else _rank01_fn
    profile_timing = bool(_SURKV_PROFILE_TIMING if _profile_timing is None else _profile_timing)
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
        return _build_allocator_timing_stats(profile_times, profile_t0, enabled=profile_timing)

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
        mean_rank_t = rank01(atom_mean)[0]
        peak_rank_t = rank01(atom_peak)[0]
        surrogate_mean_rank_t = rank01(surrogate_atom_mean)[0]
        surrogate_peak_rank_t = rank01(surrogate_atom_peak)[0]
        surrogate_risk_t = torch.maximum(surrogate_mean_rank_t, surrogate_peak_rank_t)
        current_risk_t = torch.maximum(mean_rank_t, peak_rank_t)
        if bool(predictive):
            atom_spread = torch.clamp(atom_peak - atom_mean, min=0.0)
            spread_rank_t = rank01(atom_spread)[0]
            if atom_mean.shape[1] > 1:
                left_mean = torch.cat((atom_mean[:, :1], atom_mean[:, :-1]), dim=1)
                right_mean = torch.cat((atom_mean[:, 1:], atom_mean[:, -1:]), dim=1)
                neighbor_floor = torch.maximum(left_mean, right_mean)
                local_contrast = torch.clamp(atom_peak - neighbor_floor, min=0.0)
                contrast_rank_t = rank01(local_contrast)[0]
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

    def encode_action_runs(actions_ref: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _encode_action_runs(actions_ref, prefix_len)

    def collect_stats(actions_ref: Sequence[int]) -> Tuple[Dict[str, float], List[int]]:
        return _summarize_region_actions(
            actions_ref,
            prefix_len=prefix_len,
            budget_entries=budget_entries,
            full_cost=full_cost,
        )

    def materialize_region_actions(actions_ref: Sequence[int], *, tensors: bool = True):
        return _materialize_region_actions(
            self,
            actions_ref,
            atom_start_arr=atom_start_arr,
            atom_end_arr=atom_end_arr,
            prefix_len=prefix_len,
            device=device,
            tensors=tensors,
        )

    profile_mark("define_base_helpers")

    if budget_entries >= full_cost:
        actions_full = np.full((num_atoms,), 2, dtype=np.int8)
        stats, _ = collect_stats(actions_full)
        profile_mark("full_budget_stats")
        stats.update(
            _build_full_budget_stats(
                budget_entries=budget_entries,
                full_cost=full_cost,
                tail_price_scale=tail_price_scale,
                predictive=predictive,
            )
        )
        stats.update(profile_export())
        self._last_allocator_stats = stats
        return _finalize_allocation_plan(self, stats, materialize_region_actions, actions_full, profile_t0)

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

    def coherent_residual_from_mask(
        local_mean: np.ndarray,
        local_len: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        return _coherent_residual_from_mask(local_mean, local_len, mask)

    def coherent_residual_from_prefix(start_idx: int, end_idx: int) -> float:
        return _coherent_residual_from_prefix(
            prefix_len,
            prefix_mass,
            prefix_energy,
            start_idx,
            end_idx,
        )

    def score_static_surrogate_gain(
        *,
        value: float,
        sold_loss: float,
        sold_deficit: float,
        budget_delta: int,
    ) -> float:
        return _score_surrogate_packet_gain(
            value=value,
            sold_loss=sold_loss,
            sold_deficit=sold_deficit,
            budget_delta=budget_delta,
            region_open_cost=region_open_cost,
            raw_slot_price=raw_slot_price,
            initial_buyback_prefix_len=initial_buyback_prefix_len,
            initial_buyback_prefix_value=initial_buyback_prefix_value,
        )

    static_surrogate_packet_cache: Dict[Tuple[int, int], Tuple[float, float, float, int, int] | None] = {}

    def score_static_surrogate_packet(start_idx: int, end_idx: int):
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        cache_key = (start_idx, end_idx)
        if cache_key in static_surrogate_packet_cache:
            return static_surrogate_packet_cache[cache_key]
        if start_idx >= end_idx:
            static_surrogate_packet_cache[cache_key] = None
            return None
        token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
        if token_len <= 0:
            static_surrogate_packet_cache[cache_key] = None
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
            coherent_residual_from_prefix(int(start_idx), int(end_idx)),
        )
        budget_delta = int(1 - int(sold_tokens))
        gain = score_static_surrogate_gain(
            value=float(value),
            sold_loss=float(sold_loss),
            sold_deficit=float(sold_deficit),
            budget_delta=int(budget_delta),
        )
        result = (
            float(gain),
            float(value),
            float(sold_loss),
            int(budget_delta),
            int(token_len),
        )
        static_surrogate_packet_cache[cache_key] = result
        return result

    # Candidate tuple:
    # gain, value, sold_loss, budget_delta, token_len, start, end, seed_count
    Candidate = Tuple[float, float, float, int, int, int, int, int]
    starts, ends, run_actions, _run_lens = encode_action_runs(initial_actions)
    seeds: List[Candidate] = []
    candidate_count = 0
    shadow_candidate_count = 0

    def append_surrogate_candidate(out: List[Candidate], start_idx: int, end_idx: int, seed_count: int) -> None:
        _append_surrogate_candidate(out, start_idx, end_idx, seed_count, score_static_surrogate_packet)

    profile_mark("define_candidate_helpers")

    def append_drop_run_candidates(actions_for_runs: np.ndarray) -> int:
        return _append_drop_run_surrogate_candidates(
            actions_for_runs,
            action_runs_fn=encode_action_runs,
            append_candidate_fn=lambda start, end, count: append_surrogate_candidate(seeds, start, end, count),
        )

    candidate_count += append_drop_run_candidates(initial_actions)

    # A stricter RAW frontier keeps broad residual packets available as
    # candidates. Final admission still uses the current budget and ledger.
    nested_frontier_price = float(region_open_cost) * 2.0
    if math.isfinite(nested_frontier_price) and nested_frontier_price > 0.0:
        shadow_actions = np.where(raw_density_arr >= float(nested_frontier_price), 2, 0).astype(np.int8)
        if bool(np.any(shadow_actions == 2)) and bool(np.any(shadow_actions == 0)):
            before_count = int(len(seeds))
            candidate_count += append_drop_run_candidates(shadow_actions)
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
            metrics = score_static_surrogate_packet(int(left_start), int(right_end))
            if metrics is None:
                break
            gain, value, sold_loss, budget_delta, token_len = metrics
            split_gain = float(left[0]) + float(right[0])
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
    buy_cursor = 0

    def buy_raw_until_full() -> None:
        nonlocal buy_cursor
        nonlocal current_cost
        nonlocal online_buyback_atoms, online_buyback_tokens, online_buyback_value
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
            online_buyback_atoms += 1
            online_buyback_tokens += int(atom_len)
            online_buyback_value += float(raw_value_arr[int(atom_idx)])
            buy_cursor += 1

    def buy_raw_best_effort() -> None:
        nonlocal current_cost
        nonlocal online_buyback_atoms, online_buyback_tokens, online_buyback_value
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
            online_buyback_atoms += 1
            online_buyback_tokens += int(atom_len)
            online_buyback_value += float(raw_value_arr[atom_idx])
            if int(current_cost) >= int(budget_entries):
                break

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
            coherent_residual_from_mask(local_mean, local_len, residual_mask),
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
            coherent_residual_from_prefix(int(start_idx), int(end_idx)),
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
        actions[start_idx:end_idx] = 1
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
    stats.update(_build_final_allocation_stats(locals()))
    profile_mark("stats_update")
    stats.update(profile_export())
    self._last_allocator_stats = stats
    return _finalize_allocation_plan(self, stats, materialize_region_actions, actions, profile_t0)

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .common import _rank01


class LayerBudgetMixin:
    @classmethod
    def reset_global_budget_ledger(
        cls,
        *,
        enabled: bool,
        total_layers: int,
        total_capacity: int,
        prompt_tokens: int,
    ) -> None:
        cls._GLOBAL_BUDGET_LEDGER = {
            "enabled": bool(enabled),
            "total_layers": max(0, int(total_layers)),
            "remaining_layers": max(0, int(total_layers)),
            "total_capacity": max(0, int(total_capacity)),
            "remaining_capacity": max(0, int(total_capacity)),
            "total_base_capacity": max(0, int(total_capacity)),
            "remaining_base_capacity": max(0, int(total_capacity)),
            "prompt_tokens": max(0, int(prompt_tokens)),
            "signal_ema": 0.0,
            "true_dynamic_marginal_ema": 0.0,
            "seen_layers": 0,
        }

    @classmethod
    def reset_global_layer_dynamic(
        cls,
        *,
        enabled: bool,
        total_layers: int,
        total_capacity: int,
        prompt_tokens: int,
    ) -> None:
        cls._GLOBAL_LAYER_DYNAMIC_STATE = {
            "enabled": bool(enabled),
            "total_layers": max(0, int(total_layers)),
            "total_capacity": max(0, int(total_capacity)),
            "prompt_tokens": max(0, int(prompt_tokens)),
            "records": [],
            "finalized": False,
        }

    @classmethod
    def _global_layer_dynamic_active(cls) -> bool:
        state = cls._GLOBAL_LAYER_DYNAMIC_STATE
        return bool(state.get("enabled", False))

    def _register_global_layer_dynamic_record(
        self,
        *,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        token_scores: torch.Tensor,
        q_len: int,
        recent_len: int,
        sink_len: int,
        compressible_start: int,
        past_len: int,
        base_capacity_prompt: int,
        num_key_value_groups: int,
    ) -> None:
        state = type(self)._GLOBAL_LAYER_DYNAMIC_STATE
        if not bool(state.get("enabled", False)):
            return
        layer_idx = int(self.global_budget_layer_idx)
        records = list(state.get("records", []))
        records = [record for record in records if int(record.get("layer_idx", -1)) != layer_idx]
        chunk_slices = [(int(compressible_start), int(past_len))]
        chunk_lengths = torch.tensor(
            [max(0, int(past_len) - int(compressible_start))],
            device=key_states.device,
            dtype=torch.long,
        )
        shadow_signal = 1.0
        shadow_stats: Dict[str, float] = {}
        if str(getattr(self, "mode", "")) == "surrogate_kv_dynamic_layer":
            support_signal, support_stats = self._estimate_global_layer_support_signal(
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                target_compressed_tokens=int(base_capacity_prompt),
                sink_len=int(sink_len),
                recent_len=int(recent_len),
            )
            market_signal, market_stats = self._estimate_global_layer_market_signal(
                token_scores=token_scores,
                chunk_slices=chunk_slices,
                target_compressed_tokens=int(base_capacity_prompt),
                sink_len=int(sink_len),
                recent_len=int(recent_len),
            )
            support_signal = float(support_signal) if math.isfinite(float(support_signal)) else 1.0
            market_signal = float(market_signal) if math.isfinite(float(market_signal)) else 1.0
            shadow_signal = math.sqrt(max(1e-9, support_signal) * max(1e-9, market_signal))
            shadow_stats = dict(support_stats or {})
            shadow_stats.update(market_stats or {})
            shadow_stats.update(
                {
                    "surrogate_kv_layer_support_signal": float(support_signal),
                    "surrogate_kv_layer_surrogate_aware_signal": float(market_signal),
                    "surrogate_kv_layer_combined_signal": float(shadow_signal),
                }
            )
        records.append(
            {
                "cluster": self,
                "layer_idx": layer_idx,
                "mode": str(getattr(self, "mode", "")),
                "key_states": key_states,
                "query_states": query_states,
                "value_states": value_states,
                "token_scores": token_scores,
                "q_len": int(q_len),
                "recent_len": int(recent_len),
                "sink_len": int(sink_len),
                "chunk_slices": chunk_slices,
                "chunk_lengths": chunk_lengths,
                "base_capacity": int(base_capacity_prompt),
                "num_key_value_groups": int(num_key_value_groups),
                "shadow_signal": float(shadow_signal),
                "shadow_stats": dict(shadow_stats),
            }
        )
        records.sort(key=lambda record: int(record.get("layer_idx", -1)))
        state["records"] = records
        state["finalized"] = False
        type(self)._GLOBAL_LAYER_DYNAMIC_STATE = state

    @staticmethod
    def _global_layer_dynamic_capacity_grid(
        *,
        q_len: int,
        min_capacity: int,
        base_capacity: int,
    ) -> List[int]:
        q_len = max(1, int(q_len))
        min_capacity = max(1, min(q_len, int(min_capacity)))
        base_capacity = max(min_capacity, min(q_len, int(base_capacity)))
        multipliers = (
            (1, 3),
            (1, 2),
            (2, 3),
            (4, 5),
            (1, 1),
            (5, 4),
            (3, 2),
            (2, 1),
            (5, 2),
        )
        capacities = {min_capacity, base_capacity, q_len}
        for num, den in multipliers:
            capacities.add(max(min_capacity, min(q_len, int(round(float(base_capacity) * float(num) / float(den))))))
        return sorted(capacities)

    @staticmethod
    def _global_layer_dynamic_objective(stats: Dict[str, object]) -> float:
        for key in (
            "ks_run_objective_value",
            "surrogate_kv_solver_score",
            "surrogate_kv_selected_value",
        ):
            value = stats.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        raw = stats.get("ks_run_raw_value", 0.0)
        sur = stats.get("ks_run_surrogate_projection_value", 0.0)
        try:
            return float(raw) + float(sur)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _evaluate_global_layer_dynamic_curve(cls, record: Dict[str, object]) -> List[Dict[str, float]]:
        cluster = record["cluster"]
        q_len = int(record["q_len"])
        min_capacity = max(1, int(record["recent_len"]) + int(record["sink_len"]) + 1)
        capacities = cls._global_layer_dynamic_capacity_grid(
            q_len=q_len,
            min_capacity=min_capacity,
            base_capacity=int(record["base_capacity"]),
        )
        old_stats = dict(getattr(cluster, "_last_allocator_stats", {}) or {})
        old_pack_plan = getattr(cluster, "_last_fast_pack_plan", None)
        curve: List[Dict[str, float]] = []
        for capacity in capacities:
            cluster._last_allocator_stats = {}
            cluster._last_fast_pack_plan = None
            allocated = cluster._allocate_surrogate_regions(
                token_scores=record["token_scores"],
                chunk_slices=record["chunk_slices"],
                chunk_lengths=record["chunk_lengths"],
                target_compressed_tokens=int(capacity),
                sink_len=int(record["sink_len"]),
                recent_len=int(record["recent_len"]),
            )
            stats = dict(getattr(cluster, "_last_allocator_stats", {}) or {})
            utility = cls._global_layer_dynamic_objective(stats)
            if allocated is None:
                utility = 0.0
            actual_entries = stats.get("ks_run_used_entries")
            if actual_entries is None:
                actual_capacity = int(capacity)
            else:
                actual_capacity = int(actual_entries) + int(record["sink_len"]) + int(record["recent_len"])
            actual_capacity = max(min_capacity, min(q_len, int(actual_capacity)))
            curve.append(
                {
                    "capacity": int(actual_capacity),
                    "target_capacity": int(capacity),
                    "utility": float(utility),
                }
            )
        cluster._last_allocator_stats = old_stats
        cluster._last_fast_pack_plan = old_pack_plan
        best_by_capacity: Dict[int, Dict[str, float]] = {}
        for point in curve:
            capacity = int(point["capacity"])
            previous = best_by_capacity.get(capacity)
            if previous is None or float(point["utility"]) > float(previous.get("utility", -math.inf)):
                best_by_capacity[capacity] = dict(point)
        compact = [point for _capacity, point in sorted(best_by_capacity.items())]
        filtered: List[Dict[str, float]] = []
        best_utility = -math.inf
        for point in compact:
            utility = float(point["utility"])
            if utility >= best_utility:
                filtered.append(point)
                best_utility = utility
        return filtered

    @classmethod
    def _select_global_layer_dynamic_capacities(
        cls,
        *,
        records: List[Dict[str, object]],
        total_capacity: int,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[str, float]], int]:
        curves = {
            int(record["layer_idx"]): cls._evaluate_global_layer_dynamic_curve(record)
            for record in records
        }
        selected_index = {layer_idx: 0 for layer_idx in curves}
        selected_cost_capacity = {
            layer_idx: int(points[0]["capacity"]) if points else int(records[idx]["base_capacity"])
            for idx, (layer_idx, points) in enumerate(curves.items())
        }
        selected_capacity = {
            layer_idx: int(points[0].get("target_capacity", points[0]["capacity"])) if points else int(records[idx]["base_capacity"])
            for idx, (layer_idx, points) in enumerate(curves.items())
        }
        used_capacity = sum(int(value) for value in selected_cost_capacity.values())
        remaining = max(0, int(total_capacity) - int(used_capacity))
        heap: List[Tuple[float, int, int, int]] = []
        for layer_idx, points in curves.items():
            if len(points) < 2:
                continue
            cost_delta = int(points[1]["capacity"]) - int(points[0]["capacity"])
            value_delta = float(points[1]["utility"]) - float(points[0]["utility"])
            if cost_delta > 0:
                heapq.heappush(heap, (-value_delta / float(cost_delta), int(layer_idx), 0, 1))
        while heap and remaining > 0:
            _neg_density, layer_idx, from_idx, to_idx = heapq.heappop(heap)
            points = curves[layer_idx]
            if int(selected_index[layer_idx]) != int(from_idx):
                continue
            cost_delta = int(points[to_idx]["capacity"]) - int(points[from_idx]["capacity"])
            if cost_delta <= 0:
                continue
            if cost_delta > remaining:
                continue
            selected_index[layer_idx] = int(to_idx)
            selected_cost_capacity[layer_idx] = int(points[to_idx]["capacity"])
            selected_capacity[layer_idx] = int(points[to_idx].get("target_capacity", points[to_idx]["capacity"]))
            remaining -= int(cost_delta)
            next_idx = int(to_idx) + 1
            if next_idx < len(points):
                next_cost = int(points[next_idx]["capacity"]) - int(points[to_idx]["capacity"])
                next_value = float(points[next_idx]["utility"]) - float(points[to_idx]["utility"])
                if next_cost > 0:
                    heapq.heappush(heap, (-next_value / float(next_cost), int(layer_idx), int(to_idx), int(next_idx)))
        selected_points = {
            layer_idx: dict(curves[layer_idx][int(selected_index[layer_idx])])
            for layer_idx in curves
        }
        return selected_capacity, selected_points, remaining

    @staticmethod
    def _select_global_layer_shadow_price_capacities(
        *,
        records: List[Dict[str, object]],
        total_capacity: int,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[str, float]], int]:
        """Allocate layer budgets with one common shadow price.

        Each layer contributes a single scalar demand signal from the same
        SurrogateKV RAW/SUR/DROP market.  Capacity is then water-filled under a
        shared price, so no layer receives budget because it happened to run
        earlier in the stack.
        """
        if not records:
            return {}, {}, int(total_capacity)

        layer_ids = np.asarray([int(record["layer_idx"]) for record in records], dtype=np.int64)
        base = np.asarray([max(1, int(record.get("base_capacity", 1))) for record in records], dtype=np.float64)
        q_lens = np.asarray([max(1, int(record.get("q_len", 1))) for record in records], dtype=np.float64)
        mins = np.asarray(
            [
                max(1, min(int(record.get("q_len", 1)), int(record.get("recent_len", 0)) + int(record.get("sink_len", 0)) + 1))
                for record in records
            ],
            dtype=np.float64,
        )
        maxs = np.maximum(mins, q_lens)
        target_total = int(round(float(total_capacity)))
        target_total = max(int(np.sum(mins)), min(int(np.sum(maxs)), int(target_total)))

        signals = np.asarray(
            [max(1e-9, float(record.get("shadow_signal", 1.0) or 1.0)) for record in records],
            dtype=np.float64,
        )
        log_signal = np.log(np.maximum(signals, 1e-9))
        center = float(np.mean(log_signal)) if log_signal.size else 0.0
        dispersion = float(np.std(log_signal)) if log_signal.size > 1 else 0.0
        elasticity = float(dispersion / (1.0 + dispersion)) if math.isfinite(dispersion) else 0.0
        prices = np.exp(np.clip(float(elasticity) * (log_signal - float(center)), -6.0, 6.0))
        demand = np.maximum(1e-9, base * prices)

        def capacities_for_lambda(lambda_value: float) -> np.ndarray:
            lam = max(1e-9, float(lambda_value))
            return np.minimum(maxs, np.maximum(mins, demand / lam))

        low = 1e-9
        high = max(1e-9, float(np.max(demand / np.maximum(mins, 1.0))))
        while float(np.sum(capacities_for_lambda(high))) > float(target_total) and high < 1e18:
            high *= 2.0
        for _ in range(64):
            mid = (low + high) * 0.5
            if float(np.sum(capacities_for_lambda(mid))) > float(target_total):
                low = mid
            else:
                high = mid

        raw_capacity = capacities_for_lambda(high)
        floors = np.floor(raw_capacity).astype(np.int64)
        min_int = mins.astype(np.int64)
        max_int = maxs.astype(np.int64)
        floors = np.minimum(max_int, np.maximum(min_int, floors))
        remainder = int(target_total) - int(np.sum(floors))
        fractions = raw_capacity - np.floor(raw_capacity)

        if remainder > 0:
            order = np.lexsort((layer_ids, -fractions))
            cursor = 0
            while remainder > 0 and cursor < len(order) * 4:
                idx = int(order[cursor % len(order)])
                if floors[idx] < max_int[idx]:
                    floors[idx] += 1
                    remainder -= 1
                cursor += 1
        elif remainder < 0:
            order = np.lexsort((layer_ids, fractions))
            cursor = 0
            while remainder < 0 and cursor < len(order) * 4:
                idx = int(order[cursor % len(order)])
                if floors[idx] > min_int[idx]:
                    floors[idx] -= 1
                    remainder += 1
                cursor += 1

        selected_capacity = {int(layer_ids[idx]): int(floors[idx]) for idx in range(len(records))}
        selected_points: Dict[int, Dict[str, float]] = {}
        shadow_price = float(high)
        for idx, record in enumerate(records):
            layer_idx = int(layer_ids[idx])
            capacity = int(selected_capacity[layer_idx])
            min_capacity = int(min_int[idx])
            signal = float(signals[idx])
            utility = float(signal * math.log1p(max(0, capacity - min_capacity)))
            selected_points[layer_idx] = {
                "capacity": int(capacity),
                "target_capacity": int(capacity),
                "utility": float(utility),
                "shadow_signal": float(signal),
                "shadow_price": float(shadow_price),
                "shadow_elasticity": float(elasticity),
                "shadow_dispersion": float(dispersion),
                "shadow_demand": float(demand[idx]),
            }
        remaining = int(target_total) - int(sum(selected_capacity.values()))
        return selected_capacity, selected_points, remaining

    @classmethod
    def finalize_global_layer_dynamic_records(cls):
        state = cls._GLOBAL_LAYER_DYNAMIC_STATE
        records = list(state.get("records", []))
        if not bool(state.get("enabled", False)) or not records:
            return []
        total_capacity = int(state.get("total_capacity", 0))
        if total_capacity <= 0:
            total_capacity = sum(int(record.get("base_capacity", 0)) for record in records)
        use_shadow_price = any(str(record.get("mode", "")) == "surrogate_kv_dynamic_layer" for record in records)
        if bool(use_shadow_price):
            selected_capacity, selected_points, remaining = cls._select_global_layer_shadow_price_capacities(
                records=records,
                total_capacity=total_capacity,
            )
        else:
            selected_capacity, selected_points, remaining = cls._select_global_layer_dynamic_capacities(
                records=records,
                total_capacity=total_capacity,
            )
        updates = []
        for record in records:
            cluster = record["cluster"]
            layer_idx = int(record["layer_idx"])
            target_capacity = int(selected_capacity.get(layer_idx, int(record["base_capacity"])))
            cluster._global_layer_dynamic_finalizing_capacity = int(target_capacity)
            try:
                key_out, value_out = cluster.update_kv(
                    record["key_states"],
                    record["query_states"],
                    record["value_states"],
                    None,
                    int(record["num_key_value_groups"]),
                )
            finally:
                cluster._global_layer_dynamic_finalizing_capacity = None
            stats = dict(getattr(cluster, "last_stats", {}) or {})
            point = selected_points.get(layer_idx, {})
            stats.update(
                {
                    "surkv_layer_dynamic_global_allocator": 1,
                    "surkv_layer_dynamic_layer_idx": int(layer_idx),
                    "surkv_layer_dynamic_base_capacity": int(record.get("base_capacity", target_capacity)),
                    "surkv_layer_dynamic_total_capacity": int(total_capacity),
                    "surkv_layer_dynamic_selected_capacity": int(target_capacity),
                    "surkv_layer_dynamic_selected_utility": float(point.get("utility", 0.0)),
                    "surkv_layer_dynamic_curve_capacity": int(point.get("capacity", target_capacity)),
                    "surkv_layer_dynamic_curve_target_capacity": int(point.get("target_capacity", target_capacity)),
                    "surkv_layer_dynamic_global_layers": int(len(selected_points)),
                    "surkv_layer_dynamic_budget_remaining": int(remaining),
                }
            )
            if bool(use_shadow_price):
                stats.update(
                    {
                        "surrogate_kv_dynamic_layer_shadow_allocator": 1,
                        "surrogate_kv_dynamic_layer_shadow_layer_idx": int(layer_idx),
                        "surrogate_kv_dynamic_layer_shadow_base_capacity": int(record.get("base_capacity", target_capacity)),
                        "surrogate_kv_dynamic_layer_shadow_selected_capacity": int(target_capacity),
                        "surrogate_kv_dynamic_layer_shadow_total_capacity": int(total_capacity),
                        "surrogate_kv_dynamic_layer_shadow_signal": float(point.get("shadow_signal", record.get("shadow_signal", 1.0))),
                        "surrogate_kv_dynamic_layer_shadow_price": float(point.get("shadow_price", 0.0)),
                        "surrogate_kv_dynamic_layer_shadow_elasticity": float(point.get("shadow_elasticity", 0.0)),
                        "surrogate_kv_dynamic_layer_shadow_dispersion": float(point.get("shadow_dispersion", 0.0)),
                        "surrogate_kv_dynamic_layer_shadow_demand": float(point.get("shadow_demand", 0.0)),
                    }
                )
                stats.update(dict(record.get("shadow_stats", {}) or {}))
            cluster.last_stats = stats
            updates.append((layer_idx, key_out, value_out))
        state["records"] = []
        state["finalized"] = True
        cls._GLOBAL_LAYER_DYNAMIC_STATE = state
        return updates

    def finalize_global_layer_dynamic(self):
        return type(self).finalize_global_layer_dynamic_records()

    def _global_budget_ledger_active(self) -> bool:
        if not self.global_budget_ledger:
            return False
        ledger = type(self)._GLOBAL_BUDGET_LEDGER
        if not bool(ledger.get("enabled", False)):
            return False
        if self.global_budget_num_layers <= 0 or self.global_budget_layer_idx < 0:
            return False
        return True

    def _apply_global_budget_ledger(
        self,
        *,
        q_len: int,
        base_capacity_prompt: int,
        min_capacity_prompt: int,
    ) -> int:
        if not self._global_budget_ledger_active():
            return int(base_capacity_prompt)

        ledger = type(self)._GLOBAL_BUDGET_LEDGER
        remaining_layers = max(1, int(ledger.get("remaining_layers", self.global_budget_num_layers) or 1))
        remaining_capacity = max(1, int(ledger.get("remaining_capacity", 0) or 0))
        remaining_base_capacity = max(1, int(ledger.get("remaining_base_capacity", 0) or 0))
        base_capacity = max(1, int(base_capacity_prompt))
        signal = float(getattr(self, "_last_layer_budget_signal", 0.0) or 0.0)
        if not math.isfinite(signal) or signal <= 0.0:
            signal = 1.0
        keep_frac = max(0.0, min(1.0, float(base_capacity) / max(1.0, float(q_len))))
        compression_pressure = max(0.0, min(1.0, 1.0 - float(keep_frac)))
        pressure_relaxed_ledger = (
            str(getattr(self, "mode", "")) == "surrogate_kv_dynamic_layer"
            and not bool(self.global_layer_allocator)
        )
        ledger_strength = float(compression_pressure * compression_pressure) if bool(pressure_relaxed_ledger) else 1.0

        seen_layers = max(0, int(ledger.get("seen_layers", 0) or 0))
        signal_ema = float(ledger.get("signal_ema", 0.0) or 0.0)
        if not math.isfinite(signal_ema) or signal_ema <= 0.0:
            signal_ema = signal
        relative_pressure = math.log(max(signal, 1e-9) / max(signal_ema, 1e-9))

        future_layers_for_bounds = max(0, int(remaining_layers) - 1)
        hard_min_capacity = max(
            int(min_capacity_prompt),
            int(remaining_capacity) - int(future_layers_for_bounds) * int(q_len),
        )
        if bool(pressure_relaxed_ledger):
            relaxed_min = int(
                round(
                    float(min_capacity_prompt)
                    + (float(hard_min_capacity) - float(min_capacity_prompt)) * float(ledger_strength)
                )
            )
            hard_min_capacity = max(int(min_capacity_prompt), min(int(hard_min_capacity), int(relaxed_min)))
        hard_max_capacity = min(
            int(q_len),
            max(int(hard_min_capacity), int(remaining_capacity) - int(future_layers_for_bounds) * int(min_capacity_prompt)),
        )
        hard_min_capacity = max(1, min(int(q_len), int(hard_min_capacity)))
        hard_max_capacity = max(int(hard_min_capacity), min(int(q_len), int(hard_max_capacity)))

        if remaining_layers <= 1:
            if bool(self.global_layer_allocator):
                target_capacity = remaining_capacity
            else:
                surplus = int(remaining_capacity) - int(base_capacity)
                target_capacity = int(base_capacity) + int(round(float(surplus) * float(ledger_strength)))
        elif (
            bool(self.global_layer_allocator)
            and bool(getattr(self, "global_layer_allocator_curve", False))
            and getattr(self, "_last_layer_budget_curve", None)
        ):
            curve = list(getattr(self, "_last_layer_budget_curve", []) or [])
            future_layers = max(0, int(remaining_layers) - 1)
            future_min = int(future_layers) * int(min_capacity_prompt)
            target_max = max(int(min_capacity_prompt), int(remaining_capacity) - int(future_min))
            target_max = min(int(q_len), int(target_max))
            target_min = max(1, min(int(target_max), int(min_capacity_prompt)))
            valid_points = [
                point
                for point in curve
                if int(target_min) <= int(point.get("capacity", 0)) <= int(target_max)
            ]
            if not valid_points:
                target_capacity = max(int(target_min), min(int(target_max), int(round(float(remaining_capacity) / float(remaining_layers)))))
            else:
                current_signal = float(getattr(self, "_last_layer_budget_signal", 0.0) or 0.0)
                marginal_ema = float(ledger.get("true_dynamic_marginal_ema", 0.0) or 0.0)
                if not math.isfinite(marginal_ema) or marginal_ema <= 0.0:
                    marginal_ema = max(float(current_signal), 1e-9)
                best_point = max(
                    valid_points,
                    key=lambda point: (
                        float(point.get("utility", 0.0)) - float(marginal_ema) * float(point.get("capacity", 0)),
                        float(point.get("utility", 0.0)),
                        -abs(int(point.get("capacity", 0)) - int(round(float(remaining_capacity) / float(remaining_layers)))),
                    ),
                )
                target_capacity = int(best_point.get("capacity", target_min))
                target_capacity = max(int(target_min), min(int(target_max), int(target_capacity)))

                selected_marginal = float(best_point.get("avg_marginal", 0.0) or 0.0)
                if not math.isfinite(selected_marginal) or selected_marginal <= 0.0:
                    selected_marginal = float(current_signal)
                next_marginal_ema = (
                    float(selected_marginal)
                    if seen_layers <= 0 or marginal_ema <= 0.0
                    else 0.75 * float(marginal_ema) + 0.25 * float(selected_marginal)
                )
                ledger["true_dynamic_marginal_ema"] = float(next_marginal_ema)
                ledger_stats_extra = {
                    "surrogate_kv_true_dynamic_layer_allocator": 1,
                    "surrogate_kv_true_dynamic_curve_points": int(len(curve)),
                    "surrogate_kv_true_dynamic_target_min": int(target_min),
                    "surrogate_kv_true_dynamic_target_max": int(target_max),
                    "surrogate_kv_true_dynamic_selected_capacity": int(target_capacity),
                    "surrogate_kv_true_dynamic_selected_utility": float(best_point.get("utility", 0.0)),
                    "surrogate_kv_true_dynamic_selected_marginal": float(selected_marginal),
                    "surrogate_kv_true_dynamic_marginal_ema": float(next_marginal_ema),
                }
                self._last_allocator_stats.update(ledger_stats_extra)
        elif bool(self.global_layer_allocator):
            future_layers = max(0, int(remaining_layers) - 1)
            expected_future_signal = max(float(signal_ema), 1e-9)
            denom = max(1e-9, float(signal) + float(future_layers) * float(expected_future_signal))
            target_capacity = int(round(float(remaining_capacity) * float(signal) / float(denom)))
            future_min = int(future_layers) * int(min_capacity_prompt)
            target_max = int(remaining_capacity) - int(future_min)
            if int(target_max) < int(min_capacity_prompt):
                target_max = int(remaining_capacity)
            target_capacity = max(int(hard_min_capacity), min(int(hard_max_capacity), int(target_capacity)))
        else:
            credit_per_layer = (
                (float(remaining_capacity) - float(remaining_base_capacity))
                / float(remaining_layers)
                * float(ledger_strength)
            )
            # Use one bounded adjustment without a global cross-layer top-k.
            swing_limit = max(0, int(round(float(base_capacity) * 0.08 * float(ledger_strength))))
            signal_swing = int(round(max(-1.0, min(1.0, relative_pressure)) * float(swing_limit)))
            target_capacity = int(round(float(base_capacity) + credit_per_layer + float(signal_swing)))

            future_base = max(0, int(remaining_base_capacity) - int(base_capacity))
            future_floor = int(round(float(future_base) * 0.88))
            future_ceiling = int(round(float(future_base) * 1.12))
            exact_upper = max(1, int(remaining_capacity) - future_floor)
            exact_lower = int(remaining_capacity) - future_ceiling
            if target_capacity > exact_upper:
                target_capacity = int(round(float(target_capacity) - (float(target_capacity) - float(exact_upper)) * float(ledger_strength)))
            if target_capacity < exact_lower:
                target_capacity = int(round(float(target_capacity) + (float(exact_lower) - float(target_capacity)) * float(ledger_strength)))

        target_capacity = max(int(hard_min_capacity), min(int(hard_max_capacity), int(target_capacity)))
        target_capacity = max(1, min(int(q_len), int(target_capacity)))

        next_ema = signal if seen_layers <= 0 else (0.875 * signal_ema + 0.125 * signal)
        ledger["signal_ema"] = float(next_ema)
        ledger["seen_layers"] = int(seen_layers + 1)
        ledger_stats = {
            "surrogate_kv_global_ledger_enabled": 1,
            "surrogate_kv_global_ledger_layer_idx": int(self.global_budget_layer_idx),
            "surrogate_kv_global_ledger_base_capacity": int(base_capacity),
            "surrogate_kv_global_ledger_planned_capacity": int(target_capacity),
            "surrogate_kv_global_ledger_delta": int(target_capacity) - int(base_capacity),
            "surrogate_kv_global_ledger_signal": float(signal),
            "surrogate_kv_global_ledger_signal_ema": float(next_ema),
            "surrogate_kv_global_ledger_strength": float(ledger_strength),
            "surrogate_kv_global_layer_allocator": int(bool(self.global_layer_allocator)),
            "surrogate_kv_global_ledger_remaining_layers_before": int(remaining_layers),
            "surrogate_kv_global_ledger_remaining_capacity_before": int(remaining_capacity),
        }
        for key, value in dict(getattr(self, "_last_allocator_stats", {}) or {}).items():
            if str(key).startswith("surrogate_kv_true_dynamic_"):
                ledger_stats[key] = value
        for key, value in dict(getattr(self, "_last_allocator_stats", {}) or {}).items():
            if str(key).startswith("surrogate_kv_layer_market_"):
                ledger_stats[key] = value
            elif str(key).startswith("surrogate_kv_layer_support_"):
                ledger_stats[key] = value
            elif str(key).startswith("surrogate_kv_layer_surrogate_aware_"):
                ledger_stats[key] = value
            elif str(key).startswith("surrogate_kv_layer_combined_"):
                ledger_stats[key] = value
        self._pending_global_budget_ledger_stats = dict(ledger_stats)
        self._last_allocator_stats.update(ledger_stats)
        return int(target_capacity)

    def _commit_global_budget_ledger(
        self,
        *,
        base_capacity_prompt: int,
        planned_capacity_prompt: int,
        actual_capacity_prompt: int,
    ) -> None:
        if not self._global_budget_ledger_active():
            return
        ledger = type(self)._GLOBAL_BUDGET_LEDGER
        remaining_layers = max(1, int(ledger.get("remaining_layers", self.global_budget_num_layers) or 1))
        remaining_capacity = int(ledger.get("remaining_capacity", 0) or 0)
        remaining_base_capacity = int(ledger.get("remaining_base_capacity", 0) or 0)
        next_remaining_layers = max(0, remaining_layers - 1)
        next_remaining_capacity = max(0, remaining_capacity - int(actual_capacity_prompt))
        next_remaining_base = max(0, remaining_base_capacity - int(base_capacity_prompt))
        ledger["remaining_layers"] = int(next_remaining_layers)
        ledger["remaining_capacity"] = int(next_remaining_capacity)
        ledger["remaining_base_capacity"] = int(next_remaining_base)
        ledger_stats = dict(getattr(self, "_pending_global_budget_ledger_stats", {}) or {})
        ledger_stats.update(
            {
                "surrogate_kv_global_ledger_actual_capacity": int(actual_capacity_prompt),
                "surrogate_kv_global_ledger_planned_actual_gap": int(actual_capacity_prompt) - int(planned_capacity_prompt),
                "surrogate_kv_global_ledger_remaining_layers_after": int(next_remaining_layers),
                "surrogate_kv_global_ledger_remaining_capacity_after": int(next_remaining_capacity),
            }
        )
        self._last_allocator_stats.update(ledger_stats)

    def _estimate_global_layer_market_signal(
        self,
        *,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        target_compressed_tokens: int,
        sink_len: int,
        recent_len: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Cheap RAW/SUR/DROP marginal market signal for online layer budgeting."""
        if not chunk_slices or token_scores.shape[0] != 1:
            return 1.0, {}
        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return 1.0, {}
        base_start, base_end = span
        if int(base_end) <= int(base_start):
            return 1.0, {}

        micro_len = max(1, int(self.spec.dynamic_anchor_width or (int(self.chunk_size) // 4)))
        atom_start_arr = np.arange(int(base_start), int(base_end), int(micro_len), dtype=np.int64)
        if atom_start_arr.size <= 0:
            return 1.0, {}
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
        if not mean_parts:
            return 1.0, {}

        atom_mean = torch.cat(mean_parts, dim=0).view(1, -1)
        atom_peak = torch.cat(peak_parts, dim=0).view(1, -1)
        atom_mean_abs_arr = atom_mean[0].detach().cpu().numpy().astype(np.float64)
        atom_peak_abs_arr = atom_peak[0].detach().cpu().numpy().astype(np.float64)
        mean_rank_t = _rank01(atom_mean)[0]
        peak_rank_t = _rank01(atom_peak)[0]
        atom_risk_t = torch.maximum(mean_rank_t, peak_rank_t)
        mean_risk_arr = mean_rank_t.detach().cpu().numpy().astype(np.float64) + 1e-6
        atom_risk_arr = atom_risk_t.detach().cpu().numpy().astype(np.float64) + 1e-6

        num_atoms = int(atom_start_arr.size)
        budget_entries = max(1, int(target_compressed_tokens) - int(sink_len) - int(recent_len))
        full_cost = int(atom_len_int_arr.sum())
        tail_floor = 1.0 / float(max(2, num_atoms + 1))
        mean_signal_arr = -np.log(np.maximum(float(tail_floor), 1.0 - np.clip(mean_risk_arr, 0.0, 1.0)))
        atom_signal_arr = -np.log(np.maximum(float(tail_floor), 1.0 - np.clip(atom_risk_arr, 0.0, 1.0)))
        atom_len_arr = np.maximum(1.0, atom_len_int_arr.astype(np.float64))
        raw_value_arr = atom_signal_arr * atom_signal_arr * atom_len_arr
        raw_density_arr = raw_value_arr / np.maximum(atom_len_arr, 1.0)
        atom_indices_arr = np.arange(num_atoms, dtype=np.int64)
        raw_drop_order = np.lexsort((atom_indices_arr, atom_risk_arr))
        actions = np.full((num_atoms,), 2, dtype=np.int8)
        current_cost = int(full_cost)
        for atom_idx in raw_drop_order.tolist():
            if current_cost <= budget_entries:
                break
            actions[int(atom_idx)] = 0
            current_cost -= int(atom_len_int_arr[int(atom_idx)])

        kept_density = raw_density_arr[actions == 2]
        dropped_density = raw_density_arr[actions == 0]
        remove_price = float(kept_density.min()) if kept_density.size else 0.0
        add_price = float(dropped_density.max()) if dropped_density.size else remove_price
        region_open_cost = max(float(add_price), float(remove_price))

        prefix_len = np.concatenate(([0], np.cumsum(atom_len_int_arr))).astype(np.int64)
        prefix_mass = np.concatenate(([0.0], np.cumsum(mean_signal_arr * atom_len_arr))).astype(np.float64)
        prefix_energy = np.concatenate(([0.0], np.cumsum(mean_signal_arr * mean_signal_arr * atom_len_arr))).astype(np.float64)

        def run_boundaries(actions_ref: np.ndarray):
            if actions_ref.size <= 0:
                return [], [], []
            changes = np.flatnonzero(actions_ref[1:] != actions_ref[:-1]).astype(np.int64) + 1
            starts = np.concatenate((np.asarray([0], dtype=np.int64), changes))
            ends = np.concatenate((changes, np.asarray([actions_ref.size], dtype=np.int64)))
            return starts.tolist(), ends.tolist(), actions_ref[starts].tolist()

        best_surrogate_bid = 0.0
        candidate_count = 0
        starts, ends, run_actions = run_boundaries(actions)
        for run_idx, run_action in enumerate(run_actions):
            if int(run_action) != 0:
                continue
            if int(run_idx) <= 0 or int(run_idx) + 1 >= len(run_actions):
                continue
            if int(run_actions[int(run_idx) - 1]) != 2 or int(run_actions[int(run_idx) + 1]) != 2:
                continue
            start_idx = int(starts[int(run_idx)])
            end_idx = int(ends[int(run_idx)])
            token_len = int(prefix_len[end_idx] - prefix_len[start_idx])
            if token_len <= 0:
                continue
            mass = float(prefix_mass[end_idx] - prefix_mass[start_idx])
            energy = float(prefix_energy[end_idx] - prefix_energy[start_idx])
            coherent_mass = float(mass * mass / max(1.0, float(token_len)))
            value = max(0.0, min(float(energy), float(2.0 * coherent_mass - energy)))
            gain = float(value) - float(region_open_cost)
            if gain <= 0.0:
                continue
            candidate_count += 1
            best_surrogate_bid = max(float(best_surrogate_bid), float(gain) / max(1.0, float(token_len)))

        abs_mean = float(np.maximum(atom_mean_abs_arr, 0.0).mean()) if atom_mean_abs_arr.size else 0.0
        abs_peak = float(np.maximum(atom_peak_abs_arr, 0.0).mean()) if atom_peak_abs_arr.size else 0.0
        sharpness = float(abs_peak / max(abs_mean, 1e-9)) if abs_peak > 0.0 else 1.0
        score_scale = max(1e-9, float(abs_mean + abs_peak) * (1.0 + math.log1p(max(1.0, float(sharpness)))))
        signal = max(float(add_price), float(best_surrogate_bid), 1e-9) * float(score_scale)
        stats = {
            "surrogate_kv_layer_market_signal": float(signal),
            "surrogate_kv_layer_market_add_price": float(add_price),
            "surrogate_kv_layer_market_remove_price": float(remove_price),
            "surrogate_kv_layer_market_surrogate_bid": float(best_surrogate_bid),
            "surrogate_kv_layer_market_score_scale": float(score_scale),
            "surrogate_kv_layer_market_abs_mean": float(abs_mean),
            "surrogate_kv_layer_market_abs_peak": float(abs_peak),
            "surrogate_kv_layer_market_sharpness": float(sharpness),
            "surrogate_kv_layer_market_candidate_count": int(candidate_count),
            "surrogate_kv_layer_market_budget_entries": int(budget_entries),
            "surrogate_kv_layer_market_full_cost": int(full_cost),
            "surrogate_kv_layer_market_used_entries": int(current_cost),
        }
        return float(signal), stats

    def _estimate_global_layer_support_signal(
        self,
        *,
        token_scores,
        chunk_slices: Sequence[Tuple[int, int]],
        target_compressed_tokens: int,
        sink_len: int,
        recent_len: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Cheap DynamicKV-style layer signal from the already computed scores."""
        if not chunk_slices or token_scores.shape[0] != 1:
            return 1.0, {}
        span = self._contiguous_chunk_span(chunk_slices)
        if span is None:
            return 1.0, {}
        base_start, base_end = span
        if int(base_end) <= int(base_start):
            return 1.0, {}

        scores = token_scores[:, int(base_start) : int(base_end)].detach().to(dtype=torch.float32)
        positive = torch.clamp(scores, min=0.0).reshape(-1)
        token_count = int(positive.numel())
        if token_count <= 0:
            return 1.0, {}

        total_mass_t = positive.sum()
        total_mass = float(total_mass_t.item())
        mean_score = float(positive.mean().item())
        peak_score = float(positive.max().item())
        std_score = float(positive.std(unbiased=False).item()) if token_count > 1 else 0.0
        budget_entries = max(1, int(target_compressed_tokens) - int(sink_len) - int(recent_len))
        top_k = max(1, min(int(token_count), int(budget_entries)))
        if total_mass <= 1e-12:
            signal = 1.0
            support_frac = 1.0
            effective_support = float(token_count)
            top_mean = 0.0
            top_mass_frac = 0.0
        else:
            top_values = torch.topk(positive, k=int(top_k), largest=True, sorted=False).values
            top_mean = float(top_values.mean().item())
            top_mass_frac = float((top_values.sum() / torch.clamp(total_mass_t, min=1e-12)).item())
            weights = positive / torch.clamp(total_mass_t, min=1e-12)
            entropy = -torch.sum(weights * torch.log(torch.clamp(weights, min=1e-12)))
            effective_support = float(torch.exp(entropy).item())
            support_frac = max(1.0 / float(token_count), min(1.0, effective_support / float(token_count)))
            evidence_scale = float(top_mean) + 0.5 * float(mean_score) + 0.25 * float(std_score)
            support_scale = math.sqrt(float(support_frac)) * math.log1p(float(effective_support))
            signal = max(1e-9, float(evidence_scale) * float(support_scale))

        sharpness = float(peak_score / max(mean_score, 1e-9)) if peak_score > 0.0 else 1.0
        stats = {
            "surrogate_kv_layer_market_signal": float(signal),
            "surrogate_kv_layer_market_add_price": float(top_mean),
            "surrogate_kv_layer_market_remove_price": float(mean_score),
            "surrogate_kv_layer_market_surrogate_bid": float(support_frac),
            "surrogate_kv_layer_market_score_scale": float(effective_support),
            "surrogate_kv_layer_market_abs_mean": float(mean_score),
            "surrogate_kv_layer_market_abs_peak": float(peak_score),
            "surrogate_kv_layer_market_sharpness": float(sharpness),
            "surrogate_kv_layer_market_candidate_count": int(top_k),
            "surrogate_kv_layer_market_budget_entries": int(budget_entries),
            "surrogate_kv_layer_market_full_cost": int(token_count),
            "surrogate_kv_layer_market_used_entries": int(top_k),
            "surrogate_kv_layer_support_effective_tokens": float(effective_support),
            "surrogate_kv_layer_support_fraction": float(support_frac),
            "surrogate_kv_layer_support_top_mass_fraction": float(top_mass_frac),
        }
        return float(signal), stats

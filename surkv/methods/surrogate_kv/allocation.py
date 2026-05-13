from __future__ import annotations

from dataclasses import dataclass

import torch

from ..base import AllocationPlan, SurrogateContext
from ..utils.atoms import (
    DEFAULT_ATOM_WIDTH,
    AtomAction,
    build_plan_from_actions,
    build_scored_atoms,
    iter_drop_regions,
)


SURROGATE_COVERAGE_DISCOUNT = 0.72


@dataclass(frozen=True)
class _SurrogateCandidate:
    start: int
    end: int
    value: float
    tokens: int

    @property
    def value_per_token(self) -> float:
        return self.value / max(1, self.tokens)


def plan_surrogate_cache(context: SurrogateContext) -> AllocationPlan:
    atoms = build_scored_atoms(context)
    if not atoms.spans:
        return _empty_plan(context)

    scores = atoms.scores[0].detach().to(dtype=torch.float32).cpu().tolist()
    lengths = atoms.lengths.detach().cpu().tolist()
    actions = _initial_keep_actions(scores=scores, lengths=lengths, budget=context.budget_compressible)
    used_tokens = _count_tokens(actions=actions, lengths=lengths, action=AtomAction.KEEP)

    candidates = _candidate_regions(actions=actions, scores=scores, lengths=lengths)
    release_order = _lowest_value_kept_order(actions=actions, scores=scores, lengths=lengths)
    release_cursor = 0
    released_atoms = 0
    released_tokens = 0
    released_value = 0.0

    for candidate in candidates:
        if any(actions[idx] != AtomAction.DROP for idx in range(candidate.start, candidate.end)):
            continue

        needed_tokens = max(0, used_tokens + 1 - int(context.budget_compressible))
        atoms_to_release: list[int] = []
        if needed_tokens > 0:
            atoms_to_release, freed_tokens, freed_value, next_release_cursor = _release_candidate(
                actions=actions,
                lengths=lengths,
                scores=scores,
                release_order=release_order,
                start_cursor=release_cursor,
                target_tokens=needed_tokens,
            )
            if freed_tokens < needed_tokens or candidate.value <= freed_value:
                continue

        for atom_idx in atoms_to_release:
            actions[atom_idx] = AtomAction.DROP
            used_tokens -= int(lengths[atom_idx])
            released_atoms += 1
            released_tokens += int(lengths[atom_idx])
            released_value += float(scores[atom_idx])
        if atoms_to_release:
            release_cursor = next_release_cursor

        if used_tokens + 1 <= int(context.budget_compressible):
            for atom_idx in range(candidate.start, candidate.end):
                actions[atom_idx] = AtomAction.SURROGATE
            used_tokens += 1

    used_tokens = _use_remaining_budget(
        actions=actions,
        scores=scores,
        lengths=lengths,
        budget=context.budget_compressible,
        used_tokens=used_tokens,
    )
    stats = _allocation_stats(
        actions=actions,
        lengths=lengths,
        atom_count=len(atoms.spans),
        candidate_count=len(candidates),
        budget=context.budget_compressible,
        used_tokens=used_tokens,
        released_atoms=released_atoms,
        released_tokens=released_tokens,
        released_value=released_value,
    )
    return build_plan_from_actions(context=context, atoms=atoms, actions=actions, stats=stats)


def _empty_plan(context: SurrogateContext) -> AllocationPlan:
    return AllocationPlan(
        chunk_slices=context.chunk_slices,
        chunk_lengths=context.chunk_lengths,
        replace_mask=torch.zeros_like(context.surrogate_lengths, dtype=torch.bool),
        surrogate_lengths=context.surrogate_lengths.new_zeros(context.surrogate_lengths.shape),
    )


def _candidate_regions(*, actions, scores, lengths) -> list[_SurrogateCandidate]:
    candidates: list[_SurrogateCandidate] = []
    for start, end in iter_drop_regions(actions):
        tokens = _range_tokens(lengths, start, end)
        value = float(sum(scores[start:end])) * SURROGATE_COVERAGE_DISCOUNT
        candidates.append(_SurrogateCandidate(start=start, end=end, value=value, tokens=tokens))
    candidates.sort(key=lambda candidate: candidate.value_per_token, reverse=True)
    return candidates


def _initial_keep_actions(*, scores: list[float], lengths: list[int], budget: int) -> list[AtomAction]:
    actions = [AtomAction.DROP for _ in scores]
    used_tokens = 0
    for atom_idx in sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True):
        atom_tokens = int(lengths[atom_idx])
        if used_tokens + atom_tokens > int(budget):
            continue
        actions[atom_idx] = AtomAction.KEEP
        used_tokens += atom_tokens
    return actions


def _lowest_value_kept_order(*, actions, scores, lengths) -> list[int]:
    kept_atoms = [idx for idx, action in enumerate(actions) if action == AtomAction.KEEP]
    kept_atoms.sort(key=lambda idx: float(scores[idx]) / max(1, int(lengths[idx])))
    return kept_atoms


def _release_candidate(
    *,
    actions,
    lengths,
    scores,
    release_order: list[int],
    start_cursor: int,
    target_tokens: int,
) -> tuple[list[int], int, float, int]:
    selected: list[int] = []
    released = 0
    released_value = 0.0
    cursor = int(start_cursor)
    while cursor < len(release_order) and released < int(target_tokens):
        atom_idx = release_order[cursor]
        cursor += 1
        if actions[atom_idx] != AtomAction.KEEP:
            continue
        selected.append(atom_idx)
        released += int(lengths[atom_idx])
        released_value += float(scores[atom_idx])
    return selected, released, released_value, cursor


def _use_remaining_budget(*, actions, scores, lengths, budget: int, used_tokens: int) -> int:
    fill_order = [idx for idx, action in enumerate(actions) if action == AtomAction.DROP]
    fill_order.sort(key=lambda idx: float(scores[idx]), reverse=True)
    for atom_idx in fill_order:
        atom_tokens = int(lengths[atom_idx])
        if used_tokens + atom_tokens > int(budget):
            continue
        actions[atom_idx] = AtomAction.KEEP
        used_tokens += atom_tokens
    return used_tokens


def _allocation_stats(
    *,
    actions,
    lengths,
    atom_count: int,
    candidate_count: int,
    budget: int,
    used_tokens: int,
    released_atoms: int,
    released_tokens: int,
    released_value: float,
) -> dict[str, object]:
    raw_tokens = 0
    surrogate_tokens = 0
    dropped_tokens = 0
    surrogate_regions = 0
    longest_surrogate = 0

    idx = 0
    while idx < len(actions):
        action = actions[idx]
        end = idx + 1
        while end < len(actions) and actions[end] == action:
            end += 1

        tokens = _range_tokens(lengths, idx, end)
        if action == AtomAction.KEEP:
            raw_tokens += tokens
        elif action == AtomAction.SURROGATE:
            surrogate_regions += 1
            surrogate_tokens += tokens
            longest_surrogate = max(longest_surrogate, tokens)
        else:
            dropped_tokens += tokens
        idx = end

    return {
        "surrogate_kv_allocator": 1,
        "surrogate_kv_atom_width": int(DEFAULT_ATOM_WIDTH),
        "surrogate_kv_atoms": int(atom_count),
        "surrogate_kv_candidate_regions": int(candidate_count),
        "surrogate_kv_selected_regions": int(surrogate_regions),
        "surrogate_kv_raw_tokens": int(raw_tokens),
        "surrogate_kv_covered_tokens": int(surrogate_tokens),
        "surrogate_kv_drop_tokens": int(dropped_tokens),
        "surrogate_kv_released_raw_atoms": int(released_atoms),
        "surrogate_kv_released_raw_tokens": int(released_tokens),
        "surrogate_kv_released_raw_value": float(released_value),
        "surrogate_kv_budget_gap": int(budget) - int(used_tokens),
        "region_mean_len": 0.0 if surrogate_regions == 0 else float(surrogate_tokens / surrogate_regions),
        "region_max_len": int(longest_surrogate),
        "region_count": int(surrogate_regions),
    }


def _count_tokens(*, actions, lengths, action: AtomAction) -> int:
    return sum(int(lengths[idx]) for idx, current in enumerate(actions) if current == action)


def _range_tokens(lengths, start: int, end: int) -> int:
    return sum(int(lengths[idx]) for idx in range(int(start), int(end)))


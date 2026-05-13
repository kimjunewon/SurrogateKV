from __future__ import annotations

import torch

from ..base import AllocationPlan, SurrogateContext
from ..utils.atoms import (
    DEFAULT_ATOM_WIDTH,
    AtomAction,
    build_plan_from_actions,
    build_scored_atoms,
    iter_drop_regions,
)


def plan_drop_only_cache(context: SurrogateContext) -> AllocationPlan:
    atoms = build_scored_atoms(context)
    if not atoms.spans:
        return _empty_plan(context)

    scores = atoms.scores[0].detach().to(dtype=torch.float32).cpu().tolist()
    lengths = atoms.lengths.detach().cpu().tolist()
    actions = _initial_keep_actions(scores=scores, lengths=lengths, budget=context.budget_compressible)

    raw_tokens = _count_tokens(actions=actions, lengths=lengths, action=AtomAction.KEEP)
    dropped_tokens = _count_tokens(actions=actions, lengths=lengths, action=AtomAction.DROP)
    dropped_regions = iter_drop_regions(actions)
    dropped_region_lengths = [_range_tokens(lengths, start, end) for start, end in dropped_regions]
    stats = {
        "surrogate_drop_allocator": 1,
        "surrogate_drop_atom_width": int(DEFAULT_ATOM_WIDTH),
        "surrogate_drop_atoms": int(len(atoms.spans)),
        "surrogate_drop_raw_tokens": int(raw_tokens),
        "surrogate_drop_tokens": int(dropped_tokens),
        "surrogate_drop_budget_gap": int(context.budget_compressible) - int(raw_tokens),
        "region_mean_len": _mean(dropped_region_lengths),
        "region_max_len": int(max(dropped_region_lengths, default=0)),
        "region_count": int(len(dropped_regions)),
    }
    return build_plan_from_actions(context=context, atoms=atoms, actions=actions, stats=stats)


def _empty_plan(context: SurrogateContext) -> AllocationPlan:
    return AllocationPlan(
        chunk_slices=context.chunk_slices,
        chunk_lengths=context.chunk_lengths,
        replace_mask=torch.zeros_like(context.surrogate_lengths, dtype=torch.bool),
        surrogate_lengths=context.surrogate_lengths.new_zeros(context.surrogate_lengths.shape),
    )


def _count_tokens(*, actions, lengths, action: AtomAction) -> int:
    return sum(int(lengths[idx]) for idx, current in enumerate(actions) if current == action)


def _range_tokens(lengths, start: int, end: int) -> int:
    return sum(int(lengths[idx]) for idx in range(int(start), int(end)))


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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

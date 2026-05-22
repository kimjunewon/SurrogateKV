from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch

from .registry_base import AllocationPlan, ChunkSlice, SurrogateContext


DEFAULT_ATOM_WIDTH = 4


class AtomAction(IntEnum):
    DROP = 0
    KEEP = 1
    SURROGATE = 2


@dataclass(frozen=True)
class ScoredAtoms:
    spans: list[ChunkSlice]
    lengths: torch.Tensor
    scores: torch.Tensor


def build_scored_atoms(context: SurrogateContext, *, width: int = DEFAULT_ATOM_WIDTH) -> ScoredAtoms:
    width = max(1, int(width))
    start_token = int(context.sink_len)
    end_token = int(context.past_len)
    token_count = max(0, end_token - start_token)
    if token_count <= 0:
        lengths = torch.empty((0,), device=context.key_states.device, dtype=torch.long)
        scores = context.token_scores.new_empty((context.key_states.shape[0], 0))
        return ScoredAtoms(spans=[], lengths=lengths, scores=scores)

    atom_count = (token_count + width - 1) // width
    spans = [
        (start, min(start + width, end_token))
        for start in range(start_token, end_token, width)
    ]
    lengths = torch.full((atom_count,), width, device=context.key_states.device, dtype=torch.long)
    tail_length = token_count - width * (atom_count - 1)
    lengths[-1] = int(tail_length)

    token_scores = context.token_scores[:, start_token:end_token]
    padding = atom_count * width - token_count
    if padding == 0:
        atom_scores = token_scores.reshape(token_scores.shape[0], atom_count, width)
        mean_scores = atom_scores.mean(dim=-1)
        peak_scores = atom_scores.max(dim=-1).values
        return ScoredAtoms(spans=spans, lengths=lengths, scores=0.5 * mean_scores + 0.5 * peak_scores)

    regular_scores = []
    regular_peaks = []
    regular_count = atom_count - 1
    if regular_count > 0:
        regular_tokens = regular_count * width
        regular_atoms = token_scores[:, :regular_tokens].reshape(token_scores.shape[0], regular_count, width)
        regular_scores.append(regular_atoms.mean(dim=-1))
        regular_peaks.append(regular_atoms.max(dim=-1).values)

    tail = token_scores[:, regular_count * width :]
    regular_scores.append(tail.mean(dim=-1, keepdim=True))
    regular_peaks.append(tail.max(dim=-1, keepdim=True).values)
    mean_scores = torch.cat(regular_scores, dim=-1)
    peak_scores = torch.cat(regular_peaks, dim=-1)
    return ScoredAtoms(spans=spans, lengths=lengths, scores=0.5 * mean_scores + 0.5 * peak_scores)


def iter_drop_regions(actions: Sequence[AtomAction]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = None
    for idx, action in enumerate(actions):
        if action == AtomAction.DROP and start is None:
            start = idx
        elif action != AtomAction.DROP and start is not None:
            regions.append((start, idx))
            start = None
    if start is not None:
        regions.append((start, len(actions)))
    return regions


def build_plan_from_actions(
    *,
    context: SurrogateContext,
    atoms: ScoredAtoms,
    actions: Sequence[AtomAction],
    stats: dict[str, object],
) -> AllocationPlan:
    output_spans: list[ChunkSlice] = []
    output_lengths: list[int] = []
    replace_flags: list[bool] = []
    surrogate_lengths: list[int] = []

    idx = 0
    while idx < len(actions):
        action = AtomAction(actions[idx])
        end_idx = idx + 1
        while end_idx < len(actions) and AtomAction(actions[end_idx]) == action:
            end_idx += 1

        start_token = atoms.spans[idx][0]
        end_token = atoms.spans[end_idx - 1][1]
        output_spans.append((start_token, end_token))
        output_lengths.append(end_token - start_token)

        replace_flags.append(action != AtomAction.KEEP)
        surrogate_lengths.append(1 if action == AtomAction.SURROGATE else 0)
        idx = end_idx

    device = context.key_states.device
    chunk_lengths = torch.tensor(output_lengths, device=device, dtype=torch.long)
    replace_row = torch.tensor(replace_flags, device=device, dtype=torch.bool)
    surrogate_row = torch.tensor(surrogate_lengths, device=device, dtype=torch.long)
    batch_size = context.key_states.shape[0]

    return AllocationPlan(
        chunk_slices=output_spans,
        chunk_lengths=chunk_lengths,
        replace_mask=replace_row.unsqueeze(0).expand(batch_size, -1).contiguous(),
        surrogate_lengths=surrogate_row.unsqueeze(0).expand(batch_size, -1).contiguous(),
        allocator_stats=stats,
    )


@dataclass(frozen=True)
class _SurrogateCandidate:
    start: int
    end: int
    value: float
    tokens: int
    coherence: float

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
        candidates=candidates,
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
        value, coherence = _surrogate_region_value(
            scores=scores,
            lengths=lengths,
            start=start,
            end=end,
        )
        candidates.append(
            _SurrogateCandidate(
                start=start,
                end=end,
                value=value,
                tokens=tokens,
                coherence=coherence,
            )
        )
    candidates.sort(key=lambda candidate: candidate.value_per_token, reverse=True)
    return candidates


def _surrogate_region_value(*, scores, lengths, start: int, end: int) -> tuple[float, float]:
    """Value a one-token surrogate by score coherence instead of a fixed discount."""

    start = int(start)
    end = int(end)
    if start >= end:
        return 0.0, 0.0

    local_scores = [max(0.0, float(score)) for score in scores[start:end]]
    local_lengths = [max(1, int(length)) for length in lengths[start:end]]
    base_value = float(sum(local_scores))
    if base_value <= 0.0:
        return 0.0, 0.0

    token_count = float(sum(local_lengths))
    if token_count <= 0.0:
        return 0.0, 0.0

    # A mean surrogate preserves a region best when score mass is coherent
    # across the covered atoms.  The projection term is maximal for uniform
    # mass and shrinks for peaky/evidence-like regions that should stay raw.
    mass = sum(score * length for score, length in zip(local_scores, local_lengths))
    energy = sum(score * score * length for score, length in zip(local_scores, local_lengths))
    if energy <= 1e-12:
        return 0.0, 0.0
    projection = float(mass * mass) / max(1.0, token_count)
    coherent_energy = max(0.0, min(float(energy), float(2.0 * projection - energy)))
    coherence = max(0.0, min(1.0, coherent_energy / max(float(energy), 1e-12)))
    return float(base_value * (coherence**0.5)), float(coherence)


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
    candidates,
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
        "surrogate_kv_dynamic_surrogate_value": 1,
        "surrogate_kv_fixed_coverage_discount": 0,
        "surrogate_kv_atom_width": int(DEFAULT_ATOM_WIDTH),
        "surrogate_kv_atoms": int(atom_count),
        "surrogate_kv_candidate_regions": int(candidate_count),
        "surrogate_kv_candidate_mean_coherence": (
            0.0 if not candidates else float(sum(float(item.coherence) for item in candidates) / len(candidates))
        ),
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

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch

from ..base import AllocationPlan, ChunkSlice, SurrogateContext


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

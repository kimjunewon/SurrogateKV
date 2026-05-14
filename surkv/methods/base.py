from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

import torch


ChunkSlice = Tuple[int, int]


@dataclass(frozen=True)
class SurrogateContext:
    key_states: torch.Tensor
    value_states: torch.Tensor
    token_scores: torch.Tensor
    chunk_scores: torch.Tensor
    chunk_slices: Sequence[ChunkSlice]
    chunk_lengths: torch.Tensor
    replace_mask: torch.Tensor
    surrogate_lengths: torch.Tensor
    past_len: int
    sink_len: int
    recent_len: int
    local_radius: int
    tokens_to_save: int
    budget_compressible: int
    chunk_size: int


@dataclass(frozen=True)
class AllocationPlan:
    chunk_slices: Sequence[ChunkSlice]
    chunk_lengths: torch.Tensor
    replace_mask: torch.Tensor
    surrogate_lengths: torch.Tensor
    allocator_stats: dict[str, object] = field(default_factory=dict)


SurrogateBuilder = Callable[[SurrogateContext], tuple[torch.Tensor, torch.Tensor]]
ChunkPlanner = Callable[[SurrogateContext], AllocationPlan]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    mode: str
    build_surrogates: SurrogateBuilder | None = None
    protected_sink: bool = False
    null_fastpath: bool = False
    plan_chunks: ChunkPlanner | None = None
    direct_strategy: str = "local"
    surrogate_mode: str = "mean"
    selection_strategy: str = ""
    weighted_mapping: bool = False
    anchor_residual: bool = False
    dynamic_regioning: bool = False
    dynamic_allocator: str = "legacy"
    dynamic_surrogate_variant: str = ""
    dynamic_victim_max_len: int = 0
    dynamic_micro_risk: str = "max"
    dynamic_anchor_width: int = 0
    dynamic_coverage_block: int = 512

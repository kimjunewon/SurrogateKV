from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MethodSpec:
    name: str
    mode: str
    surrogate_mode: str = "norm_rms_mean"
    dynamic_regioning: bool = True
    dynamic_allocator: str = "surrogate_kv"
    dynamic_surrogate_variant: str = ""
    dynamic_anchor_width: int = 4
    score_method: str = "attention"
    head_score_fusion: str = "mean"


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_str(name: str, default: str) -> str:
    raw = str(os.environ.get(name, "") or "").strip()
    return raw if raw else str(default)


SURROGATE_KV = MethodSpec(
    name="SurrogateKV",
    mode="surrogate_kv",
    surrogate_mode=_env_str("SURKV_SURROGATE_MODE", "norm_rms_mean"),
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=_env_int("SURKV_ATOM_SIZE", 4),
)

SURROGATE_KV_ADA = MethodSpec(
    name="SurrogateKV-Ada",
    mode="surrogate_kv_ada",
    surrogate_mode=_env_str("SURKV_SURROGATE_MODE", "norm_rms_mean"),
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=_env_int("SURKV_ATOM_SIZE", 4),
    score_method="attention",
    head_score_fusion="ada_shared",
)

SURROGATE_KV_DYNAMIC = MethodSpec(
    name="SurrogateKV-Dynamic",
    mode="surrogate_kv_dynamic_layer",
    surrogate_mode=_env_str("SURKV_SURROGATE_MODE", "norm_rms_mean"),
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=_env_int("SURKV_ATOM_SIZE", 4),
    score_method="attention",
    head_score_fusion="mean",
)

SUPPORTED_SPECS = [
    SURROGATE_KV,
    SURROGATE_KV_ADA,
    SURROGATE_KV_DYNAMIC,
]
MODE_TO_SPEC: dict[str, MethodSpec] = {spec.mode: spec for spec in SUPPORTED_SPECS}
METHOD_TO_MODE: dict[str, str] = {spec.name.lower(): spec.mode for spec in SUPPORTED_SPECS}
METHOD_TO_MODE.update(
    {
        "surrogatekv": SURROGATE_KV.mode,
        "surrogate_kv": SURROGATE_KV.mode,
        "surkv": SURROGATE_KV.mode,
        "surrogatekvsnap": SURROGATE_KV.mode,
        "surrogatekv-snap": SURROGATE_KV.mode,
        "surrogatekv_snap": SURROGATE_KV.mode,
        "surrogate_kv_snap": SURROGATE_KV.mode,
        "surkvsnap": SURROGATE_KV.mode,
        "surkv-snap": SURROGATE_KV.mode,
        "surkv_snap": SURROGATE_KV.mode,
        "surrogatekvada": SURROGATE_KV_ADA.mode,
        "surrogatekv-ada": SURROGATE_KV_ADA.mode,
        "surrogate_kv_ada": SURROGATE_KV_ADA.mode,
        "surkvada": SURROGATE_KV_ADA.mode,
        "surkv-ada": SURROGATE_KV_ADA.mode,
        "surrogatekvdynamic": SURROGATE_KV_DYNAMIC.mode,
        "surrogatekv-dynamic": SURROGATE_KV_DYNAMIC.mode,
        "surrogatekv_dynamic": SURROGATE_KV_DYNAMIC.mode,
        "surrogate_kv_dynamic": SURROGATE_KV_DYNAMIC.mode,
        "surkvdynamic": SURROGATE_KV_DYNAMIC.mode,
        "surkv-dynamic": SURROGATE_KV_DYNAMIC.mode,
        "surkv_dynamic": SURROGATE_KV_DYNAMIC.mode,
        "surrogatekvdynamiclayer": SURROGATE_KV_DYNAMIC.mode,
        "surrogatekv_dynamic_layer": SURROGATE_KV_DYNAMIC.mode,
        "surrogate_kv_dynamic_layer": SURROGATE_KV_DYNAMIC.mode,
        "surkvdynamiclayer": SURROGATE_KV_DYNAMIC.mode,
        "surkv_dynamic_layer": SURROGATE_KV_DYNAMIC.mode,
    }
)


def override_method_specs(*, surrogate_mode: str | None = None, dynamic_anchor_width: int | None = None) -> None:
    """Apply process-local ablation overrides to all SurrogateKV variants."""
    updates = {}
    if surrogate_mode:
        updates["surrogate_mode"] = str(surrogate_mode)
    if dynamic_anchor_width is not None and int(dynamic_anchor_width) > 0:
        updates["dynamic_anchor_width"] = int(dynamic_anchor_width)
    if not updates:
        return
    for mode, spec in list(MODE_TO_SPEC.items()):
        MODE_TO_SPEC[mode] = replace(spec, **updates)


__all__ = [
    "METHOD_TO_MODE",
    "MethodSpec",
    "MODE_TO_SPEC",
    "override_method_specs",
    "SUPPORTED_SPECS",
    "SURROGATE_KV",
    "SURROGATE_KV_ADA",
    "SURROGATE_KV_DYNAMIC",
]

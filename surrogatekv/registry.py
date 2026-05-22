from __future__ import annotations

from .allocation import plan_surrogate_cache
from .prototypes import build_chunk_exact_surrogates
from .registry_base import MethodSpec


SURROGATE_KV = MethodSpec(
    name="SurrogateKV",
    mode="surrogate_kv",
    build_surrogates=build_chunk_exact_surrogates,
    plan_chunks=plan_surrogate_cache,
    direct_strategy="local",
    surrogate_mode="norm_rms_mean",
    selection_strategy="dynamic",
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=4,
)

SURROGATE_KV_ADA = MethodSpec(
    name="SurrogateKV-Ada",
    mode="surrogate_kv_ada",
    build_surrogates=build_chunk_exact_surrogates,
    plan_chunks=plan_surrogate_cache,
    direct_strategy="local",
    surrogate_mode="norm_rms_mean",
    selection_strategy="dynamic",
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=4,
    score_method="attention",
    head_score_fusion="ada_shared",
)

SURROGATE_KV_DYNAMIC = MethodSpec(
    name="SurrogateKV-Dynamic",
    mode="surrogate_kv_dynamic_layer",
    build_surrogates=build_chunk_exact_surrogates,
    plan_chunks=plan_surrogate_cache,
    direct_strategy="local",
    surrogate_mode="norm_rms_mean",
    selection_strategy="dynamic",
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=4,
    score_method="attention",
    head_score_fusion="mean",
)

SUPPORTED_SPECS = [SURROGATE_KV, SURROGATE_KV_ADA, SURROGATE_KV_DYNAMIC]
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


__all__ = [
    "METHOD_TO_MODE",
    "MODE_TO_SPEC",
    "SUPPORTED_SPECS",
    "SURROGATE_KV",
    "SURROGATE_KV_ADA",
    "SURROGATE_KV_DYNAMIC",
]

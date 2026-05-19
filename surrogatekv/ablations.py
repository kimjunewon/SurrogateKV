from __future__ import annotations

from .registry_base import MethodSpec


SURROGATE_KV_RAWDROP = MethodSpec(
    name="SurrogateKV-RawDrop",
    mode="surrogate_kv_rawdrop",
    kind="surrogate",
    surrogate_mode="norm_rms_mean",
    dynamic_regioning=True,
    dynamic_allocator="ablation_rawdrop",
    dynamic_anchor_width=4,
)

SURROGATE_KV_ALLSUR = MethodSpec(
    name="SurrogateKV-AllSur",
    mode="surrogate_kv_allsur",
    kind="surrogate",
    surrogate_mode="norm_rms_mean",
    dynamic_regioning=True,
    dynamic_allocator="ablation_allsur",
    dynamic_anchor_width=4,
)

# Legacy chunk-drop adaptation. Kept runnable for comparisons, but grouped with
# ablations so the final SurrogateKV spec stays separate.
SURROGATE_KV_DROP = MethodSpec(
    name="SurrogateKV-Drop",
    mode="surrogate_kv_drop",
    kind="drop",
    null_fastpath=True,
    dynamic_allocator="surrogate_drop",
    dynamic_anchor_width=4,
)


ABLATION_SPECS = [
    SURROGATE_KV_RAWDROP,
    SURROGATE_KV_ALLSUR,
    SURROGATE_KV_DROP,
]

ABLATION_METHOD_TO_MODE = {
    "surrogatekv-rawdrop": SURROGATE_KV_RAWDROP.mode,
    "surrogatekv_rawdrop": SURROGATE_KV_RAWDROP.mode,
    "surrogate_kv_rawdrop": SURROGATE_KV_RAWDROP.mode,
    "surrogatekv-prune": SURROGATE_KV_RAWDROP.mode,
    "surrogatekv_prune": SURROGATE_KV_RAWDROP.mode,
    "surrogate_kv_prune": SURROGATE_KV_RAWDROP.mode,
    "surrogatekv-allsur": SURROGATE_KV_ALLSUR.mode,
    "surrogatekv_allsur": SURROGATE_KV_ALLSUR.mode,
    "surrogate_kv_allsur": SURROGATE_KV_ALLSUR.mode,
    "surrogatekv-nodrop": SURROGATE_KV_ALLSUR.mode,
    "surrogatekv_nodrop": SURROGATE_KV_ALLSUR.mode,
    "surrogate_kv_nodrop": SURROGATE_KV_ALLSUR.mode,
    "surrogatekv-drop": SURROGATE_KV_DROP.mode,
    "surrogatekv_drop": SURROGATE_KV_DROP.mode,
    "surrogate_kv_drop": SURROGATE_KV_DROP.mode,
}


__all__ = [
    "ABLATION_METHOD_TO_MODE",
    "ABLATION_SPECS",
    "SURROGATE_KV_ALLSUR",
    "SURROGATE_KV_DROP",
    "SURROGATE_KV_RAWDROP",
]

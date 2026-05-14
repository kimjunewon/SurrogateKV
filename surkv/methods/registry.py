from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    mode: str
    kind: str
    surrogate_mode: str = "mean"
    protected_sink: bool = False
    null_fastpath: bool = False
    dynamic_regioning: bool = False
    dynamic_allocator: str = ""
    dynamic_anchor_width: int = 0


SURROGATE_KV = MethodSpec(
    name="SurrogateKV",
    mode="surrogate_kv",
    kind="surrogate",
    surrogate_mode="norm_rms_mean",
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=4,
)

SURROGATE_KV_DROP = MethodSpec(
    name="SurrogateKV-Drop",
    mode="surrogate_kv_drop",
    kind="drop",
    null_fastpath=True,
    dynamic_allocator="surrogate_drop",
    dynamic_anchor_width=4,
)


SUPPORTED_SPECS = [SURROGATE_KV, SURROGATE_KV_DROP]

MODE_TO_SPEC: dict[str, MethodSpec] = {spec.mode: spec for spec in SUPPORTED_SPECS}
METHOD_TO_MODE: dict[str, str] = {
    "surrogatekv": SURROGATE_KV.mode,
    "surrogate_kv": SURROGATE_KV.mode,
    "surkv": SURROGATE_KV.mode,
    "surrogatekv-drop": SURROGATE_KV_DROP.mode,
    "surrogatekv_drop": SURROGATE_KV_DROP.mode,
    "surrogate_kv_drop": SURROGATE_KV_DROP.mode,
}


__all__ = ["METHOD_TO_MODE", "MODE_TO_SPEC", "SUPPORTED_SPECS", "MethodSpec"]

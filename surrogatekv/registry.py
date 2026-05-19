from __future__ import annotations

from .ablations import ABLATION_METHOD_TO_MODE, ABLATION_SPECS
from .registry_base import MethodSpec


SURROGATE_KV = MethodSpec(
    name="SurrogateKV",
    mode="surrogate_kv",
    kind="surrogate",
    surrogate_mode="norm_rms_mean",
    dynamic_regioning=True,
    dynamic_allocator="surrogate_kv",
    dynamic_anchor_width=4,
)

SUPPORTED_SPECS = [SURROGATE_KV, *ABLATION_SPECS]

MODE_TO_SPEC: dict[str, MethodSpec] = {spec.mode: spec for spec in SUPPORTED_SPECS}
METHOD_TO_MODE: dict[str, str] = {
    "surrogatekv": SURROGATE_KV.mode,
    "surrogate_kv": SURROGATE_KV.mode,
    "surkv": SURROGATE_KV.mode,
}
METHOD_TO_MODE.update(ABLATION_METHOD_TO_MODE)


__all__ = ["METHOD_TO_MODE", "MODE_TO_SPEC", "SUPPORTED_SPECS", "MethodSpec"]

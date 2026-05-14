from __future__ import annotations

from .base import MethodSpec
from .surrogate_kv import SPEC as SURROGATE_KV
from .surrogate_kv_drop import SPEC as SURROGATE_KV_DROP


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


__all__ = ["METHOD_TO_MODE", "MODE_TO_SPEC", "SUPPORTED_SPECS"]

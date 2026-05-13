from __future__ import annotations

from .base import MethodSpec
from .global_ import SPEC as GLOBAL
from .local import SPEC as LOCAL
from .null import SPEC as NULL
from .surrogate_kv import SPEC as SURROGATE_KV
from .surrogate_kv_drop import SPEC as SURROGATE_KV_DROP


SUPPORTED_SPECS = [
    NULL,
    SURROGATE_KV,
    SURROGATE_KV_DROP,
    GLOBAL,
    LOCAL,
]

MODE_TO_SPEC: dict[str, MethodSpec] = {spec.mode: spec for spec in SUPPORTED_SPECS}
METHOD_TO_MODE: dict[str, str] = {spec.name.lower(): spec.mode for spec in SUPPORTED_SPECS}
METHOD_TO_MODE.update(
    {
        "surkv": SURROGATE_KV.mode,
    }
)


__all__ = ["METHOD_TO_MODE", "MODE_TO_SPEC", "SUPPORTED_SPECS"]

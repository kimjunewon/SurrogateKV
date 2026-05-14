from __future__ import annotations

from ..base import MethodSpec


SPEC = MethodSpec(
    name="SurKVNull",
    mode="null",
    null_fastpath=True,
    direct_strategy="null",
)

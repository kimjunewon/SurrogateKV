from .core import SurKVCluster
from .registry import METHOD_TO_MODE as SURROGATEKV_METHOD_TO_MODE

SURKV_METHOD_TO_MODE = SURROGATEKV_METHOD_TO_MODE

__version__ = "0.1.0"

__all__ = ["SURKV_METHOD_TO_MODE", "SURROGATEKV_METHOD_TO_MODE", "SurKVCluster", "__version__"]

#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
if str(RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN_ROOT))

from _workspace_dispatch import dispatch_tool


dispatch_tool("run_surkv_longbench.py")

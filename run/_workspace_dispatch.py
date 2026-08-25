from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

SURROGATEKV_ROOT = Path(__file__).resolve().parents[1]


def _candidate_workspace_roots() -> list[Path]:
    raw_roots = [
        os.environ.get("SURKV_WORKSPACE_ROOT"),
        os.environ.get("SURKV_ROOT"),
        str(SURROGATEKV_ROOT.parent.parent),
        str(SURROGATEKV_ROOT.parent),
    ]
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_roots:
        if not raw:
            continue
        root = Path(raw).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return roots


def resolve_workspace_root() -> Path:
    for root in _candidate_workspace_roots():
        if (root / "tools" / "run_surkv_longbench.py").exists() and (root / "repos" / "KVCache-Factory").is_dir():
            return root
    searched = "\n".join(f"- {root}" for root in _candidate_workspace_roots())
    raise SystemExit(
        "Could not locate the SurKV experiment workspace.\n"
        "Set SURKV_WORKSPACE_ROOT to the directory containing tools/ and repos/KVCache-Factory.\n"
        f"Searched:\n{searched}"
    )


def dispatch_tool(tool_name: str) -> None:
    workspace_root = resolve_workspace_root()
    tool_path = workspace_root / "tools" / tool_name
    if not tool_path.exists():
        raise SystemExit(f"Missing workspace runner: {tool_path}")

    kvcf_root = workspace_root / "repos" / "KVCache-Factory"
    os.environ.setdefault("SURKV_WORKSPACE_ROOT", str(workspace_root))
    os.environ.setdefault("SURROGATEKV_ROOT", str(SURROGATEKV_ROOT))
    os.environ.setdefault("SURKV_KVCF_ROOT", str(kvcf_root))

    for path in reversed((workspace_root / "tools", SURROGATEKV_ROOT, workspace_root / "repos", kvcf_root)):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    previous_argv = sys.argv
    sys.argv = [str(tool_path), *sys.argv[1:]]
    try:
        runpy.run_path(str(tool_path), run_name="__main__")
    finally:
        sys.argv = previous_argv

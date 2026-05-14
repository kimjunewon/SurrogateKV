# SurrogateKV

SurrogateKV is the standalone repository for the paper-facing runtime. The
Python package is named `surrogatekv` so the public code reads like the method
name, while benchmark harnesses can still use the shorter `surkv` method alias.

```python
from surrogatekv import SurKVCluster, SURROGATEKV_METHOD_TO_MODE
```

## Layout

- `surrogatekv/`: flat runtime package for the paper-facing implementation
- `surrogatekv/core.py`: cache update orchestration
- `surrogatekv/allocation.py`: SurrogateKV raw/surrogate/drop allocator
- `surrogatekv/prototypes.py`: surrogate prototype construction
- `surrogatekv/packing.py`: compressed KV packing for SurrogateKV and Drop
- `surrogatekv/registry.py`: method names, aliases, and optional Drop registration
- `run/longbench/`: LongBench prediction, evaluation, and example launch
  scripts, matching the shape of external reference repos such as DynamicKV
- `data/`: empty dataset slots, kept only as public-repo placeholders

This repository keeps the paper-facing method code and minimal runnable
LongBench entrypoints together. Local scratch scripts, model weights, logs, and
generated outputs should still live outside this repo in the experiment
workspace.

## Install

```bash
python -m pip install -e /path/to/SurrogateKV
```

Quick import check:

```bash
python - <<'PY'
from surrogatekv import SurKVCluster, SURROGATEKV_METHOD_TO_MODE

print(sorted(SURROGATEKV_METHOD_TO_MODE))
PY
```

LongBench example, after placing JSONL files under `data/LongBench/`:

```bash
export SURKV_WORKSPACE_ROOT=/path/to/SurKV-experiment-workspace
python run/longbench/pred.py \
  --dataset qasper \
  --data_file data/LongBench/qasper.jsonl \
  --save_dir results \
  --model_path /path/to/model \
  --method SurrogateKV \
  --max_capacity_prompts 512
```

## Integration

SurrogateKV is not a fork of KVCache-Factory or KVPress. External harnesses
should import this package and connect it through a small adapter:

```text
benchmark harness
  -> attention hook / KV adapter
  -> import surrogatekv
  -> SurKVCluster.update_kv(...)
```

In the local experiment workspace, benchmark orchestration and compatibility
tools live outside this repo:

```text
run/longbench/pred.py
  -> ../../tools/run_surkv_longbench.py
../../tools/run_surkv_ruler.py
../../tools/longbench_baselines.py
../../tools/ruler_baselines.py
```

The KVCache-Factory bridge also stays outside this repository so the SurrogateKV
repo remains publishable as the method implementation itself.

## Notes

- Keep only source code, minimal packaging files, DynamicKV-style LongBench
  entrypoints, and empty data slots here.
- Do not commit model weights, datasets, experiment logs, summaries, plots, or
  workspace-only helper tools.
- New SurrogateKV variants should start as a small `registry.py` spec and keep any
  new implementation code in the runtime module it actually belongs to.

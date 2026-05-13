# SurrogateKV

SurrogateKV is the standalone repository for the SurKV runtime. The import
package is intentionally named `surkv` so benchmark harnesses can attach the
method without vendoring this repository.

```python
from surkv import SurKVCluster, SURKV_METHOD_TO_MODE
```

## Layout

- `surkv/`: core runtime package and method implementations
- `surkv/core.py`: shared tensor packing and cache update logic
- `surkv/methods/`: active methods, currently `SurrogateKV`,
  `SurrogateKV-Drop`, `SurKVNull`, `SurKVGlobal`, and `SurKVLocal`
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
from surkv import SurKVCluster, SURKV_METHOD_TO_MODE

print(sorted(SURKV_METHOD_TO_MODE))
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
  -> import surkv
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
- New SurKV variants should live under `surkv/methods/<method_name>/` and be
  registered in `surkv/methods/registry.py`.

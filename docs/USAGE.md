# Installation and usage

[Back to SurrogateKV](../README.md)

Run the commands below from the repository root. For the allocation profiles, see [Variants](METHOD.md#variants).

## Contents

- `surrogatekv/`: allocation, surrogate construction, packing, and runtime API
- `run/longbench/`: LongBench wrappers and evaluator
- `data/`: machine-readable values for reported tables and selected figures
- `tests/`: runtime and registry checks
- `scripts/validate_release.py`: consistency checks for released result files
- `scripts/build_readme_figures.py`: README figures generated from released CSVs

Model weights, benchmark datasets, generated responses, and machine-specific
logs are not included.

## Installation

```bash
git clone https://github.com/kimjunewon/SurrogateKV.git
cd SurrogateKV
python3 -m pip install -e .
```

Python 3.10 or newer is required. Install the LongBench dependencies with:

```bash
python3 -m pip install -e ".[longbench]"
```

The core framework versions reported for the paper experiments are recorded in
[`docs/PAPER_ENVIRONMENT.md`](PAPER_ENVIRONMENT.md). They are historical
environment metadata; use the standard installation above for ordinary use.

## Runtime API

The cache adapter calls `SurKVCluster` after prefill attention scores are
available. Tensors use the standard `[batch, heads, sequence, head_dim]`
layout. The current prefill path supports batch size 1.

```python
from surrogatekv import SurKVCluster

cluster = SurKVCluster(
    mode="surrogate_kv",
    window_size=32,
    max_capacity_prompt=512,
    kernel_size=7,
)

compressed_k, compressed_v = cluster.update_kv(
    key_states,
    query_states,
    value_states,
    attention_mask=None,
    num_key_value_groups=num_key_value_groups,
)
```

The paper setting uses allocation atoms of size `c = 4`, controlled by
`SURKV_ATOM_SIZE` and used by default. The runtime's `chunk_size` is a separate
coarse-region parameter and defaults to 32.

The Ada-KV variant preserves a head-specific cache layout and therefore uses
`update_kv_headwise()` instead of `update_kv()`. Calling the shared-token entry
point in `surrogate_kv_ada` mode raises an error.

## Evaluation

The result CSVs and saved-prediction evaluator are self-contained. End-to-end
LongBench prediction generation dispatches to the KVCache-Factory adapter used
in the paper, which lives in the companion experiment workspace and is not
vendored into this repository. Set `SURKV_WORKSPACE_ROOT` to a workspace that
contains `tools/run_surkv_longbench.py` and `repos/KVCache-Factory` before using
the prediction wrappers.

```bash
export SURKV_WORKSPACE_ROOT=/path/to/SurKV
export LONGBENCH_DATA_DIR=/path/to/LongBench
export MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct
export METHOD=SurrogateKV
export KV_BUDGETS=128,512

bash run/longbench/scripts/run_llama/run_llama3_8b_instruct_surkv.sh
```

Saved predictions can be evaluated independently:

```bash
python3 run/longbench/eval.py \
  --results_dir runs/longbench/meta-llama-3-8b-instruct_budget_128 \
  --datasets qasper,multifieldqa_en,hotpotqa \
  --methods SurrogateKV,SurrogateKV-Dynamic,SurrogateKV-Ada
```

Evaluate one model-and-budget directory at a time; prediction outputs are
stored as `<model>_budget_<B>/<dataset>/<method>.json`.

See [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the evaluation
scope, reported environment, and released artifacts.

## Development

```bash
python3 -m compileall -q surrogatekv run scripts
python3 -m unittest discover -s tests -v
python3 -m ruff check surrogatekv run tests scripts
```

# SurrogateKV

Official implementation and result exports for **SurrogateKV:
Representation-Preserving KV Cache Compression for Long-Context LLMs**.

Conventional KV-cache compressors decide which states to retain and which to
evict. SurrogateKV adds a third option under the same cache-slot budget: a
lower-salience contiguous region can be represented by one independently
addressable surrogate KV pair. High-salience states remain exact RAW entries,
and regions that are not admitted remain dropped.

## Highlights

- **Fixed-budget representation:** allocates cache capacity among exact RAW,
  regional SURROGATE, and DROP states.
- **One-time construction:** builds and packs surrogate entries after prefill;
  decoding then uses ordinary attention over standard KV tensors.
- **Parent-aware variants:** supports shared-token, layer-adaptive, and
  head-adaptive cache layouts.
- **Result exports:** includes compact LongBench tables and Mistral
  Needle-in-a-Haystack (NIAH) grids.

## Repository Layout

```text
surrogatekv/
  core.py                # SurKVCluster runtime entry point
  registry.py            # method registry and aliases
  schedule.py            # layer-budget schedules
  runtime/
    region_allocator.py  # RAW / SURROGATE / DROP allocation
    prototype_bank.py    # surrogate KV construction
    cache_pipeline.py    # scoring, packing, and layout metadata
    headwise_runtime.py  # head-aware Ada-KV path
    layer_budget.py      # cross-layer budget coordination
    common.py            # shared tensor helpers and flags
run/longbench/           # prediction and evaluation entry points
data/                    # compact result exports and figures
tests/                   # CPU unit and smoke tests
```

Model weights, benchmark datasets, full generations, logs, and local scratch
artifacts are intentionally excluded.

## Installation

```bash
git clone https://github.com/kimjunewon/SurrogateKV.git
cd SurrogateKV
python3 -m pip install -e .
```

The runtime requires Python 3.10 or newer, PyTorch, and NumPy. LongBench
evaluation additionally uses `transformers`, `datasets`, `rouge`, `jieba`, and
`fuzzywuzzy`; install those optional dependencies with
`python3 -m pip install -e ".[longbench]"`.

Quick import check:

```bash
python3 - <<'PY'
from surrogatekv import SurKVCluster, SURROGATEKV_METHOD_TO_MODE

print(sorted(SURROGATEKV_METHOD_TO_MODE))
print(SurKVCluster(mode="surrogate_kv").mode)
PY
```

## Paper Variants

| Method | Internal mode | Parent layout |
| --- | --- | --- |
| `SurrogateKV`, `SurrogateKV-Snap` | `surrogate_kv` | Shared-token SnapKV-style layout |
| `SurrogateKV-Ada` | `surrogate_kv_ada` | Head-adaptive Ada-KV layout |
| `SurrogateKV-Dynamic` | `surrogate_kv_dynamic_layer` | Layer-adaptive DynamicKV layout |

Common lowercase aliases are available through
`SURROGATEKV_METHOD_TO_MODE`.

## Runtime API

`SurKVCluster` is called by an attention hook or cache adapter after the
prefill attention scores are available. Inputs use the standard
`[batch, heads, sequence, head_dim]` layout.

```python
from surrogatekv import SurKVCluster

cluster = SurKVCluster(
    mode="surrogate_kv",
    window_size=32,
    max_capacity_prompt=512,
    kernel_size=7,
    chunk_size=16,
)

compressed_k, compressed_v = cluster.update_kv(
    key_states,
    query_states,
    value_states,
    attention_mask=None,
    num_key_value_groups=num_key_value_groups,
)
```

Ada-KV uses a head-specific cache layout. Its adapter must call the dedicated
head-aware entry point:

```python
ada_cluster = SurKVCluster(
    mode="surrogate_kv_ada",
    window_size=32,
    max_capacity_prompt=512,
    kernel_size=7,
    chunk_size=16,
)

compressed_k, compressed_v = ada_cluster.update_kv_headwise(
    key_states,
    query_states,
    value_states,
    attention_mask=None,
    num_key_value_groups=num_key_value_groups,
)
```

Calling shared-token `update_kv()` in `surrogate_kv_ada` mode raises an error,
preventing accidental conversion of the head-specific layout into a shared
one.

## LongBench

The launch scripts use the KVCache-Factory attention adapter from the companion
experiment workspace. Set `SURKV_WORKSPACE_ROOT` to a checkout containing
`tools/run_surkv_longbench.py` and `repos/KVCache-Factory`, and set
`LONGBENCH_DATA_DIR` to the LongBench JSONL directory.

```bash
export SURKV_WORKSPACE_ROOT=/path/to/SurKV
export LONGBENCH_DATA_DIR=/path/to/LongBench
export MODEL_PATH=/path/to/meta-llama--Meta-Llama-3-8B-Instruct
export METHOD=SurrogateKV
export KV_BUDGETS=128,512
export DATASETS_CSV=qasper,multifieldqa_en,hotpotqa

bash run/longbench/scripts/run_llama/run_llama3_8b_instruct_surkv.sh
```

Evaluate saved predictions with:

```bash
python3 run/longbench/eval.py \
  --results_dir runs/longbench \
  --datasets qasper,multifieldqa_en,hotpotqa \
  --methods SurrogateKV,SurrogateKV-Ada,SurrogateKV-Dynamic
```

## Results

The repository contains compact CSV exports for the paper tables and figures.
See [`data/README.md`](data/README.md) for file-level details.

### LongBench, LLaMA-3-8B-Instruct, $B_{\mathrm{KV}}=512$

| Method | Average | FullKV retention (%) |
| --- | ---: | ---: |
| FullKV | 41.92 | 100.00 |
| H2O | 39.68 | 94.65 |
| SnapKV | 40.26 | 96.04 |
| PyramidKV | 40.18 | 95.84 |
| DynamicKV | 40.60 | 96.86 |
| **SurrogateKV-Snap** | **40.88** | **97.52** |

Source: [`table1_longbench_k512.csv`](data/longbench/llama3_8b_instruct/table1_longbench_k512.csv)

### Mistral-7B-Instruct-v0.2 NIAH, $B_{\mathrm{KV}}=128$

| Parent method | Parent score | SurrogateKV variant | Variant score |
| --- | ---: | --- | ---: |
| SnapKV | 87.51 | SurrogateKV-Snap | **98.84** |
| DynamicKV | 98.46 | SurrogateKV-Dynamic | **98.74** |
| Ada-KV | 90.04 | SurrogateKV-Ada | **98.18** |

Each score averages 1,560 evaluated placements spanning context lengths from
1K to 32K tokens and needle depths from 0% to 100%.

Source: [`niah_average_table.csv`](data/niah/mistral_7b_instruct_v02/k128_ctx1000_32000_step200/niah_average_table.csv)

[Mistral NIAH parent and SurrogateKV comparison (PDF)](data/images/mistral_niah_k128_method_comparison.pdf)

The corrected head-aware Ada evaluation and its scope are documented in
[`CORRECTIONS.md`](CORRECTIONS.md).

## Development

Run the CPU checks with:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q surrogatekv run
```

Keep raw benchmark outputs and machine-specific paths outside this repository.
Public aliases belong in `surrogatekv/registry.py`; runtime implementation
details belong under `surrogatekv/runtime/`.

## Citation

BibTeX will be added when the publication metadata is finalized.

## Acknowledgements

This repository follows the evaluation and integration conventions of prior
KV-cache compression projects, including SnapKV, H2O, Ada-KV, DynamicKV, and
PyramidKV/KVCache-Factory.

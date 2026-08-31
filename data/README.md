# Experiment Data

This directory contains compact CSV exports for the camera-ready paper. Raw
benchmark inputs, model checkpoints, generated responses, temporary tensors,
and local run logs are excluded.

## File Index

### LongBench

`longbench/llama3_8b_instruct/`

| File | Contents |
| --- | --- |
| `budget_scores.csv` | Six-budget averages for all baseline and SurrogateKV variants |
| `per_dataset_scores.csv` | Full per-dataset scores, including FullKV and all allocation groups |
| `category_scores.csv` | Workload-category scores by method and budget |
| `allocation_by_budget.csv` | Raw coverage, surrogate coverage, drop rate, and resident surrogate-slot rate |

LongBench columns use the official metric for each dataset. `average`/`Avg`
is the unweighted mean across the listed datasets. A surrogate coverage rate
measures source positions represented by surrogates; it is distinct from the
number of resident surrogate slots.

### Needle-in-a-Haystack

`niah/mistral_7b_instruct_v02/`

- `k64_ctx1000_32000_step200/`
- `k128_ctx1000_32000_step200/`

Each directory contains a compact average table and one heatmap CSV for each
matched parent/SurrogateKV pair. Rows are needle depths, columns are context
lengths, and cells are retrieval accuracy on a 0--100 scale. The corrected
head-aware grids are the values used in the camera-ready paper; see
[`../CORRECTIONS.md`](../CORRECTIONS.md).

### Analysis and Ablations

| Path | Contents |
| --- | --- |
| `analysis/motivation_summary.csv` | Attention-mass change and KV-region coherence diagnostics |
| `attention/k64_workload_summary.csv` | Attention-shift summaries by workload and method |
| `ablations/longbench_b512.csv` | Allocation, constructor, and atom-size ablations |

The ablation table uses 20 examples per workload across the 16 listed
LongBench workloads, matching the paper's ablation setting. `full_method`
denotes selective admission with the mean-plus-norm constructor and atom size
`c = 4`.

### Comparisons, Efficiency, and Scaling

| Path | Contents |
| --- | --- |
| `comparisons/merging_longbench_summary.csv` | CaM, D2O, and SurrogateKV-Snap averages at two settings |
| `comparisons/merging_longbench_per_dataset.csv` | Per-dataset values for the same comparison |
| `efficiency/llama3_8b_instruct_b128.csv` | TTFT, decode throughput, latency, and resident KV footprint |
| `scaling/qwen25_longbench.csv` | Matched Qwen2.5-14B/72B LongBench task results |

D2O uses its released retention-ratio parameterization targeting approximately
128 and 512 states; CaM and SurrogateKV use fixed slot budgets. The serving
CSV uses `B_KV = 128`; resident-KV columns exclude temporary construction
buffers, as specified in the paper.

### Figures

`images/` contains archival PDFs and GitHub-preview SVGs for the method
overview, the LongBench budget sweep, and the Mistral NIAH comparison.

## Validation

From the repository root:

```bash
python3 scripts/validate_release.py
```

The validator checks schemas, row coverage, key paper values, LongBench
cross-file agreement, NIAH heatmap means, and accidental local paths.

# Experiment Data

This directory keeps compact CSV exports for the experiments reported with this
codebase, plus selected heatmap images used by the paper. Raw benchmark
datasets, per-example generations, logs, and local workspace paths are
intentionally excluded.

## LongBench

`longbench/llama3_8b_instruct/` contains aggregate Llama-3-8B-Instruct results,
including the SurrogateKV-Snap, SurrogateKV-Ada, and SurrogateKV-Dynamic budget
curves and raw/surrogate/drop allocation summaries.

## Needle-in-a-Haystack

`niah/mistral_7b_instruct_v02/k128_ctx1000_32000_step200/` contains the
corrected B_KV=128 Mistral-7B-Instruct-v0.2 average scores and heatmap grids for
SnapKV, DynamicKV, AdaKV, and the corresponding SurrogateKV variants.
`niah/mistral_7b_instruct_v02/k64_ctx1000_32000_step200/` contains the corrected
B_KV=64 SurrogateKV-Ada average and heatmap grid. See `../CORRECTIONS.md` for
the correction scope. The corrected runs used a different worker-concurrency
layout, so their runner wall time is marked `NA` rather than compared with the
timing column from the original exports.

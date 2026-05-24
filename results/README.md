# Result Exports

This directory keeps compact CSV exports for the experiments reported with this
codebase. Raw benchmark datasets, per-example generations, logs, and local
workspace paths are intentionally excluded.

## LongBench

`longbench/llama3_8b_instruct/` contains aggregate Llama-3-8B-Instruct results,
including the SurrogateKV-Snap, SurrogateKV-Ada, and SurrogateKV-Dynamic budget
curves and raw/surrogate/drop allocation summaries.

Source workspace:
`/home/junwon/SurKV/experiments/longbench/llama3_8b_instruct/aggregate/exports/`

## Needle-in-a-Haystack

`niah/mistral_7b_instruct_v02/k128_ctx1000_32000_step200/` contains K=128
Mistral-7B-Instruct-v0.2 average scores and heatmap grids for SnapKV, DynamicKV,
AdaKV, and the corresponding SurrogateKV variants.

Source workspace:
`/home/junwon/SurKV/experiments/niah/mistral_7b_instruct_v02/k128_ctx1000_32000_step200/exports/`

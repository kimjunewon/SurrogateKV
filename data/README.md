# Experiment Data

This directory contains compact, machine-readable exports used to reproduce
the reported tables and figures. Raw datasets, model outputs, logs, and local
workspace paths are excluded.

## LongBench

`longbench/llama3_8b_instruct/` contains:

- the six-budget accuracy curves;
- the $B_{\mathrm{KV}}=512$ task-group summary;
- per-dataset appendix values; and
- RAW, SURROGATE, and DROP allocation summaries.

Rows are aggregate scores unless a filename explicitly identifies token counts
or rates. The `SurrogateKV` row in the main table export denotes the
SnapKV-parent variant (`SurrogateKV-Snap`).

## Needle-in-a-Haystack

`niah/mistral_7b_instruct_v02/` contains Mistral-7B-Instruct-v0.2 grids. Each
heatmap CSV stores needle depth in the first column and one column per context
length. The `k128_ctx1000_32000_step200/` directory contains all parent and
SurrogateKV variants. The `k64_ctx1000_32000_step200/` directory contains the
corrected head-aware SurrogateKV-Ada export used in the correction audit.

## Figures

`images/mistral_niah_k128_method_comparison.pdf` is the current paired
comparison linked from the README. The dispatch-correction panel is retained
alongside it for traceability. Both figures use PDF to preserve vector text for
papers and slides.

The July 2026 Ada dispatch correction is described in
[`../CORRECTIONS.md`](../CORRECTIONS.md).

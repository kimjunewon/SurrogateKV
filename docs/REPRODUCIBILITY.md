# Reproducibility Notes

## Scope

This repository releases the SurrogateKV runtime package, LongBench evaluator,
paper-facing result data, and consistency checks. Model weights, benchmark
inputs, generated outputs, and machine-specific logs are not redistributed.

The scripts under `run/longbench/scripts/` are the launch interface used by the
experiment workspace. They require a companion checkout containing:

```text
tools/run_surkv_longbench.py
repos/KVCache-Factory/
```

Set `SURKV_WORKSPACE_ROOT` to that checkout. The wrapper exits with an explicit
message if the adapter is unavailable. Result CSVs and `run/longbench/eval.py`
do not require the companion workspace.

## Reported Setup

- LongBench model: LLaMA-3-8B-Instruct
- NIAH model: Mistral-7B-Instruct-v0.2
- Main cache budgets: `B_KV` in `{64, 128, 256, 512, 1024, 2048}`
- Recent-query window: `32` for the shared-token and head-wise profiles; `8`
  for the DynamicKV layer-wise profile
- Local pooling width: `7` (max pooling for the shared-token and head-wise
  profiles; average pooling for the DynamicKV layer-wise profile)
- Allocation atom size: `4`
- Decoding: deterministic, using each benchmark's official prompt and output settings
- Hardware: NVIDIA A100 80GB PCIe; Intel Xeon Gold 6338 CPU
- Core software: PyTorch 2.11.0; Transformers 4.44.2

One retained raw KV pair and one surrogate KV pair each consume one resident
slot. Generated-token KV is appended identically for all methods.

All three variants construct mean-pooled, norm-calibrated surrogates in the
paper experiments and by default in the runtime. For head-wise runs,
`SURKV_HEADWISE_SURROGATE_PROTO` may be left unset or set to `mean`.
The explicit `peak` override selects a highest-salience representative and
is an experimental alternative, not the paper configuration.

The current prefill compression paths use evaluation batch size 1. Prediction
outputs are grouped by model and budget; pass one such directory (for example,
`runs/longbench/meta-llama-3-8b-instruct_budget_128`) to
`run/longbench/eval.py`.

## Released Checks

The data validator has no third-party dependencies:

```bash
python3 scripts/validate_release.py
```

With the runtime dependencies installed:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q surrogatekv run scripts
```

README figures can be regenerated from the released CSVs with:

```bash
python3 -m pip install -e ".[plots]"
python3 scripts/build_readme_figures.py
```

## Data Provenance

The files under `data/` are compact exports of the reported tables and selected
plots. `data/README.md` records the scope and interpretation of each file.

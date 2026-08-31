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
- Local max-pooling window: `7`
- Allocation atom size: `4`
- Decoding: deterministic, using each benchmark's official prompt and output settings
- Hardware: NVIDIA A100 80GB PCIe; Intel Xeon Gold 6338 CPU
- Core software: PyTorch 2.11.0; Transformers 4.44.2

One retained raw KV pair and one surrogate KV pair each consume one resident
slot. Generated-token KV is appended identically for all methods.

The shared-token and layer-wise modes construct mean-pooled, norm-calibrated
surrogates. The released head-wise profile selects the highest-salience
representative KV pair within each admitted region. Set
`SURKV_HEADWISE_SURROGATE_PROTO=mean` to select mean-plus-norm construction for
head-wise runs.

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

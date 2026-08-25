# Evaluation Corrections

## Mistral NIAH SurrogateKV-Ada (July 13, 2026)

While investigating the inconsistent NIAH behavior highlighted during review,
we found that the Mistral NIAH evaluator dispatched SurrogateKV-Ada through the
shared-token cache path rather than the per-head path required by Ada-KV. This
collapsed head-specific capacities and selections into a common token layout.

We reran the complete affected grid after correcting only this dispatch. The
model, data, cache budgets, evaluation protocol, and method hyperparameters
were unchanged.

| $B_{\mathrm{KV}}$ | Ada-KV | Previous SurrogateKV-Ada | Corrected SurrogateKV-Ada |
| ---: | ---: | ---: | ---: |
| 64 | 72.13 | 92.60 | **97.60** |
| 128 | 90.04 | 84.66 | **98.18** |

At $B_{\mathrm{KV}}=128$, the average boundary-depth score changes from 37.56 to 99.56.
The correction affects only the Mistral NIAH SurrogateKV-Ada row and its
derived heatmaps. LongBench and all other NIAH rows are unchanged.

The corrected parent/variant comparison is shown below; select the preview to
open the [vector PDF](data/images/mistral_niah_k128_method_comparison.pdf).

<p align="center">
  <a href="data/images/mistral_niah_k128_method_comparison.pdf">
    <img src="data/images/mistral_niah_k128_method_comparison.svg" alt="Corrected Mistral NIAH parent and SurrogateKV comparison" width="680">
  </a>
</p>

The original-path audit panel is retained below for traceability; select it to
open the [vector PDF](data/images/mistral_niah_ada_dispatch_correction.pdf).

<p align="center">
  <a href="data/images/mistral_niah_ada_dispatch_correction.pdf">
    <img src="data/images/mistral_niah_ada_dispatch_correction.svg" alt="Mistral NIAH Ada dispatch correction audit" width="760">
  </a>
</p>

The runtime now rejects shared-token `update_kv()` calls for
`surrogate_kv_ada`; integrations must use `update_kv_headwise()` so that the
per-head layout is preserved.

# Head-Wise NIAH Evaluation Note

## Mistral NIAH SurrogateKV-Ada

SurrogateKV-Ada must use the per-head cache path required by Ada-KV. The
released Mistral NIAH grids preserve head-specific capacities and selections;
their average scores are 97.60 at $B_{\mathrm{KV}}=64$ and 98.18 at
$B_{\mathrm{KV}}=128$.

The released parent/variant comparison is shown below; select the preview to
open the [vector PDF](data/images/mistral_niah_k128_method_comparison.pdf).

<p align="center">
  <a href="data/images/mistral_niah_k128_method_comparison.pdf">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="data/images/mistral_niah_k128_method_comparison-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="data/images/mistral_niah_k128_method_comparison.svg">
      <img src="data/images/mistral_niah_k128_method_comparison.svg" alt="Mistral NIAH parent and SurrogateKV comparison" width="760">
    </picture>
  </a>
</p>

The runtime rejects shared-token `update_kv()` calls for
`surrogate_kv_ada`; integrations must use `update_kv_headwise()` so that the
per-head layout is preserved.

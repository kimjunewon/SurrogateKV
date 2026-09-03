# Paper Environment

The paper experiments used the following core framework versions:

| Package | Version |
| --- | --- |
| PyTorch | 2.11.0 |
| Transformers | 4.44.2 |

These versions document the historical experiment environment. They are not a
recommendation for a new deployment and may no longer contain current security
fixes. For ordinary use, install the package and optional evaluation
dependencies through `pyproject.toml`:

```bash
python3 -m pip install -e ".[longbench]"
```

The LongBench evaluator additionally uses Accelerate, Datasets, fuzzywuzzy,
jieba, rouge, and tqdm. Hardware and benchmark settings are recorded in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

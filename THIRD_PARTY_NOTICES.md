# Third-Party Notices

The notices below apply only to the identified third-party-derived portions of
this repository. They do not license SurrogateKV's original code,
documentation, data, figures, or other original material.

## SnapKV

- Source: <https://github.com/FasterDecoding/SnapKV>
- License: Apache License 2.0
- Full license text: [`legal/third-party/Apache-2.0.txt`](legal/third-party/Apache-2.0.txt)
- Modified portions: `surrogatekv/core.py`,
  `surrogatekv/runtime/cache_pipeline.py`, and
  `surrogatekv/runtime/headwise_runtime.py`

The identified portions were adapted and substantially modified for
SurrogateKV's scoring, cache representation, and runtime integration.

## Hugging Face Transformers

- Source: <https://github.com/huggingface/transformers>
- Copyright: Copyright 2022 EleutherAI and the HuggingFace Inc. team. All
  rights reserved.
- License: Apache License 2.0
- Full license text: [`legal/third-party/Apache-2.0.txt`](legal/third-party/Apache-2.0.txt)
- Modified portion: `surrogatekv/runtime/common.py` (`_repeat_kv_heads`)

## AdaKV

- Source: <https://github.com/FFY0/AdaKV>
- Copyright: Copyright (c) 2024 Yuan Feng
- License: MIT License
- Full license text: [`legal/third-party/AdaKV-MIT.txt`](legal/third-party/AdaKV-MIT.txt)
- Modified portions: `surrogatekv/core.py`,
  `surrogatekv/runtime/cache_pipeline.py`, and
  `surrogatekv/runtime/headwise_runtime.py`

## LongBench

- Source: <https://github.com/THUDM/LongBench>
- Copyright: Copyright (c) 2023 THU-KEG & Zhipu AI
- License: MIT License
- Full license text: [`legal/third-party/LongBench-MIT.txt`](legal/third-party/LongBench-MIT.txt)
- Modified portions: `run/longbench/metrics.py` and
  `run/longbench/eval.py`

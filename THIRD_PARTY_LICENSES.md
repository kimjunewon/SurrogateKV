# Third-Party Licenses

## Incorporated or Adapted Code

The following projects informed code distributed in this repository:

- [LongBench](https://github.com/THUDM/LongBench), Copyright (c) 2023
  THU-KEG & Zhipu AI. `run/longbench/metrics.py` and the evaluation flow in
  `run/longbench/eval.py` are adapted from its official evaluator.
- [PyramidKV/KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory),
  Copyright (c) 2024 Zefan Cai.
- [AdaKV](https://github.com/FFY0/AdaKV), Copyright (c) 2024 Yuan Feng.

The runtime integration notices appear in `surrogatekv/core.py`,
`surrogatekv/schedule.py`, and the corresponding modules under
`surrogatekv/runtime/`.

These projects are distributed under the following MIT License terms:

> MIT License
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The runtime interfaces were also developed with reference to
[SnapKV](https://github.com/FasterDecoding/SnapKV), which is distributed under
the Apache License 2.0. A copy of that license is included in [`LICENSE`](LICENSE).

## Linked Baselines

The README links to H2O, DynamicKV, CaM, and D2O for attribution and comparison.
Their source trees are not redistributed here and remain subject to their own
licenses.

<p align="center">
  <a href="https://www.sogang.ac.kr/"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/sogang-logo-dark.svg"><img src="docs/sogang-logo.svg" alt="Sogang University" height="44"></picture></a>
  &nbsp;
  <a href="https://discos.sogang.ac.kr/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://github.com/kimjunewon/SurrogateKV/blob/ca94f111c89df166c3421e824e22ff594281ce76/docs/discos-logo-dark.png?raw=true"><img src="https://github.com/kimjunewon/SurrogateKV/blob/ca94f111c89df166c3421e824e22ff594281ce76/docs/discos-logo.png?raw=true" alt="DISCOS Lab" height="44"></picture></a>
</p>

<h1 align="center">SurrogateKV</h1>

<p align="center">
  <strong>Representation-Preserving KV Cache Compression<br>for Long-Context LLMs</strong><br>
  Findings of EMNLP 2026
</p>

<p align="center">
  <a href="mailto:kimjunewon@sogang.ac.kr"><img src="https://img.shields.io/badge/Email-Contact_author-A50034?style=for-the-badge" alt="Email Junwon Kim" height="30"></a>
  <a href="https://discos.sogang.ac.kr/"><img src="https://img.shields.io/badge/DISCOS-Lab_website-155B96?style=for-the-badge" alt="DISCOS Lab website" height="30"></a>
  <a href="https://github.com/kimjunewon/SurrogateKV/tree/main/surrogatekv"><img src="https://img.shields.io/badge/GitHub-Source_code-24292F?style=for-the-badge&logo=github&logoColor=white" alt="GitHub source code" height="30"></a>
  <a href="docs/EMNLP2026_Find-6068.pdf"><img src="https://img.shields.io/badge/Poster-PDF-A50034?style=for-the-badge" alt="View poster PDF" height="30"></a>
</p>

<p align="center">
  <strong>Junwon Kim</strong> &middot; DISCOS Lab, Sogang University<br>
  <a href="mailto:kimjunewon@sogang.ac.kr">kimjunewon@sogang.ac.kr</a>
</p>

<p align="center">
  <a href="docs/EMNLP2026_Find-6068.pdf"><img src="docs/EMNLP2026_Find-6068_preview.png" alt="SurrogateKV EMNLP 2026 poster. Click to open the PDF." width="220"></a><br>
  <a href="docs/EMNLP2026_Find-6068.pdf"><strong>View the EMNLP 2026 poster (PDF)</strong></a>
</p>

SurrogateKV adds a third representation under the same cache-slot budget:
a contiguous historical region can be replaced by one independently addressable
surrogate KV pair. The packed cache uses standard attention during decoding.

## Documentation

- [Method overview](docs/METHOD.md)
- [Installation, runtime API and evaluation](docs/USAGE.md)
- [Results and released data](docs/RESULTS.md)
- [Citation and notices](docs/CITATION.md)

Copyright (c) 2026 Junwon Kim. All rights reserved. See [copyright and third-party notices](docs/CITATION.md#copyright-and-third-party-notices).

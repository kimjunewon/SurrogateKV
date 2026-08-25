from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="surrogatekv",
    version="0.1.0",
    description="Representation-preserving KV-cache compression for long-context LLMs",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy>=1.24", "torch>=2.1"],
    extras_require={
        "longbench": [
            "datasets",
            "fuzzywuzzy",
            "jieba",
            "rouge",
            "tqdm",
            "transformers",
        ]
    },
)

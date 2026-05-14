from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="surrogatekv",
    version="0.1.0",
    description="SurrogateKV chunk-replacement KV-cache compression runtime",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
)

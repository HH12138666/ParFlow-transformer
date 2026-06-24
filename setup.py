#!/usr/bin/env python
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements(path):
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="parflow-predformer",
    version="0.2.0",
    description="PredFormer surrogate modeling for ParFlow pressure prediction",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("configs", "tools", "scripts")),
    python_requires=">=3.10",
    install_requires=read_requirements("requirements/runtime.txt"),
    zip_safe=False,
)

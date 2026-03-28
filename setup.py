#!/usr/bin/env python
from pathlib import Path
from setuptools import find_packages, setup


def read_requirements(path):
    reqs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


setup(
    name="OpenSTL",
    version="0.1.0",
    packages=find_packages(exclude=("configs", "tools", "demo")),
    install_requires=read_requirements("requirements/runtime.txt"),
    zip_safe=False,
)

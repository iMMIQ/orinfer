"""Discover the four offline source namespaces without walking build assets."""

import os
from pathlib import Path

from setuptools import setup

packages = set()
for namespace, source in {
    "orinfer_build": "python/orinfer_build",
    "kernels": "kernels",
    "tools": "tools",
    "tests": "tests",
}.items():
    for current, directories, files in os.walk(source):
        directories[:] = [
            d
            for d in directories
            if not d.startswith(".") and d not in {"__pycache__", "environments"}
        ]
        if any(name.endswith(".py") for name in files):
            parts = Path(current).relative_to(source).parts
            packages.update(".".join((namespace, *parts[:i])) for i in range(len(parts) + 1))

setup(packages=sorted(packages))

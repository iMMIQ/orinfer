"""Resolve historical factory names only when rebuilding archived contracts."""

from functools import lru_cache
import importlib
import json
from pathlib import Path


@lru_cache(maxsize=1)
def migrations():
    return json.loads(Path(__file__).with_name("legacy_modules.json").read_text())


def canonical_module(name):
    return migrations().get(name, name)


def import_module(name):
    return importlib.import_module(canonical_module(name))

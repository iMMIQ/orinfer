"""Configurable inputs for offline real-checkpoint operator diagnostics."""
from functools import lru_cache
import os
from pathlib import Path

from tools.model.publication import file_hash


ROOT = Path(__file__).resolve().parents[1]


def reference_path(variable, name):
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else ROOT / 'artifacts/reference' / name


CHECKPOINT = reference_path('ORINFER_REFERENCE_CHECKPOINT', 'checkpoint')
SOURCE = reference_path('ORINFER_REFERENCE_SOURCE', 'vllm')
ACTIVATIONS = reference_path('ORINFER_REFERENCE_ACTIVATIONS', 'activations')


@lru_cache(maxsize=1)
def checkpoint_sha256(path):
    """Hash the supplied immutable file once, outside GPU benchmark timing."""
    return file_hash(Path(path))

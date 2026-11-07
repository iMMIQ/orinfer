"""Serialize offline GPU work and release compiler scratch before unlocking."""

from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def gpu_lock(path):
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            fcntl.flock(stream, fcntl.LOCK_UN)

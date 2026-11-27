"""Hash-cached C11 shared-library compilation and loading for offline tools."""

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def load_shared_c(
    source: Path,
    cache_dir: Path,
    *,
    prefix: str,
    openmp: bool = True,
    math_lib: bool = False,
    extra_sources: tuple[Path, ...] = (),
) -> ctypes.CDLL:
    """Compile a C11 source into a content-addressed `.so` under `cache_dir` and load it."""
    source = Path(source)
    flags = ["-O3", "-std=c11"]
    if openmp:
        flags.append("-fopenmp")
    flags.extend(["-shared", "-fPIC"])
    libs = ["-lm"] if math_lib else []

    compiler = subprocess.check_output(["cc", "--version"])
    hasher = hashlib.sha256()
    hasher.update(source.read_bytes())
    for extra in extra_sources:
        hasher.update(Path(extra).read_bytes())
    hasher.update(compiler)
    hasher.update(platform.machine().encode())
    hasher.update(" ".join(flags + libs).encode())
    key = hasher.hexdigest()

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"{prefix}-{key}.so"
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            staged = Path(temporary) / f"{prefix}.so"
            subprocess.run(
                ["cc", *flags, str(source), *libs, "-o", str(staged)],
                check=True,
            )
            os.replace(staged, library)
    return ctypes.CDLL(str(library.resolve()))

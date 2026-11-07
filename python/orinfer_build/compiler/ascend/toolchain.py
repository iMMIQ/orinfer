"""CANN 9.2 compiler identity for native A2 build and publication."""

from pathlib import Path
import re


def cann_version(home):
    info = Path(home) / "compiler/version.info"
    match = re.search(r"^Version=(\S+)", info.read_text(), re.MULTILINE)
    if match is None or match[1].split(".")[:2] != ["9", "2"]:
        raise ValueError("Native A2 compilation requires CANN 9.2")
    return match[1]


def require_cann92(specs):
    compilers = [spec["compiler"] for spec in specs]
    versions = {compiler.get("cann_version") for compiler in compilers}
    identities = {
        tuple(compiler.get(key) for key in ("cann_version", "cann_home", "bisheng", "tilelang"))
        for compiler in compilers
    }
    if (
        len(versions) != 1
        or any(
            not isinstance(version, str) or version.split(".")[:2] != ["9", "2"]
            for version in versions
        )
        or len(identities) != 1
    ):
        raise ValueError("Rebuild every referenced A2 kernel with the same CANN 9.2 toolchain")

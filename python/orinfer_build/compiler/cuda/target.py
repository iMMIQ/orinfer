"""CUDA target discovery and package-target contracts for Orin and generic discrete GPUs.

Kept free of top-level Torch/TileLang imports so CPU-side builders, validators,
and unit tests can resolve and validate targets without a CUDA toolchain.
"""

from __future__ import annotations

import os
import platform

DEFAULT_CUDA_ARCH = "sm_87"
SUPPORTED_CUDA_ARCHES = (
    "sm_80",
    "sm_86",
    "sm_87",
    "sm_89",
    "sm_90",
    "sm_100",
    "sm_103",
    "sm_120",
)
_SM_MAP = {
    "sm_80": (8, 0),
    "sm_86": (8, 6),
    "sm_87": (8, 7),
    "sm_89": (8, 9),
    "sm_90": (9, 0),
    "sm_100": (10, 0),
    "sm_103": (10, 3),
    "sm_120": (12, 0),
}
SUPPORTED_HOST_ARCHES = ("aarch64", "x86_64")


def parse_cuda_sm(arch: str) -> tuple[int, int] | None:
    """Return `(major, minor)` for a supported CUDA architecture, else `None`."""
    if not isinstance(arch, str):
        return None
    return _SM_MAP.get(arch)


def validate_cuda_arch(arch: str) -> str:
    if parse_cuda_sm(arch) is None:
        supported = ", ".join(SUPPORTED_CUDA_ARCHES)
        raise ValueError(f"Unsupported CUDA architecture {arch!r}; expected one of: {supported}")
    return arch


def host_arch(explicit: str | None = None) -> str:
    """Return canonical Linux host architecture (`aarch64` or `x86_64`)."""
    raw = explicit or os.environ.get("ORINFER_HOST_ARCH") or platform.machine()
    normalized = {"arm64": "aarch64", "amd64": "x86_64"}.get(raw.lower(), raw)
    if normalized not in SUPPORTED_HOST_ARCHES:
        raise ValueError(f"Unsupported host architecture {raw!r}")
    return normalized


def cuda_arch(explicit: str | None = None) -> str:
    """Resolve the active CUDA architecture (`ORINFER_CUDA_ARCH` -> live GPU -> `sm_87`)."""
    if explicit is not None:
        return validate_cuda_arch(explicit)
    env_arch = os.environ.get("ORINFER_CUDA_ARCH", "").strip()
    if env_arch and env_arch.lower() != "auto":
        return validate_cuda_arch(env_arch)
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            detected = f"sm_{major}{minor}"
            return validate_cuda_arch(detected)
    except ImportError:
        pass
    return DEFAULT_CUDA_ARCH


def cuda_target(arch: str | None = None) -> dict[str, str]:
    """Return the TileLang CUDA target dictionary for the active or requested SM."""
    return {"kind": "cuda", "arch": cuda_arch(arch)}


def is_orin_arch(arch: str | None = None) -> bool:
    """True when targeting Jetson AGX Orin (`sm_87`)."""
    return cuda_arch(arch) == "sm_87"


def is_generic_cuda_arch(arch: str | None = None) -> bool:
    """True when targeting a non-Orin discrete CUDA GPU (`sm_80`, `sm_86`, `sm_89`, `sm_90`, ...)."""
    return not is_orin_arch(arch)


def package_target(arch: str | None = None, host: str | None = None) -> str:
    """Format the canonical execution-package target string matching `orinfer_model_sdk::Target`.

    - Jetson AGX Orin retains its legacy `"sm_87"` target string on `aarch64`.
    - Non-Orin discrete GPUs use bare `"sm_XX"` on `x86_64` and `"sm_XX-aarch64"` on ARM hosts.
    """
    resolved_arch = cuda_arch(arch)
    resolved_host = host_arch(host)
    if resolved_arch == "sm_87":
        return "sm_87" if resolved_host == "aarch64" else f"{resolved_arch}-{resolved_host}"
    return resolved_arch if resolved_host == "x86_64" else f"{resolved_arch}-{resolved_host}"

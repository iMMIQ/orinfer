"""Checkout identity and lifecycle paths for offline work; no device access."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KINDS = frozenset({"builds", "cache", "runs", "releases", "locks"})


def work_root(root=ROOT):
    if value := os.environ.get("ORINFER_WORK_ROOT"):
        return Path(value).expanduser().resolve()
    config = Path(root) / "artifacts/layout.json"
    if config.is_file():
        value = json.loads(config.read_text())
        if value.get("schema") != 1 or not Path(value["work_root"]).is_absolute():
            raise ValueError("Invalid workspace layout")
        return Path(value["work_root"])
    return Path(root) / "artifacts"


def work_path(kind, *parts, root=ROOT):
    if kind not in KINDS or any(Path(p).is_absolute() or ".." in Path(p).parts for p in parts):
        raise ValueError("Invalid artifact lifecycle path")
    return work_root(root) / kind / Path(*parts)


def workspace_name(root=ROOT):
    config = Path(root) / "artifacts/layout.json"
    value = json.loads(config.read_text()).get("workspace") if config.exists() else None
    return value or Path(root).name


def gpu_lock_path(root=ROOT):
    if value := os.environ.get("ORINFER_GPU_LOCK"):
        return Path(value).expanduser().resolve()
    # Existing workers and archived commands must retain the same lock inode.
    legacy = Path(root) / "artifacts/gpu-experiment.lock"
    return (
        legacy.resolve()
        if legacy.exists()
        else work_path("locks", "gpu-experiment.lock", root=root)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--kind", choices=sorted(KINDS))
    parser.add_argument("--gpu-lock", action="store_true")
    parser.add_argument("--work-root", action="store_true")
    parser.add_argument("parts", nargs="*")
    args = parser.parse_args()
    print(
        work_root(args.root)
        if args.work_root
        else gpu_lock_path(args.root)
        if args.gpu_lock
        else work_path(args.kind, *args.parts, root=args.root)
    )

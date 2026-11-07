"""Initialize isolated checkout outputs and inventory worktrees without deleting data."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from orinfer_build.common.files import atomic_json
from orinfer_build.common.paths import ROOT, KINDS, work_root


def link(source, destination):
    if source.exists() or source.is_symlink():
        if source.resolve() != destination.resolve():
            raise ValueError(f"Preserve existing path: {source}; migrate it before initialization")
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(destination, target_is_directory=destination.is_dir())


def initialize(root, output, name):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Workspace name must be a single safe path component")
    if output == root or root in output.parents:
        raise ValueError("Use an artifact root outside the source checkout")
    links = {root / "target": output / "builds" / name / "cargo"}
    links.update({root / "artifacts" / key: output / key for key in KINDS})
    links.update(
        {
            root / "artifacts" / key: output / "cache" / key
            for key in ["models", "execution-packages", "ascend-runtime"]
        }
    )
    links[root / "artifacts/gpu-experiment.lock"] = output / "locks/gpu-experiment.lock"
    # Validate every source before creating a link or changing checkout settings.
    for source, destination in links.items():
        if (source.exists() or source.is_symlink()) and source.resolve() != destination.resolve():
            raise ValueError(f"Preserve existing path: {source}")
    for key in KINDS:
        (output / key).mkdir(parents=True, exist_ok=True)
    for source, destination in links.items():
        if destination.name == "gpu-experiment.lock":
            destination.touch(exist_ok=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
        link(source, destination)
    value = dict(schema=1, work_root=str(output), workspace=name)
    atomic_json(root / "artifacts/layout.json", value)
    return value


def inventory(root):
    raw = subprocess.check_output(
        ["git", "-C", str(root), "worktree", "list", "--porcelain"], text=True
    )
    trees = []
    for block in raw.strip().split("\n\n"):
        record = dict(
            line.split(" ", 1) if " " in line else (line, True) for line in block.splitlines()
        )
        status = subprocess.check_output(
            ["git", "-C", record["worktree"], "status", "--porcelain"], text=True
        )
        record["dirty"] = bool(status)
        trees.append(record)
    return dict(work_root=str(work_root(root)), worktrees=trees)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--work-root", type=Path, required=True)
    init.add_argument("--name", required=True)
    sub.add_parser("inventory")
    args = parser.parse_args()
    value = (
        initialize(args.repository, args.work_root, args.name)
        if args.command == "init"
        else inventory(args.repository)
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()

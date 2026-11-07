"""Shared immutable model cloning and validated, atomic publication."""

from orinfer_build.common.paths import ROOT


from contextlib import contextmanager
from orinfer_build.common.files import write_json

import hashlib
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def generated_outputs(directory):
    """Installed derived files are rebuilt on install, not redistributed."""
    manifest = Path(directory) / "package.json"
    if not manifest.exists():
        return set()
    outputs = set()
    for asset in json.loads(manifest.read_text()).get("generated_assets", []):
        name = asset["output"]["file"]
        if (
            not name
            or Path(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise ValueError("Unsafe generated asset output")
        if name in outputs:
            raise ValueError("Duplicate generated asset output")
        outputs.add(name)
    return outputs


def source_path(base, file):
    path = Path(file)
    if not file or path.is_absolute() or any(p in (".", "..") for p in path.parts):
        raise ValueError(f"Unsafe artifact path: {file}")
    # Check lexical components too: pathlib normalizes away "." components.
    if any(p in ("", ".", "..") for p in file.split("/")):
        raise ValueError(f"Unsafe artifact path: {file}")
    resolved = (base / path).resolve(strict=True)
    if not resolved.is_relative_to(base):
        raise ValueError(f"Artifact leaves its directory: {file}")
    return resolved


class AssetReader:
    """Verify standalone or named U8 kernel assets, reusing bundle metadata."""

    def __init__(self, base):
        self.base = Path(base).resolve(strict=True)
        self.headers = {}

    def read(self, asset):
        path = source_path(self.base, asset["file"])
        name = asset.get("tensor")
        if name is None:
            raw = path.read_bytes()
        else:
            from orinfer_build.weights.safetensors_source import read_header

            with path.open("rb") as stream:
                if path not in self.headers:

                    def read(offset, size):
                        stream.seek(offset)
                        return stream.read(size)

                    self.headers[path] = read_header(read, file_size=path.stat().st_size)
                begin, header = self.headers[path]
                info = header.get(name)
                if not name or info is None or info.get("dtype") != "U8":
                    raise ValueError("Missing or invalid bundled kernel asset")
                first, end = info["data_offsets"]
                if info["shape"] != [end - first] or first == end:
                    raise ValueError("Bundled kernel asset must be a nonempty U8 vector")
                stream.seek(begin + first)
                raw = stream.read(end - first)
        if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
            raise ValueError("Operator asset sha256 mismatch")
        return raw


def link_or_copy(source, destination):
    destination = Path(destination)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        # A staged package may share this inode with its immutable source.
        # Replace the link, never overwrite bytes through the shared inode.
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def clone_cpu_assets(source, destination):
    """CPU input metadata and immutable tables travel with prepared model data."""
    assets = source / "cache/cpu"
    if assets.is_dir():
        shutil.copytree(
            assets,
            destination / "cache/cpu",
            copy_function=lambda src, dst: (
                link_or_copy(src, dst)
                if Path(src).suffix == ".safetensors"
                else shutil.copyfile(src, dst)
            ),
        )


def load_model(model, *, plan=None):
    model = model.resolve(strict=True)
    data = json.loads((model / "cache/model.json").read_text())
    digest = data["execution_package"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid operator package digest")
    root = Path(
        os.environ.get(
            "ORINFER_EXECUTION_CACHE",
            str(
                Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                / "orinfer/packages"
            ),
        )
    )
    origin = root / digest
    if not origin.exists():
        origin = model / "cache/packages" / digest
    raw = (origin / "package.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Source operator package digest mismatch")
    package = json.loads(raw)
    if "catalog" in package:
        # Offline tools reuse the registered Rust binding generator. This is
        # compiler staging data, never an alternative online package format.
        engine = Path(os.environ.get("ORINFER_ENGINE", str(ROOT / "target/release/orinfer")))
        completed = subprocess.run(
            [str(engine), "plan-model", str(model)], check=True, capture_output=True, text=True
        )
        report = json.loads(completed.stdout)
        manifest = report["manifest"]
        if plan is not None:
            plan.update(manifest)
        package.pop("catalog")
        package["kernels"] = manifest["kernels"]
        package["dynamic_batch_kernels"] = list(manifest["dynamic_batch_kernels"].values())
        package["plan_kernel_contracts"] = report["plan_kernel_contracts"]
    return data, origin, package


def clone_model(source, destination, operator):
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Destination must be outside the immutable source model")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
    shutil.copytree(
        source / "cache/weights",
        destination / "cache/weights",
        copy_function=lambda src, dst: (
            link_or_copy(src, dst)
            if Path(src).suffix == ".safetensors"
            else shutil.copyfile(src, dst)
        ),
    )
    clone_cpu_assets(source, destination)
    target = destination / "cache/packages/.building"
    shutil.copytree(operator, target, copy_function=link_or_copy)
    return target


def commit_package(model, operator, data, package):
    from orinfer_build.contracts.catalog import seal

    package = seal(package, operator)
    raw = (json.dumps(package, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    # Metadata can be hardlinked to the source. Always replace its inode.
    with tempfile.NamedTemporaryFile(dir=operator, delete=False) as stream:
        temporary = Path(stream.name)
        os.fchmod(stream.fileno(), (operator / "package.json").stat().st_mode & 0o777)
        stream.write(raw)
    try:
        temporary.replace(operator / "package.json")
    finally:
        temporary.unlink(missing_ok=True)
    operator.rename(operator.with_name(digest))
    data["execution_package"] = digest
    write_json(model / "cache/model.json", data)
    return digest


@contextmanager
def staged_directory(destination):
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as directory:
        staging = Path(directory) / "model"
        yield staging
        if destination.exists() or destination.is_symlink():
            raise ValueError("Destination appeared during publication")
        staging.rename(destination)


@contextmanager
def atomic_model(destination, engine=None, command="plan-model"):
    with staged_directory(destination) as staging:
        yield staging
        cli = engine if engine is not None else ROOT / "target/release/orinfer"
        subprocess.run(
            [str(cli.resolve(strict=True)), command, str(staging)],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def seed_compile_cache(source_cache, cache):
    """Seed completed TileLang hash entries without touching live staging files."""
    source_cache = source_cache.resolve(strict=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        cache.symlink_to(source_cache, target_is_directory=True)
    elif cache.resolve() != source_cache:
        for entry in source_cache.glob("*/**/kernels/*"):
            if not entry.is_dir():
                continue
            destination = cache / entry.relative_to(source_cache)
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".seed-", dir=cache) as temp:
                staged = Path(temp) / "kernel"
                shutil.copytree(entry, staged, copy_function=link_or_copy)
                try:
                    os.rename(staged, destination)
                except OSError as error:
                    if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                        raise

"""Build and verify a release from a clean, exact Git revision; publish only with --publish."""

import argparse
import lzma
import json
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
# Support direct script invocation before loading project modules.
sys.path.insert(0, str(ROOT))
from tools.model.package import execution_identity, install  # noqa: E402
from tools.build.files import file_hash  # noqa: E402


def command(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def archive(directory, output, epoch):
    with (
        output.open("xb") as destination,
        lzma.LZMAFile(destination, mode="wb", preset=6) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for path in [directory, *sorted(directory.rglob("*"))]:
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(directory.parent)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = epoch
                if info.isfile():
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
                else:
                    tar.addfile(info)


def verify(directory):
    checksums = directory / "SHA256SUMS"
    for line in checksums.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name or file_hash(directory / name) != digest:
            raise ValueError(f"Release checksum mismatch: {name}")


def notices(metadata):
    text = ["License notices for the locked Cargo dependency graph.\n"]
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        text.append(
            f"\n{'=' * 72}\n{package['name']} {package['version']}\nLicense: {package.get('license')}\nRepository: {package.get('repository')}\nAuthors: {', '.join(package['authors'])}\n"
        )
        root = Path(package["manifest_path"]).parent
        paths = {
            p
            for p in root.iterdir()
            if p.is_file() and p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
        }
        if package.get("license_file"):
            paths.add(root / package["license_file"])
        for path in sorted(paths):
            text.append(f"\n--- {path.name} ---\n{path.read_text(errors='replace')}\n")
    return "".join(text)


def execution_asset_identity(package, tag):
    """Resolve a readable architecture specification from the package contract."""
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?", tag):
        raise ValueError("Unsafe release tag")
    if package.get("backend") == "ascend":
        specification = "deepseek-v41-flash-1017"
        return f"orinfer-exec-{specification}-w4a8-ascend910b-{tag}.tar.xz", specification
    fields = [package[k] for k in ("architecture", "compute_policy", "target")]
    if any(not isinstance(f, str) or not re.fullmatch(r"[a-z0-9_]+", f) for f in fields):
        raise ValueError("Unsafe execution package asset identity")
    architecture, policy, target = fields
    families = json.loads((ROOT / "configs/architecture-contract.json").read_text())["families"]
    variants = families.get(architecture, {}).get("release_variants", {})
    signature = package.get("config_signature", {})
    text = signature.get("text", signature.get("text_config", signature))
    matches = [
        name
        for name, selector in variants.items()
        if all(text.get(key) == value for key, value in selector.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Register one release architecture specification for {architecture}")
    specification = matches[0]
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", specification):
        raise ValueError("Unsafe release architecture specification")
    compute = "a8" if policy == "int8_quality" else policy.replace("_", "-")
    return (
        f"orinfer-exec-{specification}-{compute}-{target.replace('_', '')}-{tag}.tar.xz",
        specification,
    )


def execution_assets(archives, cache, binary, tag):
    """Validate each package independently and name assets from their actual contract."""
    assets = []
    names = set()
    for archive_path in archives:
        archive_path = archive_path.resolve(strict=True)
        if archive_path.is_dir():
            root = archive_path
            package = json.loads((root / "package.json").read_text())
            if package.get("backend") != "ascend" or file_hash(root / "package.json") != root.name:
                raise ValueError("Expected a digest-named native A2 execution directory")
            if package.get("schema_version") != 1:
                raise ValueError("Unsupported native A2 execution schema")
            name, specification = execution_asset_identity(package, tag)
            if name in names:
                raise ValueError("Duplicate execution package asset in release")
            names.add(name)
            assets.append(
                dict(
                    directory=root,
                    metadata=dict(
                        file=name,
                        digest=root.name,
                        architecture="deepseek_v41",
                        architecture_id=specification,
                        compute_policy="w4a8",
                        target="ascend910b",
                    ),
                )
            )
            continue
        digest = install(archive_path, cache)
        root = cache / digest
        package = json.loads((root / "package.json").read_text())
        identity = execution_identity(
            root / package["execution"]["library"]["file"],
            binary,
            package["architecture"],
            package["compute_policy"],
        )
        if identity != package["execution"]:
            raise ValueError("Native execution identity differs from package manifest")
        name, specification = execution_asset_identity(package, tag)
        if name in names:
            raise ValueError("Duplicate execution package asset in release")
        names.add(name)
        assets.append(
            dict(
                directory=root,
                metadata=dict(
                    file=name,
                    digest=digest,
                    architecture=package["architecture"],
                    architecture_id=specification,
                    compute_policy=package["compute_policy"],
                    target=package["target"],
                    execution=identity,
                ),
            )
        )
    return assets


def build(args):
    revision = command("git", "rev-parse", "HEAD")
    if command("git", "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Release requires a clean worktree, including new source files")
    metadata = json.loads(
        command("cargo", "metadata", "--locked", "--offline", "--format-version", "1")
    )
    version = next(p["version"] for p in metadata["packages"] if p["name"] == "orinfer-cli")
    tag = "v" + version
    if args.version != tag:
        raise ValueError(f"Tag must match workspace version: {tag}")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--offline", "-p", "orinfer-cli"],
        cwd=ROOT,
        check=True,
    )
    binary = ROOT / "target/release/orinfer"
    elf = binary.read_bytes()[:20]
    if elf[:4] != b"\x7fELF" or int.from_bytes(elf[18:20], "little") != 183:
        raise ValueError("Binary must target Linux aarch64")
    if command(str(binary), "--version") != f"orinfer {version}":
        raise ValueError("Binary version mismatch")
    if command("git", "rev-parse", "HEAD") != revision or command("git", "status", "--porcelain"):
        raise ValueError("Source changed during build")
    with tempfile.TemporaryDirectory(prefix="orinfer-release-") as temp:
        packages = execution_assets(args.operators, Path(temp) / "operator-cache", binary, tag)
        bundle = Path(temp) / f"orinfer-{tag}-linux-aarch64"
        (bundle / "bin").mkdir(parents=True)
        shutil.copy2(binary, bundle / "bin/orinfer")
        for name in (
            "README.md",
            "README.zh-CN.md",
            "README.ja.md",
            "README.ko.md",
            "LICENSE",
            "COPYING",
            "COPYING.LESSER",
            "THIRD_PARTY_NOTICES.md",
        ):
            if (ROOT / name).exists():
                shutil.copyfile(ROOT / name, bundle / name)
        # Ship complete offline tools and their imports; only tracked files are included.
        for name in command(
            "git", "ls-files", "docs", "examples", "tools", "kernels", "configs", "fixtures"
        ).splitlines():
            (bundle / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, bundle / name)
        source = bundle / "source.tar.xz"
        source_tar = Path(temp) / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={source_tar}", revision],
            cwd=ROOT,
            check=True,
        )
        with (
            source_tar.open("rb") as source_stream,
            lzma.open(source, "xb", preset=6) as compressed,
        ):
            shutil.copyfileobj(source_stream, compressed)
        source_tar.unlink()
        (bundle / "SOURCE.txt").write_text(
            f"Revision: {revision}\nSource: source.tar.xz\nBuild: cargo build --release --locked --offline -p orinfer-cli\n"
        )
        (bundle / "THIRD_PARTY_LICENSES.txt").write_text(notices(metadata))
        (bundle / "RELEASE.json").write_text(
            json.dumps(
                dict(
                    tag=tag,
                    revision=revision,
                    execution_packages=[p["metadata"] for p in packages],
                    binary_sha256=file_hash(binary),
                ),
                indent=2,
            )
            + "\n"
        )
        epoch = int(command("git", "show", "-s", "--format=%ct", revision))
        archive(bundle, output / f"{bundle.name}.tar.xz", epoch)
        for package in packages:
            archive(package["directory"], output / package["metadata"]["file"], epoch)
            if package["metadata"].get("target") == "ascend910b":
                # The release binary validates schema, native ABI and every A2 asset.
                command(
                    str(binary),
                    "package",
                    "install",
                    str(output / package["metadata"]["file"]),
                    "--cache",
                    str(Path(temp) / "a2-install-cache"),
                )
    assets = sorted(output.glob("*.tar.xz"))
    (output / "SHA256SUMS").write_text(
        "".join(f"{file_hash(path)}  {path.name}\n" for path in assets)
    )
    verify(output)
    if args.publish:
        if not args.notes or not args.notes.is_file():
            raise ValueError("--publish requires --notes FILE")
        subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if command("git", "rev-parse", f"{tag}^{{}}") != revision:
            raise ValueError("Annotated release tag differs from packaged revision")
        subprocess.run(["git", "push", "origin", f"refs/tags/{tag}"], cwd=ROOT, check=True)
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                *map(str, assets),
                str(output / "SHA256SUMS"),
                "--title",
                tag,
                "--notes-file",
                str(args.notes),
            ],
            cwd=ROOT,
            check=True,
        )
        with tempfile.TemporaryDirectory(prefix="orinfer-release-download-") as temp:
            subprocess.run(["gh", "release", "download", tag, "--dir", temp], cwd=ROOT, check=True)
            verify(Path(temp))
            for path in [*assets, output / "SHA256SUMS"]:
                if file_hash(Path(temp) / path.name) != file_hash(path):
                    raise ValueError(f"Uploaded asset differs from local build: {path.name}")
    return dict(tag=tag, revision=revision, output=str(output), verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--operators",
        type=Path,
        action="append",
        required=True,
        help="Execution archive or native A2 package directory; repeat for each architecture/policy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--publish", action="store_true")
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

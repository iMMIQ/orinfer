"""Validate, archive and install schema-2 CUDA execution packages."""

from orinfer_build.common.paths import ROOT


import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


from orinfer_build.packaging.publication import AssetReader, generated_outputs, source_path
from orinfer_build.common.files import file_hash


CONTRACT = json.loads((ROOT / "configs/architecture-contract.json").read_text())
TEXT_KEYS = CONTRACT["families"]["qwen3_5"]["text_keys"]


def read_bindings(package, directory):
    """Read and verify the typed registry and reconstruct the compiler bindings."""
    from orinfer_build.contracts.binding_templates import expand as expand_registry
    from orinfer_build.contracts.catalog import expand

    if (
        package.get("schema_version") != 2
        or "catalog" not in package
        or any(key in package for key in ("kernels", "runtime_abi", "execution"))
    ):
        raise ValueError("Expected a schema 2 execution package with data bindings")
    identity = package["bindings"]
    if identity["file"] != "bindings.json" or identity.get("tensor") is not None:
        raise ValueError("Expected a standalone bindings.json")
    raw = AssetReader(directory).read(identity)
    catalog = package["catalog"]
    kernels, dynamic, contracts = expand(catalog, expand_registry(json.loads(raw)))
    return kernels, dynamic, contracts


def verify_package(package, directory):
    read_bindings(package, directory)
    reader = AssetReader(directory)
    checked = set()
    for asset in package["catalog"]["assets"]:
        if asset["file"] != "assets.safetensors" or not asset.get("tensor"):
            raise ValueError("Expected indexed execution assets in assets.safetensors")
        key = (asset["tensor"], asset["sha256"])
        if key not in checked:
            reader.read(asset)
            checked.add(key)


def config_signature(config):
    if config.get("model_type") == "qwen3":
        from orinfer_build.models.qwen_reranker.config import signature

        return signature(config)
    if config.get("model_type") == "mistral3":
        from orinfer_build.models.nemotron_embed.config import signature

        return signature(config)
    text = config.get("text_config", config)
    family = "qwen3_asr" if config.get("model_type") == "qwen3_asr" else "qwen3_5"
    return {
        **({"audio": config["audio_config"]} if family == "qwen3_asr" else {}),
        "text": {k: text[k] for k in CONTRACT["families"][family]["text_keys"] if k in text},
        "vision": config.get("vision_config"),
    }


def validate_plan(directory, source, mapping, engine):
    """Require exact graph, ABI and launch equivalence before publishing."""
    completed = subprocess.run(
        [str(engine.resolve(strict=True)), "plan-model", str(directory)],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)["manifest"]
    stages = {stage for profile in plan["prefill_plans"] for stage in profile.get("stages", [])}
    if set(plan["programs"]) != set(source["programs"]) | stages:
        raise ValueError("Registered architecture does not cover all source programs")
    before = {k["name"]: k for k in source["kernels"]}
    after = {k["name"]: k for k in plan["kernels"]}
    for program, original_ops in source["programs"].items():
        generated_ops = plan["programs"][program]
        if len(original_ops) != len(generated_ops):
            raise ValueError(f"{program}: registered operation count differs")
        for index, (left, right) in enumerate(zip(original_ops, generated_ops)):
            if left["kind"] != "kernel":
                equal = left == right
            else:
                equal = (
                    right["kind"] == "kernel"
                    and mapping[right["name"]] == left["name"]
                    and {k: v for k, v in before[left["name"]].items() if k != "name"}
                    == {k: v for k, v in after[right["name"]].items() if k != "name"}
                )
            if not equal:
                raise ValueError(f"{program}/{index}: registered operation or binding differs")


def archive(package, output):
    package = package.resolve(strict=True)
    if not output.name.endswith(".tar.xz"):
        raise ValueError("New execution archives must use .tar.xz")
    if output.exists():
        raise ValueError("Archive already exists")
    temporary = output.with_suffix(output.suffix + ".tmp")
    omitted = generated_outputs(package)

    def distributed(info):
        return None if str(Path(info.name).relative_to(package.name)) in omitted else info

    try:
        with tarfile.open(temporary, "w:xz", preset=6) as stream:
            stream.add(package, arcname=package.name, filter=distributed)
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def install(archive_path, cache):
    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=cache))
    try:
        with tarfile.open(archive_path, "r:*") as stream:
            members = stream.getmembers()
            roots = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
            if len(roots) != 1 or any(not (m.isfile() or m.isdir()) for m in members):
                raise ValueError("Expected one package directory containing regular files")
            digest = roots.pop()
            if not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("Malformed operator package digest")
            if (cache / digest).exists():
                raise ValueError("Operator package already installed")
            for member in members:
                p = Path(member.name)
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("Unsafe archive path")
            stream.extractall(staging)
        package = staging / digest
        if file_hash(package / "package.json") != digest:
            raise ValueError("Operator package digest mismatch")
        manifest = json.loads((package / "package.json").read_text())
        verify_package(manifest, package)
        package.rename(cache / digest)
        return digest
    finally:
        shutil.rmtree(staging)


def pin_assets(directory):
    """Explicitly seal active frontend files, without changing any weight/kernel payload."""
    directory = directory.resolve(strict=True)
    path = directory / "cache/model.json"
    descriptor = json.loads(path.read_text())
    names = (
        ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja")
        if descriptor["metadata"].get("embedding")
        else (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        )
        if descriptor["metadata"].get("reranking")
        else ("tokenizer.json", "chat_template.jinja", "generation_config.json")
    )
    identities = {name: file_hash(source_path(directory, name)) for name in names}
    descriptor["frontend_assets"] = identities
    fd, temporary = tempfile.mkstemp(prefix=".model-", suffix=".json", dir=path.parent)
    try:
        os.fchmod(fd, path.stat().st_mode & 0o777)
        with os.fdopen(fd, "w") as stream:
            json.dump(descriptor, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("archive")
    pack.add_argument("package", type=Path)
    pack.add_argument("output", type=Path)
    add = sub.add_parser("install")
    add.add_argument("archive", type=Path)
    add.add_argument("cache", type=Path)
    seal = sub.add_parser("pin-assets", help="Explicitly pin existing native model frontend files")
    seal.add_argument("model", type=Path)
    args = parser.parse_args()
    if args.command == "pin-assets":
        print(json.dumps(pin_assets(args.model), indent=2))
    elif args.command == "archive":
        archive(args.package, args.output)
    else:
        print(install(args.archive, args.cache))


if __name__ == "__main__":
    main()

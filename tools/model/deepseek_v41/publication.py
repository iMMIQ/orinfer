"""Publish compact native A2 execution assets separately from model weights."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from tools.build.elf import strip_library
from tools.build.files import atomic_json, file_hash
from tools.model.publication import AssetReader, source_path


def manifests(model):
    names = set()
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    names.add(step["manifest"])
                    if step.get("uncompacted_manifest"):
                        names.add(step["uncompacted_manifest"])
    for choices in model.get("row_selections", {}).values():
        for choice in choices:
            names.update(choice["kernels"])
    return sorted(names)


def prepare_execution(model, source, stage):
    source, stage = Path(source).resolve(), Path(stage)
    assets = {model["engram"]["file"], *(v["file"] for v in model["rope"].values())}
    assets.update(b["data"]["file"] for r in model["ranks"] for b in r["buffers"] if b.get("data"))
    for name in sorted(assets):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(source, name), target)
    copied = {}
    originals = {}
    for name in manifests(model):
        path = source_path(source, name)
        spec = json.loads(path.read_text())
        library = spec["library"]
        if library.get("tensor"):
            raise ValueError("Native A2 libraries must be standalone files")
        relative = str((path.parent / library["file"]).relative_to(source))
        if relative in originals and originals[relative] != library["sha256"]:
            raise ValueError("Conflicting native library identities")
        originals[relative] = library["sha256"]
        if relative not in copied:
            data = AssetReader(path.parent).read(library)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            strip_library(target)
            copied[relative] = file_hash(target)
        spec["library"]["sha256"] = copied[relative]
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(target, spec, compact=True)
    atomic_json(stage / "model.json", model, compact=True)
    atomic_json(
        stage / "package.json",
        dict(
            schema_version=1,
            backend="ascend",
            plan=dict(file="model.json", sha256=file_hash(stage / "model.json")),
        ),
        compact=True,
    )


def publish_execution(source, cache):
    source, cache = Path(source).resolve(strict=True), Path(cache)
    package = json.loads((source / "package.json").read_text())
    if package.get("schema_version") != 1 or package.get("backend") != "ascend":
        raise ValueError("Expected native A2 schema-1 execution package")
    model = json.loads(AssetReader(source).read(package["plan"]))
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=cache) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        prepare_execution(model, source, stage)
        shutil.copytree(source / "licenses", stage / "licenses")
        digest = file_hash(stage / "package.json")
        destination = cache / digest
        if destination.exists():
            raise ValueError("Execution package already published")
        stage.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    args = parser.parse_args()
    print(publish_execution(args.source, args.output_cache))


if __name__ == "__main__":
    main()

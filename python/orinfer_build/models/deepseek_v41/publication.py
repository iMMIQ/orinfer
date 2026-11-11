"""Publish compact native A2 execution assets separately from model weights."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from orinfer_build.common.elf import strip_library
from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.common.shared_json import decode
from orinfer_build.packaging.publication import AssetReader, source_path
from orinfer_build.compiler.ascend.toolchain import require_cann92


def manifest_bindings(model):
    """Yield mutable references to every executable kernel in the published model."""
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    yield step, "manifest"
                    if step.get("uncompacted_manifest"):
                        yield step, "uncompacted_manifest"
    for choices in model.get("row_selections", {}).values():
        for choice in choices:
            for index in range(len(choice["kernels"])):
                yield choice["kernels"], index
    for op in model.get("vision", {}).get("body", []):
        yield op, "manifest"
    sampling = model.get("sampling", {})
    for key in ("local", "merge", "processed_local", "processed_merge"):
        if sampling.get(key):
            yield sampling, key
    rejection = sampling.get("rejection", {})
    for key in ("statistics", "merge_statistics", "normalize", "select"):
        if rejection.get(key):
            yield rejection, key
    inputs = model.get("device_inputs")
    if inputs:
        for key in (
            "merge",
            "commit",
            "save_draft",
            "hashes",
            "scatter_embedding",
            "draft_embedding",
        ):
            yield inputs, key
        for key in inputs["embeddings"]:
            yield inputs["embeddings"], key
        for preparation in inputs["preparations"].values():
            for key in preparation:
                yield preparation, key


def manifests(model):
    return sorted({binding[key] for binding, key in manifest_bindings(model)})


def prepare_execution(model, source, stage):
    source, stage = Path(source).resolve(), Path(stage)
    descriptions = {
        name: json.loads(source_path(source, name).read_text()) for name in manifests(model)
    }
    if descriptions:
        require_cann92(descriptions.values())
    sources = {}
    for spec in descriptions.values():
        if sources.setdefault(spec["op"], spec["source_sha256"]) != spec["source_sha256"]:
            raise ValueError("Publish one source identity per A2 operator; rebuild the full plan")
    assets = {model["engram"]["file"], *(v["file"] for v in model["rope"].values())}
    assets.update(b["data"]["file"] for r in model["ranks"] for b in r["buffers"] if b.get("data"))
    for name in sorted(assets):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(source, name), target)
    copied = {}
    originals = {}
    for name, spec in descriptions.items():
        path = source_path(source, name)
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


def publish_execution(
    source,
    cache,
    *,
    packet_boundary=None,
    verification_capacity=None,
    model_dir=None,
    compact_rope=False,
    engine=None,
):
    source, cache = Path(source).resolve(strict=True), Path(cache)
    package = json.loads((source / "package.json").read_text())
    if package.get("schema_version") != 1 or package.get("backend") != "ascend":
        raise ValueError("Expected native A2 schema-1 execution package")
    model = decode(json.loads(AssetReader(source).read(package["plan"])))
    if packet_boundary is not None or verification_capacity is not None:
        from orinfer_build.models.deepseek_v41.assembly import (
            retarget_packet_boundary,
            validate_rows,
        )

        if packet_boundary is not None:
            model = retarget_packet_boundary(model, *packet_boundary)
        if verification_capacity is not None:
            if not model.get("batch"):
                raise ValueError("Verification capacity requires a native batch contract")
            model["batch"]["verification_capacity"] = verification_capacity
        validate_rows(model, source, model_dir=model_dir)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=cache) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        prepare_execution(model, source, stage)
        shutil.copytree(source / "licenses", stage / "licenses")
        digest = file_hash(stage / "package.json")
        if compact_rope:
            from orinfer_build.models.deepseek_v41.compact import compact

            if engine is None:
                raise ValueError("Compact RoPE publication requires --engine")
            encoded_source = stage.with_name(digest)
            stage.rename(encoded_source)
            return compact(encoded_source, cache, engine)
        destination = cache / digest
        if destination.exists():
            raise ValueError("Execution package already published")
        stage.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--packet-boundary", type=int, nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument("--verification-capacity", type=int)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--compact-rope", action="store_true")
    parser.add_argument("--engine", type=Path, help="Engine with installation asset generation")
    args = parser.parse_args()
    print(
        publish_execution(
            args.source,
            args.output_cache,
            packet_boundary=args.packet_boundary,
            verification_capacity=args.verification_capacity,
            model_dir=args.model_dir,
            compact_rope=args.compact_rope,
            engine=args.engine,
        )
    )


if __name__ == "__main__":
    main()

"""Compact published A2 packages with exact installation-generated RoPE tables.

The Rust installer recreates ordinary assets inside its atomic staging tree.
The model plan, kernel libraries, tensor identities and inference bytes stay exact.
"""

import argparse
import json
import lzma
from pathlib import Path
import shutil
import subprocess
import tempfile

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.common.shared_json import decode, encode
from orinfer_build.packaging.publication import AssetReader, source_path


def compact(source, output_cache, engine):
    source = Path(source).resolve(strict=True)
    output_cache = Path(output_cache).resolve()
    engine = Path(engine).resolve(strict=True)
    if output_cache.is_relative_to(source):
        raise ValueError("Compact output cache must be outside the source package")
    package = json.loads((source / "package.json").read_text())
    if package.get("schema_version") != 1 or package.get("backend") != "ascend":
        raise ValueError("Expected a published native A2 package")
    if package.get("generated_assets"):
        raise ValueError("Package already declares generated assets")
    if file_hash(source / "package.json") != source.name:
        raise ValueError("Published package digest differs")
    model = decode(json.loads(AssetReader(source).read(package["plan"])))
    rope = model["rope"]
    if set(rope) != {"base_cos", "base_sin", "long_cos", "long_sin"}:
        raise ValueError("Expected base and long cosine/sine tables")
    omitted = {id["file"] for id in rope.values()}
    if any(id.get("tensor") for id in rope.values()):
        raise ValueError("RoPE tables must be standalone files")
    output_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compact-", dir=output_cache) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        # Copy published files without re-stripping or mutating shared inodes.
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if path.is_symlink():
                raise ValueError("Expected published regular files")
            if str(relative) in omitted or path.is_dir():
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Expected published regular files")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        derived = []
        for prefix in ("base", "long"):
            directory = Path(temporary) / (prefix + "-encoded")
            cos = source_path(source, rope[prefix + "_cos"]["file"])
            sin = source_path(source, rope[prefix + "_sin"]["file"])
            for component, path in (("cos", cos), ("sin", sin)):
                if file_hash(path) != rope[prefix + "_" + component]["sha256"]:
                    raise ValueError("Original RoPE identity differs")
            encoded = json.loads(
                subprocess.check_output(
                    [
                        str(engine),
                        "package",
                        "encode-rope",
                        "--cos",
                        str(cos),
                        "--sin",
                        str(sin),
                        "--columns",
                        "32",
                        "--output",
                        str(directory),
                    ],
                    text=True,
                )
            )
            if encoded["rows"] != model["capacity"] or len(encoded["frequencies"]) != 32:
                raise ValueError("Encoded RoPE geometry differs from the model")
            for component in ("cos", "sin"):
                relative = f"generated/{prefix}_{component}.xor.xz"
                target = stage / relative
                target.parent.mkdir(exist_ok=True)
                with (
                    Path(encoded[component + "_corrections"]).open("rb") as stream,
                    lzma.open(target, "wb", preset=6) as output,
                ):
                    shutil.copyfileobj(stream, output)
                if target.stat().st_size >= cos.stat().st_size // 4:
                    raise ValueError("RoPE correction stream is not compact enough")
                derived.append(
                    dict(
                        output=rope[prefix + "_" + component],
                        recipe=dict(
                            kind="rope_f32_v1",
                            rows=encoded["rows"],
                            frequencies=encoded["frequencies"],
                            component=component,
                            corrections=dict(file=relative, sha256=file_hash(target)),
                        ),
                    )
                )
        # Final plans intern common values across ranks, layers and packet ranges.
        # Expansion changes no array order, scalar, operand or launch expression.
        document = encode(model)
        if decode(document) != model:
            raise ValueError("Shared plan differs from the source contract")
        atomic_json(stage / package["plan"]["file"], document, compact=True)
        package["plan"]["sha256"] = file_hash(stage / package["plan"]["file"])
        package["generated_assets"] = derived
        atomic_json(stage / "package.json", package, compact=True)
        digest = file_hash(stage / "package.json")
        destination = output_cache / digest
        if destination.exists():
            raise ValueError("Compact execution package already exists")
        stage.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()
    print(compact(args.source, args.output_cache, args.engine))


if __name__ == "__main__":
    main()

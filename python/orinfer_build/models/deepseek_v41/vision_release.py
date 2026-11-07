"""Publish vision assets alongside immutable language shards and a separate execution package."""

from orinfer_build.common.paths import ROOT


import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.models.deepseek_v41.publication import prepare_execution
from orinfer_build.weights.safetensors_source import local_header


def publish(plan, source_model, vision_weights, output_model, execution_cache):
    plan, source_model, vision_weights = (
        plan.resolve(),
        source_model.resolve(),
        vision_weights.resolve(),
    )
    output_model, execution_cache = output_model.resolve(), execution_cache.resolve()
    if (
        output_model.exists()
        or output_model.is_relative_to(execution_cache)
        or execution_cache.is_relative_to(output_model)
    ):
        raise ValueError("Use a new model directory and a separate execution cache")
    model = json.loads(plan.read_text())
    if not model.get("vision"):
        raise ValueError("The plan must contain the validated native vision program")
    index = json.loads((source_model / model["index"]["file"]).read_text())
    added = json.loads((vision_weights / "model.safetensors.index.json").read_text())["weight_map"]
    if index["weight_map"].keys() & added.keys():
        raise ValueError("Vision tensors conflict with the language checkpoint")
    if set(added.values()) != {"vision.safetensors"}:
        raise ValueError("Require one semantic vision shard")
    index["weight_map"].update(added)
    release = deepcopy(json.loads((source_model / "release-manifest.json").read_text()))
    record = json.loads((vision_weights / "vision.json").read_text())
    if record["target_basis"] != "ascend_preprocessed":
        raise ValueError("Vision and language target bases differ")
    certificate = record.get("value_compute", {})
    if (
        certificate.get("dtype") != "f16"
        or not 0 <= certificate.get("maximum", float("inf")) < 65504
    ):
        raise ValueError("Export vision assets with a certified FP16 V range")
    # Offline release metadata only. Normal model startup never scans payload hashes.
    start, header = local_header(vision_weights / "vision.safetensors")
    hashes = {}
    with (vision_weights / "vision.safetensors").open("rb") as file:
        for name in sorted(added):
            begin, end = header[name]["data_offsets"]
            file.seek(start + begin)
            remaining, digest = end - begin, hashlib.sha256()
            while remaining:
                chunk = file.read(min(remaining, 8 << 20))
                if not chunk:
                    raise ValueError("Truncated vision release shard")
                digest.update(chunk)
                remaining -= len(chunk)
            hashes[name] = digest.hexdigest()
    release.setdefault("logical_shapes", {}).update(record["logical_shapes"])
    if "value_compute" in record:
        release["vision_value_compute"] = record["value_compute"]
    release["files"]["vision.safetensors"] = dict(
        bytes=record["bytes"], sha256=record["sha256"], tensors=hashes
    )
    execution_cache.mkdir(parents=True, exist_ok=True)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vision-model-", dir=output_model.parent) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir()
        for name in set(index["weight_map"].values()):
            source = vision_weights / name if name == "vision.safetensors" else source_model / name
            target = stage / name
            if source.resolve().is_relative_to(
                vision_weights if name == "vision.safetensors" else source_model
            ) is False or not target.resolve().is_relative_to(stage):
                raise ValueError("Weight shard leaves its model directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(source)
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
            "LICENSE",
            "SOURCE-README.md",
        ]:
            if (source_model / name).exists():
                shutil.copyfile(source_model / name, stage / name)
        atomic_json(stage / "model.safetensors.index.json", index)
        model["index"]["sha256"] = file_hash(stage / "model.safetensors.index.json")
        with tempfile.TemporaryDirectory(
            prefix=".vision-execution-", dir=execution_cache
        ) as package_temp:
            package_stage = Path(package_temp) / "package"
            package_stage.mkdir()
            prepare_execution(model, plan.parent, package_stage)
            shutil.copytree(plan.parent / "licenses", package_stage / "licenses")
            shutil.copyfile(
                ROOT / "THIRD_PARTY_NOTICES.md",
                package_stage / "licenses/THIRD_PARTY_NOTICES.md",
            )
            digest = file_hash(package_stage / "package.json")
            package = execution_cache / digest
            if package.exists():
                raise ValueError("Execution package already exists")
            package_stage.rename(package)
        descriptor = json.loads((source_model / "cache/model.json").read_text())
        descriptor["execution_package"] = digest
        (stage / "cache").mkdir()
        atomic_json(stage / "cache/model.json", descriptor)
        atomic_json(stage / "release-manifest.json", release)
        stage.rename(output_model)
    return dict(
        model=str(output_model),
        execution_package=str(package),
        execution_digest=digest,
        vision_tensors=len(added),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["plan", "source-model", "vision-weights", "output-model", "execution-cache"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish(
                args.plan,
                args.source_model,
                args.vision_weights,
                args.output_model,
                args.execution_cache,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

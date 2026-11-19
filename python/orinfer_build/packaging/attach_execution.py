"""Attach a compatible, digest-named execution package to a prepared model."""

import argparse
import json
from pathlib import Path

from orinfer_build.common.files import file_hash, write_json
from orinfer_build.packaging.package import verify_package
from orinfer_build.packaging.publication import atomic_model, clone_model, source_path


def attach(source, output, package, engine=None):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Destination must be outside the immutable source model")
    package = package.resolve(strict=True)
    manifest = source_path(package, "package.json")
    digest = file_hash(manifest)
    if digest != package.name:
        raise ValueError("Execution package digest mismatch")
    execution = json.loads(manifest.read_text())
    verify_package(execution, package)
    descriptor = json.loads(source_path(source, "cache/model.json").read_text())
    if descriptor.get("schema_version") != 1 or any(
        descriptor[key] != execution[key] for key in ("architecture", "compute_policy")
    ):
        raise ValueError("Execution package architecture or policy differs from the model")
    with atomic_model(output, engine) as staging:
        target = clone_model(source, staging, package)
        target.rename(target.with_name(digest))
        descriptor["execution_package"] = digest
        write_json(staging / "cache/model.json", descriptor)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()
    print(attach(args.model, args.output, args.package, args.engine))


if __name__ == "__main__":
    main()

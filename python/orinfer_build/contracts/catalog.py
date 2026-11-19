"""Factor expanded compiler bindings into a data-only execution package."""

import copy
import json
from pathlib import Path
import re

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.contracts import binding_templates


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def construction_contracts(package):
    """Retain bindings whose presence changes a registered execution program.

    Older Qwen packages use a separate convolution-history writer. Its presence
    must reach the constructor even though it has no row-selection suffix.
    In-place convolution packages have no such binding to retain.
    """
    return set(package.get("plan_kernel_contracts", [])) | {
        k["name"]
        for k in package["kernels"]
        if re.fullmatch(r"qwen_rows/state/layer[0-9]+/history", k["name"])
    }


def factor(package):
    package = copy.deepcopy(package)
    selected = construction_contracts(package)
    kernels = package.pop("kernels")
    package.pop("plan_kernel_contracts", None)
    dynamic = package.pop("dynamic_batch_kernels", [])
    assets, definitions, templates = [], [], []
    asset_ids, definition_ids, template_ids = {}, {}, {}

    def intern(value, values, indices):
        key = canonical(value)
        if key not in indices:
            indices[key] = len(values)
            values.append(value)
        return indices[key]

    launch = {}
    for item in sorted(dynamic, key=lambda item: item["name"]):
        item = copy.deepcopy(item)
        name = item.pop("name")
        item.setdefault("capacity", 128)
        if name in launch:
            raise ValueError("Duplicate dynamic launch binding")
        launch[name] = intern(item, templates, template_ids)
    bindings = []
    seen = set()
    for k in sorted(kernels, key=lambda k: k["name"]):
        if k["name"] in seen:
            raise ValueError("Duplicate native binding")
        seen.add(k["name"])
        definition = {
            field: k[field] for field in ("symbol", "block", "shared_memory_bytes", "cooperative")
        }
        if "launch" in k:
            definition["launch"] = k["launch"]
        for field in ("module", "source", "host_abi"):
            definition[field] = intern(k[field], assets, asset_ids)
        definition["parameters"] = [a["kind"] for a in k["args"]]
        if k.get("readonly_arguments"):
            if len(k["readonly_arguments"]) != len(k["args"]):
                raise ValueError("Argument access contract differs from ABI")
            definition["readonly_arguments"] = k["readonly_arguments"]
        bindings.append(
            dict(
                name=k["name"],
                definition=intern(definition, definitions, definition_ids),
                grid=k["grid"],
                args=k["args"],
                dynamic=launch.get(k["name"]),
                plan_contract=k["name"] in selected,
            )
        )
    if not selected.issubset(seen) or not launch.keys() <= seen or not bindings:
        raise ValueError("Missing native plan/dynamic binding")
    catalog = dict(assets=assets, definitions=definitions, dynamic=templates)
    package["catalog"] = catalog
    return package, bindings


def expand(catalog, bindings):
    """Offline exact roundtrip proof, before publication."""
    kernels = []
    dynamic = []
    contracts = []
    for binding in bindings:
        d = copy.deepcopy(binding_templates.lookup(catalog["definitions"], binding["definition"]))
        parameters = d.pop("parameters")
        if parameters != [a["kind"] for a in binding["args"]]:
            raise ValueError("Native parameters differ from the compiled ABI")
        for field in ("module", "source", "host_abi"):
            d[field] = binding_templates.lookup(catalog["assets"], d[field])
        kernels.append(dict(d, name=binding["name"], grid=binding["grid"], args=binding["args"]))
        if binding["dynamic"] is not None:
            dynamic.append(
                dict(
                    binding_templates.lookup(catalog["dynamic"], binding["dynamic"]),
                    name=binding["name"],
                )
            )
        if binding["plan_contract"]:
            contracts.append(binding["name"])
    return kernels, dynamic, contracts


def seal(package, directory):
    """Write the compact binding registry atomically and pin its identity."""
    factored, bindings = factor(package)
    registry = binding_templates.factor(bindings)
    reconstructed, dynamic, contracts = expand(
        factored["catalog"], binding_templates.expand(registry)
    )

    def normalize(k):
        normalized = {
            field: value for field, value in k.items() if field != "readonly_arguments" or value
        }
        normalized["args"] = [
            binding_templates.encode_argument(a, lambda name: name) for a in k["args"]
        ]
        return normalized

    if {k["name"]: normalize(k) for k in reconstructed} != {
        k["name"]: normalize(k) for k in package["kernels"]
    }:
        raise ValueError("Compiled catalog failed exact binding roundtrip")
    if {v["name"]: v for v in dynamic} != {
        v["name"]: dict({"capacity": 128}, **v) for v in package.get("dynamic_batch_kernels", [])
    } or set(contracts) != construction_contracts(package):
        raise ValueError("Compiled catalog failed dynamic/plan contract roundtrip")
    path = Path(directory) / "bindings.json"
    atomic_json(path, registry, compact=True)
    path.chmod(0o644)
    factored["schema_version"] = 2
    factored["bindings"] = dict(file=path.name, sha256=file_hash(path))
    return factored


def finalize(model, output, engine=None):
    """Rewrite execution metadata, hardlinking unchanged weight assets."""
    from orinfer_build.packaging.publication import (
        atomic_model,
        clone_model,
        commit_package,
        load_model,
    )

    model = Path(model).resolve(strict=True)
    descriptor, origin, package = load_model(model)
    spec = descriptor["metadata"].get("mtp") or {}
    if spec.get("packet_table") and not descriptor["metadata"].get("packet_table"):
        descriptor["metadata"]["packet_table"] = spec.pop("packet_table")
    with atomic_model(Path(output), engine) as staging:
        operator = clone_model(model, staging, origin)
        digest = commit_package(staging, operator, descriptor, package)
    return dict(execution_package=digest, model=str(output), weight_payloads_unchanged=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.model, args.output, args.engine), indent=2))


if __name__ == "__main__":
    main()

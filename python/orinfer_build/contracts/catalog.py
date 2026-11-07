"""Compile typed model bindings into Rust; publish only shared kernel definitions.

Input expanded bindings are offline compiler output, never an online format.
The generated Rust module uses SDK role IDs and shared argument groups. It is
sealed to the exact catalog and contains no weight payload or execution order.
Registered Rust adapters continue to compose and validate execution programs.
"""

from orinfer_build.common.paths import ROOT, work_path, workspace_name


import copy
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import tempfile


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
    signature = hashlib.sha256(canonical(dict(catalog=catalog, bindings=bindings))).hexdigest()
    catalog["binding_signature"] = signature
    package["catalog"] = catalog
    return package, bindings


def expand(catalog, bindings):
    """Offline exact roundtrip proof, before compiling the native mapping."""
    kernels = []
    dynamic = []
    contracts = []
    for binding in bindings:
        d = copy.deepcopy(catalog["definitions"][binding["definition"]])
        parameters = d.pop("parameters")
        if parameters != [a["kind"] for a in binding["args"]]:
            raise ValueError("Native parameters differ from the compiled ABI")
        for field in ("module", "source", "host_abi"):
            d[field] = catalog["assets"][d[field]]
        kernels.append(dict(d, name=binding["name"], grid=binding["grid"], args=binding["args"]))
        if binding["dynamic"] is not None:
            dynamic.append(dict(catalog["dynamic"][binding["dynamic"]], name=binding["name"]))
        if binding["plan_contract"]:
            contracts.append(binding["name"])
    return kernels, dynamic, contracts


def rust_source(catalog, bindings):
    roles, role_ids = [], {}

    def role(name):
        if name not in role_ids:
            role_ids[name] = len(roles)
            roles.append(name)
        return role_ids[name]

    def argument(a):
        kind = a["kind"]
        if kind == "buffer":
            return f"A::Buffer({role(a['name'])})"
        if kind == "buffer_slice":
            return f"A::BufferSlice({role(a['name'])},{a['offset']})"
        if kind == "f32":
            bits = struct.unpack("<I", struct.pack("<f", a["value"]))[0]
            return f"A::F32({bits})"
        constructors = {"i32": "I32", "u32": "U32", "i64": "I64", "u64": "U64"}
        if kind not in constructors:
            raise ValueError("Unknown native argument type")
        return f"A::{constructors[kind]}({a['value']})"

    from orinfer_build.contracts.binding_templates import factor as layer_templates

    registry = layer_templates(bindings)
    parameters = [
        "&[" + ",".join(argument(a) for a in group) + "]" for group in registry["parameters"]
    ]
    choices = []
    for table in registry["choices"]:
        records = []
        for variant in table:
            dynamic = "None" if variant["dynamic"] is None else f"Some({variant['dynamic']})"
            grid = ",".join(str(v) for v in variant["grid"])
            indices = ",".join(str(v) for v in variant["arguments"])
            records.append(
                f"V{{suffix:{json.dumps(variant['suffix'])},definition:{variant['definition']},"
                f"grid:[{grid}],arguments:&[{indices}],dynamic:{dynamic},"
                f"plan_contract:{str(variant['plan_contract']).lower()}}}"
            )
        choices.append("&[" + ",".join(records) + "]")
    templates = []
    for template in registry["templates"]:
        layers = []
        for instance in template["layers"]:
            layer = "None" if instance["layer"] is None else f"Some({instance['layer']})"
            layers.append(f"L{{layer:{layer},parameters:{instance['parameters']}}}")
        templates.append(
            f"T{{prefix:{json.dumps(template['prefix'])},suffix:{json.dumps(template['suffix'])},"
            f"choice:{template['choice']},layers:&[{','.join(layers)}]}}"
        )
    identity = hashlib.sha256(
        canonical({k: v for k, v in catalog.items() if k != "binding_signature"})
    ).hexdigest()
    # JSON string quoting is also valid for these restricted compiler role names.
    if any(any(ord(c) < 32 or c in '\\"' for c in name) for name in roles):
        raise ValueError("Invalid compiler tensor role name")
    return (
        "use orinfer_model_sdk::catalog::{RoleArgument as A,BindingRegistry as R,BindingTemplate as T,LayerBinding as L,BindingVariant as V};\n"
        f'pub const SIGNATURE: &str = "{catalog["binding_signature"]}";\n'
        f'pub const CATALOG_ID: &str = "{identity}";\n'
        "pub const ROLES: &[&str] = &["
        + ",".join(json.dumps(v, ensure_ascii=False) for v in roles)
        + "];\n"
        "pub const PARAMETERS: &[&[A]] = &[\n" + ",\n".join(parameters) + "\n];\n"
        "pub const CHOICES: &[&[V]] = &[\n" + ",\n".join(choices) + "\n];\n"
        "pub const TEMPLATES: &[T] = &[\n" + ",\n".join(templates) + "\n];\n"
        "pub const REGISTRY: R = R{roles:ROLES,parameters:PARAMETERS,choices:CHOICES,templates:TEMPLATES};\n"
    )


def seal(package, directory, engine=None):
    """Finalize an offline expanded package. Requires the pinned Rust toolchain."""
    factored, bindings = factor(package)
    reconstructed, dynamic, contracts = expand(factored["catalog"], bindings)

    def normalize(k):
        return {
            field: value for field, value in k.items() if field != "readonly_arguments" or value
        }

    if {k["name"]: normalize(k) for k in reconstructed} != {
        k["name"]: normalize(k) for k in package["kernels"]
    }:
        raise ValueError("Compiled catalog failed exact binding roundtrip")
    if {v["name"]: v for v in dynamic} != {
        v["name"]: dict({"capacity": 128}, **v) for v in package.get("dynamic_batch_kernels", [])
    } or set(contracts) != construction_contracts(package):
        raise ValueError("Compiled catalog failed dynamic/plan contract roundtrip")
    factored["execution"] = build_library(factored, bindings, directory, engine)
    return factored


def build_library(package, bindings, directory, engine=None):
    from orinfer_build.packaging.package import execution_identity

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "lib/model.so"
    library.parent.mkdir(exist_ok=True)
    build_root = work_path("builds", workspace_name())
    build_root.mkdir(parents=True, exist_ok=True)
    # One shared Cargo target; retain the lock through copying its exact output.
    with (
        (build_root / "native-build.lock").open("w") as lock,
        tempfile.TemporaryDirectory(prefix="orinfer-native-", dir=build_root) as temp,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = Path(temp) / "bindings.rs"
        source.write_text(rust_source(package["catalog"], bindings))
        environment = dict(os.environ, ORINFER_BINDINGS_SOURCE=str(source))
        target = build_root / "native-cargo"
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--offline",
                "-p",
                "orinfer-models",
                "--target-dir",
                str(target),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        # Never overwrite a shared hardlink to an existing immutable library.
        import shutil

        temporary = library.with_suffix(".tmp")
        shutil.copyfile(target / "release/liborinfer_models.so", temporary)
        temporary.replace(library)
        # Corresponding source for the generated LGPL native binding module.
        generated = library.with_name("bindings.rs")
        generated_temporary = generated.with_suffix(".tmp")
        shutil.copyfile(source, generated_temporary)
        generated_temporary.replace(generated)
        instructions = library.with_name("BUILD.txt")
        instructions_temporary = instructions.with_suffix(".tmp")
        instructions_temporary.write_text(
            "Use the matching release source.tar.xz (included in the CLI archive).\n"
            "From that workspace, rebuild this library with:\n"
            "ORINFER_BINDINGS_SOURCE=/absolute/path/to/lib/bindings.rs "
            "cargo build --release --locked --offline -p orinfer-models\n"
            "The generated module contains tensor roles and ABI bindings, no weights.\n"
        )
        instructions_temporary.replace(instructions)
    return execution_identity(library, engine, package["architecture"], package["compute_policy"])


def finalize(model, output, engine=None):
    """Rebuild only native metadata/library, hardlinking unchanged weight assets."""
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

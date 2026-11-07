"""Rebuild short Qwen attention from a package's measured decode geometry.

The native package, large-prefill choices, weights and request buffer contract
are reused. Run through tools/environments/cuda/run.sh with the fixed compiler image.
"""

from orinfer_build.common.paths import ROOT


import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from orinfer_build.common.files import source_hash, write_json
from orinfer_build.packaging.publication import atomic_model, clone_model, load_model
from orinfer_build.compiler.host_compiler import Compiler, private_roles


def decode_geometry(kernel, source, dimension):
    call = re.search(
        r"= attn_split_count\(1,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,[^;]*)?\);", source
    )
    if not call:
        raise ValueError("Expected a measured single-query attention split contract")
    group, block_m, target, splits = map(int, call.groups())
    if target != 16 or block_m != 16:
        raise ValueError("Rebuild requires the measured compact decode tile")
    # The compiled shared tile has one K and one V allocation. Its offset
    # difference certifies block_n independently of source-local array names.
    k = re.search(r"(?:void|half_t|half)\* k = [^;]+buf_dyn_shmem \+ (\d+)", source)
    v = re.search(r"(?:void|half_t|half)\* v = [^;]+buf_dyn_shmem \+ (\d+)", source)
    if k is None or v is None:
        raise ValueError("Expected certified shared K/V tile offsets")
    block_n = abs(int(v[1]) - int(k[1])) // (dimension * 2)
    if block_n not in (32, 64):
        raise ValueError("Unsupported measured KV tile")
    return dict(
        block_m=block_m,
        block_n=block_n,
        head_group=group,
        max_splits=splits,
        threads=kernel["block"][0],
        stages=int("packed_k" in source),
        independent_queries=True,
    )


def short_alias(name):
    match = re.fullmatch(r"(.*/attention(?:_merge)?)/rows/(\d+)-(\d+)", name)
    if match and int(match[2]) <= 8:
        return None if int(match[3]) <= 8 else f"{match[1]}/rows/9-{match[3]}"
    return name


def rebuild(args):
    import tilelang
    import torch
    from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
    from orinfer_build.compiler.compile import instantiate
    from orinfer_build.compiler.private_rows import wrap_private
    from orinfer_build.models.qwen35.parallel import attention_pointers, factory
    from orinfer_build.packaging.compact import compact_package

    source = args.model.resolve(strict=True)
    data, origin, package = load_model(source)
    if data["architecture"] not in ("qwen3_5", "qwen3_asr"):
        raise ValueError("Expected a native Qwen text execution package")
    metadata = data["metadata"]
    if (metadata.get("mtp") or {}).get("verification_table"):
        raise ValueError("Use the full Qwen builder for owner-parallel verification tables")
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    qh, kvh = text["num_attention_heads"], text["num_key_value_heads"]
    dimension = text.get("head_dim", text["hidden_size"] // qh)
    original = {k["name"]: k for k in package["kernels"]}
    sample = next(
        k
        for k in package["kernels"]
        if re.fullmatch(r"qwen_rows/state/layer\d+/attention", k["name"])
    )
    identity = source_hash(
        ROOT,
        [
            ROOT / p
            for p in (
                "kernels/cuda/model/attention_rows.py",
                "kernels/cuda/model/kv_int8.py",
                "kernels/shared/rows.py",
                "kernels/cuda/model/independent_rows.py",
                "python/orinfer_build/compiler/compile.py",
                "python/orinfer_build/compiler/private_rows.py",
                "python/orinfer_build/compiler/host_compiler.py",
                "python/orinfer_build/models/qwen35/parallel.py",
                "python/orinfer_build/models/qwen35/rebuild_attention.py",
                "tools/environments/cuda/uv.lock",
            )
        ],
    )
    changed = []
    with atomic_model(args.target) as stage:
        operator = clone_model(source, stage, origin)
        buffers = {b["name"]: b for b in metadata["buffers"]}
        compiler = Compiler(origin, operator, args.output, buffers)
        saved_geometry = package["toolchain"].get("causal_attention")
        geometry = (
            json.loads(saved_geometry)
            if saved_geometry
            else decode_geometry(sample, compiler.read(sample["source"]).decode(), dimension)
        )
        if geometry["head_group"] != qh // kvh:
            raise ValueError("Expected the measured complete GQA head group")
        compiler_identity = dict(
            tilelang=tilelang.__version__, cuda=torch.version.cuda, target="sm_87"
        )
        identity = hashlib.sha256(
            json.dumps(
                dict(
                    source=identity,
                    geometry=geometry,
                    context=metadata["max_context"],
                    query_heads=qh,
                    kv_heads=kvh,
                    dimension=dimension,
                    compiler=compiler_identity,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        dynamic = {d["name"]: d for d in package["dynamic_batch_kernels"]}
        renamed = {name: short_alias(name) for name in original}
        package["kernels"] = [
            dict(k, name=renamed[k["name"]])
            for k in package["kernels"]
            if renamed[k["name"]] is not None
        ]
        package["dynamic_batch_kernels"] = [
            dict(d, name=renamed[d["name"]])
            for d in dynamic.values()
            if renamed[d["name"]] is not None
        ]
        package["plan_kernel_contracts"] = sorted(
            {renamed.get(n, n) for n in package["plan_kernel_contracts"] if renamed.get(n, n)}
        )
        columns = metadata["batch_layout"]["state_columns"]
        layers = len(metadata["batch_layout"]["layers"])
        for old in list(original.values()):
            name = old["name"]
            normal = re.fullmatch(r"qwen_rows/state/layer(\d+)/(attention|attention_merge)", name)
            shared = re.fullmatch(
                r"qwen_rows/(?:layer\d+/private|mtp)/(attention|attention_merge)", name
            )
            if not normal and not shared:
                continue
            role = normal[2] if normal else shared[1]
            if normal:
                fn = factory(role, metadata["max_context"], geometry, text)
                source_text = compiler.read(old["source"]).decode()
                roles, offsets = private_roles(source_text, columns, layers * len(columns))
                modes, pointers = {}, compiler.pointers(old)
                for buf in fn.buffer_map.values():
                    if buf.name in roles:
                        modes[buf.name] = roles[buf.name]
                    elif buf.name in pointers:
                        modes[buf.name] = ("row", pointers[buf.name]["name"])
                fn, bindings = wrap_private(
                    fn,
                    128,
                    modes,
                    columns,
                    pointers,
                    "QwenRowPointers",
                    table_offset=int(normal[1]) * len(columns) * 8,
                    row_stride=layers * len(columns),
                    offsets=offsets,
                    batch_parameters=("independent_instances",),
                )
                emitted, launch = compiler.emit(
                    old,
                    name,
                    bindings,
                    128,
                    build=lambda fn=fn: fn,
                    key=f"causal-owner-{role}-{identity}",
                )
                package["kernels"] = [k for k in package["kernels"] if k["name"] != name]
                package["dynamic_batch_kernels"] = [
                    k for k in package["dynamic_batch_kernels"] if k["name"] != name
                ]
            else:
                name += "/rows/1-8"
                bindings = compiler.pointers(old)
                if role == "attention":
                    prefix = old["name"].rsplit("/", 1)[0]
                    prepared = original[
                        prefix + ("/kv_prepare" if prefix.endswith("/mtp") else "/prepare")
                    ]
                    bindings = attention_pointers(bindings, compiler.pointers(prepared))
                options = {
                    k: v for k, v in geometry.items() if k not in ("block_n", "threads", "stages")
                }
                options.update(
                    query_heads=qh,
                    kv_heads=kvh,
                    dimension=dimension,
                    gated=data["architecture"] != "qwen3_asr",
                )
                fn = causal_attention_int8 if role == "attention" else merge_attention_int8
                if role == "attention":
                    options.update(
                        context=metadata["max_context"],
                        block_n=geometry["block_n"],
                        threads=geometry["threads"],
                        stages=geometry["stages"],
                    )
                emitted, launch = compiler.emit(
                    old,
                    name,
                    bindings,
                    8,
                    build=lambda fn=fn, options=options: instantiate(fn, 8, **options),
                    key=f"causal-shared-{role}-{identity}",
                )
                package["plan_kernel_contracts"].append(name)
            package["kernels"].append(emitted)
            package["dynamic_batch_kernels"].append(launch)
            changed.append(name)
        if not changed:
            raise ValueError("No Qwen attention bindings rebuilt")
        package["plan_kernel_contracts"] = sorted(set(package["plan_kernel_contracts"]))
        package["toolchain"] = dict(
            package["toolchain"],
            causal_attention=json.dumps(geometry, sort_keys=True),
            causal_attention_source=identity,
            causal_attention_compiler=json.dumps(compiler_identity, sort_keys=True),
        )
        metadata["toolchain"] = package["toolchain"]
        digest = compact_package(operator, package, stage / "cache/packages")
        data["execution_package"] = digest
        write_json(stage / "cache/model.json", data)
        shutil.rmtree(operator)
    result = dict(
        model=str(args.target.resolve()),
        execution_package=digest,
        base_model=str(source),
        geometry=geometry,
        rebuilt=changed,
        weight_payloads_unchanged=True,
    )
    write_json(args.output / "result.json", result)
    print("COMPLETE", digest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    rebuild(parser.parse_args())


if __name__ == "__main__":
    main()

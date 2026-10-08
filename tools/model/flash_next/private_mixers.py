"""Publish request-parallel GDN/QSA kernels into a native Flash batch package.

Weights and their precision stay unchanged. The existing TileLang M1 functions
are wrapped before lowering, retaining their CTA arithmetic and synchronization.
Only the offline publisher imports the compiler; serving stays in Rust.
"""

import argparse
import ast
import math
import re
from pathlib import Path

from tools.model.publication import (
    atomic_model,
    clone_model,
    commit_package,
    file_hash,
    load_model,
    source_path,
)

TABLE = "FlashBatchPointers"
SCRATCH = "FlashPrivateScratch"


def identity(name):
    return re.sub(r"^State_\d+_", "State_", name)


def factory(label, capacity):
    from kernels.model import flash_next, qsa, qsa_attention
    from kernels.model.gdn_sequence import gdn_sequence
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from tools.model.flash_next.batching import kind

    name = kind(label)
    mrope = label.startswith("vision-")
    if name == "gdn-conv":
        return gdn_conv_prep.__wrapped__(
            B=1,
            tokens=1,
            normalize_round_fp16=False,
            conv_product_round_fp16=False,
            weight_dtype="float32",
        )
    if name == "gdn-sequence":
        return gdn_sequence.__wrapped__(1, in_place=True)
    if name == "qsa-prepare":
        return flash_next.qsa_prepare.__wrapped__(
            1, capacity, is_neox_style=True, staged=True, mrope=mrope
        )
    if name == "qsa-kv-store":
        return qsa.kv_store.__wrapped__(1, capacity)
    if name == "index-query":
        return qsa.index_query.__wrapped__(1, capacity if mrope else 0)
    if name == "index-compress":
        return qsa.index_compress.__wrapped__(1, capacity, mrope=mrope)
    if name == "index-pending":
        return qsa.index_pending.__wrapped__(1)
    if name == "index-scores":
        return qsa.index_scores.__wrapped__(1, capacity)
    if name in ("index-hist", "index-choose"):
        fn = qsa.radix_histogram if name == "index-hist" else qsa.radix_choose
        return fn.__wrapped__(1, capacity, int(label.rsplit("-", 1)[1]))
    if name in ("index-counts", "index-offsets", "index-scatter"):
        fn = {
            "index-counts": qsa.selection_counts,
            "index-offsets": qsa.selection_offsets,
            "index-scatter": qsa.selection_scatter,
        }[name]
        return fn.__wrapped__(1, capacity)
    if name == "qsa-sparse":
        if label != "qsa-sparse-packed-1-8-32-128-False":
            raise ValueError("Unsupported M1 QSA geometry")
        return qsa_attention.sparse_attention.__wrapped__(
            1, capacity, packed=True, single_buffer=False
        )
    if name == "qsa-merge":
        return qsa.sparse_merge.__wrapped__(1, 8)
    raise ValueError(f"Unsupported private Flash factory: {label}")


def upgrade(model, destination, output):
    import shutil
    import subprocess
    import tilelang
    import torch
    from kernels.model.independent_rows import independent_rows
    from tools.model.flash_next.batching import PROFILES, kind
    from tools.model.flash_next.prepare import DTYPES, Publisher
    from tools.operators.abi import parse_host
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    metadata = data["metadata"]
    if data["architecture"] != "flash_next" or metadata.get("batch_profiles") != list(PROFILES):
        raise ValueError("Expected a native Flash decode-batch package")
    buffers = {b["name"]: b for b in metadata["buffers"]}
    if TABLE in buffers:
        raise ValueError("Private Flash batching already installed")
    operator = clone_model(model, destination, origin)
    boundary = set(metadata["batch_layout"]["row_strides"])
    layers = {}
    private = set()
    scratch = {}
    element_bytes = {
        "f16": 2,
        "bf16": 2,
        "f32": 4,
        "i8": 1,
        "u8": 1,
        "i32": 4,
        "u32": 4,
        "i64": 8,
        "u64": 8,
    }
    for kernel in package["kernels"]:
        if (
            not kernel["name"].startswith("flash_batch_m2/layer")
            or "/private/" not in kernel["name"]
        ):
            continue
        label = Path(kernel["module"]["file"]).parent.name
        if kind(label) is None or kind(label).startswith("ple-"):
            continue
        layer = int(kernel["name"].split("/")[1][5:])
        path = source_path(origin, kernel["host_abi"]["file"])
        (launch,) = parse_host(path.read_text())
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "call"
        )
        api = [a.arg for a in node.args.args if a.arg not in ("kernels", "stream")]
        roles = {}
        for arg, abi in zip(kernel["args"], launch["ordered_arguments"]):
            if arg["kind"] != "buffer":
                raise ValueError("M1 private mixer must use complete static buffers")
            name = arg["name"]
            if name.startswith("BatchM2_"):
                name = name.removeprefix("BatchM2_")
                if name not in boundary:
                    raise ValueError("Unknown private mixer row boundary")
                mode = "row"
            elif data["buffer_scopes"][name] == "weights":
                mode = "weight"
            else:
                mode = "private"
                if data["buffer_scopes"][name] == "workspace":
                    b = buffers[name]
                    scratch[name] = math.prod(b["shape"]) * element_bytes[b["dtype"]]
                    private.add(SCRATCH)
                else:
                    private.add(identity(name))
            variable = abi["value"].removesuffix(".data_ptr()")
            if variable not in api:
                raise ValueError("Unknown M1 host argument")
            roles[variable] = (mode, name)
        layers.setdefault(layer, []).append((label, roles))
    if set(layers) != set(range(48)):
        raise ValueError("Missing private mixer layers")
    scratch_offsets = {}
    scratch_bytes = 0
    for name, size in sorted(scratch.items()):
        scratch_bytes = (scratch_bytes + 255) // 256 * 256
        scratch_offsets[name] = scratch_bytes
        scratch_bytes += size
    buffers[SCRATCH] = dict(
        name=SCRATCH,
        dtype="u8",
        shape=[(scratch_bytes + 255) // 256 * 256],
        layout="native_contiguous",
        alignment=256,
        access="read_write",
    )
    data["buffer_scopes"][SCRATCH] = "sequence"
    columns = {name: i for i, name in enumerate(sorted(private))}
    metadata["batch_layout"]["state_columns"] = list(columns)
    buffers[TABLE] = dict(
        name=TABLE,
        dtype="u64",
        shape=[128, 48, len(columns)],
        layout="native_contiguous",
        alignment=256,
        access="read_write",
    )
    data["buffer_scopes"][TABLE] = "workspace"
    types = {v[0]: getattr(torch, k) for k, v in DTYPES.items()}
    types["u64"] = torch.uint64
    tensors = {
        name: torch.empty(b["shape"], dtype=types[b["dtype"]], device="meta")
        for name, b in buffers.items()
    }
    logical = {t.untyped_storage()._cdata: name for name, t in tensors.items()}

    class ShapeModel:
        def __init__(self):
            self.output = output

    class MixerPublisher(Publisher):
        def buffer(self, tensor, name=None, scope="workspace"):
            return logical[
                tensor.untyped_storage()._cdata
            ], tensor.storage_offset() * tensor.element_size()

        def bind(self, *args):
            op = super().bind(*args)
            for field in ("module", "source", "host_abi"):
                asset = self.kernels[-1][field]
                if not asset["file"].startswith("flash-private/"):
                    asset["file"] = "flash-private/" + asset["file"]
            return op

    publish = MixerPublisher(ShapeModel(), destination)
    publish.cache = operator / "flash-private"
    (publish.cache / "kernels").mkdir(parents=True)
    compiled = {}
    signatures = {}
    for rows in PROFILES:
        for layer, entries in layers.items():
            section = f"layer{layer}"
            for label, old_roles in entries:
                key = f"independent-dynamic-{label}"
                if key not in compiled:
                    fn = factory(label, metadata["max_context"])
                    roles = {
                        name: (
                            mode,
                            SCRATCH
                            if value in scratch
                            else identity(value)
                            if mode == "private"
                            else value,
                        )
                        for name, (mode, value) in old_roles.items()
                    }
                    offsets = {
                        name: scratch_offsets[value]
                        for name, (_, value) in old_roles.items()
                        if value in scratch
                    }
                    fn, inputs = independent_rows(
                        fn,
                        128,
                        roles,
                        columns,
                        row_stride=48 * len(columns),
                        dynamic=True,
                        offsets=offsets,
                    )
                    kernel = tilelang.compile(
                        fn,
                        out_idx=[],
                        execution_backend="nvrtc",
                        target={"kind": "cuda", "arch": "sm_87"},
                    )
                    export_kernel(kernel, output / "aot" / key)
                    compiled[key] = kernel
                    signatures[key] = inputs
                args = []
                for name in signatures[key]:
                    if name is None:
                        args.append(tensors[TABLE][:, layer])
                    elif name == "__rows__":
                        args.append(rows)
                    else:
                        mode, role = old_roles[name]
                        args.append(tensors[f"BatchM{rows}_{role}" if mode == "row" else role])
                op = publish.bind(f"flash_private_m{rows}", section, key, args, rows)
                publish.groups[f"flash_private_m{rows}"].setdefault(section, []).append(op)
                if kind(label) == "gdn-conv":
                    from kernels.model.independent_rows import history_commit

                    copy_key = "independent-history-dynamic"
                    if copy_key not in compiled:
                        if scratch_offsets["M1_HistoryOut"] != 0:
                            raise ValueError(
                                "Convolution history must start the packed scratch arena"
                            )
                        kernel = history_commit(
                            len(columns),
                            columns[SCRATCH],
                            columns["State_conv"],
                            row_stride=48 * len(columns),
                        )
                        export_kernel(kernel, output / "aot" / copy_key)
                        compiled[copy_key] = kernel
                    op = publish.bind(
                        f"flash_private_m{rows}",
                        section,
                        copy_key,
                        [tensors[TABLE][:, layer], rows],
                        rows,
                    )
                    publish.groups[f"flash_private_m{rows}"].setdefault(section, []).append(op)
            print(
                "PRIVATE STAGE",
                rows,
                layer,
                len(publish.groups[f"flash_private_m{rows}"][section]),
                flush=True,
            )
    metadata["buffers"] = list(buffers.values())
    package["kernels"].extend(publish.kernels)
    package["buffer_contracts"] = [
        {k: v for k, v in b.items() if k != "data"} for b in metadata["buffers"]
    ]
    from tools.model.package import model_library

    path = operator / "lib/model.so"
    path.unlink()
    shutil.copyfile(model_library(), path)
    package["execution"]["library"]["sha256"] = file_hash(path)
    commit_package(destination, operator, data, package)
    subprocess.run(
        [str(Path("target/release/orinfer").resolve()), "plan-model", str(destination)],
        check=True,
        stdout=(output / "registered-plan.json").open("w"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-cache", type=Path)
    a = p.parse_args()
    if a.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(a.compile_cache, a.output / "cache/0.1.15")
    with atomic_model(a.model_output) as staging:
        upgrade(a.model.resolve(), staging, a.output)


if __name__ == "__main__":
    main()

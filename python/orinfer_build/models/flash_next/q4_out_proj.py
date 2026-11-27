"""Convert Flash Next ssm_out (36L) and attn_output (12L) projections from W8 to single-resident row-Q4."""

from itertools import pairwise
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


_DTYPE_BYTES = {
    "i8": 1,
    "u8": 1,
    "f16": 2,
    "bf16": 2,
    "u16": 2,
    "i16": 2,
    "f32": 4,
    "i32": 4,
    "u32": 4,
    "u64": 8,
}

# GPU-measured fastest Q4 decode tiles for N=2560, K=6144 on Orin SM87
DEFAULT_ROW_SPECS = {
    "1-1": (1, False, dict(algorithm="dp4a", block_m=1, block_n=16)),
    "2-2": (2, False, dict(algorithm="dp4a", block_m=2, block_n=8)),
    "3-3": (3, False, dict(algorithm="dp4a", block_m=3, block_n=16)),
    "4-4": (4, False, dict(algorithm="dp4a", block_m=4, block_n=16)),
    "5-7": (
        1024,
        True,
        dict(algorithm="tensor_core", block_m=8, block_n=32, block_k=256, stages=1),
    ),
    "8-8": (8, False, dict(algorithm="tensor_core", block_m=8, block_n=32, block_k=256, stages=1)),
    "9-15": (
        1024,
        True,
        dict(algorithm="tensor_core", block_m=16, block_n=32, block_k=256, stages=1),
    ),
    "16-16": (
        16,
        False,
        dict(algorithm="tensor_core", block_m=16, block_n=32, block_k=256, stages=1),
    ),
    "17-31": (
        1024,
        True,
        dict(algorithm="tensor_core", block_m=32, block_n=32, block_k=256, stages=2),
    ),
    "32-32": (
        32,
        False,
        dict(algorithm="tensor_core", block_m=32, block_n=32, block_k=256, stages=2),
    ),
}


def _tensor_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _quantize_shards(weights_dir: Path, data: dict, package: dict, compiler):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from orinfer_build.quantization.row_q4_gpu import fit

    index_path = weights_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    wmap = index["weight_map"]

    by_shard: dict[str, list[tuple[int, str]]] = {}
    for layer in range(48):
        role = "attn_output" if layer % 4 == 3 else "ssm_out"
        w_name = f"W_blk.{layer}.{role}.weight_weight"
        s_name = f"W_blk.{layer}.{role}.weight_scale"
        if w_name not in compiler.buffers or s_name not in compiler.buffers:
            raise ValueError(f"Missing W8 source buffer for layer {layer} {role}")
        shard = wmap[compiler.buffers[w_name]["data"]["tensor"]]
        if wmap[compiler.buffers[s_name]["data"]["tensor"]] != shard:
            raise ValueError(f"Mismatched shard for {w_name} and {s_name}")
        by_shard.setdefault(shard, []).append((layer, role))

    retired_buffers = set()
    new_buffer_entries = []
    new_contract_entries = []

    for shard, entries in sorted(by_shard.items()):
        shard_path = weights_dir / shard
        tensors = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        for layer, role in entries:
            w_name = f"W_blk.{layer}.{role}.weight_weight"
            s_name = f"W_blk.{layer}.{role}.weight_scale"
            w_tensor_key = compiler.buffers[w_name]["data"]["tensor"]
            s_tensor_key = compiler.buffers[s_name]["data"]["tensor"]
            w_i8 = tensors.pop(w_tensor_key)
            w_sc = tensors.pop(s_tensor_key)
            if list(w_i8.shape) != [2560, 6144] or w_i8.dtype != torch.int8:
                raise ValueError(
                    f"Unexpected W8 shape/dtype for {w_name}: {w_i8.shape} {w_i8.dtype}"
                )
            if list(w_sc.shape) != [2560] or w_sc.dtype != torch.float16:
                raise ValueError(f"Unexpected W8 scale for {s_name}: {w_sc.shape} {w_sc.dtype}")

            raw = w_i8.numpy().astype(np.float32) * w_sc.numpy().astype(np.float32)[:, None]
            rng = np.random.default_rng(20261002 + 1000 + layer)
            signs = rng.choice(np.array([-1, 1], np.int8), 6144)
            fitted = fit(raw, signs)
            packed_u16 = np.ascontiguousarray(fitted.packed).view(np.uint16)
            scales_f16 = np.ascontiguousarray(fitted.scales)
            signs_i8 = np.ascontiguousarray(fitted.signs)

            del wmap[w_tensor_key]
            del wmap[s_tensor_key]
            meta.pop(f"orin.layout.{w_tensor_key}", None)
            meta.pop(f"orin.layout.{s_tensor_key}", None)
            retired_buffers.add(w_name)
            retired_buffers.add(s_name)

            prefix = f"Q4_blk.{layer}.{role}"
            for suffix, dtype_str, arr in (
                ("packed", "u16", packed_u16),
                ("scale", "f16", scales_f16),
                ("signs", "i8", signs_i8),
            ):
                bname = f"{prefix}_{suffix}"
                tensors[bname] = torch.from_numpy(arr.copy())
                wmap[bname] = shard
                meta[f"orin.layout.{bname}"] = "native_contiguous"
                data_ref = dict(tensor=bname, sha256=_tensor_sha256(arr))
                desc = dict(
                    name=bname,
                    dtype=dtype_str,
                    shape=list(arr.shape),
                    alignment=256,
                    layout="native_contiguous",
                    access="read",
                    data=data_ref,
                )
                new_buffer_entries.append(desc)
                new_contract_entries.append({k: v for k, v in desc.items() if k != "data"})

        tmp = shard_path.with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata=meta)
        tmp.replace(shard_path)
        del tensors
        torch.cuda.empty_cache()

    # Update metadata buffers, buffer_scopes, buffer_contracts, and exact weight_bytes
    data["metadata"]["buffers"] = [
        b for b in data["metadata"]["buffers"] if b["name"] not in retired_buffers
    ] + new_buffer_entries
    for dead in retired_buffers:
        data["buffer_scopes"].pop(dead, None)
        compiler.buffers.pop(dead, None)
    for b in new_buffer_entries:
        data["buffer_scopes"][b["name"]] = "weights"
        compiler.buffers[b["name"]] = b
    package["buffer_contracts"] = [
        b for b in package["buffer_contracts"] if b["name"] not in retired_buffers
    ] + new_contract_entries

    total_weight_bytes = sum(
        math.prod(b["shape"]) * _DTYPE_BYTES[b["dtype"]]
        for b in data["metadata"]["buffers"]
        if b.get("data") is not None
    )
    data["metadata"]["weight_bytes"] = total_weight_bytes
    if "metadata" in index and "total_size" in index["metadata"]:
        index["metadata"]["total_size"] = total_weight_bytes
    tmp_idx = index_path.with_suffix(".tmp")
    tmp_idx.write_text(json.dumps(index, indent=2) + "\n")
    tmp_idx.replace(index_path)


def publish(data, package, manifest, compiler, *, weights_dir: Path, row_specs=None):
    from kernels.cuda.model.int8_projection import int8_projection
    from kernels.cuda.model.q4_projection import q4_decode, unpack_q4
    from kernels.cuda.model.rotation_a8 import rotate_activation_a8

    if package["architecture"] != "flash_next":
        raise ValueError("Q4 output projection conversion requires Flash Next")
    specs = DEFAULT_ROW_SPECS if row_specs is None else row_specs

    _quantize_shards(weights_dir, data, package, compiler)

    # Register shared temporary INT8 expansion workspace for large prefill (>=769 rows)
    exp_name = "FlashPrefillExpandedOutProj"
    exp_desc = dict(
        name=exp_name,
        dtype="i8",
        shape=[2560, 6144],
        alignment=256,
        layout="native_contiguous",
        access="read_write",
    )
    if exp_name not in compiler.buffers:
        data["metadata"]["buffers"].append(dict(exp_desc, data=None))
        data["buffer_scopes"][exp_name] = "workspace"
        package["buffer_contracts"].append(exp_desc)
        compiler.buffers[exp_name] = exp_desc
    expanded_op = dict(kind="buffer", name=exp_name)

    kernels = {k["name"]: k for k in package["kernels"]}
    launches = {d["name"]: d for d in package["dynamic_batch_kernels"]}
    contracts = set(package["plan_kernel_contracts"])
    replaced, added, new_launches = {}, {}, {}

    def q4_ops(layer: int):
        role = "attn_output" if layer % 4 == 3 else "ssm_out"
        prefix = f"Q4_blk.{layer}.{role}"
        return (
            dict(kind="buffer", name=f"{prefix}_packed"),
            dict(kind="buffer", name=f"{prefix}_scale"),
            dict(kind="buffer", name=f"{prefix}_signs"),
        )

    def replace_rot(rot_name: str, signs_op: dict, capacity: int, symbolic: bool):
        old = kernels[rot_name]
        p = compiler.pointers(old)
        if set(p) != {"Q", "S", "X"}:
            raise ValueError(f"Unexpected quantize_a8 ABI for {rot_name}: {set(p)}")
        bindings = {"X": p["X"], "Signs": signs_op, "Q": p["Q"], "S": p["S"]}
        item, launch = compiler.emit(
            old,
            rot_name,
            bindings,
            capacity,
            build=partial(
                rotate_activation_a8.__wrapped__,
                capacity,
                6144,
                dynamic_rows=symbolic,
                registers=True,
            ),
            key=f"q4-rot-reg-k6144-c{capacity}-{'dyn' if symbolic else 'static'}",
            symbolic=symbolic,
        )
        replaced[rot_name] = item
        if symbolic:
            new_launches[rot_name] = launch

    def proj_bindings(old, packed_op, scale_op):
        p = compiler.pointers(old)
        if set(p) != {"A", "Output", "TokenScale", "Weight", "WeightScale"}:
            raise ValueError(f"Unexpected W8 projection ABI for {old['name']}: {set(p)}")
        return {
            "A": p["A"],
            "AS": p["TokenScale"],
            "W": packed_op,
            "S": scale_op,
            "Out": p["Output"],
        }

    # Find predecessor + projection slots in decode, flash_decode, and prefill_m4096
    layer_re = re.compile(r"W_blk\.(\d+)\.(ssm_out|attn_output)\.weight_weight")
    proj_slots = {}
    for prog_name, ops in manifest["programs"].items():
        if prog_name not in ("decode", "prefill") and not (
            prog_name.startswith("flash_decode/layer") and prog_name.endswith("/post")
        ):
            continue
        kops = [op["name"] for op in ops if op["kind"] == "kernel"]
        for prev_name, cur_name in pairwise(kops):
            cur = kernels[cur_name]
            matched = None
            for a in cur["args"]:
                m = layer_re.fullmatch(a.get("name", ""))
                if m:
                    matched = int(m.group(1))
                    break
            if matched is not None:
                proj_slots[(prog_name, matched)] = (prev_name, cur_name)

    if len(proj_slots) != 48 * 3:
        raise ValueError(
            f"Expected 144 (decode/flash_decode/prefill) out_proj slots, found {len(proj_slots)}"
        )

    for (prog_name, layer), (rot_name, proj_name) in sorted(proj_slots.items()):
        packed_op, scale_op, signs_op = q4_ops(layer)
        old_proj = kernels[proj_name]
        pb = proj_bindings(old_proj, packed_op, scale_op)

        if prog_name == "decode":
            replace_rot(rot_name, signs_op, 1, False)
            item, _ = compiler.emit(
                old_proj,
                proj_name,
                pb,
                1,
                build=partial(
                    q4_decode,
                    1,
                    2560,
                    6144,
                    algorithm="dp4a",
                    block_m=1,
                    block_n=16,
                    output_dtype="float16",
                    dynamic_rows=False,
                ),
                key="q4-outproj-2560x6144-m1-dp4a-bn16",
                symbolic=False,
            )
            replaced[proj_name] = item

        elif prog_name.startswith("flash_decode/"):
            cap = launches[proj_name]["capacity"]
            replace_rot(rot_name, signs_op, cap, True)
            base_item, base_launch = compiler.emit(
                old_proj,
                proj_name,
                pb,
                cap,
                build=partial(
                    q4_decode,
                    cap,
                    2560,
                    6144,
                    algorithm="tensor_core",
                    block_m=64,
                    block_n=64,
                    block_k=128,
                    stages=2,
                    output_dtype="float16",
                    dynamic_rows=True,
                ),
                key=f"q4-outproj-2560x6144-tc-m64-n64-k128-s2-c{cap}",
                symbolic=True,
            )
            replaced[proj_name] = base_item
            new_launches[proj_name] = base_launch

            prefix = f"{proj_name}/rows/"
            for kname in list(kernels):
                if not kname.startswith(prefix):
                    continue
                suffix = kname[len(prefix) :]
                if suffix not in specs:
                    raise ValueError(f"Unsupported flash_decode out_proj row range: {kname}")
                rows, symbolic, cfg = specs[suffix]
                key_parts = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
                item, launch = compiler.emit(
                    kernels[kname],
                    kname,
                    pb,
                    rows if not symbolic else cap,
                    build=partial(
                        q4_decode,
                        rows if not symbolic else cap,
                        2560,
                        6144,
                        output_dtype="float16",
                        dynamic_rows=symbolic,
                        **cfg,
                    ),
                    key=f"q4-outproj-2560x6144-{suffix}-{key_parts}",
                    symbolic=symbolic,
                )
                replaced[kname] = item
                if not symbolic:
                    launch["capacity"] = cap
                new_launches[kname] = launch

        elif prog_name == "prefill":
            cap = launches[proj_name]["capacity"]
            if cap != 4096:
                raise ValueError(f"Unexpected prefill out_proj capacity: {cap}")
            replace_rot(rot_name, signs_op, cap, True)
            base_item, base_launch = compiler.emit(
                old_proj,
                proj_name,
                pb,
                cap,
                build=partial(
                    q4_decode,
                    cap,
                    2560,
                    6144,
                    algorithm="tensor_core",
                    block_m=64,
                    block_n=64,
                    block_k=128,
                    stages=2,
                    output_dtype="float16",
                    dynamic_rows=True,
                ),
                key=f"q4-outproj-2560x6144-tc-m64-n64-k128-s2-c{cap}",
                symbolic=True,
            )
            replaced[proj_name] = base_item
            new_launches[proj_name] = base_launch

            for first, last, block_m in ((769, cap - 1, 128), (cap, cap, 64)):
                base = f"{proj_name}/choice/{first}-{last}/1"
                for idx, (bindings, key, build, symbolic) in enumerate(
                    (
                        (
                            {"W": packed_op, "Output": expanded_op},
                            "q4-expand-2560x6144",
                            partial(unpack_q4.__wrapped__, 2560, 6144),
                            False,
                        ),
                        (
                            {
                                "A": pb["A"],
                                "Weight": expanded_op,
                                "WeightScale": scale_op,
                                "TokenScale": pb["AS"],
                                "Output": pb["Out"],
                            },
                            f"q4-expanded-int8-{cap}-2560x6144-bm{block_m}",
                            partial(
                                int8_projection.__wrapped__,
                                cap,
                                2560,
                                6144,
                                block_m=block_m,
                                block_n=128,
                                block_k=128,
                                dynamic_rows=True,
                            ),
                            True,
                        ),
                    )
                ):
                    cname = f"{base}/k{idx}"
                    item, launch = compiler.emit(
                        old_proj,
                        cname,
                        bindings,
                        cap,
                        build=build,
                        key=key,
                        symbolic=symbolic,
                    )
                    added[cname] = item
                    if symbolic:
                        new_launches[cname] = launch
                    contracts.add(cname)

    kernels.update(replaced)
    kernels.update(added)
    launches.update(new_launches)
    package["kernels"] = list(kernels.values())
    package["dynamic_batch_kernels"] = list(launches.values())
    package["plan_kernel_contracts"] = sorted(contracts)
    return sorted(replaced.keys() | added.keys())

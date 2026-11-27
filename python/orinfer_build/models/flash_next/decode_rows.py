"""Publish measured flash_decode row specializations (small-batch projections, fast HC inject, fused HC combine-norm)."""

from itertools import pairwise
from functools import partial


def publish(package, manifest, compiler):
    from kernels.cuda.model.hyperconnection import (
        bf16_projection_small,
        hc_combine_norm,
        hc_injection_fast,
    )
    from kernels.cuda.model.int8_swiglu import int8_swiglu
    from kernels.cuda.model.q4_projection import q4_decode

    kernels = {k["name"]: k for k in package["kernels"]}
    launches = {d["name"]: d for d in package["dynamic_batch_kernels"]}
    added = {}

    def add_choice(old, suffix, build, key, *, pointers=None, allow_replace=False):
        name = f"{old['name']}/rows/{suffix}"
        if name in kernels and not allow_replace:
            raise ValueError(f"Overlapping flash_decode row specialization: {name}")
        capacity = launches[old["name"]]["capacity"]
        bindings = compiler.pointers(old) if pointers is None else pointers
        item, launch = compiler.emit(
            old,
            name,
            bindings,
            capacity,
            build=build,
            key=key,
            symbolic=True,
        )
        added[name] = item
        launches[name] = launch

    for name, old in list(kernels.items()):
        if not name.startswith("flash_decode/") or "/rows/" in name or "/skip/" in name:
            continue
        if name not in launches:
            continue
        capacity = launches[name]["capacity"]
        pointers = compiler.pointers(old)
        roles = set(pointers)

        # 0. Fast vectorized HC injection (4 x 10240 BF16): replace /rows/1-87
        if roles == {"A", "W", "C"} and name.startswith("flash_decode/layer"):
            w = compiler.buffers[pointers["W"]["name"]]
            if w["dtype"] == "bf16" and w["shape"] == [4, 10240]:
                add_choice(
                    old,
                    "1-87",
                    partial(
                        hc_injection_fast.__wrapped__,
                        capacity,
                        10240,
                        outputs=4,
                        dynamic_rows=True,
                    ),
                    f"flash-hc-inject-fast-v2-10240-{capacity}",
                    pointers={"A": pointers["A"], "Weight": pointers["W"], "Output": pointers["C"]},
                    allow_replace=True,
                )
            continue

        # 1. BF16 HC down (320 x 10240, splits=4) and MoE router (512 x 2560, splits=1)
        if old["symbol"] == "bf16_rows_kernel_kernel" and roles == {"A", "Weight", "Output"}:
            w = compiler.buffers[pointers["Weight"]["name"]]
            out = compiler.buffers[pointers["Output"]["name"]]
            if w["dtype"] != "bf16" or out["dtype"] != "f32" or len(w["shape"]) != 2:
                continue
            n_dim, k_dim = w["shape"]
            if (n_dim, k_dim) == (320, 10240) and old["grid"][1:] == [5, 4]:
                add_choice(
                    old,
                    "3-32",
                    partial(
                        bf16_projection_small.__wrapped__,
                        capacity,
                        n_dim,
                        k_dim,
                        splits=4,
                        block_m=8,
                        block_n=64,
                        block_k=64,
                        stages=2,
                        threads=128,
                        dynamic_rows=True,
                    ),
                    f"flash-bf16-hc-down-m8-n64-k64-s2-{capacity}",
                )
            elif (n_dim, k_dim) == (512, 2560) and old["grid"][1:] == [8, 1]:
                add_choice(
                    old,
                    "4-32",
                    partial(
                        bf16_projection_small.__wrapped__,
                        capacity,
                        n_dim,
                        k_dim,
                        splits=1,
                        block_m=8,
                        block_n=64,
                        block_k=64,
                        stages=2,
                        threads=128,
                        dynamic_rows=True,
                    ),
                    f"flash-bf16-router-m8-n64-k64-s2-{capacity}",
                )
            continue

        # 2. Shared expert SwiGLU W8 (640 x 2560): fill gaps 5-7, 9-15, and 17-31
        if roles == {"A", "Gate", "Up", "GateScale", "UpScale", "TokenScale", "Output"}:
            w = compiler.buffers[pointers["Gate"]["name"]]
            if w["dtype"] == "i8" and w["shape"] == [640, 2560]:
                for suffix, bm in (("5-7", 16), ("9-15", 16), ("17-31", 32)):
                    add_choice(
                        old,
                        suffix,
                        partial(
                            int8_swiglu.__wrapped__,
                            capacity,
                            640,
                            2560,
                            64,
                            True,
                            algorithm="tensor_core",
                            block_m=bm,
                        ),
                        f"flash-swiglu-tc-m{bm}-n64-{capacity}",
                    )
            continue

        # 3. LM Head Q4 (248320 x 2560, f32): fill gaps 5-7, 9-15, and 17-31
        if name == "flash_decode/head/k6" and roles == {"A", "W", "S", "AS", "Out"}:
            w = compiler.buffers[pointers["W"]["name"]]
            if w["dtype"] == "u16" and w["shape"] == [248320, 640]:
                add_choice(
                    old,
                    "5-7",
                    partial(
                        q4_decode,
                        capacity,
                        248320,
                        2560,
                        algorithm="tensor_core",
                        block_m=8,
                        block_n=32,
                        block_k=256,
                        stages=2,
                        output_dtype="float32",
                        dynamic_rows=True,
                    ),
                    f"flash-head-q4-tc-m8-n32-k256-s2-{capacity}",
                )
                add_choice(
                    old,
                    "9-15",
                    partial(
                        q4_decode,
                        capacity,
                        248320,
                        2560,
                        algorithm="tensor_core",
                        block_m=16,
                        block_n=64,
                        block_k=256,
                        stages=1,
                        output_dtype="float32",
                        dynamic_rows=True,
                    ),
                    f"flash-head-q4-tc-m16-n64-k256-s1-{capacity}",
                )
                add_choice(
                    old,
                    "17-31",
                    partial(
                        q4_decode,
                        capacity,
                        248320,
                        2560,
                        algorithm="tensor_core",
                        block_m=32,
                        block_n=64,
                        block_k=256,
                        stages=1,
                        output_dtype="float32",
                        dynamic_rows=True,
                    ),
                    f"flash-head-q4-tc-m32-n64-k256-s1-{capacity}",
                )

    # 4. Dynamic HC combine + norm fusion across all 48 intra-post and 46 inter-layer pairs
    #    (layer 0->1 has PLE in between; layer 47->head copies residual to FlashMixedHeadResidual).
    def is_combine_norm_pair(first_name, second_name):
        c_k = kernels.get(first_name)
        n_k = kernels.get(second_name)
        if c_k is None or n_k is None or first_name not in launches or second_name not in launches:
            return None
        if launches[first_name]["capacity"] != launches[second_name]["capacity"]:
            return None
        c_p = compiler.pointers(c_k)
        n_p = compiler.pointers(n_k)
        if (
            set(c_p) == {"Block", "Inject", "Output", "Residual"}
            and set(n_p) == {"Normed", "Residual", "Weight"}
            and c_p["Output"] == n_p["Residual"]
        ):
            return c_k, n_k, c_p, n_p
        return None

    fuse_pairs = []
    programs = manifest["programs"]
    for layer in range(48):
        post_ops = [
            op["name"]
            for op in programs.get(f"flash_decode/layer{layer}/post", [])
            if op["kind"] == "kernel"
        ]
        for a, b in pairwise(post_ops):
            if is_combine_norm_pair(a, b) is not None:
                fuse_pairs.append((a, b))
        if 1 <= layer <= 46 and post_ops:
            next_pre = [
                op["name"]
                for op in programs.get(f"flash_decode/layer{layer + 1}/pre", [])
                if op["kind"] == "kernel"
            ]
            if next_pre and is_combine_norm_pair(post_ops[-1], next_pre[0]) is not None:
                fuse_pairs.append((post_ops[-1], next_pre[0]))

    if len(fuse_pairs) != 94:
        raise ValueError(f"Expected 94 flash_decode HC combine-norm pairs, found {len(fuse_pairs)}")

    for comb_name, norm_name in fuse_pairs:
        c_k, n_k, c_p, n_p = is_combine_norm_pair(comb_name, norm_name)
        cap = launches[comb_name]["capacity"]
        bindings = {
            "Block": c_p["Block"],
            "Residual": c_p["Residual"],
            "Inject": c_p["Inject"],
            "Weight": n_p["Weight"],
            "Output": c_p["Output"],
            "Normed": n_p["Normed"],
        }
        add_choice(
            c_k,
            f"1-{cap}",
            partial(
                hc_combine_norm.__wrapped__,
                cap,
                2560,
                streams=4,
                dtype="float16",
                dynamic_rows=True,
            ),
            f"flash-hc-combine-norm-rows-v2-2560-4-{cap}",
            pointers=bindings,
            allow_replace=True,
        )
        skip_name = f"{norm_name}/skip/1-{cap}"
        added[skip_name] = dict(n_k, name=skip_name)
        launches[skip_name] = dict(launches[norm_name], name=skip_name)

    if not added:
        raise ValueError("No flash_decode small-row specializations produced")
    kernels.update(added)
    package["kernels"] = list(kernels.values())
    package["dynamic_batch_kernels"] = list(launches.values())
    package["plan_kernel_contracts"] = sorted(set(package["plan_kernel_contracts"]) | added.keys())
    return sorted(added)

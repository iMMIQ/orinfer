"""Publish the exact HC prefill path for non-aligned row counts."""

import math
from functools import partial


def publish(package, manifest, compiler, *, scopes):
    from kernels.cuda.model.hc_prefill import hc_up_mix_prefill

    capacity, hidden, rank, tile = 4096, 2560, 320, 128
    plan = next(p for p in manifest["prefill_plans"] if p["chunk_tokens"] == capacity)
    program = {
        op["name"] for op in manifest["programs"][plan["prefill_program"]] if op["kind"] == "kernel"
    }
    shapes = {
        "A": ("bf16", capacity * rank),
        "Residual": ("f16", capacity * 4 * hidden),
        "InvRms": ("f32", capacity * 4),
        "Out": ("f16", capacity * hidden),
    }
    launches = {v["name"]: v for v in package["dynamic_batch_kernels"]}
    kernels = {k["name"]: k for k in package["kernels"]}
    replacements, small = {}, {}
    for old in package["kernels"]:
        name = old["name"]
        if name not in program or old["symbol"] != "hc_prefill_up_mix_kernel":
            continue
        pointers = compiler.pointers(old)
        if set(pointers) != set(shapes) | {"W", "Weight"}:
            raise ValueError(f"Unexpected HC prefill arguments: {name}")
        for role, (dtype, elements) in shapes.items():
            operand = pointers[role]
            buf = compiler.buffers[operand["name"]]
            if (
                operand["kind"] != "buffer"
                or scopes[operand["name"]] != "workspace"
                or buf["dtype"] != dtype
                or buf["layout"] != "native_contiguous"
                or math.prod(buf["shape"]) < elements
            ):
                raise ValueError(f"HC prefill requires a full workspace for {role}: {name}")
        if len({pointers[role]["name"] for role in shapes}) != len(shapes):
            raise ValueError(f"Aliased HC prefill workspace: {name}")
        if launches[name]["capacity"] != capacity:
            raise ValueError(f"Incompatible HC prefill row capacity: {name}")

        # The final tile may read inactive workspace rows. MMA keeps rows
        # independent and the kernel stores only the live output extent.
        rebuilt, launch = compiler.emit(
            old,
            name,
            pointers,
            capacity,
            build=partial(
                hc_up_mix_prefill.__wrapped__,
                capacity,
                hidden,
                rank,
                block_m=tile,
                dynamic_rows=True,
                padded_inputs=True,
            ),
            key="flash-hc-prefill-up-padded",
        )
        if rebuilt["args"] != old["args"] or launch != launches[name]:
            raise ValueError(f"Changed HC prefill operand or launch contract: {name}")
        # Keep the original path for very short tails where the padding
        # costs more than the masked operands. The aligned choice starts at 128.
        short = f"{name}/choice/1-15/1/k0"
        if short not in kernels:
            small[short] = dict(old, name=short)
            launches[short] = dict(launch, name=short)
        replacements[name] = rebuilt
    if not replacements:
        raise ValueError("Missing native HC prefill up/mix bindings")
    kernels.update(replacements)
    kernels.update(small)
    package["kernels"] = list(kernels.values())
    package["dynamic_batch_kernels"] = list(launches.values())
    package["plan_kernel_contracts"] = sorted(set(package["plan_kernel_contracts"]) | small.keys())
    return sorted(replacements)

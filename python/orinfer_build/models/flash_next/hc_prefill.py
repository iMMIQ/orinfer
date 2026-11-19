"""Publish exact HC prefill and large packet projections."""

import math
from functools import partial


def _publish_packet_up_mix(kernels, launches, compiler, scopes):
    from kernels.cuda.model.hc_prefill import hc_up_mix_packet

    capacity, first, hidden, rank = 1024, 97, 2560, 320
    rows = {
        "A": capacity * rank,
        "Normed": capacity * 4 * hidden,
        "Out": capacity * hidden,
    }
    additions, obsolete = {}, set()
    for name, old in kernels.items():
        if (
            not name.startswith("flash_decode/layer")
            or "/rows/" in name
            or old["symbol"] != "hc_up13_vector_project_kernel"
        ):
            continue
        pointers = compiler.pointers(old)
        if not rows.keys() <= pointers.keys() or launches[name]["capacity"] != capacity:
            raise ValueError(f"Invalid packet HC geometry: {name}")
        bindings = {role: pointers[role] for role in rows}
        for role, extent in rows.items():
            operand = bindings[role]
            buffer = compiler.buffers[operand["name"]]
            if (
                operand["kind"] != "buffer"
                or scopes[operand["name"]] != "workspace"
                or buffer["dtype"] != "f16"
                or buffer["layout"] != "native_contiguous"
                or math.prod(buffer["shape"]) != extent
            ):
                raise ValueError(f"Invalid packet HC workspace for {role}: {name}")

        # The HC13 cache encodes this same immutable BF16 bank losslessly.
        low = pointers["Lo"]["name"]
        weight = low.removeprefix("HC13_").removesuffix("_Lo")
        buffer = compiler.buffers[weight]
        if (
            low != f"HC13_{weight}_Lo"
            or pointers["Hi"]["name"] != f"HC13_{weight}_Hi"
            or buffer["dtype"] != "bf16"
            or buffer["shape"] != [4 * hidden, rank]
            or buffer["layout"] != "native_contiguous"
            or scopes[weight] != "weights"
            or buffer.get("data") is None
        ):
            raise ValueError(f"Invalid packet HC BF16 source: {name}")
        bindings["W"] = dict(kind="buffer", name=weight)

        prefix = name + "/rows/"
        for choice in kernels.keys():
            if choice.startswith(prefix):
                low, high = map(int, choice[len(prefix) :].split("-"))
                if high < first:
                    continue
                if low < first or compiler.pointers(kernels[choice]) != bindings:
                    raise ValueError(f"Overlapping packet HC choice: {choice}")
                obsolete.add(choice)
        choice = f"{prefix}{first}-{capacity}"
        additions[choice], launches[choice] = compiler.emit(
            old,
            choice,
            bindings,
            capacity,
            build=partial(hc_up_mix_packet.__wrapped__, capacity, hidden, rank, dynamic_rows=True),
            key="flash-hc-packet-up-bf16-64",
        )
    for name in obsolete - additions.keys():
        del kernels[name]
        del launches[name]
    kernels.update(additions)
    return additions.keys(), obsolete


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
    packet, obsolete = _publish_packet_up_mix(kernels, launches, compiler, scopes)
    package["kernels"] = list(kernels.values())
    package["dynamic_batch_kernels"] = list(launches.values())
    package["plan_kernel_contracts"] = sorted(
        (set(package["plan_kernel_contracts"]) - obsolete) | small.keys() | packet
    )
    return sorted(replacements.keys() | packet)

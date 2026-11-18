"""Rebuild the exact Flash prefill recurrence with register-resident operands."""

from functools import partial
from math import prod


def publish(package, manifest, compiler, *, scopes):
    from kernels.cuda.model.gdn_sequence import _gdn_sequence

    capacity, first = 4096, 32
    text = package["config_signature"]["text"]
    hk, hv = text["linear_num_key_heads"], text["linear_num_value_heads"]
    if (text["linear_key_head_dim"], text["linear_value_head_dim"]) != (128, 128):
        raise ValueError("Unsupported Flash prefill GDN head dimensions")
    plan = next(p for p in manifest["prefill_plans"] if p["chunk_tokens"] == capacity)
    program = {
        op["name"] for op in manifest["programs"][plan["prefill_program"]] if op["kind"] == "kernel"
    }
    kernels = {k["name"]: k for k in package["kernels"]}
    launches = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    geometry = {
        "Q": ("f16", hk * capacity * 128),
        "K": ("f16", hk * capacity * 128),
        "V": ("f16", hv * capacity * 128),
        "G": ("f32", capacity * hv),
        "Beta": ("f32", capacity * hv),
        "State": ("f32", hv * 128 * 128),
        "Out": ("f16", capacity * hv * 128),
    }
    build = partial(
        _gdn_sequence,
        capacity,
        in_place=True,
        value_tile=16,
        threads=32,
        prefetch_qk=True,
        dynamic_rows=True,
        token_unroll=8,
        key_heads=hk,
        value_heads=hv,
        register_qk=True,
    )
    extra = {}
    for name in sorted(program):
        old = kernels[name]
        if not any(a["kind"] == "buffer" and a["name"].endswith("_gdn") for a in old["args"]):
            continue
        operands = compiler.pointers(old)
        launch = launches[name]
        if (
            set(operands) != set(geometry)
            or len({v["name"] for v in operands.values()}) != len(operands)
            or launch["capacity"] != capacity
        ):
            raise ValueError(f"Unexpected Flash GDN recurrence contract: {name}")
        for role, (dtype, elements) in geometry.items():
            operand = operands[role]
            buffer = compiler.buffers[operand["name"]]
            if (
                operand["kind"] != "buffer"
                or scopes[operand["name"]] != ("sequence" if role == "State" else "workspace")
                or buffer["dtype"] != dtype
                or buffer["layout"] != "native_contiguous"
                or prod(buffer["shape"]) != elements
            ):
                raise ValueError(f"Invalid Flash GDN prefill operand {role}: {name}")
        choice = f"{name}/choice/{first}-{capacity}/1/k0"
        if any(n.startswith(f"{name}/choice/") and n != choice for n in kernels):
            raise ValueError(f"Overlapping Flash GDN prefill choices: {name}")
        rebuilt, rebuilt_launch = compiler.emit(
            old,
            choice,
            operands,
            capacity,
            build=build,
            key=f"causal-gdn-flash-register-{capacity}-{hk}-{hv}",
        )
        if rebuilt["args"] != old["args"] or rebuilt_launch != dict(launch, name=choice):
            raise ValueError(f"Changed Flash GDN operand or launch contract: {name}")
        extra[choice], launches[choice] = rebuilt, rebuilt_launch
    if len(extra) != 36:
        raise ValueError("Missing Flash prefill GDN recurrence bindings")
    kernels.update(extra)
    package["kernels"] = list(kernels.values())
    package["dynamic_batch_kernels"] = list(launches.values())
    package["plan_kernel_contracts"] = sorted(set(package["plan_kernel_contracts"]) | extra.keys())
    return sorted(extra)

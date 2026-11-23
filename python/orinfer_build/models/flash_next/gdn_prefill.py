"""Rebuild the Flash prefill GDN recurrence with register-resident short path + exact-decay chunkwise TC."""

from functools import partial
from math import prod


def publish(data, package, manifest, compiler):
    from kernels.cuda.model.gdn_chunk import prepare_precise, recurrent_precise
    from kernels.cuda.model.gdn_sequence import _gdn_sequence

    scopes = data["buffer_scopes"]
    capacity, short_first, chunk_first, chunk = 4096, 32, 128, 16
    text = package["config_signature"]["text"]
    hk, hv = text["linear_num_key_heads"], text["linear_num_value_heads"]
    if (text["linear_key_head_dim"], text["linear_value_head_dim"]) != (128, 128):
        raise ValueError("Unsupported Flash prefill GDN head dimensions")
    plan = next(p for p in manifest["prefill_plans"] if p["chunk_tokens"] == capacity)
    program = {
        op["name"] for op in manifest["programs"][plan["prefill_program"]] if op["kind"] == "kernel"
    }

    # Register reusable chunkwise GDN scratch workspaces (L, D, GP)
    ws = {}
    for role, stride in (("L", chunk), ("D", chunk), ("GP", 2)):
        buf_name = f"FlashPrefillGdn{role}"
        desc = dict(
            name=buf_name,
            dtype="f32",
            shape=[hv, capacity, stride],
            alignment=256,
            layout="native_contiguous",
            access="read_write",
        )
        if buf_name in compiler.buffers:
            existing = compiler.buffers[buf_name]
            if (
                any(existing.get(k) != v for k, v in desc.items())
                or existing.get("data") is not None
                or scopes.get(buf_name) != "workspace"
            ):
                raise ValueError(f"Invalid GDN chunk workspace: {buf_name}")
        else:
            data["metadata"]["buffers"].append(dict(desc, data=None))
            scopes[buf_name] = "workspace"
            package["buffer_contracts"].append(desc)
            compiler.buffers[buf_name] = desc
        ws[role] = dict(kind="buffer", name=buf_name)

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
    build_seq = partial(
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
    build_prep = partial(prepare_precise, capacity, HK=hk, HV=hv, C=chunk, dynamic_rows=True)
    build_rec = partial(
        recurrent_precise, capacity, HK=hk, HV=hv, C=chunk, BV=64, dynamic_rows=True
    )

    extra, retired = {}, set()
    slots = 0
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
        for existing in list(kernels):
            if existing.startswith(f"{name}/choice/"):
                retired.add(existing)
        short = f"{name}/choice/{short_first}-{chunk_first - 1}/1/k0"
        rebuilt, rebuilt_launch = compiler.emit(
            old,
            short,
            operands,
            capacity,
            build=build_seq,
            key=f"causal-gdn-flash-register-{capacity}-{hk}-{hv}",
        )
        if rebuilt["args"] != old["args"] or rebuilt_launch != dict(launch, name=short):
            raise ValueError(f"Changed Flash GDN operand or launch contract: {name}")
        extra[short], launches[short] = rebuilt, rebuilt_launch

        bindings = operands | ws
        for i, (factory, roles, key) in enumerate(
            (
                (
                    build_prep,
                    ("Q", "K", "G", "Beta", "L", "D", "GP"),
                    f"gdn-precise-prep-{capacity}-{hk}-{hv}-{chunk}",
                ),
                (
                    build_rec,
                    ("Q", "K", "V", "Beta", "L", "D", "GP", "State", "Out"),
                    f"gdn-precise-rec-bv64-{capacity}-{hk}-{hv}-{chunk}",
                ),
            )
        ):
            choice = f"{name}/choice/{chunk_first}-{capacity}/1/k{i}"
            extra[choice], launches[choice] = compiler.emit(
                old,
                choice,
                {r: bindings[r] for r in roles},
                capacity,
                build=factory,
                key=key,
            )
        slots += 1
    if slots != 36:
        raise ValueError(f"Expected 36 Flash prefill GDN recurrence bindings, found {slots}")
    for dead in retired - extra.keys():
        kernels.pop(dead, None)
        launches.pop(dead, None)
    kernels.update(extra)
    package["kernels"] = list(kernels.values())
    package["dynamic_batch_kernels"] = list(launches.values())
    package["plan_kernel_contracts"] = sorted(
        (set(package["plan_kernel_contracts"]) - retired) | extra.keys()
    )
    return sorted(extra)

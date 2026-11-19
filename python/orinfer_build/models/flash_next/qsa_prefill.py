"""Stage short-context FP16 KV for the native prefill attention pass."""

from functools import partial


def publish(data, package, manifest, compiler):
    from kernels.cuda.model.qsa import sparse_merge
    from kernels.cuda.model.qsa_attention import expand_prefill_kv
    from orinfer_build.tuning.auxiliary_tuning import (
        _QSA_ATTENTION,
        _QSA_PREFILL_PASSES,
        _QSA_STATE,
        _packed_qsa,
    )

    context = manifest["max_context"]
    limit = min(context, 8192)
    first = 256
    expanded = {}
    for role in ("K", "V"):
        name = f"FlashPrefillQsaExpanded{role}"
        desc = dict(
            name=name,
            dtype="f16",
            shape=[limit, 2, 256],
            alignment=256,
            layout="native_contiguous",
            access="read_write",
        )
        scopes = data["buffer_scopes"]
        if name in compiler.buffers:
            if any(compiler.buffers[name].get(k) != v for k, v in desc.items()) or (
                scopes[name] != "workspace" or compiler.buffers[name].get("data") is not None
            ):
                raise ValueError(f"Invalid QSA expansion workspace: {name}")
        else:
            data["metadata"]["buffers"].append(dict(desc, data=None))
            scopes[name] = "workspace"
            package["buffer_contracts"].append(desc)
            compiler.buffers[name] = desc
        expanded[f"Expanded{role}"] = dict(kind="buffer", name=name)

    kernels = {k["name"]: k for k in package["kernels"]}
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    added, launches = {}, {}
    for plan in manifest["prefill_plans"]:
        rows = plan["chunk_tokens"]
        operations = manifest["programs"][plan["prefill_program"]]
        for op, following in zip(operations, operations[1:]):
            if op["kind"] != "kernel":
                continue
            name = op["name"]
            old = kernels[name]
            if old["grid"] != [rows, 2, 8]:
                continue
            operands = compiler.pointers(old)
            if operands.keys() != _QSA_ATTENTION:
                continue
            if dynamic[name]["capacity"] != rows or old["block"] != [128, 1, 1]:
                raise ValueError(f"Unexpected native QSA attention geometry: {name}")
            merge = kernels.get(following.get("name"))
            if merge is None or merge["grid"] != [rows, 24, 1]:
                raise ValueError(f"Missing native QSA partition merge: {name}")
            merge_args = compiler.pointers(merge)
            position = operands["Position"]
            if (
                merge_args.keys() - {"Position"} != {"Max", "Den", "Out", "Gate", "Output"}
                or any(operands[role] != merge_args[role] for role in ("Max", "Den", "Out"))
                or merge_args.get("Position", position) != position
                or dynamic[merge["name"]]["capacity"] != rows
            ):
                raise ValueError(f"Unexpected native QSA merge contract: {merge['name']}")
            stage = {role: operands[role] for role in _QSA_STATE} | expanded
            # The merge and attention choices use the same row interval.
            # The merge reads only partitions the attention CTA populated.
            for kernel, i, role, bindings, build in (
                (old, 0, "expand", stage, partial(expand_prefill_kv, rows, context, limit)),
                (
                    old,
                    1,
                    "attention",
                    operands | expanded,
                    partial(_packed_qsa, rows, context, limit, skip_empty=True),
                ),
                (
                    merge,
                    0,
                    "merge",
                    merge_args | {"Position": position},
                    partial(sparse_merge.__wrapped__, rows, dynamic_rows=True, split_size=256),
                ),
            ):
                choice = f"{kernel['name']}/choice/{first}-{rows}/1/k{i}"
                key = f"flash-qsa-live-{role}-{context}-{limit}-{rows}"
                added[choice], launches[choice] = compiler.emit(
                    kernel,
                    choice,
                    bindings,
                    rows,
                    build=build,
                    key=key,
                    pass_configs=_QSA_PREFILL_PASSES,
                )
    if not added:
        raise ValueError("Missing native QSA prefill expansion bindings")
    kernels.update(added)
    dynamic.update(launches)
    package["kernels"] = list(kernels.values())
    package["dynamic_batch_kernels"] = list(dynamic.values())
    package["plan_kernel_contracts"] = sorted(set(package["plan_kernel_contracts"]) | added.keys())
    return sorted(added)

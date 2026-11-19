"""Publish Flash Next native prefill QSA selections and FP16 KV staging."""

from functools import partial


def dense_selection(local, merge, compiler, rows, context):
    from kernels.cuda.model.qsa_selection import select, merge as merge_selection

    producer = compiler.pointers(local)
    consumer = compiler.pointers(merge)
    if (
        producer.keys() != {"Scores", "Position", "Selected"}
        or consumer.keys() != {"Candidates", "Position", "Selected"}
        or producer["Selected"] != consumer["Candidates"]
        or producer["Position"] != consumer["Position"]
        or local["grid"] != [rows, 2, 1]
        or merge["grid"] != [rows, 1, 1]
        or any(kernel["block"] != [128, 1, 1] for kernel in (local, merge))
    ):
        raise ValueError(f"Unexpected native QSA selection pair: {local['name']}")
    direct_blocks = 2048
    return (
        (
            local,
            producer,
            partial(
                select.__wrapped__,
                rows,
                context,
                True,
                query_major=True,
                dense_scores=True,
                direct_blocks=direct_blocks,
            ),
            "partial",
        ),
        (
            merge,
            consumer | {"Scores": producer["Scores"]},
            partial(
                merge_selection.__wrapped__,
                rows,
                2,
                True,
                threads=256,
                warp_sort=True,
                score_capacity=context,
                direct_blocks=direct_blocks,
            ),
            "merge",
        ),
    )


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
            if old["symbol"] == "qsa_select_partial_kernel":
                merge = kernels.get(following.get("name"))
                if merge is None or merge["symbol"] != "qsa_select_merge_kernel":
                    raise ValueError(f"Missing native QSA candidate merge: {name}")
                for kernel, bindings, build, role in dense_selection(
                    old, merge, compiler, rows, context
                ):
                    choice = f"{kernel['name']}/choice/1-{rows}/1/k0"
                    if dynamic[kernel["name"]]["capacity"] != rows:
                        raise ValueError(f"Unexpected native QSA row capacity: {kernel['name']}")
                    added[choice], launches[choice] = compiler.emit(
                        kernel,
                        choice,
                        bindings,
                        rows,
                        build=build,
                        key=f"flash-qsa-bounded-select-{role}-{context}-{rows}",
                    )
                continue
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
                    "attention-paired",
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

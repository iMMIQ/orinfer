"""Publish expert dispatch and wide expert tiles for Flash Next native prefill."""

from functools import partial

from orinfer_build.tuning.expert_tuning import tile_capacity


def _token_workspace(data, package, compiler, rows, width, top):
    shapes = {
        "Low": ("u32", [width // 64, rows, 8]),
        "High": ("u32", [width // 64, rows, 8]),
        "Sum": ("i32", [rows]),
        "Sources": ("i32", [rows * top]),
    }
    workspace = {}
    for role, (dtype, shape) in shapes.items():
        name = f"FlashPrefillExpertToken{role}"
        desc = dict(
            name=name,
            dtype=dtype,
            shape=shape,
            alignment=256,
            layout="native_contiguous",
            access="read_write",
        )
        if name in compiler.buffers:
            buffer = compiler.buffers[name]
            if (
                any(buffer.get(key) != value for key, value in desc.items())
                or buffer.get("data") is not None
                or data["buffer_scopes"].get(name) != "workspace"
            ):
                raise ValueError(f"Invalid expert token workspace: {name}")
        else:
            compiler.buffers[name] = desc
            data["metadata"]["buffers"].append(dict(desc, data=None))
            data["buffer_scopes"][name] = "workspace"
            package["buffer_contracts"].append(desc)
        workspace[role] = dict(kind="buffer", name=name)
    return workspace


def _dispatch_choices(data, package, compiler, preceding, projection, kernels, dynamic, rows):
    from kernels.cuda.model.e8p_prefill import expand_expert_slots, pack_expert_tokens

    first = 256
    for op in preceding:
        slot = op["name"]
        wide = f"{slot}/choice/56-{rows}/1/k0"
        small = f"{slot}/choice/56-{first - 1}/1/k0"
        old = kernels.get(wide, kernels.get(small))
        if old is None or old["symbol"] != "dispatch_planes_kernel":
            continue
        old_args = compiler.pointers(old)
        expected = {
            "A",
            "Scale",
            "IDs",
            "RelativeSlot",
            "RowOffsets",
            "Low",
            "High",
            "Sum",
            "DS",
            "SlotMap",
        }
        if old_args.keys() != expected or dynamic[old["name"]]["capacity"] != rows:
            raise ValueError(f"Unexpected native expert dispatch contract: {slot}")
        projection_args = compiler.pointers(projection)
        if any(old_args[role] != projection_args[role] for role in ("Low", "High", "Sum")) or (
            old_args["DS"] != projection_args["AS"]
            or old_args["RowOffsets"] != projection_args["Offsets"]
        ):
            raise ValueError(f"Expert dispatch and projection views differ: {slot}")
        text = package["config_signature"]["text"]
        width, experts, top = text["hidden_size"], text["num_experts"], text["num_experts_per_tok"]
        workspace = _token_workspace(data, package, compiler, rows, width, top)
        pack_args = {
            role: old_args[role] for role in ("A", "IDs", "RelativeSlot", "RowOffsets", "SlotMap")
        } | workspace
        expand_args = {role: old_args[role] for role in ("Scale", "Low", "High", "Sum", "DS")} | {
            **{f"Src{role}": workspace[role] for role in ("Low", "High", "Sum")},
            "Sources": workspace["Sources"],
        }
        bindings = {small: dict(old, name=small)}
        launches = {small: dict(dynamic[old["name"]], name=small)}
        for i, arguments, build in (
            (0, pack_args, partial(pack_expert_tokens, rows, width, experts, top)),
            (1, expand_args, partial(expand_expert_slots, rows, width, top)),
        ):
            name = f"{slot}/choice/{first}-{rows}/1/k{i}"
            bindings[name], launches[name] = compiler.emit(
                old,
                name,
                arguments,
                rows,
                build=build,
                key=f"flash-expert-token-dispatch-{rows}-{width}-{experts}-{top}-{i}",
            )
        return bindings, launches, {wide} & kernels.keys()
    return {}, {}, set()


def publish(data, package, manifest, compiler):
    from kernels.cuda.model import e8p_prefill, moe

    # SM87 gate/up benefits from smaller output tiles and fixed short-tail
    # loops. Real prefill routing favors the wider row tile from 1200 rows.
    first, block_m = 1200, 32
    text = package["config_signature"]["text"]
    experts, top = text["num_experts"], text["num_experts_per_tok"]
    kernels = {k["name"]: k for k in package["kernels"]}
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    added, launches, obsolete = {}, {}, set()
    for plan in manifest["prefill_plans"]:
        rows = plan["chunk_tokens"]
        if rows < first:
            continue
        capacity = tile_capacity(rows, experts, top, block_m)
        suffix = f"/choice/{first}-{rows}/1/k0"
        groups = {}
        for op in manifest["programs"][plan["prefill_program"]]:
            if op["kind"] == "kernel":
                name = op["name"]
                groups.setdefault(name.rsplit("/", 1)[0], []).append(kernels[name])
        for operations in groups.values():
            projections = [k for k in operations if k["symbol"] == "exact_e8p_int4_kernel"]
            if not projections:
                continue
            if len(projections) != 2:
                raise ValueError("Expected paired Flash prefill expert projections")
            preceding = operations[: operations.index(projections[0])]
            prefix, tiles = None, None
            for old in preceding:
                pointers = compiler.pointers(old)
                if set(pointers) == {"Counts", "RowOffsets", "TileOffsets", "TileCount"}:
                    prefix = old, pointers
                if set(pointers) == {"Counts", "TileOffsets", "TileExpert", "TileRow"}:
                    tiles = old, pointers
            dispatch, dispatch_launches, replaced = _dispatch_choices(
                data, package, compiler, preceding, projections[0], kernels, dynamic, rows
            )
            added.update(dispatch)
            launches.update(dispatch_launches)
            obsolete.update(replaced)
            if prefix is None or tiles is None:
                raise ValueError("Missing Flash prefill expert routing")
            prefix_kernel, prefix_args = prefix
            tile_kernel, tile_args = tiles
            for role in ("Counts", "TileOffsets"):
                if prefix_args[role] != tile_args[role]:
                    raise ValueError("Flash prefill expert routing views differ")
            for role in ("TileExpert", "TileRow"):
                operand = tile_args[role]
                desc = compiler.buffers[operand["name"]]
                if (
                    operand["kind"] != "buffer"
                    or desc["dtype"] != "i32"
                    or desc["shape"][0] < capacity
                ):
                    raise ValueError("Wide expert tile exceeds its routing workspace")

            def emit(old, pointers, build, key, multiplier=1):
                name = old["name"] + suffix
                previous = {old["name"] + f"/choice/{start}-{rows}/1/k0" for start in (1280, 2048)}
                choices = {n for n in kernels if n.startswith(old["name"] + "/choice/")}
                if choices - ({name} | previous):
                    raise ValueError("Overlapping Flash prefill expert choices")
                obsolete.update(choices & previous)
                if old["name"] in dynamic and dynamic[old["name"]]["capacity"] != rows:
                    raise ValueError("Unexpected Flash prefill expert capacity")
                added[name], launches[name] = compiler.emit(
                    old,
                    name,
                    pointers,
                    rows,
                    build=build,
                    key=key,
                    symbolic=old["name"] in dynamic,
                    multiplier=multiplier,
                )

            # The route scans use compiler-inserted barriers. The expert-wide
            # projection keys select the explicitly synchronized mainloop.
            emit(
                prefix_kernel,
                prefix_args,
                lambda: moe.expert_offsets.__wrapped__(experts, block_m),
                f"flash-routing-offsets-{experts}-{block_m}",
            )
            emit(
                tile_kernel,
                tile_args,
                lambda: moe.expert_tiles.__wrapped__(
                    rows, experts, top, block_m, dynamic_rows=True
                ),
                f"flash-routing-tiles-{rows}-{experts}-{top}-{block_m}",
            )
            for projection, gate_up in zip(projections, (True, False), strict=True):
                pointers = compiler.pointers(projection)
                for consumer, source in (
                    ("Counts", prefix_args["Counts"]),
                    ("Offsets", prefix_args["RowOffsets"]),
                    ("TC", prefix_args["TileCount"]),
                    ("TE", tile_args["TileExpert"]),
                    ("TR", tile_args["TileRow"]),
                ):
                    if pointers[consumer] != source:
                        raise ValueError("Flash prefill expert routing and projection views differ")
                bank = compiler.buffers[pointers["P"]["name"]]
                count, blocks, n, words = bank["shape"]
                if bank["dtype"] != "u16" or count != experts or words != 8:
                    raise ValueError("Expected a compressed K64 Q2 expert bank")
                k = blocks * 64
                emit(
                    projection,
                    pointers,
                    lambda: e8p_prefill.e8p_prefill_grouped.__wrapped__(
                        rows * top,
                        experts,
                        capacity,
                        n,
                        k,
                        dynamic_rows=True,
                        block_m=block_m,
                        block_n=64 if gate_up else 128,
                        specialize_tails=gate_up,
                    ),
                    f"expert-wide-{'gate-up-ldm4' if gate_up else 'down'}-{rows}-{experts}-{top}-{n}-{k}",
                    multiplier=top,
                )
    if not added:
        raise ValueError("Missing wide Flash prefill expert bindings")
    for name in obsolete:
        del kernels[name]
        del dynamic[name]
    kernels.update(added)
    dynamic.update(launches)
    package["kernels"] = list(kernels.values())
    package["dynamic_batch_kernels"] = list(dynamic.values())
    package["plan_kernel_contracts"] = sorted(
        (set(package["plan_kernel_contracts"]) - obsolete) | added.keys()
    )
    return sorted(added)

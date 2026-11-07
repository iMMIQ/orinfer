"""Qwen MoE FFN: shared routing, grouped integer projection and row transforms."""

import inspect
import hashlib
import math
import numpy as np
from orinfer_build.compiler.compile import primitive


def workspace(b):
    w, c, e, top, f, h = (
        b.weights,
        4096,
        b.text["num_experts"],
        b.text["num_experts_per_tok"],
        b.f,
        b.h,
    )
    if "MoeLogits" in w.buffers:
        return
    tiles = math.ceil(c * top / 16) + min(e, c * top)
    for name, dtype, shape in [
        ("Logits", "f32", [c, e]),
        ("Gate", "f16", [c]),
        ("IDs", "i32", [c, top]),
        ("Prob", "f32", [c, top]),
        ("Partial", "i32", [e, math.ceil(c * top / 1024)]),
        ("SegmentOffsets", "i32", [e, math.ceil(c * top / 1024)]),
        ("Relative", "i32", [c, top]),
        ("Counts", "i32", [e]),
        ("Offsets", "i32", [e]),
        ("TileOffsets", "i32", [e]),
        ("TileExpert", "i32", [tiles]),
        ("TileRow", "i32", [tiles]),
        ("TileCount", "i32", [1]),
        ("Slots", "i32", [c, top]),
        ("A8", "i8", [c * top, h]),
        ("AS", "f16", [c * top]),
        ("GateUp", "f16", [c * top, 2 * f]),
        ("DownA8", "i8", [c * top, f]),
        ("DownAS", "f16", [c * top]),
        ("Output", "f16", [c * top, h]),
        ("Shared", "f16", [c, h]),
    ]:
        w.fixed("Moe" + name, dtype, shape)


def banks(b, prefix, draft):
    w, e = b.weights, b.text["num_experts"]
    if not draft:
        return [
            (
                w.immutable(prefix + ".experts." + name + ".packed"),
                w.immutable(prefix + ".experts." + name + ".scale"),
            )
            for name in ("gate_up_proj", "down_proj")
        ]
    result = []
    for role in ("gate_up", "down"):
        operands = []
        scales = []
        for i in range(e):
            names = [
                prefix + f".experts.{i}." + x + "_proj.weight"
                for x in (("gate", "up") if role == "gate_up" else ("down",))
            ]
            operands.append(np.concatenate([w.array(n + ".int8") for n in names]))
            scales.append(np.concatenate([w.array(n + ".scale") for n in names]))
        name = "Runtime." + prefix + "." + role
        result.append(
            (
                w.constant(name + ".int8", np.stack(operands)),
                w.constant(name + ".scale", np.stack(scales)),
            )
        )
    return result


def selection(b, banks):
    if not hasattr(b, "moe_selection"):
        import torch
        from orinfer_build.tuning.expert_tuning import MoeTuner
        from orinfer_build.common.gpu import gpu_lock

        w = b.weights
        with gpu_lock(b.lock):
            actual = dict(
                book=torch.from_numpy(w.extra[b.book].copy()).cuda(),
                banks=[
                    tuple(torch.from_numpy(w.array(name).copy()).cuda() for name in bank)
                    for bank in banks
                ],
            )
            tuner = MoeTuner(b.output / "moe-tuning", b.aot.identity)
            b.moe_selection = tuner.routed(
                b.h,
                b.f,
                b.text["num_experts"],
                b.text["num_experts_per_tok"],
                actual,
                [w.buffers[name]["data"]["sha256"] for bank in banks for name in bank],
                routing_samples=getattr(b, "moe_routing", None),
            )
    return b.moe_selection


def emitter(b, section, rows):
    from kernels.cuda.operators.op30_activation_quantization import _compile as a8_compile

    def emit(
        role, fn, args, *dimensions, multiplier=1, suffix="", compiler_options=None, **options
    ):
        identity = repr((rows, dimensions, options, compiler_options)).encode()
        key = "moe-" + inspect.unwrap(getattr(fn, "func", fn)).__name__
        key += "-" + hashlib.sha256(identity).hexdigest()
        b.emit(
            section + "/" + role + suffix,
            key,
            lambda: primitive(
                fn,
                *dimensions,
                compiler_options=compiler_options,
                **({} if fn is a8_compile else dict(dynamic_rows=True)),
                **options,
            ),
            args,
            rows * multiplier,
            row_multiplier=multiplier,
        )

    return emit


def projections(b, section, bank_up, bank_down, *, rows=4096, draft=False):
    from kernels.cuda.model import moe
    from orinfer_build.tuning.expert_tuning import (
        expert_projection,
        expert_passes,
        cold_expert_config,
        tile_capacity,
    )

    w, h, f = b.weights, b.h, b.f
    e, top = b.text["num_experts"], b.text["num_experts_per_tok"]
    codec = "w8" if draft else "e8p_q4"
    emit = emitter(b, section, rows)
    selected = (
        [("", dict(block_m=16, expand=False))]
        if draft
        else list(b.row_choices(selection(b, [bank_up, bank_down]), rows))
    )
    expanded = any(config["expand"] for _, config in selected)
    if expanded and "MoeExpanded" not in w.buffers:
        w.fixed("MoeExpanded", "i8", [e * 2 * f * h])
    planes = any(config.get("folded") for _, config in selected)
    cold = any(config.get("cold_rows") for _, config in selected)
    cold_config = cold_expert_config()
    if (planes or cold) and "MoeLow" not in w.buffers:
        for name in ("MoeLow", "MoeHigh"):
            w.fixed(name, "i8", [4096 * top * max(h, f) // 2])
        w.fixed("MoeSum", "i32", [4096 * top])
    if cold and "MoeColdTileCount" not in w.buffers:
        capacity = tile_capacity(4096, e, top, cold_config["block_m"])
        for name, shape in [
            ("Offsets", [e]),
            ("Expert", [capacity]),
            ("Row", [capacity]),
            ("Count", [1]),
        ]:
            w.fixed("MoeColdTile" + name, "i32", shape)

    from orinfer_build.contracts.row_selection import project

    routing_selection = (
        dict(
            b.moe_selection,
            configs=[dict(c, cold_rows=c.get("cold_rows", 0)) for c in b.moe_selection["configs"]],
        )
        if not draft
        else None
    )
    routing = (
        selected
        if draft
        else b.row_choices(project(routing_selection, ("block_m", "cold_rows")), rows)
    )
    for suffix, config in routing:
        tile = config["block_m"]
        if config.get("cold_rows"):
            from kernels.cuda.model import expert_partition

            bindings = dict(
                Counts="MoeCounts",
                RowOffsets="MoeOffsets",
                TileOffsets="MoeTileOffsets",
                TileCount="MoeTileCount",
                ColdOffsets="MoeColdTileOffsets",
                ColdCount="MoeColdTileCount",
            )
            b.emit(
                section + "/offsets" + suffix,
                f"moe-partition-offsets-{e}-{tile}-{config['cold_rows']}",
                lambda tile=tile, limit=config["cold_rows"]: primitive(
                    expert_partition.offsets, e, tile, cold_config["block_m"], limit
                ),
                bindings,
                rows,
                dynamic=False,
            )
            emit(
                "tiles",
                expert_partition.tiles,
                dict(
                    Counts="MoeCounts",
                    TileOffsets="MoeTileOffsets",
                    TileExpert="MoeTileExpert",
                    TileRow="MoeTileRow",
                    ColdOffsets="MoeColdTileOffsets",
                    ColdExpert="MoeColdTileExpert",
                    ColdRow="MoeColdTileRow",
                ),
                rows,
                e,
                top,
                tile,
                cold_config["block_m"],
                config["cold_rows"],
                suffix=suffix,
            )
            continue
        b.emit(
            section + "/offsets" + suffix,
            f"moe-offsets-{e}-{tile}",
            lambda tile=tile: primitive(moe.expert_offsets, e, tile),
            dict(
                Counts="MoeCounts",
                RowOffsets="MoeOffsets",
                TileOffsets="MoeTileOffsets",
                TileCount="MoeTileCount",
            ),
            rows,
            dynamic=False,
        )
        emit(
            "tiles",
            moe.expert_tiles,
            dict(
                Counts="MoeCounts",
                TileOffsets="MoeTileOffsets",
                TileExpert="MoeTileExpert",
                TileRow="MoeTileRow",
            ),
            rows,
            e,
            top,
            tile,
            suffix=suffix,
        )
    common = dict(
        Book=b.book,
        Counts="MoeCounts",
        Offsets="MoeOffsets",
        TileExpert="MoeTileExpert",
        TileRow="MoeTileRow",
        TileCount="MoeTileCount",
    )

    def conditional_step(construct, enabled):
        construct()
        from orinfer_build.contracts.row_selection import choices

        for suffix, config in choices(b.moe_selection, rows, bounded=True):
            if not enabled(config):
                construct(suffix.replace("/rows/", "/skip/"))

    def expert(role, n, k, bank, activation, scale, output):
        if cold:
            from kernels.cuda.model.e8p_prefill import split_routed_prefill_a8

            def cold_planes(suffix=""):
                emit(
                    role + "_cold_planes",
                    split_routed_prefill_a8,
                    dict(
                        A=activation,
                        Counts="MoeCounts",
                        Offsets="MoeOffsets",
                        TE="MoeColdTileExpert",
                        TR="MoeColdTileRow",
                        TC="MoeColdTileCount",
                        Low="MoeLow",
                        High="MoeHigh",
                        Sum="MoeSum",
                    ),
                    rows * top,
                    k,
                    e,
                    cold_config["block_m"],
                    multiplier=top,
                    suffix=suffix,
                )

            def cold_product(suffix=""):
                emit(
                    role + "_cold",
                    expert_projection,
                    dict(
                        Low="MoeLow",
                        High="MoeHigh",
                        Sum="MoeSum",
                        P=bank[0],
                        Book=b.book,
                        WS=bank[1],
                        AS=scale,
                        Counts="MoeCounts",
                        Offsets="MoeOffsets",
                        TE="MoeColdTileExpert",
                        TR="MoeColdTileRow",
                        TC="MoeColdTileCount",
                        Out=output,
                    ),
                    rows,
                    n,
                    k,
                    e,
                    top,
                    multiplier=top,
                    suffix=suffix,
                    codec="e8p_q4",
                    config=cold_config,
                    compiler_options=expert_passes("e8p_q4", cold_config),
                )

            conditional_step(cold_planes, lambda c: bool(c.get("cold_rows")))
            conditional_step(cold_product, lambda c: bool(c.get("cold_rows")))
        if expanded:

            def expand(suffix=""):
                b.emit(
                    section + "/" + role + "_expand" + suffix,
                    f"moe-expand-{e}-{n}-{k}",
                    lambda: primitive(moe.expand_experts, n, k, e),
                    dict(W=bank[0], Book=b.book, Counts="MoeCounts", Out="MoeExpanded"),
                    rows,
                    dynamic=False,
                )

            conditional_step(expand, lambda c: c["expand"])
        if planes:
            from kernels.cuda.model.e8p_prefill import split_prefill_a8

            def pack(suffix=""):
                emit(
                    role + "_planes",
                    split_prefill_a8,
                    dict(A=activation, Low="MoeLow", High="MoeHigh", Sum="MoeSum"),
                    rows * top,
                    k,
                    multiplier=top,
                    suffix=suffix,
                )

            conditional_step(pack, lambda c: c.get("folded", False))
        for suffix, config in selected:
            if config.get("folded"):
                emit(
                    role,
                    expert_projection,
                    dict(
                        Low="MoeLow",
                        High="MoeHigh",
                        Sum="MoeSum",
                        P=bank[0],
                        Book=b.book,
                        WS=bank[1],
                        AS=scale,
                        Counts="MoeCounts",
                        Offsets="MoeOffsets",
                        TE="MoeTileExpert",
                        TR="MoeTileRow",
                        TC="MoeTileCount",
                        Out=output,
                    ),
                    rows,
                    n,
                    k,
                    e,
                    top,
                    multiplier=top,
                    suffix=suffix,
                    codec="e8p_q4",
                    config=config,
                    compiler_options=expert_passes("e8p_q4", config),
                )
                continue
            emit(
                role,
                expert_projection,
                dict(
                    common,
                    A=activation,
                    W="MoeExpanded" if config["expand"] else bank[0],
                    WS=bank[1],
                    AS=scale,
                    Out=output,
                ),
                rows,
                n,
                k,
                e,
                top,
                codec="w8" if config["expand"] else codec,
                compiler_options=expert_passes(codec, config),
                suffix=suffix,
                config=config,
            )

    expert("expert_up", 2 * f, h, bank_up, "MoeA8", "MoeAS", "MoeGateUp")
    expert("expert_down", h, f, bank_down, "MoeDownA8", "MoeDownAS", "MoeOutput")


def build(b, section, prefix, *, rows=4096, draft=False):
    from kernels.cuda.model import moe, rotation_a8
    from kernels.cuda.operators.op30_activation_quantization import _compile as a8_compile

    workspace(b)
    w, h, f = b.weights, b.h, b.f
    e, top = b.text["num_experts"], b.text["num_experts_per_tok"]
    norm = "MtpNorm" if draft else "Norm"
    output = "MtpHidden" if draft else "Hidden"
    gate_up = "MtpGateUp" if draft else "GateUp"
    codec = "w8" if draft else "e8p_q4"
    bank_up, bank_down = banks(b, prefix, draft)
    emit = emitter(b, section, rows)

    router = w.combined(
        prefix + ".router", [prefix + ".gate.weight", prefix + ".shared_expert_gate.weight"]
    )
    from orinfer_build.tuning.expert_tuning import MoeTuner
    from orinfer_build.common.gpu import gpu_lock

    if not hasattr(b, "moe_router_tuner"):
        b.moe_router_tuner = MoeTuner(b.output / "moe-tuning", b.aot.identity)
    with gpu_lock(b.lock):
        router_selection = b.moe_router_tuner.router(h, e)
        topk_selection = b.moe_router_tuner.topk(e, top)
    for suffix, config in b.row_choices(router_selection, rows):
        emit(
            "router",
            moe.router_projection,
            dict(X=norm, W=router, Logits="MoeLogits", SharedGate="MoeGate"),
            rows,
            h,
            e,
            suffix=suffix,
            **config,
        )
    for suffix, config in b.row_choices(topk_selection, rows):
        options = {k: v for k, v in config.items() if k != "algorithm"}
        emit(
            "topk",
            moe.router_topk_warp if config["algorithm"] == "warp" else moe.router_topk,
            dict(Logits="MoeLogits", IDs="MoeIDs", Prob="MoeProb"),
            rows,
            e,
            top,
            suffix=suffix,
            **options,
        )
    emit(
        "histogram_local",
        moe.expert_histogram_local,
        dict(IDs="MoeIDs", Partial="MoePartial", Slots="MoeRelative"),
        rows,
        e,
        top,
    )
    emit(
        "histogram_offsets",
        moe.expert_histogram_offsets,
        dict(Partial="MoePartial", Offsets="MoeSegmentOffsets", Counts="MoeCounts"),
        rows,
        e,
        top,
    )
    emit(
        "histogram_finish",
        moe.expert_histogram_finish,
        dict(IDs="MoeIDs", Offsets="MoeSegmentOffsets", Slots="MoeRelative"),
        rows,
        e,
        top,
    )
    emit(
        "dispatch",
        moe.expert_dispatch,
        dict(
            A="A8",
            Scale="AS",
            IDs="MoeIDs",
            RelativeSlot="MoeRelative",
            RowOffsets="MoeOffsets",
            D="MoeA8",
            DS="MoeAS",
            SlotMap="MoeSlots",
        ),
        rows,
        h,
        e,
        top,
        h,
    )
    for role, offset in (("gate", 0), ("up", f if draft else 4096 * f)):
        b.projection(
            section + "/" + role,
            prefix + ".shared_expert." + role + "_proj.weight",
            f,
            h,
            gate_up,
            stride=2 * f if draft else f,
            offset=offset,
            rows=rows,
            kind=codec,
        )
    if draft:
        emit(
            "down_quant",
            a8_compile,
            dict(X=gate_up, Mask="UnusedA8Mask", Q="A8", S="AS"),
            f,
            f,
            False,
            True,
            256,
        )
    else:
        b.rotation(section + "/swiglu_rotate", f, gate_up, swiglu=True, plane_rows=4096)
    b.projection(
        section + "/down",
        prefix + ".shared_expert.down_proj.weight",
        h,
        f,
        "MoeShared",
        rows=rows,
        kind=codec,
    )
    if draft:
        emit(
            "expert_rotate",
            a8_compile,
            dict(X="MoeGateUp", Mask="UnusedA8Mask", Q="MoeDownA8", S="MoeDownAS"),
            f,
            f,
            False,
            True,
            256,
            multiplier=top,
        )
    else:
        with gpu_lock(b.lock):
            rotation = b.row_tuner.activation_rotation(f, True, multiplier=top)
        rotation_config = rotation["configs"][int(rotation["default"])]
        emit(
            "expert_rotate",
            rotation_a8.rotate_activation_a8,
            dict(X="MoeGateUp", Signs=b.signs[f], Q="MoeDownA8", S="MoeDownAS"),
            rows * top,
            f,
            True,
            multiplier=top,
            **rotation_config,
        )
    projections(b, section, bank_up, bank_down, rows=rows, draft=draft)
    emit(
        "combine",
        moe.moe_combine,
        dict(
            Expert="MoeOutput",
            SlotMap="MoeSlots",
            Prob="MoeProb",
            Shared="MoeShared",
            SharedGate="MoeGate",
            Output=output,
        ),
        rows,
        h,
        rows * top,
        top,
    )

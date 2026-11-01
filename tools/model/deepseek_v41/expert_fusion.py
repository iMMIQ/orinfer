"""Fuse expert gather/INT4 planes and grouped-GMM activation boundaries offline."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.assembly import (
    check_intermediate_liveness,
    publish_package,
    validate_rows,
)
from tools.model.deepseek_v41.batch import compile_task


def selected_chain(program, index, describe):
    """Require the complete existing gather/up/SwiGLU contract and geometry."""
    chain = program[index : index + 3]
    if len(chain) != 3:
        raise ValueError("Incomplete expert chain")
    gather, up, activation = chain
    g, a = describe(gather), describe(activation)
    if (
        not g
        or not a
        or g["op"] != "memory.gather_quantized_rows"
        or a["op"] not in ("model_ops.swiglu_packed_quant", "model_ops.swiglu_packed_quant_active")
        or up["op"] != "w4"
        or up.get("weight_layout") != "transposed_nz"
        or not up.get("groups")
        or up.get("expected_expert_rows") is None
        or set(gather) != {"op", "buffers", "manifest"}
        or set(activation) != {"op", "buffers", "manifest"}
    ):
        raise ValueError("Expert fusion requires the shared routed NZ GMM boundary")
    rows, reduction, columns, experts = (up[k] for k in ("rows", "input", "output", "experts"))
    gp, ap = g["parameters"], a["parameters"]
    capacity = gp.get("_row_capacity")
    if (
        not capacity
        or rows != capacity * 6
        or columns % 512
        or gp["rows"] != rows
        or gp["width"] != reduction
        or gp["table_rows"] != capacity
        or ap["rows"] != rows
        or ap["width"] * 2 != columns
        or ap.get("limit", 10.0) != 10.0
        or ap.get("selected", 6) != 6
        or ap.get("_row_capacity") != capacity
        or gather["buffers"][3:5] != [up["buffers"][0], up["buffers"][4]]
        or activation["buffers"][0] != up["buffers"][5]
    ):
        raise ValueError("Expert fusion geometry or dataflow differs")
    return chain, (capacity, rows, reduction, columns // 2, experts)


def build(
    plan,
    output,
    workers=4,
    minimum=32,
    maximum=511,
    model_dir=None,
    *,
    epilogue_tile=2304,
    expected_expert_rows=None,
):
    """Replace complete selected packets; reuse all state, banks and other packets."""
    import numpy as np
    from tools.operators.ascend.build import operand

    if output.exists() or not 1 <= minimum <= maximum <= 4096:
        raise ValueError("Require a new output directory and a valid row interval")
    if (
        epilogue_tile < 32
        or epilogue_tile % 32
        or (expected_expert_rows is not None and expected_expert_rows < 1)
    ):
        raise ValueError("Require an aligned epilogue tile and positive expert row estimate")
    model = json.loads(plan.read_text())
    base = output / "source"
    publish_package(model, plan.parent, base)
    descriptions, tasks, changed = {}, {}, 0

    def describe(step):
        if step.get("op") != "kernel":
            return None
        path = step["manifest"]
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    for rank in model["ranks"]:
        definitions = {v["name"]: v for v in rank["buffers"]}

        def buffer(name, size, data=None):
            definition = dict(name=name, bytes=size)
            if data is not None:
                path = base / "data" / (name + ".bin")
                path.parent.mkdir(exist_ok=True)
                payload = data.tobytes()
                if not path.exists():
                    path.write_bytes(payload)
                elif path.read_bytes() != payload:
                    raise ValueError("Shared expert constant differs")
                definition["data"] = dict(file=str(path.relative_to(base)), sha256=file_hash(path))
            if name not in definitions:
                definitions[name] = definition
                rank["buffers"].append(definition)
            elif definitions[name] != definition:
                raise ValueError("Shared expert buffer differs")
            return name

        for packet in rank["packets"]:
            low, high = packet.get("min_rows", 1), packet.get("max_rows", packet["rows"])
            if packet.get("batch_decode", False) or packet.get("role", "target") != "target":
                continue
            if high < minimum or low > maximum:
                continue
            if low < minimum or high > maximum:
                raise ValueError("Select complete existing packet intervals")
            old, result, index = packet["body"], [], 0
            while index < len(old):
                spec = describe(old[index])
                if not spec or spec["op"] != "memory.gather_quantized_rows":
                    result.append(old[index])
                    index += 1
                    continue
                (gather, up, activation), geometry = selected_chain(old, index, describe)
                capacity, rows, reduction, width, experts = geometry
                if capacity != packet["rows"]:
                    raise ValueError("Expert fusion requires the original packet capacity")
                check_intermediate_liveness(
                    old[index + 3 :],
                    [up["buffers"][0], up["buffers"][5]],
                    describe,
                    model["batch"],
                    definitions,
                )
                prefix = f"expert.split.m{capacity}.k{reduction}.n{width}.e{experts}"
                groups = buffer(prefix + ".groups", experts * 8)
                partial = buffer(prefix + ".partial", rows * 2 * width * 2 * 2)
                unit_tokens = buffer(
                    prefix + ".token", rows * 2 * 4, np.ones(rows * 2, dtype="<f4")
                )
                key = json.dumps(geometry)
                gather_key, finish_key = "@gather_split" + key, "@expert_finish" + key
                if gather_key not in tasks:
                    common = dict(module="tools.operators.ascend.expert", capacity=capacity)
                    tasks[gather_key] = dict(
                        common,
                        key=gather_key,
                        name="gather_split",
                        arguments=dict(
                            rows=rows, table_rows=capacity, width=reduction, experts=experts
                        ),
                        dynamic=dict(rows=6, table_rows=1),
                        specs=[
                            operand("table", "i8", [capacity, reduction]),
                            operand("scales", "f32", [capacity]),
                            operand("indices", "i32", [rows]),
                            operand("groups", "i64", [experts]),
                            operand("packed", "i8", [rows * 2, reduction // 2], "write"),
                            operand("out_scales", "f32", [rows], "write"),
                            operand("errors", "i32", [rows], "write"),
                            operand("split_groups", "i64", [experts], "write"),
                        ],
                    )
                    tasks[finish_key] = dict(
                        common,
                        key=finish_key,
                        name="finish",
                        arguments=dict(
                            rows=rows, width=width, experts=experts, tile_n=epilogue_tile, slots=24
                        ),
                        dynamic=dict(rows=6),
                        specs=[
                            operand("raw", "f16", [rows * 2, width * 2]),
                            operand("assist", "f32", [experts, width * 2]),
                            operand("token_scale", "f32", [rows]),
                            operand("groups", "i64", [experts]),
                            operand("out", "i8", [rows, width], "write"),
                            operand("scales", "f32", [rows], "write"),
                        ],
                    )
                replacement = deepcopy(up)
                replacement.update(
                    op="w4_a4",
                    rows=rows * 2,
                    groups=groups,
                    expected_expert_rows=expected_expert_rows or up["expected_expert_rows"],
                )
                replacement["buffers"] = [
                    *up["buffers"][:4],
                    unit_tokens,
                    partial,
                ]
                result.extend(
                    [
                        dict(
                            op="kernel",
                            manifest=gather_key,
                            buffers=[
                                *gather["buffers"][:3],
                                up["groups"],
                                *gather["buffers"][3:],
                                groups,
                            ],
                        ),
                        replacement,
                        dict(
                            op="kernel",
                            manifest=finish_key,
                            buffers=[
                                partial,
                                *up["buffers"][3:5],
                                up["groups"],
                                *activation["buffers"][1:3],
                            ],
                        ),
                    ]
                )
                index += 3
                changed += 1
            packet["body"] = result
    if not changed:
        raise ValueError("No matching expert chains")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        compiled = dict(pool.map(compile_task, [(task, base) for task in tasks.values()]))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"]:
                if step.get("manifest") in compiled:
                    step["manifest"] = compiled[step["manifest"]]
    validate_rows(model, base, model_dir=model_dir)
    assets = publish_package(model, base, output / "package")
    atomic_json(
        output / "result.json",
        dict(changed_chains=changed, compiled_kernels=len(tasks), assets=assets),
    )
    return output / "package" / "model.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--min-rows", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=511)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--epilogue-tile", type=int, default=2304)
    parser.add_argument("--expected-expert-rows", type=int)
    args = parser.parse_args()
    print(
        build(
            args.plan,
            args.output,
            args.compile_workers,
            args.min_rows,
            args.max_rows,
            args.model_dir,
            epilogue_tile=args.epilogue_tile,
            expected_expert_rows=args.expected_expert_rows,
        )
    )


if __name__ == "__main__":
    main()

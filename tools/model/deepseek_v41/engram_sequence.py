"""Exchange Engram column shards directly into the existing sequence partitions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.deepseek_v41.batch import compile_task, DTYPES
from tools.model.deepseek_v41.converge import compact_buffers, publish_package, validate_rows
from tools.model.row_contract import expression


def build(plan, output, workers=4, minimum=512):
    from kernels.model.rows import row_count
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes

    if output.exists():
        raise ValueError("Require a new output directory")
    model = json.loads(plan.read_text())
    if len(model["ranks"]) != 8:
        raise ValueError("Engram transport requires the existing TP8 sequence layout")
    base = output / "source"
    publish_package(model, plan.parent, base)
    descriptions, tasks, counts = {}, {}, []

    def describe(step):
        if step["op"] != "kernel":
            return None
        path = step["manifest"]
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    def kernel(kind, params, capacity, buffers, access, rows):
        module, name = kind.split(".")
        function = getattr(importlib.import_module("tools.operators.ascend." + module), name)(
            **dict(params, rows=row_count(capacity, True))
        )
        tensors = [function.buffer_map[v] for v in function.params if v in function.buffer_map]
        specs = [
            operand(t.name, DTYPES[str(t.dtype)], shape, port)
            for t, shape, port in zip(
                tensors, physical_shapes(function, capacity), access, strict=True
            )
        ]
        identity = json.dumps([kind, params, capacity, specs], sort_keys=True)
        if identity not in tasks:
            tasks[identity] = dict(
                key=f"@engram_sequence{len(tasks)}",
                module="tools.operators.ascend." + module,
                name=name,
                arguments=dict(params, rows=capacity),
                capacity=capacity,
                dynamic=dict(rows=1),
                specs=specs,
            )
        return dict(
            op="kernel",
            manifest=tasks[identity]["key"],
            buffers=buffers,
            row_expression=expression(rows),
        )

    for rank in model["ranks"]:
        definitions = {v["name"]: v for v in rank["buffers"]}

        def buffer(name, size, view=None):
            item = dict(name=name, bytes=size)
            if view is not None:
                item["view"] = view
            if name not in definitions:
                definitions[name] = item
                rank["buffers"].append(item)
            elif definitions[name] != item:
                raise ValueError("Engram sequence buffer contract differs: " + name)
            return name

        for packet in rank["packets"]:
            if packet.get("batch_decode") or packet.get("min_rows", 1) < minimum:
                continue
            capacity = packet["rows"]
            if capacity % 8:
                raise ValueError("Engram sequence capacity must be divisible by8")
            local = capacity // 8
            old, body, i, changed = packet["body"], [], 0, 0
            while i < len(old):
                step = old[i]
                spec = describe(step)
                if (
                    spec is None
                    or spec["op"] != "dense.projection"
                    or not step["buffers"][4].endswith(".engram.local")
                ):
                    body.append(step)
                    i += 1
                    continue
                chain = old[i + 1 : i + 7]
                if len(chain) != 6 or [v["op"] for v in chain] != [
                    "gather",
                    "kernel",
                    "kernel",
                    "gather",
                    "kernel",
                    "kernel",
                ]:
                    raise ValueError("Require complete original Engram communication/gate chain")
                gather, columns, split, residual, hadamard, gate = chain
                if [describe(v)["op"] for v in [columns, split, hadamard, gate]] != [
                    "memory.gather_columns",
                    "model_ops.split",
                    "engram.hadamard32",
                    "engram.gate",
                ]:
                    raise ValueError("Unsupported Engram consumer chain")
                p = spec["parameters"]
                width = p["output"]
                if (
                    width != 3200
                    or gather["source"] != step["buffers"][4]
                    or residual["destination"] != gate["buffers"][0]
                    or hadamard["buffers"][0] != gate["buffers"][0]
                ):
                    raise ValueError("Engram sequence inputs do not share the original geometry")
                hidden = residual["source"]
                target = "batch.prefill.sequence.view." + gate["buffers"][-1].removeprefix(
                    "batch.prefill."
                )
                if (
                    hidden not in definitions
                    or target not in definitions
                    or definitions[hidden]["bytes"] != local * 20480 * 2
                    or definitions[target]["bytes"] != local * 20480 * 2
                ):
                    raise ValueError("Require complete existing local residual views")
                prefix = "batch.prefill.sequence." + step["buffers"][4].removeprefix(
                    "batch.prefill."
                ).removesuffix(".local")
                received = buffer(prefix + ".received", capacity * width * 2)
                projected = buffer(prefix + ".projected", local * 25600 * 2)
                key = buffer(prefix + ".key", local * 20480 * 2)
                value = buffer(prefix + ".value", local * 5120 * 2)
                original = buffer(prefix + ".original", local * 20480 * 4)
                mask_parent = gate["buffers"][5]
                mask = buffer(
                    prefix + ".mask",
                    local,
                    dict(
                        parent=mask_parent,
                        offset=rank["rank"] * local,
                        row_capacity=capacity,
                        offset_expression=expression(f"((m + 7) // 8) * {rank['rank']}"),
                    ),
                )
                body.append(step)
                body.append(
                    kernel(
                        "memory.zero_row_tail",
                        dict(capacity=capacity, width=width, dtype="bfloat16", partitions=8),
                        capacity,
                        [packet["coordinates"]["slots"], gather["source"]],
                        ["read", "read_write"],
                        "m",
                    )
                )
                body.append(
                    dict(
                        op="exchange",
                        source=gather["source"],
                        destination=received,
                        elements=local * width,
                        fp32=False,
                        elements_expression=expression(f"((m + 7) // 8) * {width}"),
                    )
                )
                body.append(
                    kernel(
                        "memory.gather_columns",
                        dict(width=25600),
                        local,
                        [received, projected],
                        ["read", "write"],
                        "(m + 7) // 8",
                    )
                )
                body.append(
                    kernel(
                        "model_ops.split",
                        dict(left=20480, right=5120),
                        local,
                        [projected, key, value],
                        ["read", "write", "write"],
                        "(m + 7) // 8",
                    )
                )
                hp = {
                    k: v
                    for k, v in describe(hadamard)["parameters"].items()
                    if not k.startswith("_") and k != "rows"
                }
                body.append(
                    kernel(
                        "engram.hadamard32",
                        hp,
                        local,
                        [hidden, hadamard["buffers"][1], original],
                        ["read", "read", "write"],
                        "(m + 7) // 8",
                    )
                )
                gp = {
                    k: v
                    for k, v in describe(gate)["parameters"].items()
                    if not k.startswith("_") and k != "rows"
                }
                body.append(
                    kernel(
                        "engram.gate",
                        gp,
                        local,
                        [hidden, original, key, value, gate["buffers"][4], mask, target],
                        ["read"] * 6 + ["write"],
                        "(m + 7) // 8",
                    )
                )
                changed += 1
                i += 7
            if changed:
                counts.append(changed)
            packet["body"] = body
    if not counts or len(set(counts)) != 1 or len(counts) != 8:
        raise ValueError("Require matching complete Engram sequence transforms on all ranks")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replacements = dict(pool.map(compile_task, [(v, str(base)) for v in tasks.values()]))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step.get("manifest") in replacements:
                    step["manifest"] = replacements[step["manifest"]]
    compact_buffers(model, owner_tables=True)
    validate_rows(model, base)
    assets = publish_package(model, base, output / "package")
    report = dict(
        changed_chains=sum(counts),
        kernels=len(tasks),
        assets=assets,
        model=str(output / "package/model.json"),
    )
    atomic_json(output / "engram-sequence.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--min-rows", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.output, args.compile_workers, args.min_rows)))


if __name__ == "__main__":
    main()

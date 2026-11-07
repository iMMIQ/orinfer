"""Promote native BF16 sums with shared dynamic casts and BF16 model boundaries."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

from orinfer_build.common.files import atomic_json
from orinfer_build.contracts.row_contract import expression, resolve


def promote(model, kernel):
    """Keep collective partitioning and cast the final sum once to BF16."""
    model = deepcopy(model)
    peers = len(model["ranks"])
    changes = 0
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        parents, views = {}, {}

        def view(kind, elements):
            parent = "collective.fp32_sum." + kind
            name = parent + f".{elements}"
            size = elements * 4
            if parent in definitions or name in definitions:
                raise ValueError("Reserved FP32 sum workspace name already exists")
            if parent not in parents:
                parents[parent] = dict(name=parent, bytes=size)
            parents[parent]["bytes"] = max(parents[parent]["bytes"], size)
            views[name] = dict(name=name, bytes=size, view=dict(parent=parent, offset=0))
            return name

        def cast(source, destination, rows, width, source_dtype, destination_dtype, extent):
            manifest = kernel(rows, width, source_dtype, destination_dtype)
            step = dict(op="kernel", manifest=manifest, buffers=[source, destination])
            if extent != expression("rows"):
                step["row_expression"] = extent
            return step

        for packet in rank["packets"]:
            for section in ("body", "head"):
                capacity, steps = packet["rows"], []
                for step in packet[section]:
                    if step["op"] == "limit_rows":
                        capacity = step["rows"]
                    if step["op"] != "reduce" or step.get("fp32", False):
                        steps.append(step)
                        continue
                    if step.get("fixed_rows", False):
                        raise ValueError("FP32 sum promotion requires row-wise collectives")
                    multiplier = peers if step.get("scatter", False) else 1
                    elements = step["elements"]
                    if min(elements, capacity, peers) < 1 or elements * multiplier % capacity:
                        raise ValueError("Sum row geometry differs from packet capacity")
                    width = elements * multiplier // capacity
                    extent = step.get("elements_expression")
                    if extent is not None:
                        if extent.get("op") != "multiply":
                            raise ValueError("Sum extent must expose its row width")
                        left, right = extent["lhs"], extent["rhs"]
                        if right == expression(str(width)):
                            output_rows = left
                        elif left == expression(str(width)):
                            output_rows = right
                        else:
                            raise ValueError("Sum expression width differs")
                        if resolve(extent, capacity) != elements:
                            raise ValueError("Sum expression capacity differs")
                    elif multiplier == 1:
                        output_rows = expression("rows")
                    else:
                        raise ValueError("ReduceScatter requires an explicit row expression")
                    input_rows = output_rows
                    if multiplier > 1:
                        input_rows = dict(
                            op="multiply", lhs=output_rows, rhs=expression(str(multiplier))
                        )
                    source, destination = step["source"], step["destination"]
                    source_elements = definitions[source]["bytes"] // 2
                    destination_elements = definitions[destination]["bytes"] // 2
                    if (
                        definitions[source]["bytes"] % (2 * width)
                        or definitions[destination]["bytes"] % (2 * width)
                        or source_elements < elements * multiplier
                        or destination_elements < elements
                    ):
                        raise ValueError("BF16 sum operand extent differs")
                    send = view("send", source_elements)
                    # AllReduce supports an identical input/output span. Only
                    # ReduceScatter needs an independent, smaller receive bank.
                    receive = view("receive", destination_elements) if multiplier > 1 else send
                    restored = receive if multiplier > 1 else view("send", destination_elements)
                    steps.append(
                        cast(
                            source,
                            send,
                            source_elements // width,
                            width,
                            "bf16",
                            "f32",
                            input_rows,
                        )
                    )
                    steps.append(dict(step, source=send, destination=receive, fp32=True))
                    steps.append(
                        cast(
                            restored,
                            destination,
                            destination_elements // width,
                            width,
                            "f32",
                            "bf16",
                            output_rows,
                        )
                    )
                    changes += 1
                packet[section] = steps
        rank["buffers"].extend(parents.values())
        rank["buffers"].extend(views.values())
    return model, changes


def compile_promoted(model, base, workers):
    from orinfer_build.models.deepseek_v41.build import compile_task, kernel_rebuild_task
    from orinfer_build.compiler.ascend.build import operand

    tasks = {}

    def kernel(rows, width, source, destination):
        key = (rows, width, source, destination)
        if key not in tasks:
            tasks[key] = dict(
                placeholder=f"@sum-cast-{len(tasks)}",
                module="kernels.ascend.model_ops",
                name="convert",
                arguments=dict(
                    rows=rows,
                    width=width,
                    source_dtype={"bf16": "bfloat16", "f32": "float32"}[source],
                    destination_dtype={"bf16": "bfloat16", "f32": "float32"}[destination],
                ),
                specs=[
                    operand("x", source, (rows, width)),
                    operand("out", destination, (rows, width), "write"),
                ],
                row_capacity=rows,
                dynamic_arguments=dict(rows=1),
            )
        return tasks[key]["placeholder"]

    result, changes = promote(model, kernel)
    bindings = [
        (step, key)
        for rank in result["ranks"]
        for packet in rank["packets"]
        for step in packet["body"] + packet["head"]
        for key in ("manifest", "uncompacted_manifest")
        if step.get(key) is not None
    ]
    bindings.extend((step, "manifest") for step in result.get("vision", {}).get("body", []))
    for variants in result.get("row_selections", {}).values():
        for variant in variants:
            bindings.extend((variant["kernels"], i) for i in range(len(variant["kernels"])))
    promoted_templates = len(tasks)
    if changes:
        # The released package requires one source identity per operator. Reuse
        # the existing rebuild contract for older conversions in this plan too.
        paths = {binding[key] for binding, key in bindings}
        paths.update(result.get("row_selections", {}))
        for path in sorted(paths):
            if path.startswith("@sum-cast-"):
                continue
            spec = json.loads((base / path).read_text())
            if spec["op"] == "model_ops.convert":
                tasks[("existing", path)] = kernel_rebuild_task(spec, path)
    if tasks:
        unique, aliases = {}, {}
        for task in tasks.values():
            identity = json.dumps(
                dict(
                    module=task["module"],
                    name=task["name"],
                    arguments=task["arguments"],
                    specs=task["specs"],
                    row_capacity=task.get("row_capacity"),
                    dynamic_arguments=task.get("dynamic_arguments"),
                    grid_limit=task.get("grid_limit", 64),
                ),
                sort_keys=True,
            )
            previous = unique.setdefault(identity, task)
            aliases[task["placeholder"]] = previous["placeholder"]
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            compiled = dict(pool.map(compile_task, [(t, str(base)) for t in unique.values()]))
        replacements = {old: compiled[new] for old, new in aliases.items()}
        for binding, key in bindings:
            binding[key] = replacements.get(binding[key], binding[key])
        if "row_selections" in result:
            result["row_selections"] = {
                replacements.get(slot, slot): variants
                for slot, variants in result["row_selections"].items()
            }
    return result, dict(
        collectives=changes,
        cast_templates=promoted_templates,
        rebuilt_conversions=len(tasks) - promoted_templates,
    )


def build(plan, output, workers=4, model_dir=None):
    from orinfer_build.models.deepseek_v41.assembly import (
        compact_buffers,
        publish_package,
        validate_rows,
    )

    if output.exists() or workers < 1:
        raise ValueError("Require a new output directory and positive compiler count")
    model = json.loads(plan.read_text())
    base = output / "source"
    publish_package(model, plan.parent, base)
    model, report = compile_promoted(model, base, workers)
    compact_buffers(model)
    validate_rows(model, base, model_dir=model_dir)
    atomic_json(base / "model.json", model)
    report["assets"] = publish_package(model, base, output / "package")
    atomic_json(output / "sum-precision.json", report)
    return output / "package/model.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    print(build(args.plan, args.output, args.compile_workers, args.model_dir))


if __name__ == "__main__":
    main()

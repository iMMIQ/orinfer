"""Recompile one private primitive schedule while retaining the compiled model.

The existing owner-range work tables select disjoint schedules by real owner
length. Compilation, ABI wrapping, validation and final asset publication reuse
the shared native batch builder. Model arithmetic and decode assets stay fixed.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.batch import DTYPES, SIZES, compile_task
from tools.model.deepseek_v41.converge import compact_buffers, publish_package, validate_rows


def rebuild(plan, output, operator, overrides, first):
    from kernels.model.rows import row_count
    from tools.operators.ascend.private_rows import indirect
    from tools.operators.ascend.geometry import physical_shapes

    model = json.loads(plan.read_text())
    if output.exists() or not model.get("batch") or first < 1 or "rows" in overrides:
        raise ValueError(
            "Require a native rows plan, positive owner length and new output directory"
        )
    base = output / "source"
    publish_package(model, plan.parent, base)
    module_name, name = operator.split(".")
    module = importlib.import_module("tools.operators.ascend." + module_name)
    factory = getattr(module, name)
    expected_op = "private_rows." + operator.replace(".", "_")
    contract = model["batch"]
    descriptions, replacements, schedules = {}, {}, {}
    validate_rows(model, plan.parent)
    changed = 0

    def describe(path):
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    def work(rank, grid, lower, upper):
        key = json.dumps([grid, lower, upper], sort_keys=True)
        role = "batch.work." + hashlib.sha256(key.encode()).hexdigest()[:16]
        contract["work"][role] = grid
        contract.setdefault("work_ranges", {})[role] = dict(min_rows=lower, max_rows=upper)
        if not any(buffer["name"] == role for buffer in rank["buffers"]):
            rank["buffers"].append(dict(name=role, bytes=8192 * 2 * 4))
        return role

    for rank in model["ranks"]:
        for packet in rank["packets"]:
            if packet.get("batch_decode"):
                continue
            body = []
            for original in packet["body"]:
                if original["op"] != "kernel":
                    body.append(original)
                    continue
                descriptor = describe(original["manifest"])
                if descriptor["op"] != expected_op:
                    body.append(original)
                    continue
                if original["manifest"] in model.get("row_selections", {}):
                    raise ValueError(
                        "Retarget selected variants through their original tuning entrypoint"
                    )
                parameters = descriptor["parameters"]
                capacity = parameters["_owner_token_capacity"]
                extent = contract.get("work_ranges", {}).get(
                    original["buffers"][0], dict(min_rows=1, max_rows=capacity)
                )
                lower, upper = max(first, extent["min_rows"]), extent["max_rows"]
                if lower > upper:
                    body.append(original)
                    continue
                signature = json.dumps(
                    [original["manifest"], original["buffers"][1]], sort_keys=True
                )
                if signature in schedules:
                    key, grid = schedules[signature]
                else:
                    arguments = {
                        key: value for key, value in parameters.items() if not key.startswith("_")
                    }
                    arguments.update(overrides)
                    function = factory(**dict(arguments, rows=row_count(capacity, True)))
                    tensors = [
                        function.buffer_map[p] for p in function.params if p in function.buffer_map
                    ]
                    columns = {tensor.name: i for i, tensor in enumerate(tensors)}
                    aliases = {tensor.name + "_owner": str(tensor.dtype) for tensor in tensors}
                    accesses = parameters["_owner_access"]
                    if (
                        aliases != parameters["_owner_aliases"]
                        or list(aliases) != list(parameters["_owner_aliases"])
                        or [a["column"] for a in accesses] != list(range(len(tensors)))
                    ):
                        raise ValueError("Private primitive tensor order/dtype differs")
                    mapping = contract["pointers"][original["buffers"][1]]
                    if len(mapping) != len(columns):
                        raise ValueError(
                            "Private schedule requires the complete existing pointer table"
                        )
                    definitions = {buffer["name"]: buffer for buffer in rank["buffers"]}
                    for live_rows in (1, capacity):
                        shapes = physical_shapes(function, live_rows)
                        for tensor, shape, column_index in zip(
                            tensors, shapes, mapping, strict=True
                        ):
                            column = contract["columns"][column_index]
                            if column["row_alignment"] != 1:
                                raise ValueError(
                                    "Private schedule requires an unchanged unpadded row ABI"
                                )
                            actual = math.prod(shape) * SIZES[DTYPES[str(tensor.dtype)]]
                            expected = (
                                live_rows * column["row_bytes"]
                                if column["row_bytes"] is not None
                                else definitions[column["buffer"]]["bytes"]
                            )
                            if actual != expected:
                                raise ValueError(
                                    "Private schedule changes an operand's physical extent"
                                )
                    _, identities, grid = indirect(
                        function, columns, contract["owners"], capacity, table_columns=len(mapping)
                    )
                    if any(value is not None for value in identities[3:]):
                        raise ValueError("Private schedule introduces external operands")
                    metadata = {
                        key: value for key, value in parameters.items() if key.startswith("_owner_")
                    }
                    metadata["_owner_source_sha256"] = file_hash(Path(inspect.getfile(factory)))
                    key = json.dumps([arguments, metadata, descriptor["operands"]], sort_keys=True)
                    if key not in replacements:
                        _, manifest = compile_task(
                            (
                                dict(
                                    key=key,
                                    module=module.__name__,
                                    name=name,
                                    arguments=arguments,
                                    capacity=capacity,
                                    dynamic=dict(rows=1),
                                    columns=columns,
                                    owners=contract["owners"],
                                    table_columns=len(mapping),
                                    specs=descriptor["operands"],
                                    metadata=metadata,
                                ),
                                base,
                            )
                        )
                        replacements[key] = manifest
                    schedules[signature] = key, grid
                if extent["min_rows"] < lower:
                    short = deepcopy(original)
                    short["buffers"][0] = work(
                        rank, original["owner_work"], extent["min_rows"], lower - 1
                    )
                    body.append(short)
                step = deepcopy(original)
                step["manifest"] = replacements[key]
                step["owner_work"] = grid
                step["buffers"][0] = work(rank, grid, lower, upper)
                body.append(step)
                changed += 1
            packet["body"] = body
    if not changed:
        raise ValueError("No private prefill primitive matched")
    live = {
        step["buffers"][0]
        for rank in model["ranks"]
        for packet in rank["packets"]
        for step in packet["body"] + packet["head"]
        if step["op"] == "kernel" and "owner_work" in step
    }
    contract["work"] = {name: grid for name, grid in contract["work"].items() if name in live}
    contract["work_ranges"] = {
        name: value for name, value in contract.get("work_ranges", {}).items() if name in live
    }
    model = compact_buffers(model)
    validate_rows(model, base)
    atomic_json(base / "model-scheduled.json", model)
    publish_package(model, base, output / "package")
    return dict(
        changed_bindings=changed,
        selected_kernels=len(replacements),
        model=str(output / "package/model.json"),
        operator=operator,
        parameters=overrides,
        first_owner_rows=first,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--parameters", type=json.loads, required=True)
    parser.add_argument("--min-owner-rows", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            rebuild(args.plan, args.output, args.operator, args.parameters, args.min_owner_rows)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

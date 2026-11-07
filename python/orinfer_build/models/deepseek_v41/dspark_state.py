"""Resize persistent DSpark history using the existing INT8 scatter/gather factories.

The 128-row read window and noise bank stay fixed. A larger physical ring lets
the shared state journal retain live history without copying rejected writes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

from orinfer_build.common.files import atomic_json

HISTORY_ROWS = 256


def resize(model, root, storage_rows, compiler):
    """Rebuild every producer/consumer together; keep owner bindings and weights."""
    from orinfer_build.models.deepseek_v41.build import compile_task, kernel_rebuild_task

    if storage_rows not in (128, 256, 512, 1024, 2048, 4096):
        raise ValueError("Draft history storage must be a power of two in128..4096")
    contract = model["batch"]
    persistent = {
        state["buffer"]: state
        for state in contract["decoder_replay"]["decoder_state"]
        if state["buffer"].startswith("cache.dspark.swa.")
    }
    if len(persistent) != 6 or any(
        state.get("window") != 128
        or not state.get("circular")
        or state.get("tokens_per_row") != 1
        or state["row_bytes"] != (16 if name.endswith(".scale") else 512)
        for name, state in persistent.items()
    ):
        raise ValueError("Require three persistent INT8 DSpark history banks and FP16 scales")
    descriptions, tasks, updates = {}, {}, []
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        original = {
            name: definitions[name]["bytes"] // state["row_bytes"]
            for name, state in persistent.items()
        }
        if (
            any(
                definitions[name].get("view")
                or definitions[name]["bytes"] != original[name] * state["row_bytes"]
                for name, state in persistent.items()
            )
            or len(set(original.values())) != 1
        ):
            raise ValueError("Persistent draft history requires matching physical bank geometry")
        old_rows = next(iter(original.values()))
        if old_rows not in (128, 256, 512, 1024, 2048, 4096):
            raise ValueError("Invalid source draft history ring")
        readers, writers = set(), set()
        for packet in rank["packets"]:
            if packet.get("role") == "draft":
                coordinates = packet["coordinates"]
                if coordinates.get("draft_swa_storage_rows", 128) != old_rows:
                    raise ValueError("Draft coordinates differ from physical history storage")
                coordinates["draft_swa_storage_rows"] = storage_rows
            for step in packet["body"] + packet["head"]:
                if step["op"] != "kernel":
                    continue
                path = step["manifest"]
                if path not in descriptions:
                    descriptions[path] = json.loads((root / path).read_text())
                spec = descriptions[path]
                parameters = spec["parameters"]
                private = spec["op"].startswith("private_rows.")
                if private:
                    mapping = contract["pointers"][step["buffers"][1]]
                    bindings = {
                        alias.removesuffix("_owner"): contract["columns"][
                            mapping[access["column"]]
                        ]["buffer"]
                        for alias, access in zip(
                            parameters["_owner_aliases"], parameters["_owner_access"], strict=True
                        )
                    }
                else:
                    bindings = {
                        port["name"]: name
                        for port, name in zip(spec["operands"], step["buffers"], strict=True)
                    }
                live = {port: name for port, name in bindings.items() if name in persistent}
                if not live:
                    continue
                op = spec["op"]
                scatter = op in ("kv_int8.scatter", "private_rows.kv_int8_scatter")
                gather = op in (
                    "kv_int8.gather_attention",
                    "private_rows.kv_int8_gather_attention",
                    "attention_fused.selected_int8",
                    "private_rows.attention_fused_selected_int8",
                )
                if scatter and set(live) == {"cache", "scales"}:
                    if (
                        parameters["table_rows"] != old_rows
                        or parameters.get("logical_capacity", 0) < storage_rows
                        or not parameters.get("retain_tail")
                    ):
                        raise ValueError("Persistent draft producer must retain the absolute tail")
                    overrides = dict(table_rows=storage_rows)
                    writers.update(live.values())
                elif gather and set(live) == {"swa", "swa_scales"}:
                    if (
                        packet.get("role") != "draft"
                        or (parameters.get("swa_storage_rows") or parameters["capacity"])
                        != old_rows
                    ):
                        raise ValueError("Draft consumer differs from physical history storage")
                    overrides = dict(
                        capacity=max(parameters["capacity"], storage_rows),
                        swa_storage_rows=storage_rows,
                    )
                    readers.update(live.values())
                else:
                    raise ValueError("Unrecognized persistent draft history access: " + op)
                if old_rows == storage_rows:
                    continue
                if step.get("uncompacted_manifest") or path in model.get("row_selections", {}):
                    raise ValueError("Resize draft history before row selection/operand compaction")
                if path not in tasks:
                    task = kernel_rebuild_task(deepcopy(spec), path)
                    task["arguments"].update(overrides)
                    if not private:
                        for port in task["specs"]:
                            if port["name"] in live:
                                port["shape"][0] = storage_rows
                    tasks[path] = task
                updates.append((step, path))
        if readers != persistent.keys() or writers != persistent.keys():
            raise ValueError("Every persistent draft bank needs both a producer and a consumer")
        for name, state in persistent.items():
            definitions[name]["bytes"] = storage_rows * state["row_bytes"]
    compiled = dict(compiler.map(compile_task, ((t, root) for t in tasks.values())))
    for step, path in updates:
        step["manifest"] = compiled[path]
    return dict(storage_rows=storage_rows, kernels=len(tasks), bindings=len(updates))


def build(plan, output, storage_rows, workers, model_dir):
    from orinfer_build.models.deepseek_v41.assembly import publish_package, validate_rows

    if output.exists():
        raise ValueError("Require a new history package directory")
    model = json.loads(plan.read_text())
    publish_package(model, plan.parent, output / "source")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as compiler:
        report = resize(model, output / "source", storage_rows, compiler)
    validate_rows(model, output / "source", model_dir=model_dir)
    atomic_json(output / "source/model-resized.json", model)
    publish_package(model, output / "source", output / "package")
    atomic_json(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--storage-rows", type=int, default=HISTORY_ROWS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(build(args.plan, args.output, args.storage_rows, args.workers, args.model_dir))


if __name__ == "__main__":
    main()

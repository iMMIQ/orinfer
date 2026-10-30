"""Build decoder SWA bounded replay from native A2 plans.

The shared KV source runs before compaction; --source-kv-boundary can defer
layer 20's query/FFN work. Owner-row plans reuse the shared dynamic templates
for each retained suffix; the legacy single-owner
experiment takes its non-SP tail from --tail-template. This is an approximate
state policy with separate model-quality acceptance.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import importlib
import json
import math
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.row_contract import expression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tail-template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep",
        type=int,
        default=140,
        help="Retained decoder rows per owner (128..512 for owner-row plans)",
    )
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument(
        "--final-prompt-only",
        action="store_true",
        help="Run decoder replay only for the final prompt window; reserve that window on prefix restore",
    )
    parser.add_argument(
        "--source-kv-boundary",
        action="store_true",
        help="Select query rows immediately after the last shared KV writes",
    )
    args = parser.parse_args()
    if json.loads(args.plan.read_text()).get("batch"):
        from tools.model.deepseek_v41.decoder_rows import build

        print(
            json.dumps(
                build(
                    args.plan,
                    args.output,
                    args.keep,
                    args.compile_workers,
                    final_prompt_only=args.final_prompt_only,
                    source_kv_boundary=args.source_kv_boundary,
                )
            )
        )
        return
    if args.tail_template is None:
        parser.error("The legacy single-owner experiment requires --tail-template")
    if args.final_prompt_only or args.source_kv_boundary:
        parser.error("Final-prompt replay requires an owner-row plan")
    if any(
        p.parent.resolve() != args.plan.parent.resolve() for p in (args.tail_template, args.output)
    ):
        parser.error("Plans must share the asset directory")
    if not 2 <= args.keep <= 512:
        parser.error("Replay suffix must lie in 2..512")
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.build import compile_task
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes

    base = args.plan.parent
    model = json.loads(args.plan.read_text())
    template = json.loads(args.tail_template.read_text())
    if len(model["ranks"]) != 8 or len(template["ranks"]) != 8:
        raise ValueError("Decoder replay expects the native TP8 source and tail plans")
    if model["index"] != template["index"]:
        raise ValueError("Decoder replay source and tail use different checkpoint identities")
    descriptors, tasks, functions = {}, {}, {}
    sizes = dict(bf16=2, f16=2, f32=4, i8=1, i32=4, i64=8)

    def descriptor(step):
        path = step["manifest"]
        if path not in descriptors:
            descriptors[path] = json.loads((base / path).read_text())
        return descriptors[path]

    def kernel(op, parameters, specs, capacity, symbolic):
        key = json.dumps([op, parameters, specs, capacity, symbolic], sort_keys=True)
        if key not in tasks:
            module, name = op.split(".")
            tasks[key] = dict(
                module="tools.operators.ascend." + module,
                name=name,
                arguments=parameters,
                specs=specs,
                row_capacity=capacity,
                dynamic_arguments=symbolic,
                placeholder=f"@decoder_replay{len(tasks)}",
            )
        return tasks[key]["placeholder"]

    def cut(body):
        return (
            next(
                i
                for i, step in enumerate(body)
                if step["op"] == "snapshot" and step["name"] == "layer20"
            )
            + 1
        )

    for rank, reference in zip(model["ranks"], template["ranks"], strict=True):
        if rank["rank"] != reference["rank"]:
            raise ValueError("Decoder replay source and tail ranks differ")
        definitions = {b["name"]: b for b in rank["buffers"]}
        original = {b["name"]: b for b in reference["buffers"]}
        tail_source = next(p for p in reference["packets"] if p.get("min_rows") == 2)
        capacity = tail_source["rows"]
        prefix = f"m{capacity}."
        tail = tail_source["body"][cut(tail_source["body"]) :]
        live, written, mapped = {}, set(), {}

        def add(name, size):
            definition = dict(name=name, bytes=size)
            if name not in definitions:
                rank["buffers"].append(definition)
                definitions[name] = definition
            elif definitions[name] != definition:
                raise ValueError("Conflicting decoder replay buffer: " + name)
            return name

        replay_start = add(f"replay{args.keep}.start", 4)

        def remap(name, spec=None):
            if not name.startswith(prefix):
                if name in mapped:
                    return mapped[name]
                if spec is not None:
                    size = math.prod(spec["shape"]) * sizes[spec["dtype"]]
                    if size != definitions[name]["bytes"]:
                        if (
                            not name.startswith("ones")
                            or spec["dtype"] != "f32"
                            or size > original[name]["bytes"]
                        ):
                            raise ValueError(
                                "A shared replay operand changed its physical geometry: " + name
                            )
                        target = f"replay{args.keep}." + name
                        definition = dict(name=target, bytes=size, view=dict(parent=name, offset=0))
                        if target not in definitions:
                            rank["buffers"].append(definition)
                            definitions[target] = definition
                        elif definitions[target] != definition:
                            raise ValueError("Conflicting replay unit-scale view")
                        mapped[name] = target
                        return target
                return name
            if spec is None:
                if name not in mapped:
                    raise ValueError("Missing compiled replay allocation: " + name)
                return mapped[name]
            old = original[name]
            if any(key in old for key in ("weight", "prepared_weight", "data")):
                raise ValueError("Request row state unexpectedly owns weights")
            if "view" in old and old["view"]["parent"] in written:
                raise ValueError("Decoder replay requires a standalone row view")
            size = math.prod(spec["shape"]) * sizes[spec["dtype"]]
            target = add(f"replay{args.keep}." + name, size)
            mapped[name] = target
            return target

        converted = []
        for step in tail + tail_source["head"]:
            step = deepcopy(step)
            if step["op"] == "kernel":
                spec = descriptor(step)
                old_parameters = spec["parameters"]
                dynamic = old_parameters.get("_row_capacity") is not None
                if (dynamic and old_parameters["_row_capacity"] != capacity) or step.get(
                    "fixed_rows"
                ):
                    raise ValueError("Expected the existing non-SP dynamic tail template")
                parameters = {k: v for k, v in old_parameters.items() if not k.startswith("_")}
                symbolic = {
                    key: value // capacity
                    for key, value in parameters.items()
                    if key in ("rows", "tokens", "expert_rows") and value % capacity == 0
                }
                module, name = spec["op"].split(".")
                if (
                    module == "memory"
                    and name in ("gather_rows", "gather_quantized_rows")
                    and parameters.get("table_rows") == capacity
                ):
                    symbolic["table_rows"] = 1
                parameters.update({key: factor * args.keep for key, factor in symbolic.items()})
                op = spec["op"]
                if op == "attention_prefill.softmax":
                    op, name = "attention_prefill.replay_softmax", "replay_softmax"
                function_key = json.dumps([op, parameters, symbolic], sort_keys=True)
                if function_key not in functions:
                    factory = getattr(
                        importlib.import_module("tools.operators.ascend." + module), name
                    )
                    functions[function_key] = factory(
                        **dict(
                            parameters,
                            **{
                                key: row_count(args.keep, True) * factor
                                for key, factor in symbolic.items()
                            },
                        )
                    )
                function = functions[function_key]
                shapes = physical_shapes(function, args.keep)
                specs = deepcopy(spec["operands"])
                buffers = step["buffers"]
                if op == "attention_prefill.replay_softmax":
                    specs.append(operand("replay_start", "i32", [1]))
                    buffers.append(replay_start)
                if len(shapes) != len(specs):
                    raise ValueError("Replay tensor ABI differs from the source template")
                for shape, binding in zip(shapes, specs, strict=True):
                    binding["shape"] = shape
                for source, binding in zip(buffers, spec["operands"], strict=False):
                    if (
                        source not in written
                        and binding["access"] in ("read", "read_write")
                        and source.startswith(prefix)
                    ):
                        live[source] = binding
                step["buffers"] = [
                    remap(source, binding) for source, binding in zip(buffers, specs, strict=True)
                ]
                for source, binding in zip(buffers, specs, strict=True):
                    if binding["access"] in ("write", "read_write", "scratch"):
                        written.add(source)
                step["manifest"] = kernel(
                    op, parameters, specs, args.keep if dynamic else None, symbolic
                )
            elif step["op"] == "reduce":
                if step.get("scatter") or step.get("fixed_rows") or step["elements"] % capacity:
                    raise ValueError("Unexpected collective in the non-SP tail")
                step["elements"] = step["elements"] // capacity * args.keep
                for key in ("source", "destination"):
                    source = step[key]
                    remap(
                        source,
                        dict(dtype="f32" if step["fp32"] else "bf16", shape=[step["elements"]]),
                    )
                    step[key] = remap(source)
                written.add(step["destination"].removeprefix(f"replay{args.keep}."))
            elif step["op"] == "w4":
                step["rows"] = args.keep * 6
                step["buffers"] = [
                    remap(
                        b,
                        dict(
                            dtype="i8" if i == 0 else "f32" if i == 4 else "bf16",
                            shape=[
                                args.keep * 6,
                                step["input"] if i == 0 else 1 if i == 4 else step["output"],
                            ],
                        ),
                    )
                    if i in (0, 4, 5)
                    else b
                    for i, b in enumerate(step["buffers"])
                ]
                written.add(step["buffers"][5].removeprefix(f"replay{args.keep}."))
                step["groups"] = remap(step["groups"])
            elif step["op"] == "snapshot":
                step["buffer"] = remap(step["buffer"])
            elif step["op"] not in ("fork", "switch", "join"):
                raise ValueError("Unsupported operation across decoder replay: " + step["op"])
            converted.append(step)
        body_tail, head = converted[: len(tail)], converted[len(tail) :]
        for packet in rank["packets"]:
            if packet.get("min_rows", 0) == 0:
                continue
            if packet["rows"] != capacity:
                raise ValueError("Prefill replay templates disagree on row capacity")
            boundary = cut(packet["body"])
            body = packet["body"][:boundary]
            for source in live:
                private = "sequence." + source
                if packet.get("min_rows", 0) >= 512 and private in definitions:
                    size = original[source]["bytes"]
                    width = size // capacity // sizes[live[source]["dtype"]]
                    transport = dict(
                        op="gather",
                        source=private,
                        destination=source,
                        elements=capacity // 8 * width,
                        fp32=live[source]["dtype"] == "f32",
                        fixed_rows=True,
                    )
                    if any(s.get("row_expression") for s in body):
                        transport.pop("fixed_rows")
                        transport["elements_expression"] = expression(f"((m + 7) // 8) * {width}")
                    body.append(transport)
            path = kernel(
                "decoder_replay.replay_start",
                dict(rows=capacity, keep=args.keep),
                [operand("positions", "i32", [capacity]), operand("output", "i32", [1], "write")],
                capacity,
                {"rows": 1},
            )
            body.append(
                dict(
                    op="kernel",
                    manifest=path,
                    buffers=[packet["coordinates"]["slots"], replay_start],
                )
            )
            for source, spec in live.items():
                width = original[source]["bytes"] // capacity // sizes[spec["dtype"]]
                parameters = dict(
                    rows=capacity,
                    width=width,
                    keep=args.keep,
                    dtype=dict(bf16="bfloat16", f32="float32", i32="int32")[spec["dtype"]],
                )
                path = kernel(
                    "decoder_replay.compact_suffix",
                    parameters,
                    [
                        operand("source", spec["dtype"], [capacity, width]),
                        operand("output", spec["dtype"], [args.keep, width], "write"),
                    ],
                    capacity,
                    {"rows": 1},
                )
                body.append(dict(op="kernel", manifest=path, buffers=[source, remap(source)]))
            packet["body"] = body + [dict(op="limit_rows", rows=args.keep)] + deepcopy(body_tail)
            packet["head"] = [dict(op="limit_rows", rows=args.keep)] + deepcopy(head)
            for key in ("best_value", "best_index", "logits"):
                if packet.get(key):
                    packet[key] = remap(packet[key])
        print(
            dict(rank=rank["rank"], keep=args.keep, carry=list(live), kernels=len(tasks)),
            flush=True,
        )
    replacements = {}
    with ProcessPoolExecutor(
        max_workers=args.compile_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for placeholder, path in pool.map(compile_task, [(t, str(base)) for t in tasks.values()]):
            replacements[placeholder] = path
            print(dict(compiled=len(replacements), total=len(tasks)), flush=True)
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    step["manifest"] = replacements.get(step["manifest"], step["manifest"])
    atomic_json(args.output, model)


if __name__ == "__main__":
    main()

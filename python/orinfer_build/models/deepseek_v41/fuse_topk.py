"""Fuse the existing three-level TopK tree with the same ordered merge helper."""

from copy import deepcopy
import json

from orinfer_build.models.deepseek_v41.batch import compile_task
from orinfer_build.models.deepseek_v41.assembly import (
    check_intermediate_liveness,
    selected_target_packet,
)
from orinfer_build.models.deepseek_v41.transform import (
    compile_and_replace,
    finalize_plan,
    plan_parser,
    stage_plan,
)


def merge_chain(steps, index, describe):
    if steps[index]["op"] != "kernel" or describe(steps[index])["op"] != "topk.merge":
        return None
    first = describe(steps[index])["parameters"]
    if first["sources"] not in (32, 64):
        return None
    chain = steps[index : index + 3]
    sources = first["sources"]
    for level, step in enumerate(chain):
        if step["op"] != "kernel":
            raise ValueError("TopK fusion requires three adjacent merge levels")
        spec = describe(step)
        p = spec["parameters"]
        if (
            spec["op"] != "topk.merge"
            or p["sources"] != sources
            or p.get("selected", 512) != first.get("selected", 512)
        ):
            raise ValueError("TopK merge tree geometry differs")
        if level and (
            step["buffers"][0] != chain[level - 1]["buffers"][1]
            or {k: step.get(k) for k in ["row_expression", "fixed_rows"]}
            != {k: chain[0].get(k) for k in ["row_expression", "fixed_rows"]}
        ):
            raise ValueError("TopK merge tree changes row binding or intermediate address")
        sources = (sources + 3) // 4
    if len(chain) != 3 or sources != 1:
        raise ValueError("TopK fusion requires a complete three-level tree")
    return chain


def causal_lengths(producer, chain, describe, batch, definitions):
    spec, first = describe(producer), describe(chain[0])
    p = spec["parameters"]
    if (
        spec["op"] != "private_rows.indexer_fused_causal_topk"
        or first["parameters"].get("selected", 512) != 512
        or p["length"] != first["parameters"]["sources"] * 4096
        or p["_owner_token_capacity"] != first["parameters"]["_row_capacity"]
    ):
        raise ValueError("TopK causal mask requires the matching canonical leaf producer")
    aliases = list(p["_owner_aliases"])
    table = batch["pointers"][producer["buffers"][1]]
    length, candidates = (
        batch["columns"][table[aliases.index(alias)]]
        for alias in ("indices_owner", "candidates_owner")
    )
    if (
        candidates["buffer"] != chain[0]["buffers"][0]
        or candidates.get("row_partition") != 8
        or candidates["row_bytes"] != first["parameters"]["sources"] * 1024 * 4
        or length.get("row_partition") != 8
        or length["row_bytes"] != 4
        or definitions[length["buffer"]]["bytes"] != p["_owner_token_capacity"] * 4
    ):
        raise ValueError("TopK causal lengths and candidate rows differ")
    return length["buffer"]


def build(plan, output, workers=2, minimum=512, maximum=None, model_dir=None):
    model, base, describe = stage_plan(
        plan,
        output,
        workers=workers,
        minimum=minimum,
        maximum=maximum,
        require_batch=True,
        validate_source=True,
        model_dir=model_dir,
    )
    tasks, counts = {}, []

    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        for packet in rank["packets"]:
            if not selected_target_packet(packet, minimum, maximum):
                continue
            old, body, index, changed = packet["body"], [], 0, 0
            while index < len(old):
                chain = merge_chain(old, index, describe)
                if chain is None:
                    body.append(old[index])
                    index += 1
                    continue
                check_intermediate_liveness(
                    old[index + 3 :] + packet["head"],
                    [s["buffers"][1] for s in chain[:2]],
                    describe,
                    model["batch"],
                    definitions,
                )
                first, last = describe(chain[0]), describe(chain[-1])
                if not index:
                    raise ValueError("TopK causal tree has no leaf producer")
                lengths = causal_lengths(
                    old[index - 1], chain, describe, model["batch"], definitions
                )
                p = first["parameters"]
                args = {k: v for k, v in p.items() if not k.startswith("_")}
                specs = [
                    deepcopy(first["operands"][0]),
                    dict(name="lengths", dtype="i32", shape=[p["_row_capacity"], 1], access="read"),
                    deepcopy(last["operands"][1]),
                ]
                task = dict(
                    module="kernels.ascend.topk",
                    name="merge_tree",
                    arguments=args,
                    specs=specs,
                    capacity=p["_row_capacity"],
                    dynamic=dict(rows=1),
                )
                identity = json.dumps(task, sort_keys=True)
                if identity not in tasks:
                    tasks[identity] = dict(task, key=f"@topk_tree{len(tasks)}")
                launch = dict(
                    chain[0],
                    manifest=tasks[identity]["key"],
                    buffers=[chain[0]["buffers"][0], lengths, chain[-1]["buffers"][1]],
                )
                body.append(launch)
                changed += 1
                index += 3
            packet["body"] = body
            counts.append(dict(rank=rank["rank"], family=packet["min_rows"], trees=changed))
    if not tasks:
        raise ValueError("No eligible TopK merge trees")
    compile_and_replace(model, tasks, base, workers, compiler=compile_task)
    return finalize_plan(
        model,
        base,
        output,
        owner_tables=True,
        model_dir=model_dir,
        report_name="topk-tree.json",
        report=dict(
            minimum=minimum,
            kernels=len(tasks),
            families=counts,
            additional_owner_bytes=0,
        ),
    )


def main():
    args = plan_parser(__doc__, default_workers=2, min_rows=512, max_rows=True).parse_args()
    print(
        json.dumps(
            build(
                args.plan,
                args.output,
                args.compile_workers,
                args.min_rows,
                args.max_rows,
                args.model_dir,
            )
        )
    )


if __name__ == "__main__":
    main()

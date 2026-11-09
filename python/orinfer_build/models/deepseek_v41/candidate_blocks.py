"""Compile HF candidate-block sharing into every native target row family.

Candidate rows are packet workspace, regenerated at the source layer for each
query. They follow the existing owner/decoder/TP row mapping and never become
persistent KV or prefix state. All consumers use the common INT8 score factory.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import inspect
import json
import math
import multiprocessing
from pathlib import Path

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.common.shared_json import decode
from orinfer_build.models.deepseek_v41.assembly import (
    compact_buffers,
    publish_package,
    validate_rows,
)
from orinfer_build.models.deepseek_v41.batch import compile_task, DTYPES


INDEX_OPS = {"private_rows.indexer_cube_causal_logits", "private_rows.indexer_fused_causal_topk"}


def index_chains(body, describe):
    """Find complete score/selection chains, without crossing collective edges."""
    result = []
    for first, step in enumerate(body):
        spec = describe(step)
        if not spec or spec["op"] not in INDEX_OPS:
            continue
        end = first + 1
        while end < len(body):
            following = describe(body[end])
            if not following or not following["op"].startswith("topk."):
                raise ValueError("Candidate transform requires a complete adjacent TopK chain")
            end += 1
            if following["op"] == "topk.finalize":
                break
        else:
            raise ValueError("Candidate transform found an incomplete index selection")
        result.append((first, end, body[first:end]))
    return result


def build(plan, output, model_dir, workers=4, index_chunk=1024):
    from kernels.ascend import candidate, indexer_cube, indexer_fused, topk
    from kernels.ascend.private_rows import indirect
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import operand
    from orinfer_build.contracts.ascend.geometry import physical_shapes

    if index_chunk not in (512, 1024, 2048, 4096):
        raise ValueError("Invalid index chunk size")
    if output.exists():
        raise ValueError("Require a new candidate-block output directory")
    config = json.loads((model_dir / "config.json").read_text())["text_config"]
    source = config["candidate_source_layer_id"]
    layers = config["index_source_layer_ids"]
    selected, block = config["candidate_topk_blocks"], config["candidate_block_size"]
    if source != 20 or layers != [2, 8, 14, 20, 24, 28, 32, 36] or (selected, block) != (2048, 8):
        raise ValueError("Unsupported candidate-block topology")
    model = decode(json.loads(plan.read_text()))
    contract = model["batch"]
    if any(".candidate_blocks" in c["buffer"] for c in contract["columns"]):
        raise ValueError("Candidate-block topology is already compiled")
    base = output / "source"
    publish_package(model, plan.parent, base)
    descriptions, tasks, registered, registered_grids = {}, {}, {}, {}

    def describe(step):
        if step.get("op") != "kernel":
            return None
        path = step["manifest"]
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    def register(factory, params, writes, *, columns=None):
        signature = json.dumps(
            [factory.__module__, factory.__name__, params, sorted(writes), columns is not None],
            sort_keys=True,
        )
        if signature in registered:
            return registered[signature]
        function = factory(**dict(params, rows=row_count(params["rows"], True)))
        tensors = [function.buffer_map[v] for v in function.params if v in function.buffer_map]
        shapes = physical_shapes(function, params["rows"])
        task = dict(
            module=factory.__module__,
            name=factory.__name__,
            arguments=params,
            capacity=params["rows"],
            dynamic=dict(rows=1),
        )
        if columns is None:
            task["specs"] = [
                operand(
                    t.name,
                    DTYPES[str(t.dtype)],
                    shape,
                    ("read_write" if factory is candidate.restore else "write")
                    if t.name in writes
                    else "read",
                )
                for t, shape in zip(tensors, shapes, strict=True)
            ]
        else:
            names = {t.name: i for i, t in enumerate(tensors)}
            _, _, grid = indirect(function, names, contract["owners"], params["rows"])
            task.update(
                columns=names,
                owners=contract["owners"],
                table_columns=len(columns),
                specs=[
                    operand("OwnerWork", "i32", [8192, 2]),
                    operand("OwnerPointers", "u64", [contract["owners"], len(columns) + 1]),
                    operand("PacketActive", "i32", [1]),
                ],
                metadata=dict(
                    _owner_token_capacity=params["rows"],
                    _owner_source_sha256=file_hash(Path(inspect.getfile(factory))),
                    _owner_aliases={t.name + "_owner": str(t.dtype) for t in tensors},
                    _owner_access=[
                        dict(column=i, write=t.name in writes) for i, t in enumerate(tensors)
                    ],
                    _owner_writes=sorted(writes),
                ),
            )
        identity = json.dumps(task, sort_keys=True)
        if identity not in tasks:
            tasks[identity] = dict(task, key=f"@candidate{len(tasks)}")
        if columns is not None:
            registered_grids[tasks[identity]["key"]] = grid
        registered[signature] = tasks[identity]["key"], shapes
        return registered[signature]

    changed = 0
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}

        def define(name, size):
            if name not in definitions:
                definition = dict(name=name, bytes=size)
                definitions[name] = definition
                rank["buffers"].append(definition)
            elif definitions[name]["bytes"] != size:
                raise ValueError("Candidate workspace extent differs: " + name)
            return name

        def pointer(old, columns):
            ids = []
            for column in columns:
                if column not in contract["columns"]:
                    contract["columns"].append(column)
                ids.append(contract["columns"].index(column))
            name = (
                "batch.candidate.pointers."
                + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
            )
            previous = old["buffers"][1]
            contract["pointers"][name] = ids
            partitions = contract.get("partitioned_tables", {})
            if previous in partitions:
                partitions[name] = partitions[previous]
            replay = contract.get("decoder_replay")
            if replay and previous in replay["tables"]:
                replay["tables"] = sorted(set(replay["tables"]) | {name})
            define(name, contract["owners"] * (len(ids) + 1) * 8)
            return name

        references = {}
        chains = {}
        for i, packet in enumerate(rank["packets"]):
            if packet.get("role", "target") != "target":
                continue
            chains[i] = index_chains(packet["body"], describe)
            if len(chains[i]) != len(layers):
                raise ValueError("Candidate transform requires all configured index source layers")
            for _, _, chain in chains[i]:
                step = chain[0]
                if describe(step)["op"] == "private_rows.indexer_cube_causal_logits":
                    ids = contract["pointers"][step["buffers"][1]]
                    references[tuple(ids[:6])] = chain

        for packet_index, packet_chains in chains.items():
            packet = rank["packets"][packet_index]
            body, cursor = [], 0
            for layer, (first, end, original) in zip(layers, packet_chains, strict=True):
                body.extend(packet["body"][cursor:first])
                cursor = end
                if layer < source:
                    body.extend(original)
                    continue
                old = original[0]
                ids = contract["pointers"][old["buffers"][1]]
                reference = deepcopy(references.get(tuple(ids[:6])))
                if reference is None:
                    raise ValueError(
                        "Require an unfused score reference for the same owner row geometry"
                    )
                ref = reference[0]
                params = {
                    k: v for k, v in describe(ref)["parameters"].items() if not k.startswith("_")
                }
                if not params.get("length_input") or params.get("candidate_blocks"):
                    raise ValueError("Require original length-driven index scores")
                columns = [
                    deepcopy(contract["columns"][i])
                    for i in contract["pointers"][ref["buffers"][1]]
                ]
                if columns[2]["buffer"] != "cache.key.20":
                    raise ValueError("Candidate consumers must share the source key cache")
                count, length = params["rows"], params["length"]
                candidate_name = define(
                    columns[5]["buffer"] + ".candidate_blocks", count * (selected + 8) * 4
                )
                launch = {k: deepcopy(v) for k, v in reference[1].items() if k == "row_expression"}
                if reference[-1]["buffers"][1] != original[-1]["buffers"][1]:
                    raise ValueError("Candidate reference changes the shared selected-index view")
                if layer == source:
                    manifest, _ = register(
                        candidate.select_blocks,
                        dict(rows=count, length=length, selected=selected, block_size=block),
                        {"out"},
                    )
                    body.extend(
                        [
                            reference[0],
                            dict(
                                op="kernel",
                                manifest=manifest,
                                buffers=[
                                    columns[6]["buffer"],
                                    columns[5]["buffer"],
                                    candidate_name,
                                ],
                                **launch,
                            ),
                            *reference[1:],
                        ]
                    )
                else:
                    width = min(length, selected * block)
                    prefix = columns[5]["buffer"] + ".candidate_blocks"
                    columns[5].update(buffer=candidate_name, row_bytes=(selected + 8) * 4)
                    groups = (width + index_chunk - 1) // index_chunk
                    partial = define(prefix + ".local", count * groups * 1024 * 4)
                    columns[6].update(buffer=partial, row_bytes=groups * 1024 * 4)
                    cube = define(prefix + ".cube", 64 * 32 * 128 * 4)
                    scores = define(prefix + ".tile_scores", 64 * index_chunk * 4)
                    for name in (cube, scores):
                        if name not in contract["private"]:
                            contract["private"].append(name)
                    columns = columns[:7] + [
                        dict(buffer=name, row_bytes=None, row_alignment=1, decode=None)
                        for name in (cube, scores)
                    ]
                    params = dict(
                        rows=count,
                        length=length,
                        tile_rows=1,
                        candidate_blocks=selected,
                        candidate_block_size=block,
                        chunk_size=index_chunk,
                    )
                    manifest, shapes = register(
                        indexer_fused.causal_topk,
                        params,
                        {"candidates", "cube_workspace", "score_workspace"},
                        columns=columns,
                    )
                    for column, shape, item_size in zip(
                        columns, shapes, (1, 2, 1, 2, 2, 4, 4, 4, 4), strict=True
                    ):
                        expected = (
                            column["row_bytes"] * count
                            if column["row_bytes"] is not None
                            else definitions[column["buffer"]]["bytes"]
                        )
                        if math.prod(shape) * item_size != expected:
                            raise ValueError(
                                "Candidate score operand changes physical geometry: "
                                + column["buffer"]
                            )
                    table = pointer(ref, columns)
                    grid = registered_grids[manifest]
                    previous = ref["buffers"][0]
                    partition = contract.get("partitioned_tables", {}).get(previous)
                    replay = contract.get("decoder_replay")
                    decoder = replay is not None and previous in replay["tables"]
                    work = (
                        "batch.candidate.work."
                        + hashlib.sha256(
                            json.dumps([grid, partition, decoder]).encode()
                        ).hexdigest()[:16]
                    )
                    contract["work"][work] = grid
                    if partition is not None:
                        contract["partitioned_tables"][work] = partition
                    if previous in contract.get("work_ranges", {}):
                        contract["work_ranges"][work] = deepcopy(contract["work_ranges"][previous])
                    if decoder:
                        replay["tables"] = sorted(set(replay["tables"]) | {work})
                    define(work, 8192 * 2 * 4)
                    body.append(
                        dict(
                            old,
                            manifest=manifest,
                            owner_work=grid,
                            buffers=[work, table, contract["active"]],
                        )
                    )

                    if groups in (16, 32, 64):
                        following = define(prefix + ".merged", count * 1024 * 4)
                        manifest, _ = register(
                            topk.merge_tree,
                            dict(rows=count, sources=groups, leaf_size=index_chunk),
                            {"out"},
                        )
                        body.append(
                            dict(
                                op="kernel",
                                manifest=manifest,
                                buffers=[partial, contract["columns"][ids[5]]["buffer"], following],
                                **launch,
                            )
                        )
                        partial, groups = following, 1
                    level = 0
                    while groups > 1:
                        following = define(
                            prefix + f".merge{level}", count * ((groups + 3) // 4) * 1024 * 4
                        )
                        manifest, _ = register(
                            topk.merge, dict(rows=count, sources=groups), {"out"}
                        )
                        body.append(
                            dict(
                                op="kernel",
                                manifest=manifest,
                                buffers=[partial, following],
                                **launch,
                            )
                        )
                        partial, groups, level = following, (groups + 3) // 4, level + 1
                    destination, values = original[-1]["buffers"][1:]
                    manifest, _ = register(
                        candidate.finalize, dict(rows=count), {"indices", "values"}
                    )
                    body.append(
                        dict(
                            op="kernel",
                            manifest=manifest,
                            buffers=[candidate_name, partial, destination, values],
                            **launch,
                        )
                    )
                changed += 1
            body.extend(packet["body"][cursor:])
            packet["body"] = body

    # Rebuild every surviving instance of the shared score factory so the
    # package retains one source identity per operator, including plain decode.
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step.get("manifest", "").startswith("@"):
                    continue
                spec = describe(step)
                if not spec:
                    continue
                params = {k: v for k, v in spec["parameters"].items() if not k.startswith("_")}
                if spec["op"] == "private_rows.indexer_cube_causal_logits":
                    columns = [
                        contract["columns"][i] for i in contract["pointers"][step["buffers"][1]]
                    ]
                    step["manifest"], _ = register(
                        indexer_cube.causal_logits, params, {"out", "workspace"}, columns=columns
                    )
                elif spec["op"] == "private_rows.indexer_fused_causal_topk":
                    columns = [
                        contract["columns"][i] for i in contract["pointers"][step["buffers"][1]]
                    ]
                    step["manifest"], _ = register(
                        indexer_fused.causal_topk,
                        params,
                        {"candidates", "cube_workspace", "score_workspace"},
                        columns=columns,
                    )
                elif spec["op"] in {"topk.local", "topk.merge", "topk.finalize", "topk.merge_tree"}:
                    name = spec["op"].split(".")[1]
                    step["manifest"], _ = register(
                        getattr(topk, name), params, {"out", "indices", "values"}
                    )
    jobs = [(task, str(base)) for task in tasks.values()]
    if workers == 1:
        replacements = dict(compile_task(job) for job in jobs)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            replacements = dict(pool.map(compile_task, jobs))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step.get("manifest") in replacements:
                    step["manifest"] = replacements[step["manifest"]]
    compact_buffers(model, owner_tables=True)
    validate_rows(model, base, model_dir=model_dir)
    publish_package(model, base, output / "package")
    report = dict(
        changed_chains=changed,
        kernels=len(tasks),
        index_chunk_size=index_chunk,
        source_layer=source,
        consumer_index_layers=[v for v in layers if v > source],
        model=str(output / "package/model.json"),
    )
    atomic_json(output / "candidate-blocks.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--index-chunk", type=int, choices=(512, 1024, 2048, 4096), default=1024)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.plan, args.output, args.model_dir, args.compile_workers, args.index_chunk)
        )
    )


if __name__ == "__main__":
    main()

"""Group owner-private indexer queries while retaining the exact FP32 head sum."""

from copy import deepcopy
import hashlib
import json

from orinfer_build.common.files import atomic_json
from orinfer_build.models.deepseek_v41.publication import manifest_bindings
from orinfer_build.models.deepseek_v41.transform import (
    finalize_plan,
    plan_parser,
    stage_plan,
)


INDEXER = "private_rows.indexer_fused_causal_topk"
PREPARE = "private_rows.indexer_fused_prepare_queries"


def group_queries(model, descriptions, query_group=4):
    """Bind one shared grouping template to each full-history owner packet."""
    if query_group not in (2, 4):
        raise ValueError("Require two or four queries per indexer group")
    selected = {
        path: desc
        for path, desc in descriptions.items()
        if desc["op"] == INDEXER and not desc["parameters"].get("candidate_blocks", 0)
    }
    capacities = {d["parameters"]["_owner_token_capacity"] for d in selected.values()}
    if len(capacities) != 1 or any(
        d["parameters"].get("query_group", 1) != 1 or d["parameters"]["tile_rows"] % query_group
        for d in selected.values()
    ):
        raise ValueError("Require one converged full-history query capacity and compatible tiles")
    capacity = next(iter(capacities))
    contract = model["batch"]
    owners = contract["owners"]
    names = ["batch.prefill.indexer.packed." + name for name in ("query", "scale", "weights")]
    sizes = [capacity * 32 * 128, capacity * 32 * 2, capacity * 32 * 2]
    if any(name in contract["private"] for name in names):
        raise ValueError("Indexer query grouping is already present")
    packed_columns = []
    for name in names:
        contract["private"].append(name)
        packed_columns.append(len(contract["columns"]))
        contract["columns"].append(dict(buffer=name, row_bytes=None, row_alignment=1, decode=None))
    prototype = "cache/grouped-prepare-prototype/kernel.json"
    prepare = deepcopy(next(iter(selected.values())))
    prepare["op"] = PREPARE
    params = prepare["parameters"] = {
        key: value for key, value in prepare["parameters"].items() if key.startswith("_owner_")
    }
    params.update(rows=capacity, query_group=query_group)
    ports = [
        ("query", "int8"),
        ("scales", "float16"),
        ("weights", "bfloat16"),
        ("packed_query", "int8"),
        ("packed_scales", "float16"),
        ("packed_weights", "bfloat16"),
    ]
    params["_owner_aliases"] = {name + "_owner": dtype for name, dtype in ports}
    params["_owner_access"] = [dict(column=i, write=i >= 3) for i in range(6)]
    params["_owner_writes"] = [name for name, _ in ports[3:]]
    prepare["operands"][1]["shape"] = [owners, 7]
    changed = 0
    for rank in model["ranks"]:
        definitions = {value["name"]: value for value in rank["buffers"]}

        def buffer(name, size):
            if name not in definitions:
                value = dict(name=name, bytes=size)
                rank["buffers"].append(value)
                definitions[name] = value
            if definitions[name]["bytes"] != size:
                raise ValueError("Grouped indexer buffer extent differs: " + name)

        for name, size in zip(names, sizes, strict=True):
            buffer(name, size)

        def pointer(indices, source):
            digest = hashlib.sha256(json.dumps(indices).encode()).hexdigest()[:16]
            name = "batch.indexer.grouped.pointers." + digest
            contract["pointers"][name] = indices
            contract["partitioned_tables"][name] = contract["partitioned_tables"][source]
            replay = contract.get("decoder_replay")
            if replay and source in replay["tables"]:
                replay["tables"] = sorted(set(replay["tables"]) | {name})
            buffer(name, owners * (len(indices) + 1) * 8)
            return name

        for packet in rank["packets"]:
            body = []
            for step in packet["body"]:
                if step.get("manifest") not in selected:
                    body.append(step)
                    continue
                old_pointer = step["buffers"][1]
                indices = contract["pointers"][old_pointer]
                if len(indices) != 9 or step["owner_work"] != dict(op="constant", value=64):
                    raise ValueError("Require the bounded 64-CTA owner indexer ABI")
                prep = deepcopy(step)
                prep["manifest"] = prototype
                prep["buffers"][1] = pointer(
                    [indices[0], indices[1], indices[4], *packed_columns], old_pointer
                )
                grouped = deepcopy(step)
                grouped["buffers"][1] = pointer(
                    [*packed_columns[:2], *indices[2:4], packed_columns[2], *indices[5:]],
                    old_pointer,
                )
                body.extend((prep, grouped))
                changed += 1
            packet["body"] = body
    if not changed:
        raise ValueError("No full-history indexer instructions matched")
    for desc in selected.values():
        desc["parameters"]["query_group"] = query_group
    return {**selected, prototype: prepare}, dict(
        changed=changed, additional_owner_bytes=sum(sizes), query_group=query_group
    )


def build(plan, output, workers=4, query_group=4, model_dir=None):
    from orinfer_build.models.deepseek_v41.build import rebuild_kernels

    model, base, _ = stage_plan(plan, output, workers=workers)
    paths = {binding[key] for binding, key in manifest_bindings(model)}
    descriptions = {path: json.loads((base / path).read_text()) for path in paths}
    updates, report = group_queries(model, descriptions, query_group)
    for path, desc in updates.items():
        (base / path).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(base / path, desc)
    atomic_json(base / "model.json", model)
    rebuild_kernels(base / "model.json", workers, operators=[INDEXER, PREPARE])
    model = json.loads((base / "model-selected.json").read_text())
    return finalize_plan(
        model,
        base,
        output,
        owner_tables=True,
        model_dir=model_dir,
        report_name="indexer-grouping.json",
        report=report,
    )


def main():
    parser = plan_parser(__doc__, default_workers=4)
    parser.add_argument("--query-group", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.plan, args.output, args.compile_workers, args.query_group, args.model_dir)
        )
    )


if __name__ == "__main__":
    main()

"""Build and validate one native A2 row execution from the pinned model contract.

The canonical unsharded prefill body supplies all layers. Existing operator
factories build the selected SP/attention/replay subgraphs; real row expressions
bind kernels, views and communication. No tuning or Python runs online.
"""

import argparse
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile

from tools.build.files import atomic_json, file_hash, source_hash
from tools.model.row_contract import resolve


def referenced_buffers(rank, batch=None):
    used = set()
    if batch is not None:
        # Indirect kernels name pointer/work tables in their instructions;
        # the owner ABI also keeps their pointees and private views alive.
        used.update(batch["private"])
        used.update(column["buffer"] for column in batch["columns"])
        used.update(batch["pointers"])
        used.update(batch["work"])
        used.update((batch["active"], batch["endpoints"]))
    for packet in rank["packets"]:
        used.update(v for v in packet["coordinates"].values() if isinstance(v, str))
        used.update(packet["reset"])
        used.update(packet[k] for k in ("best_value", "best_index", "logits") if packet.get(k))
        for step in packet["body"] + packet["head"]:
            used.update(step.get("buffers", []))
            for key in (
                "source",
                "destination",
                "buffer",
                "indices",
                "gather",
                "restore",
                "workspace",
                "values",
                "scales",
                "mask",
            ):
                if isinstance(step.get(key), str):
                    used.add(step[key])
            groups = step.get("groups")
            used.update([groups] if isinstance(groups, str) else groups or [])
    definitions = {v["name"]: v for v in rank["buffers"]}
    pending = list(used)
    while pending:
        definition = definitions[pending.pop()]
        if "view" in definition:
            parent = definition["view"]["parent"]
            if parent not in used:
                used.add(parent)
                pending.append(parent)
    return used


def compact_buffers(model, *, owner_tables=False):
    batch = model.get("batch")
    if batch is not None and owner_tables:
        indirect = [
            step
            for rank in model["ranks"]
            for packet in rank["packets"]
            for step in packet["body"] + packet["head"]
            if step["op"] == "kernel" and "owner_work" in step
        ]
        pointers = {step["buffers"][1] for step in indirect}
        work = {step["buffers"][0] for step in indirect}
        live = sorted({column for name in pointers for column in batch["pointers"][name]})
        remap = {old: new for new, old in enumerate(live)}
        batch["columns"] = [batch["columns"][index] for index in live]
        batch["pointers"] = {
            name: [remap[index] for index in mapping]
            for name, mapping in batch["pointers"].items()
            if name in pointers
        }
        batch["work"] = {name: grid for name, grid in batch["work"].items() if name in work}
        for field in ("work_ranges", "partitioned_tables"):
            if field in batch:
                batch[field] = {
                    name: value
                    for name, value in batch[field].items()
                    if name in (work if field == "work_ranges" else work | pointers)
                }
    for rank in model["ranks"]:
        used = referenced_buffers(rank, model.get("batch"))
        rank["buffers"] = [b for b in rank["buffers"] if b["name"] in used]
    used = {
        s["manifest"]
        for r in model["ranks"]
        for p in r["packets"]
        for s in p["body"] + p["head"]
        if s["op"] == "kernel"
    }
    if "row_selections" in model:
        selections = model["row_selections"]
        domains, expressions = {}, {}
        for rank in model["ranks"]:
            for packet in rank["packets"]:
                first = max(1, packet.get("min_rows", 0))
                last = packet.get("max_rows", packet["rows"])
                for program in (packet["body"], packet["head"]):
                    active = tuple(range(first, last + 1))
                    for step in program:
                        if step["op"] == "limit_rows":
                            active = tuple(sorted({min(m, step["rows"]) for m in active}))
                        elif step["op"] == "kernel" and step["manifest"] in selections:
                            if "fixed_rows" in step:
                                rows = [step["fixed_rows"]]
                            elif "row_expression" in step:
                                expr = step["row_expression"]
                                key = (json.dumps(expr, sort_keys=True), active)
                                if key not in expressions:
                                    expressions[key] = {resolve(expr, m) for m in active}
                                rows = expressions[key]
                            else:
                                rows = active
                            domains.setdefault(step["manifest"], set()).update(rows)
        selected = {}
        for slot, variants in selections.items():
            if slot not in used:
                continue
            intervals = []
            for variant in variants:
                multiple = variant.get("multiple_of", 1)
                for rows in sorted(domains[slot]):
                    if not variant["min_rows"] <= rows <= variant["max_rows"] or rows % multiple:
                        continue
                    if (
                        intervals
                        and intervals[-1]["max_rows"] + multiple == rows
                        and intervals[-1]["kernels"] == variant["kernels"]
                        and intervals[-1].get("multiple_of", 1) == multiple
                    ):
                        intervals[-1]["max_rows"] = rows
                    else:
                        intervals.append(dict(variant, min_rows=rows, max_rows=rows))
            if intervals:
                selected[slot] = intervals
        model["row_selections"] = selected
    return model


def canonical(model):
    model = deepcopy(model)
    if model.get("schema_version") != 1 or model.get("layers") != 40 or len(model["ranks"]) != 8:
        raise ValueError("Expected the pinned complete DSv4.1 TP8 model contract")
    for rank in model["ranks"]:
        source = next(p for p in rank["packets"] if p.get("min_rows") == 2)
        if any(s["op"] == "limit_rows" for s in source["body"]):
            raise ValueError("Canonical input requires the complete non-replay reference")
        source = deepcopy(source)
        source.pop("max_rows", None)
        rank["packets"] = [p for p in rank["packets"] if p.get("min_rows", 0) == 0] + [source]
        if any(name.startswith("sequence.") for name in referenced_buffers(rank)):
            raise ValueError("Canonical input requires an unsharded small-row reference")
    return compact_buffers(model)


def validate_rows(model, base):
    """Offline full-domain arithmetic and physical span checks, deduplicated."""
    expressions, specs = {}, {}
    sizes = dict(i8=1, u8=1, f16=2, bf16=2, f32=4, i32=4, i64=8, u64=8)

    def values(expr, capacity):
        key = (json.dumps(expr, sort_keys=True), capacity)
        if key not in expressions:
            expressions[key] = [resolve(expr, rows) for rows in range(1, capacity + 1)]
        return expressions[key]

    def spec_for(path):
        if path not in specs:
            specs[path] = json.loads((base / path).read_text())
        return specs[path]

    for slot, variants in model.get("row_selections", {}).items():
        source = spec_for(slot)
        previous = 0
        for variant in variants:
            if (
                variant["min_rows"] <= previous
                or variant["min_rows"] > variant["max_rows"]
                or variant["max_rows"] > source["parameters"]["_row_capacity"]
                or variant.get("multiple_of", 1) != 1
                or len(variant["kernels"]) != 1
            ):
                raise ValueError("Invalid local row selection")
            previous = variant["max_rows"]
            replacement = spec_for(variant["kernels"][0])
            knobs = {"tile_m", "tile_n", "block_k", "pipeline_stages", "_grid_limit"}

            def math_parameters(spec):
                return {k: v for k, v in spec["parameters"].items() if k not in knobs}

            if (
                source["op"] != "dense.projection"
                or replacement["op"] != source["op"]
                or source.get("target") != replacement.get("target")
                or source.get("source_sha256") != replacement.get("source_sha256")
                or source.get("compiler") != replacement.get("compiler")
                or len(source["operands"]) != len(replacement["operands"])
                or any(
                    {k: v for k, v in a.items() if k != "shape"}
                    != {k: v for k, v in b.items() if k != "shape"}
                    or (a.get("access") != "scratch" and a["shape"] != b["shape"])
                    for a, b in zip(source["operands"], replacement["operands"], strict=True)
                )
                or math_parameters(source) != math_parameters(replacement)
            ):
                raise ValueError("Local replacement changes arithmetic or physical operands")

    for rank in model["ranks"]:
        buffers = {}
        for definition in rank["buffers"]:
            name, size = definition["name"], definition["bytes"]
            if size <= 0 or name in buffers:
                raise ValueError("Invalid or duplicate buffer")
            if "view" in definition:
                view = definition["view"]
                parent = buffers[view["parent"]]
                offsets = [view["offset"]]
                if "offset_expression" in view:
                    offsets = values(view["offset_expression"], view["row_capacity"])
                    if offsets[-1] != view["offset"]:
                        raise ValueError("View capacity expression differs")
                if any(offset + size > parent for offset in offsets):
                    raise ValueError("Dynamic view exceeds its parent: " + name)
            buffers[name] = size
        contract = model.get("batch", {})
        for column in contract.get("columns", []):
            if column.get("row_bytes") is None:
                continue
            if column.get("decode") not in (True, False):
                raise ValueError("Owner row column requires a family")
            family = next(
                p for p in rank["packets"] if p.get("batch_decode", False) == column["decode"]
            )
            stride = column.get("row_partition", 1)
            if stride < 1:
                raise ValueError("Invalid owner row partition")
            rows = (family["rows"] + stride - 1) // stride
            rows += contract["owners"] * (column.get("row_alignment", 1) - 1)
            if rows * column["row_bytes"] != buffers[column["buffer"]]:
                raise ValueError("Owner column physical row stride differs")
        for packet in rank["packets"]:
            for program in (packet["body"], packet["head"]):
                capacity = packet["rows"]
                for step in program:
                    if step["op"] == "limit_rows":
                        if not 1 <= step["rows"] <= capacity:
                            raise ValueError("Invalid compacted row boundary")
                        capacity = step["rows"]
                    elif step["op"] == "kernel":
                        path = step["manifest"]
                        spec = spec_for(path)
                        if len(step["buffers"]) != len(spec["operands"]):
                            raise ValueError("Kernel pointer count differs")
                        for name, operand in zip(step["buffers"], spec["operands"], strict=True):
                            expected = math.prod(operand["shape"]) * sizes[operand["dtype"]]
                            if (
                                buffers[name] < expected
                                if operand.get("access") == "scratch"
                                else buffers[name] != expected
                            ):
                                raise ValueError("Kernel physical operand differs: " + name)
                        for variant in model.get("row_selections", {}).get(path, []):
                            for name, operand in zip(
                                step["buffers"],
                                spec_for(variant["kernels"][0])["operands"],
                                strict=True,
                            ):
                                required = math.prod(operand["shape"]) * sizes[operand["dtype"]]
                                if operand.get("access") == "scratch" and required > buffers[name]:
                                    raise ValueError("Selected scratch exceeds allocation: " + name)
                        bound = step.get("fixed_rows", capacity)
                        if step.get("owner_work") is not None:
                            bound = 4096
                            tables = model["batch"].get("partitioned_tables", {})
                            if (
                                tables
                                or "_owner_token_capacity" in spec["parameters"]
                                or spec.get("op", "").startswith("private_rows.")
                            ):
                                if len(step["buffers"]) < 3:
                                    raise ValueError(
                                        "Owner token capacity requires complete owner tables"
                                    )
                                stride = tables.get(step["buffers"][0], 1)
                                if (
                                    not isinstance(stride, int)
                                    or isinstance(stride, bool)
                                    or stride < 1
                                    or tables.get(step["buffers"][1], 1) != stride
                                    or spec["parameters"].get("_owner_token_capacity")
                                    != (packet["rows"] + stride - 1) // stride
                                ):
                                    raise ValueError(
                                        "Owner token capacity differs from its row partition"
                                    )
                        elif step.get("owners"):
                            bound = model["batch"]["owners"]
                        if "row_expression" in step:
                            if "fixed_rows" in step:
                                raise ValueError("Ambiguous kernel row contract")
                            live = values(step["row_expression"], capacity)
                            bound = live[-1]
                            if min(live) < 1 or max(live) > bound:
                                raise ValueError("Kernel row expression exceeds capacity")
                        if (
                            expected := spec["parameters"].get("_row_capacity")
                        ) is not None and expected != bound:
                            raise ValueError("Kernel row capacity differs")
                    elif step["op"] in ("gather", "reduce", "exchange"):
                        count = step["elements"]
                        if "elements_expression" in step:
                            if step.get("fixed_rows"):
                                raise ValueError("Ambiguous collective row contract")
                            live = values(step["elements_expression"], capacity)
                            if live[-1] != count or min(live) < 1 or max(live) > count:
                                raise ValueError("Collective row capacity differs")
                        elif not step.get("fixed_rows") and count % capacity:
                            raise ValueError("Unbounded collective extent")
                        if step.get("fp32") and step.get("int32"):
                            raise ValueError("Ambiguous collective dtype")
                        byte_count = count * (4 if step.get("fp32") or step.get("int32") else 2)
                        read = byte_count * (
                            8 if step.get("scatter") or step["op"] == "exchange" else 1
                        )
                        write = byte_count * (8 if step["op"] in ("gather", "exchange") else 1)
                        if read > buffers[step["source"]] or write > buffers[step["destination"]]:
                            raise ValueError("Collective span exceeds allocation")
    return dict(
        kernels=len(specs),
        expressions=len(expressions),
        checked_rows=sum(map(len, expressions.values())),
    )


def publish_package(model, base, destination):
    """Publish referenced native assets with hard-linked prepared weight banks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    assets = {model["engram"]["file"]} | {v["file"] for v in model["rope"].values()}
    for rank in model["ranks"]:
        for definition in rank["buffers"]:
            if "prepared_weight" in definition:
                assets.add(definition["prepared_weight"])
            if "data" in definition:
                assets.add(definition["data"]["file"])
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    path = Path(step["manifest"])
                    assets.add(str(path))
                    assets.add(
                        str(path.parent / json.loads((base / path).read_text())["library"]["file"])
                    )
    for variants in model.get("row_selections", {}).values():
        for variant in variants:
            for name in variant["kernels"]:
                path = Path(name)
                assets.add(str(path))
                assets.add(
                    str(path.parent / json.loads((base / path).read_text())["library"]["file"])
                )
    if destination.exists():
        if json.loads((destination / "model.json").read_text()) != model:
            raise ValueError("Existing package differs from the selected contract")
        for relative in assets:
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Package asset path must remain relative")
            if not (destination / relative).samefile(base / relative):
                raise ValueError("Existing package asset identity differs")
        actual = {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()}
        if actual != assets | {"model.json"}:
            raise ValueError("Existing package contains unselected assets")
        return len(assets)
    with tempfile.TemporaryDirectory(prefix=".a2-package-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        for relative in sorted(assets):
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Package asset path must remain relative")
            source = (base / relative).resolve(strict=True)
            source.relative_to(base.resolve())
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(source)
        atomic_json(staging / "model.json", model)
        staging.rename(destination)
    return len(assets)


def run_transform(module, source, output, workers, options=()):
    """Compose the existing offline transforms without another compiler path."""
    previous = sys.argv
    try:
        sys.argv = [
            module,
            "--plan",
            str(source),
            "--output",
            str(output),
            "--compile-workers",
            str(workers),
            *options,
        ]
        importlib.import_module("tools.model.deepseek_v41." + module).main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-output", type=Path)
    parser.add_argument("--sequence-min-rows", type=int, default=512)
    parser.add_argument("--reduce-scatter", action="store_true")
    parser.add_argument("--shared-attention", action="store_true")
    parser.add_argument("--fused-attention", action="store_true")
    parser.add_argument("--fused-min-rows", type=int, default=512)
    parser.add_argument("--replay-keep", type=int)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--tune-projections", choices=("off", "local", "all"), default="off")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--packet-cache-capacity", type=int, default=64)
    args = parser.parse_args()
    if args.plan.parent.resolve() != args.output.parent.resolve():
        parser.error("Plans must share their asset directory")
    if args.plan.resolve() == args.output.resolve():
        parser.error("Output must preserve the input reference")
    if args.tune_projections != "off" and args.model_dir is None:
        parser.error("Real-weight projection tuning requires --model-dir")
    if args.tune_projections == "local" and not args.shared_attention:
        parser.error("Local projection tuning requires --shared-attention")
    if not 4 <= args.packet_cache_capacity <= 4096:
        parser.error("Packet cache capacity must be in 4..4096")
    if args.replay_keep is not None and (not 8 <= args.replay_keep <= 512 or args.replay_keep % 8):
        parser.error("Replay suffix must be divisible by eight, within 8..512")
    base = args.plan.parent
    root = Path(__file__).resolve().parents[3]
    source = source_hash(
        root,
        [p for p in Path(__file__).parent.glob("*.py") if not p.name.startswith("test_")]
        + [
            root / "tools/model/row_contract.py",
            root / "tools/model/row_selection.py",
            root / "tools/operators/ascend/geometry.py",
        ],
    )
    model = canonical(json.loads(args.plan.read_text()))
    model["packet_cache_capacity"] = args.packet_cache_capacity
    policy = {
        k: v
        for k, v in vars(args).items()
        if k not in ("plan", "output", "package_output", "compile_workers", "model_dir")
    }
    identity = dict(source=source, input=file_hash(args.plan), policy=policy)
    import hashlib

    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    prefix = base / (".converge-" + key)
    reference = prefix.with_name(prefix.name + "-reference.json")
    atomic_json(reference, model)
    current, stages = reference, []

    def run(module, options):
        nonlocal current
        output = prefix.with_name(prefix.name + f"-{len(stages)}.json")
        run_transform(module, current, output, args.compile_workers, options)
        stages.append(dict(stage=module, sha256=file_hash(output)))
        current = output

    run("shard_indexer", ["--min-rows", str(args.sequence_min_rows)])
    run(
        "sequence_parallel",
        ["--min-rows", str(args.sequence_min_rows)]
        + (["--reduce-scatter"] if args.reduce_scatter else [])
        + (["--shared-attention"] if args.shared_attention else []),
    )
    if args.fused_attention:
        run(
            "fuse_attention",
            [
                "--min-rows",
                str(args.fused_min_rows),
                "--tile-m",
                "64",
                "--tile-k",
                "256",
                "--reuse-kv",
            ],
        )
    if args.replay_keep:
        run("decoder_replay", ["--tail-template", str(reference), "--keep", str(args.replay_keep)])
        run(
            "sequence_parallel",
            ["--decoder-tail"]
            + (["--reduce-scatter"] if args.reduce_scatter else [])
            + (["--shared-attention"] if args.shared_attention else []),
        )
    if args.tune_projections != "off":
        previous = sys.argv
        try:
            sys.argv = [
                "tune",
                "--plan",
                str(current),
                "--model-dir",
                str(args.model_dir),
                "--output",
                str(base / ".tuning-projections"),
                "--joint",
                "--compile-workers",
                str(args.compile_workers),
            ]
            if args.tune_projections == "local":
                sys.argv.append("--local-only")
            importlib.import_module("tools.model.deepseek_v41.tune").main()
        finally:
            sys.argv = previous
        current = current.with_name(current.stem + "-tuned.json")
        stages.append(dict(stage="tune", sha256=file_hash(current)))
    model = compact_buffers(json.loads(current.read_text()))
    validation = validate_rows(model, base)
    assets = publish_package(model, base, args.package_output) if args.package_output else None
    atomic_json(args.output, model)
    receipt = dict(
        identity=identity,
        stages=stages,
        validation=validation,
        package_assets=assets,
        output_sha256=file_hash(args.output),
    )
    atomic_json(args.output.with_suffix(".build.json"), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

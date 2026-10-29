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
from tools.model.row_selection import guarded_winner


def select_schedule(
    reports,
    operator,
    parameter,
    defaults,
    geometries,
    source_sha256,
    measurement_sha256,
    maximum_regression=0.1,
):
    """Choose one same-source tile and one measured small-owner boundary.

    Complete 1..128 measurements protect the small-owner domain. Larger rows
    share one configuration, with a per-case guard across every geometry and
    length in the supplied record. The existing work ranges publish the choice.
    """
    reference = defaults[parameter]

    def geometry(parameters):
        p = dict(defaults, **{k: v for k, v in parameters.items() if not k.startswith("_")})
        p.pop("rows", None)
        p.pop(parameter)
        return json.dumps(p, sort_keys=True)

    expected = {geometry(p) for p in geometries}
    groups = {}
    for report in reports:
        if report["op"] != operator:
            continue
        if (
            report["source_sha256"] != source_sha256
            or (report.get("measurement_identity") or {}).get("measurement_source_sha256")
            != measurement_sha256
            or not all(report.get(k) for k in ("exact_reference", "graph_replay", "changed_owners"))
        ):
            raise ValueError("Tuning record lacks current-source numerical/Graph validation")
        p = report["parameters"]
        key = geometry(p)
        config = json.dumps(p.get(parameter, reference))
        measured = {r["rows"]: r["milliseconds"] for r in report["measurements"]}
        if (
            len(measured) != len(report["measurements"])
            or not set(range(1, 129)) <= measured.keys()
        ):
            raise ValueError("Tuning needs unique complete small-owner measurements1..128")
        variants = groups.setdefault(key, {})
        if config in variants:
            raise ValueError("Duplicate measured private schedule")
        variants[config] = measured
    if set(groups) != expected or not expected:
        raise ValueError("Measured private geometries differ from the model")
    variants = next(iter(groups.values()))
    configs, lengths = set(variants), set(next(iter(variants.values())))
    capacity = max(p["rows"] for p in geometries)
    if capacity not in lengths or any(
        not isinstance(r, int) or not 1 <= r <= capacity for r in lengths
    ):
        raise ValueError("Private tuning must cover the model capacity and valid row domain")
    if any(
        set(v) != configs or any(set(m) != lengths for m in v.values()) for v in groups.values()
    ):
        raise ValueError("Private schedules need matching configurations and row lengths")
    baseline = json.dumps(reference)
    if baseline not in configs:
        raise ValueError("Missing original private schedule measurements")
    if any(p.get(parameter, reference) != reference for p in geometries):
        raise ValueError("Private tuning requires one original reference configuration")
    policies = [(baseline, 1)] + [
        (config, first) for config in sorted(configs - {baseline}) for first in range(1, 130)
    ]
    cases = [
        dict(
            rows=rows,
            times={
                str(i): measured[config if rows >= first else baseline][rows]
                for i, (config, first) in enumerate(policies)
            },
        )
        for measured in groups.values()
        for rows in sorted(lengths)
    ]
    chosen = int(guarded_winner(cases, maximum_regression=maximum_regression))
    config, first = policies[chosen]
    return dict(
        parameters={parameter: json.loads(config)},
        first_owner_rows=first,
        geometries=len(groups),
        measured_lengths=sorted(lengths),
        maximum_regression=maximum_regression,
    )


def reusable_long_gathers(rank, contract, describe):
    """Prove packet-local gather reuse, invalidating on any aliased write.

    Views conservatively alias their entire root allocation. Owner ranges are
    tracked separately: a short-owner variant cannot initialize a large-owner
    variant. This only removes repeated conversion; it never changes KV state
    or carries scratch contents from a previous packet.
    """
    definitions = {b["name"]: b for b in rank["buffers"]}

    def root(name):
        seen = set()
        while "view" in definitions[name]:
            if name in seen:
                raise ValueError("Cyclic buffer view")
            seen.add(name)
            name = definitions[name]["view"]["parent"]
        return name

    def written(step, spec):
        if step["op"] == "kernel":
            writes = {
                name
                for name, operand in zip(step["buffers"], spec["operands"], strict=True)
                if operand["access"] in ("write", "read_write", "scratch")
            }
            if "owner_work" in step:
                mapping = contract["pointers"][step["buffers"][1]]
                writes.update(
                    contract["columns"][mapping[a["column"]]]["buffer"]
                    for a in spec["parameters"]["_owner_access"]
                    if a["write"]
                )
        elif step["op"] in ("embed", "gather", "reduce"):
            writes = {step["destination"]}
        elif step["op"] == "w4":
            writes = {step["buffers"][-1], step["workspace"]}
        elif step["op"] == "lookup":
            writes = {step[k] for k in ("values", "scales", "mask")}
        elif step["op"] in ("fork", "switch", "join", "snapshot"):
            writes = set()
        else:
            return None  # Unknown instructions or row changes invalidate all.
        return {root(name) for name in writes}

    selected = set()
    for packet_index, packet in enumerate(rank["packets"]):
        if packet.get("batch_decode"):
            continue
        live = {}
        for step_index, step in enumerate(packet["body"]):
            spec = describe(step["manifest"]) if step["op"] == "kernel" else None
            key, extent, protected = None, None, None
            if (
                spec is not None
                and spec["op"] == "private_rows.kv_int8_gather_attention"
                and spec["parameters"]["compressed"]
            ):
                p = spec["parameters"]
                if p.get("reuse_long", False):
                    raise ValueError("Reuse analysis requires complete reference gathers")
                mapping = contract["pointers"][step["buffers"][1]]
                if len(mapping) != 8 or p["_owner_access"] != [
                    dict(column=i, write=i >= 6) for i in range(8)
                ]:
                    raise ValueError("Unexpected gather pointer contract")
                columns = [contract["columns"][mapping[i]] for i in (2, 3, 5, 6, 7)]
                input_roots = {root(c["buffer"]) for c in columns[:3]}
                output_roots = {root(c["buffer"]) for c in columns[3:]}
                if input_roots & output_roots or len(output_roots) != 2:
                    raise ValueError("Gather output aliases a source or another output")
                key = json.dumps(
                    [
                        columns,
                        [
                            p[k]
                            for k in ("capacity", "compressed_capacity", "_owner_token_capacity")
                        ],
                        step.get("row_expression"),
                        step.get("fixed_rows"),
                    ],
                    sort_keys=True,
                )
                extent = contract.get("work_ranges", {}).get(
                    step["buffers"][0], dict(min_rows=1, max_rows=p["_owner_token_capacity"])
                )
                lower, upper = extent["min_rows"], extent["max_rows"]
                ranges = live.get(key, (set(), []))[1]
                cursor = lower
                for first, last in sorted(ranges):
                    if first <= cursor:
                        cursor = max(cursor, last + 1)
                if cursor > upper:
                    selected.add((packet_index, step_index))
                protected = input_roots | output_roots
            writes = written(step, spec)
            live = {
                k: value
                for k, value in live.items()
                if writes is not None and (k == key or not value[0] & writes)
            }
            if key is not None:
                ranges = live.get(key, (set(), []))[1]
                live[key] = protected, ranges + [(extent["min_rows"], extent["max_rows"])]
    return selected


def rebuild(
    plan, output, operator, overrides, first, *, reuse_long_kv=False, measured_identity=None
):
    from kernels.model.rows import row_count
    from tools.operators.ascend.private_rows import indirect
    from tools.operators.ascend.geometry import physical_shapes

    model = json.loads(plan.read_text())
    if "reuse_long" in overrides or (
        reuse_long_kv and (operator != "kv_int8.gather_attention" or first != 1)
    ):
        raise ValueError("Use --reuse-long-kv with the complete KV gather owner domain")
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
        reusable = reusable_long_gathers(rank, contract, describe) if reuse_long_kv else None
        for packet_index, packet in enumerate(rank["packets"]):
            if packet.get("batch_decode"):
                continue
            body = []
            for step_index, original in enumerate(packet["body"]):
                if reusable is not None and (packet_index, step_index) not in reusable:
                    body.append(original)
                    continue
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
                    if reuse_long_kv:
                        arguments["reuse_long"] = True
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
                        if measured_identity is not None:
                            compiled = describe(manifest)
                            identity = {k: compiled[k] for k in measured_identity}
                            if identity != measured_identity:
                                raise ValueError(
                                    "Measured source/compiler/target differs from compilation"
                                )
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
        reuse_long_kv=reuse_long_kv,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--parameters", type=json.loads)
    selection.add_argument("--tuning-record", type=Path)
    parser.add_argument("--tuning-parameter")
    parser.add_argument("--maximum-regression", type=float, default=0.1)
    parser.add_argument("--min-owner-rows", type=int, default=1)
    parser.add_argument("--reuse-long-kv", action="store_true")
    args = parser.parse_args()
    selected = None
    measured_identity = None
    if args.tuning_record:
        if not args.tuning_parameter or args.min_owner_rows != 1 or args.reuse_long_kv:
            parser.error(
                "Tuning selection requires one parameter and the complete reference domain"
            )
        module_name, name = args.operator.split(".")
        module = importlib.import_module("tools.operators.ascend." + module_name)
        factory = getattr(module, name)
        defaults = {
            k: p.default
            for k, p in inspect.signature(factory).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        if args.tuning_parameter not in defaults:
            parser.error("Tuning parameter requires a factory default")
        model = json.loads(args.plan.read_text())
        descriptions, geometries = {}, []
        for packet in model["ranks"][0]["packets"]:
            if packet.get("batch_decode"):
                continue
            for step in packet["body"]:
                if step["op"] != "kernel":
                    continue
                path = step["manifest"]
                if path not in descriptions:
                    descriptions[path] = json.loads((args.plan.parent / path).read_text())
                spec = descriptions[path]
                if spec["op"] == "private_rows." + args.operator.replace(".", "_"):
                    geometries.append(spec["parameters"])
        from tools.operators.ascend.validate_private_rows import measurement_source

        reports = json.loads(args.tuning_record.read_text())
        for report in reports:
            if report["op"] != args.operator:
                continue
            path = Path(report["private_manifest"])
            if file_hash(path) != report["measurement_identity"]["manifest_sha256"]:
                raise ValueError("Measured private manifest changed")
            spec = json.loads(path.read_text())
            identity = {k: spec[k] for k in ("source_sha256", "compiler", "target")}
            if measured_identity is not None and identity != measured_identity:
                raise ValueError("Private tuning candidates have different compiler identities")
            measured_identity = identity
        selected = select_schedule(
            reports,
            args.operator,
            args.tuning_parameter,
            defaults,
            geometries,
            file_hash(Path(inspect.getfile(factory))),
            measurement_source(),
            args.maximum_regression,
        )
        args.parameters, args.min_owner_rows = selected["parameters"], selected["first_owner_rows"]
    elif args.tuning_parameter:
        parser.error("--tuning-parameter requires --tuning-record")
    print(
        json.dumps(
            dict(
                rebuild(
                    args.plan,
                    args.output,
                    args.operator,
                    args.parameters,
                    args.min_owner_rows,
                    reuse_long_kv=args.reuse_long_kv,
                    measured_identity=measured_identity,
                ),
                selection=selected,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

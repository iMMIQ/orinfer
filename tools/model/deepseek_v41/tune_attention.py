"""Choose one shared head tile for native selected attention using published AOT controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import ctypes
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.batch import compile_task, DTYPES
from tools.model.deepseek_v41.assembly import publish_package, validate_rows
from tools.model.row_selection import guarded_winner
from tools.model.tuning import read_record, record_path


def attention_groups(model, base, minimum):
    groups, seen = {}, set()
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            if packet.get("batch_decode") or packet["min_rows"] < minimum:
                continue
            for step in packet["body"]:
                name = step.get("manifest")
                if name and name not in seen:
                    seen.add(name)
                    spec = json.loads((base / name).read_text())
                    if spec["op"] == "private_rows.attention_fused_selected_int8":
                        groups[name] = spec
    if not groups:
        raise ValueError("No large-family selected attention in plan")
    return groups


def compile_plan(plan, output, workers, minimum):
    from tools.operators.ascend.attention_fused import selected_int8
    from tools.operators.ascend.geometry import physical_shapes

    model = json.loads(plan.read_text())
    base = output / "source"
    if base.exists():
        if json.loads((base / "model.json").read_text()) != model:
            raise ValueError("Attention tuning input changed")
    else:
        validate_rows(model, plan.parent)
        publish_package(model, plan.parent, base)
    groups, tasks = [], []
    for name, spec in attention_groups(model, base, minimum).items():
        p = spec["parameters"]
        if p.get("softmax_heads", 1) != 1:
            raise ValueError("Require the published one-head baseline")
        arguments = {k: v for k, v in p.items() if not k.startswith("_")}
        function = selected_int8(**arguments)
        tensors = [function.buffer_map[v] for v in function.params if v in function.buffer_map]
        operands = [
            dict(name=t.name, dtype=DTYPES[str(t.dtype)], shape=shape)
            for t, shape in zip(tensors, physical_shapes(function, arguments["rows"]), strict=True)
        ]
        group = dict(baseline=name, operands=operands, candidates={"1": name})
        groups.append(group)
        aliases = list(p["_owner_aliases"])
        metadata = {k: v for k, v in p.items() if k.startswith("_owner_")}
        metadata["_owner_source_sha256"] = file_hash(Path(selected_int8.__code__.co_filename))
        for tile in (4, 8):
            task = dict(
                key=f"{len(groups) - 1}:{tile}",
                module="tools.operators.ascend.attention_fused",
                name="selected_int8",
                arguments=dict(arguments, softmax_heads=tile),
                specs=deepcopy(spec["operands"]),
                capacity=p["_owner_token_capacity"],
                dynamic=dict(rows=1),
                columns={alias.removesuffix("_owner"): i for i, alias in enumerate(aliases)},
                owners=spec["operands"][1]["shape"][0],
                table_columns=spec["operands"][1]["shape"][1] - 1,
                metadata=metadata,
                grid_limit=p["_grid_limit"],
            )
            tasks.append(task)
    rejected = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(compile_task, (task, str(base))): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                key, manifest = future.result()
                group, tile = key.split(":")
                groups[int(group)]["candidates"][tile] = manifest
                print(json.dumps(dict(compiled=key, manifest=manifest)), flush=True)
            except Exception as error:
                rejected.append(dict(candidate=task["key"], reason=str(error)))
    report = dict(minimum=minimum, groups=groups, rejected=rejected)
    atomic_json(output / "compiled.json", report)
    return report


def select_plan(plan, output, device, minimum):
    import torch
    import torch_npu
    from tools.model.deepseek_v41.tune import npu_benchmark

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(device)
    base = output / "source"
    model = json.loads(plan.read_text())
    if json.loads((base / "model.json").read_text()) != model:
        raise ValueError("Attention tuning input changed")
    compiled = json.loads((output / "compiled.json").read_text())
    if compiled["minimum"] != minimum:
        raise ValueError("Attention family boundary changed")
    groups = compiled["groups"]
    configs = sorted(set.intersection(*(set(g["candidates"]) for g in groups)))
    identity = dict(
        kernels={
            name: json.loads((base / name).read_text())
            for g in groups
            for name in g["candidates"].values()
        },
        device=torch.npu.get_device_name(device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        rows=[1, 17, 64, 68, 128, 256],
        configs=configs,
        seed=20261002,
        calls_per_replay=4,
        maximum_regression=0.1,
        helpers={
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path("tools/model/tuning.py"),
                Path("tools/model/row_selection.py"),
                Path("tools/model/deepseek_v41/tune.py"),
            ]
        },
    )

    def valid(record):
        return all(
            file_hash((base / name).parent / spec["library"]["file"]) == spec["library"]["sha256"]
            for name, spec in identity["kernels"].items()
        )

    (output / "records").mkdir(parents=True, exist_ok=True)
    record = record_path(output / "records", identity)
    result = read_record(record, identity, valid)
    if result is None:
        dtype = dict(
            i8=torch.int8,
            i32=torch.int32,
            bf16=torch.bfloat16,
            f16=torch.float16,
            f32=torch.float32,
        )
        records, libraries, rejected = [], [], []
        legal_configs = set(configs)
        for group in groups:
            spec = identity["kernels"][group["baseline"]]
            p = spec["parameters"]
            capacity = p["_owner_token_capacity"]
            if capacity != 256 or p["heads"] != 64:
                raise ValueError("Tuning fixture requires the TP8 H64/256-row query family")
            owned = [a.removesuffix("_owner") for a in p["_owner_aliases"]]
            writes = set(p["_owner_writes"])
            inputs, states, calls = {}, {}, {}
            for operand in group["operands"]:
                name, shape, kind = operand["name"], operand["shape"], dtype[operand["dtype"]]
                if name in writes:
                    continue
                if name.endswith("indices"):
                    bound = p["capacity"] if name == "swa_indices" else p["compressed_capacity"]
                    data = torch.randint(0, bound, shape, dtype=kind, device="npu")
                    data[:, ::19] = -1
                    data[:, 3] = bound
                    data[6::7] = -1
                elif kind == torch.int8:
                    data = torch.randint(-16, 17, shape, dtype=kind, device="npu")
                elif name.endswith("scales"):
                    data = torch.full(shape, 0.125, dtype=kind, device="npu")
                else:
                    data = (torch.randn(shape, dtype=torch.float32, device="npu") * 0.5).to(kind)
                inputs[name] = data
            active = torch.ones(1, dtype=torch.int32, device="npu")
            work = torch.empty((8192, 2), dtype=torch.int32, device="npu")
            for config in sorted(legal_configs):
                values = dict(inputs)
                for operand in group["operands"]:
                    if operand["name"] in writes:
                        values[operand["name"]] = torch.full(
                            operand["shape"], 17, dtype=dtype[operand["dtype"]], device="npu"
                        )
                if p.get("reuse_long"):
                    # These are the existing retained long-KV inputs, independent of softmax.
                    values["kv_workspace"][:, 128:].fill_(0.25)
                    values["mask_workspace"][:, 128:].fill_(0)
                table = torch.zeros(spec["operands"][1]["shape"], dtype=torch.int64, device="npu")
                values["table"] = table
                states[config] = values
                path = base / group["candidates"][config]
                desc = identity["kernels"][group["candidates"][config]]
                library = ctypes.CDLL(str((path.parent / desc["library"]["file"]).resolve()))
                libraries.append(library)
                entry = library.orinfer_ascend_kernel_v2
                entry.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                ]
                entry.restype = ctypes.c_int32
                pointers = (ctypes.c_void_p * 4)(
                    work.data_ptr(), table.data_ptr(), active.data_ptr(), values["sink"].data_ptr()
                )
                calls[config] = (entry, pointers)

            for rows in identity["rows"]:
                for values in states.values():
                    addresses = [rows] + [values[name].data_ptr() for name in owned]
                    values["table"][0].copy_(torch.tensor(addresses, dtype=torch.int64))
                    values["output"].fill_(17)
                    values["probability_workspace"].fill_(17)
                items = [[0, i] for i in range(min(rows, 64))]
                if len(items) % 2:
                    items.append([-1, 0])
                work[: len(items)].copy_(torch.tensor(items, dtype=torch.int32))
                length = len(items) // 2

                def launch(config):
                    entry, pointers = calls[config]
                    for _ in range(identity["calls_per_replay"]):
                        status = entry(
                            pointers, 4, length, torch.npu.current_stream()._as_parameter_
                        )
                        if status:
                            raise RuntimeError(f"Selected attention launch failed: {status}")

                def check():
                    torch.npu.synchronize()
                    invalid = {}
                    for config, values in states.items():
                        for name in ["output", "probability_workspace"]:
                            try:
                                torch.testing.assert_close(
                                    values[name][:rows],
                                    states["1"][name][:rows],
                                    rtol=0,
                                    atol=0,
                                    msg=f"Attention head tile {config}: {name}",
                                )
                            except AssertionError as error:
                                if config == "1":
                                    raise
                                invalid[config] = dict(
                                    geometry=group["baseline"],
                                    rows=rows,
                                    port=name,
                                    reason=str(error),
                                )
                                names = [
                                    "query",
                                    "kv_workspace",
                                    "mask_workspace",
                                    "score_workspace",
                                    "probability_workspace",
                                    "output_workspace",
                                    "output",
                                ]
                                torch.save(
                                    {
                                        c: {n: v[n][:rows].cpu() for n in names}
                                        | {"sink": v["sink"].cpu()}
                                        for c, v in states.items()
                                    },
                                    output / f"mismatch-{config}.pt",
                                )
                                break
                            assert bool((values[name][rows:] == 17).all()), (
                                f"Attention wrote {name} tail"
                            )

                    for config, reason in invalid.items():
                        rejected.append(dict(candidate=config, **reason))
                        legal_configs.remove(config)
                        graphs.pop(config).reset()
                        states.pop(config)
                        calls.pop(config)
                        times.pop(config)
                        for prior in records:
                            prior["times"].pop(config, None)
                        print(json.dumps(dict(rejected=rejected[-1])), flush=True)

                times, graphs = {}, {}
                for config in sorted(legal_configs):
                    timing, graph = npu_benchmark(lambda: launch(config), 9, 2)
                    times[config] = timing["median_ms"] / identity["calls_per_replay"]
                    graphs[config] = graph
                check()
                saved = states["1"]["output"][:rows].clone()
                inputs["query"].neg_()
                inputs["sink"].add_(0.125)
                inputs["swa_scales"].mul_(0.5)
                for graph in graphs.values():
                    graph.replay()
                check()
                assert not torch.equal(saved, states["1"]["output"][:rows]), (
                    "Graph ignored changed attention inputs"
                )
                for graph in graphs.values():
                    graph.reset()
                row = dict(
                    geometry=p,
                    rows=rows,
                    times=times,
                    exact_probability=True,
                    exact_output=True,
                    changed_graph=True,
                    tail_unchanged=True,
                )
                records.append(row)
                atomic_json(
                    output / "validation-progress.json",
                    dict(identity=identity, measurements=records, rejected=rejected),
                )
                print(json.dumps(row), flush=True)
            calls = states = values = graphs = inputs = None
            torch.npu.synchronize()
            torch.npu.empty_cache()
        selected = guarded_winner(records, reference="1", maximum_regression=0.1)
        result = dict(identity=identity, measurements=records, selected=selected, rejected=rejected)
        atomic_json(record, result)
    selected = result["selected"]
    replacements = {g["baseline"]: g["candidates"][selected] for g in groups}
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            if packet.get("batch_decode") or packet["min_rows"] < minimum:
                continue
            for step in packet["body"]:
                if step.get("manifest") in replacements:
                    step["manifest"] = replacements[step["manifest"]]
    validate_rows(model, base)
    assets = publish_package(model, base, output / "package")
    report = dict(
        selected=selected,
        geometries=len(groups),
        cases=len(result["measurements"]),
        assets=assets,
        record=str(record),
        model=str(output / "package/model.json"),
        batch_contract_equal=True,
        rejected=compiled["rejected"] + result["rejected"],
    )
    atomic_json(output / "attention-tuned.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--compile-workers", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=512)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        report = compile_plan(args.plan, args.output, args.compile_workers, args.min_rows)
    else:
        report = select_plan(args.plan, args.output, args.device, args.min_rows)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Validate packed index scores, TopK, tails and changed-input Graph replay."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.row_contract import resolve


def validate_private(args, original, sharded, descriptor, launch, allocate):
    """Compare published private chains with distinct owner keys on every shard."""
    import torch

    kinds = {
        "int8": torch.int8,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }

    def chains(packet):
        result = {}
        for i, step in enumerate(packet["body"]):
            if (
                step["op"] != "kernel"
                or descriptor(step)["op"] != "private_rows.indexer_cube_causal_logits"
            ):
                continue
            length = descriptor(step)["parameters"]["length"]
            if length in result:
                continue
            start = i
            while (
                start
                and packet["body"][start - 1]["op"] == "kernel"
                and descriptor(packet["body"][start - 1])["op"] == "memory.partition_rows"
            ):
                start -= 1
            end = i + 1
            while (
                packet["body"][end]["op"] == "kernel"
                and descriptor(packet["body"][end])["op"] != "topk.finalize"
            ):
                end += 1
            assert descriptor(packet["body"][end])["op"] == "topk.finalize"
            result[length] = (
                step,
                packet["body"][start : end + 1],
                packet["body"][end + 2] if start != i else None,
            )
        return result

    before = chains(next(p for p in original["ranks"][0]["packets"] if not p["batch_decode"]))
    after = [
        chains(next(p for p in rank["packets"] if not p["batch_decode"]))
        for rank in sharded["ranks"]
    ]
    assert before and all(set(c) == set(before) for c in after), "Incomplete indexer geometries"
    reports = []
    for length, (reference_index, reference_steps, _) in before.items():
        capacity = descriptor(reference_index)["parameters"]["_owner_token_capacity"]
        ref_columns = [
            original["batch"]["columns"][c]
            for c in original["batch"]["pointers"][reference_index["buffers"][1]]
        ]
        alias_types = list(descriptor(reference_index)["parameters"]["_owner_aliases"].values())
        reference_data = allocate(reference_steps)
        for column, kind in zip(ref_columns, alias_types, strict=True):
            if column["row_bytes"] is not None:
                reference_data[column["buffer"]] = torch.empty(
                    (
                        capacity,
                        column["row_bytes"] // torch.empty((), dtype=kinds[kind]).element_size(),
                    ),
                    dtype=kinds[kind],
                    device="npu",
                )
        private_keys = [
            (
                torch.randint(-127, 128, (length, 128), dtype=torch.int8).npu(),
                (torch.rand(length) * 0.01 + 0.001).half().npu(),
            )
            for _ in range(3)
        ]
        inputs = [ref_columns[i]["buffer"] for i in (0, 1, 4, 5)]
        reference_data[inputs[0]].copy_(
            torch.randint(-127, 128, reference_data[inputs[0]].shape, dtype=torch.int8)
        )
        reference_data[inputs[1]].copy_(
            (torch.rand(reference_data[inputs[1]].shape) * 0.01 + 0.001).half()
        )
        reference_data[inputs[2]].copy_(torch.randn(reference_data[inputs[2]].shape).bfloat16())
        # Zero queries produce exact ties and exercise deterministic TopK ordering.
        reference_data[inputs[0]][::13].zero_()

        def bind(model, index_step, data, lengths, order, rank, private_keys=private_keys):
            contract = model["batch"]
            pointer = index_step["buffers"][1]
            stride = contract.get("partitioned_tables", {}).get(pointer, 1)
            table = torch.zeros((contract["owners"], 9), dtype=torch.int64)
            work = []
            start = 0
            for owner, (q, key_owner) in enumerate(zip(lengths, order, strict=True)):
                first = len(range(rank, start, stride)) if stride > 1 else start
                end = len(range(rank, start + q, stride)) if stride > 1 else start + q
                count = end - first
                addresses = []
                for slot, c in enumerate(contract["pointers"][pointer]):
                    column = contract["columns"][c]
                    tensor = (
                        private_keys[key_owner][slot - 2]
                        if slot in (2, 3)
                        else data[column["buffer"]]
                    )
                    addresses.append(
                        tensor.data_ptr()
                        + (first * column["row_bytes"] if column["row_bytes"] is not None else 0)
                    )
                table[owner] = torch.tensor([count] + addresses, dtype=torch.int64)
                if count:
                    work.extend(
                        [
                            [owner, b]
                            for b in range(min(resolve(index_step["owner_work"], count), 64))
                        ]
                    )
                start += q
            if not work:
                work = [[-1, 0], [-1, 0]]
            if len(work) % 2:
                work.append([-1, 0])
            data[pointer].copy_(table)
            data[index_step["buffers"][0]][: len(work)].copy_(torch.tensor(work, dtype=torch.int32))
            data[index_step["buffers"][2]].fill_(1)
            return dict(index_step, fixed_rows=len(work) // 2)

        for rows in args.rows or (1, 2, 7, 8, 9, 17, 511, 512, 513, 2047, 2048):
            if rows > capacity:
                continue
            lengths = (
                [1]
                if rows == 1
                else ([1, 1] if rows == 2 else [1, min(7, rows - 2), rows - 1 - min(7, rows - 2)])
            )
            order = list(range(len(lengths)))
            base = length - rows if rows == 9 else 1024
            causal = torch.cat(
                [
                    torch.arange(q, dtype=torch.int32) + base + owner * 17
                    for owner, q in enumerate(lengths)
                ]
            ).clamp_max(length)
            reference_data[inputs[3]][:rows, 0].copy_(causal)
            max_key = int(causal.max())
            query = reference_data[inputs[0]].clone()
            expected = []
            for phase in range(3):
                reference_data[inputs[0]].copy_(query if phase == 0 else -query)
                active_order = order if phase < 2 else list(reversed(order))
                reference_data[ref_columns[6]["buffer"]].fill_(17)
                bound = bind(original, reference_index, reference_data, lengths, active_order, 0)
                for step in reference_steps:
                    launch(bound if step is reference_index else step, reference_data, rows)
                torch.npu.synchronize()
                expected.append(
                    (
                        reference_data[ref_columns[6]["buffer"]][:rows, :max_key].cpu(),
                        reference_data[reference_steps[-1]["buffers"][1]][:rows].cpu(),
                        reference_data[reference_steps[-1]["buffers"][2]][:rows].cpu(),
                    )
                )
            local_rows = (rows + len(after) - 1) // len(after)
            assembled = [[] for _ in range(3)]
            for rank, chain in enumerate(after):
                index_step, steps, interleave = chain[length]
                columns = [
                    sharded["batch"]["columns"][c]
                    for c in sharded["batch"]["pointers"][index_step["buffers"][1]]
                ]
                count = descriptor(index_step)["parameters"]["_owner_token_capacity"]
                data = allocate(steps, reference_data)
                for column, kind in zip(columns, alias_types, strict=True):
                    if column["row_bytes"] is not None and column["buffer"] not in data:
                        data[column["buffer"]] = torch.empty(
                            (
                                count,
                                column["row_bytes"]
                                // torch.empty((), dtype=kinds[kind]).element_size(),
                            ),
                            dtype=kinds[kind],
                            device="npu",
                        )
                score = data[columns[6]["buffer"]].view(count, length)
                bound = bind(sharded, index_step, data, lengths, order, rank)
                score.fill_(17)
                data[index_step["buffers"][2]].zero_()
                launch(bound, data, rows)
                torch.npu.synchronize()
                assert bool((score == 17).all()), "Disabled private partition wrote scores"

                def run(data=data, steps=steps, bound=bound, index_step=index_step, rows=rows):
                    for step in steps:
                        launch(bound if step is index_step else step, data, rows)

                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    run()
                ids = rank + torch.arange(local_rows) * len(after)
                valid = ids < rows
                for phase in range(3):
                    data[inputs[0]].copy_(query if phase == 0 else -query)
                    active_order = order if phase < 2 else list(reversed(order))
                    bound = bind(sharded, index_step, data, lengths, active_order, rank)
                    score.fill_(17)
                    graph.replay()
                    torch.npu.synchronize()
                    actual = score[:local_rows, :max_key].cpu()
                    torch.testing.assert_close(
                        actual[valid], expected[phase][0][ids[valid]], atol=0, rtol=0
                    )
                    selection = data[steps[-1]["buffers"][1]][:local_rows].cpu()
                    values = data[steps[-1]["buffers"][2]][:local_rows].cpu()
                    torch.testing.assert_close(
                        selection[valid], expected[phase][1][ids[valid]], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        values[valid], expected[phase][2][ids[valid]], atol=0, rtol=0
                    )
                    assert bool((selection[~valid] == -1).all()), "Noncanonical padded TopK"
                    assert bool((score[local_rows:] == 17).all()), (
                        "Private score wrote its capacity tail"
                    )
                    assembled[phase].append(selection)
                del graph, data, run
            interleave = after[0][length][2]
            data = allocate([interleave])
            for phase in range(3):
                gathered = torch.cat(assembled[phase])
                data[interleave["buffers"][0]][: len(gathered)].copy_(gathered)
                data[interleave["buffers"][1]].fill_(-7)
                launch(interleave, data, rows)
                torch.npu.synchronize()
                result = data[interleave["buffers"][1]].cpu()
                torch.testing.assert_close(result[:rows], expected[phase][1], atol=0, rtol=0)
                assert bool((result[rows:] == -7).all()), "Interleave wrote the global tail"
            reports.append(
                dict(
                    length=length,
                    rows=rows,
                    owner_lengths=lengths,
                    ranks=len(after),
                    exact_scores=True,
                    exact_topk=True,
                    exact_values=True,
                    graph_replay=True,
                    changed_query=True,
                    reversed_private_keys=True,
                    capture_guard=True,
                    padded_rows=True,
                )
            )
            print(reports[-1], flush=True)
            del data, expected, query
        del reference_data, private_keys
        torch.npu.empty_cache()
    atomic_json(args.output / "validation.json", reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--rows", type=int, nargs="+")
    args = parser.parse_args()
    if args.rows and not 1 <= min(args.rows) <= max(args.rows) <= 2048:
        parser.error("rows must be in1..2048")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.model.deepseek_v41.validate import reference

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    original = json.loads(args.baseline.read_text())
    sharded = json.loads(args.plan.read_text())
    base = args.plan.parent
    dtype = {
        "i8": torch.int8,
        "i32": torch.int32,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
        "u64": torch.int64,
    }
    descriptors, entries, paths = {}, {}, {}

    def descriptor(step):
        name = step["manifest"]
        if name not in descriptors:
            paths[name] = base / name
            if not paths[name].exists():
                paths[name] = args.baseline.parent / name
            descriptors[name] = json.loads(paths[name].read_text())
        return descriptors[name]

    def launch(step, tensors, rows):
        name = step["manifest"]
        if name not in entries:
            descriptor(step)
            library = ctypes.CDLL(str(paths[name].parent / "kernel.so"))
            entry = library.orinfer_ascend_kernel_v2
            entry.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            entry.restype = ctypes.c_int32
            entries[name] = (library, entry)
        pointers = (ctypes.c_void_p * len(step["buffers"]))(
            *[tensors[n].data_ptr() for n in step["buffers"]]
        )
        status = entries[name][1](
            pointers,
            len(pointers),
            resolve(step["row_expression"], rows)
            if "row_expression" in step
            else step.get("fixed_rows", rows),
            torch.npu.current_stream()._as_parameter_,
        )
        if status:
            raise RuntimeError(f"{name}: launch failed with {status}")

    def allocate(steps, existing=None):
        tensors = {} if existing is None else dict(existing)
        for step in steps:
            for name, spec in zip(step["buffers"], descriptor(step)["operands"], strict=True):
                if name not in tensors:
                    tensors[name] = torch.empty(
                        spec["shape"], dtype=dtype[spec["dtype"]], device="npu"
                    )
        return tensors

    def chains(packet):
        found = {}
        for i, step in enumerate(packet["body"]):
            if step["op"] != "kernel" or descriptor(step)["op"] != "indexer_cube.causal_logits":
                continue
            length = descriptor(step)["parameters"]["length"]
            if length in found:
                continue
            start = i
            while (
                start > 0
                and packet["body"][start - 1]["op"] == "kernel"
                and descriptor(packet["body"][start - 1])["op"] == "memory.partition_rows"
            ):
                start -= 1
            end = i + 1
            while descriptor(packet["body"][end])["op"] != "topk.finalize":
                end += 1
            found[length] = packet["body"][start : end + 1]
        return found

    if original.get("batch"):
        validate_private(args, original, sharded, descriptor, launch, allocate)
        return

    reports = []
    baseline_chains = chains(original["ranks"][0]["packets"][1])
    shard_chains = [chains(rank["packets"][-1]) for rank in sharded["ranks"]]
    for length, baseline in baseline_chains.items():
        capacity = descriptor(baseline[0])["parameters"]["rows"]
        count = capacity // 8
        shared = allocate(baseline)
        index_step = baseline[0]
        slots = index_step["buffers"]
        host = [
            torch.randint(-127, 128, (capacity, 32, 128), dtype=torch.int8),
            (torch.rand(capacity, 32) * 0.01).half(),
            torch.randint(-127, 128, (length, 128), dtype=torch.int8),
            (torch.rand(length) * 0.01).half(),
            torch.randn(capacity, 32).bfloat16(),
            torch.empty(capacity, length, dtype=torch.int32),
        ]
        for name, value in zip(slots[:5], host[:5], strict=True):
            shared[name].copy_(value.reshape(shared[name].shape))
        length_name = baseline[1]["buffers"][1]
        shared[length_name].zero_()
        for rows in args.rows or (1, 2, 17, 31, 32, 511, 512, 513, 2047, 2048):
            if rows > capacity:
                continue
            offset = 0 if rows % 2 else length - capacity
            lengths = (torch.arange(capacity) + offset + 1).clamp(0, length)
            host[5] = torch.where(
                torch.arange(length)[None, :] < lengths[:, None], torch.arange(length)[None, :], -1
            ).int()
            shared[slots[5]].copy_(host[5])
            shared[length_name].copy_(lengths.int())
            for step in baseline:
                launch(step, shared, rows)
            torch.npu.synchronize()
            original_scores = shared[slots[6]][:rows].cpu()
            selected_name = baseline[-1]["buffers"][1]
            expected_selected = shared[selected_name][:rows].cpu()
            live = [t[:2] if j in (0, 1, 4, 5) else t for j, t in enumerate(host)]
            expected = reference("index_quantized", live, {})[0]
            torch.testing.assert_close(
                original_scores[: min(rows, 2)], expected[: min(rows, 2)], atol=0, rtol=0
            )
            dynamic = "row_expression" in next(
                s
                for s in shard_chains[0][length]
                if descriptor(s)["op"] == "indexer_cube.causal_logits"
            )
            local_rows = (rows + 7) // 8 if dynamic else count
            assembled = torch.empty(local_rows * 8, 512, dtype=torch.int32)
            for rank in range(8):
                steps = shard_chains[rank][length]
                tensors = allocate(steps, shared)
                local_index = next(
                    s for s in steps if descriptor(s)["op"] == "indexer_cube.causal_logits"
                )
                global_ids = rank + torch.arange(local_rows) * 8
                valid = global_ids < rows
                for step in steps:
                    for name, port in zip(
                        step["buffers"], descriptor(step)["operands"], strict=True
                    ):
                        if port["access"] != "read":
                            tensors[name].fill_(17)

                def run():
                    for step in steps:
                        launch(step, tensors, rows)

                run()
                torch.npu.synchronize()
                for step in steps:
                    m = descriptor(step)
                    if m["op"] == "memory.partition_rows":
                        width = m["parameters"]["width"]
                        source, output = step["buffers"]
                        actual = tensors[output].cpu().reshape(count, width)[:local_rows]
                        expected_pack = (
                            tensors[source]
                            .cpu()
                            .reshape(capacity, width)[global_ids.clamp_max(rows - 1)]
                            .clone()
                        )
                        if not m["parameters"]["repeat_last"]:
                            expected_pack[~valid] = 0
                        torch.testing.assert_close(actual, expected_pack, atol=0, rtol=0)
                        if dynamic:
                            assert bool((tensors[output][local_rows:] == 17).all()), (
                                "Partition wrote its capacity tail"
                            )
                actual = tensors[local_index["buffers"][6]].cpu()[:local_rows]
                torch.testing.assert_close(
                    actual[valid], original_scores[global_ids[valid]], atol=0, rtol=0
                )
                selection = tensors[steps[-1]["buffers"][1]].cpu()[:local_rows]
                if dynamic:
                    assert bool((tensors[steps[-1]["buffers"][1]][local_rows:] == 17).all()), (
                        "TopK wrote its capacity tail"
                    )
                torch.testing.assert_close(
                    selection[valid], expected_selected[global_ids[valid]], atol=0, rtol=0
                )
                assert bool((selection[~valid] == -1).all()), "Padded TopK rows must be canonical"
                assembled[rank::8] = selection
                if rows in (17, 513, 2048) and rank in (0, 7):
                    _, graph = npu_benchmark(run, 1, 1)
                    query_name = slots[0]
                    tensors[query_name].neg_()
                    graph.replay()
                    torch.npu.synchronize()
                    changed = [t.clone() for t in host]
                    changed[0].neg_()
                    sample_ids = global_ids[valid][:2]
                    if sample_ids.numel():
                        live = [
                            t[sample_ids] if j in (0, 1, 4, 5) else t for j, t in enumerate(changed)
                        ]
                        expected = reference("index_quantized", live, {})[0]
                        torch.testing.assert_close(
                            tensors[local_index["buffers"][6]].cpu()[: len(sample_ids)],
                            expected,
                            atol=0,
                            rtol=0,
                        )
                    tensors[query_name].neg_()
                    del graph
                del tensors
            torch.testing.assert_close(assembled[:rows], expected_selected, atol=0, rtol=0)
            gather_step_index = (
                sharded["ranks"][0]["packets"][-1]["body"].index(shard_chains[0][length][-1]) + 2
            )
            interleave = sharded["ranks"][0]["packets"][-1]["body"][gather_step_index]
            assert descriptor(interleave)["op"] == "memory.interleave_rows"
            tensors = allocate([interleave])
            gathered = torch.cat([assembled[r::8] for r in range(8)])
            tensors[interleave["buffers"][0]][: len(gathered)].copy_(gathered)
            tensors[interleave["buffers"][1]].fill_(-7)
            launch(interleave, tensors, rows)
            torch.npu.synchronize()
            result = tensors[interleave["buffers"][1]].cpu()
            torch.testing.assert_close(result[:rows], expected_selected, atol=0, rtol=0)
            assert bool((result[rows:] == -7).all()), "Interleave wrote outside actual rows"
            if rows in (17, 513, 2048):
                _, graph = npu_benchmark(lambda: launch(interleave, tensors, rows), 1, 1)
                tensors[interleave["buffers"][0]][: len(gathered)].copy_(gathered.flip(0))
                graph.replay()
                torch.npu.synchronize()
                shuffled = (
                    gathered.flip(0)
                    .reshape(8, local_rows, 512)
                    .permute(1, 0, 2)
                    .reshape(local_rows * 8, 512)
                )
                torch.testing.assert_close(
                    tensors[interleave["buffers"][1]].cpu()[:rows], shuffled[:rows], atol=0, rtol=0
                )
                del graph
            reports.append(
                dict(
                    length=length,
                    rows=rows,
                    ranks=8,
                    exact_scores=True,
                    exact_topk=True,
                    padded_rows=True,
                    changed_input_graph=rows in (17, 513, 2048),
                )
            )
            print(reports[-1], flush=True)
        del shared
        torch.npu.empty_cache()
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()

"""Validate packed index scores, TopK, tails and changed-input Graph replay."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.row_contract import resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
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
    }
    descriptors, entries = {}, {}

    def descriptor(step):
        name = step["manifest"]
        if name not in descriptors:
            descriptors[name] = json.loads((base / name).read_text())
        return descriptors[name]

    def launch(step, tensors, rows):
        name = step["manifest"]
        if name not in entries:
            library = ctypes.CDLL(str((base / name).parent / "kernel.so"))
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
        for rows in (1, 2, 17, 31, 32, 511, 512, 513, 2047, 2048):
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

"""Check exact Engram sequence repartition, padding and changed-input Graphs."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    libraries, metadata = [], []

    def stages(plan, local):
        model = json.loads(plan.read_text())
        packet = model["ranks"][0]["packets"][-1]
        result = {}
        for step in packet["body"]:
            if step["op"] != "kernel":
                continue
            path = plan.parent / step["manifest"]
            d = json.loads(path.read_text())
            if d["op"] not in [
                "memory.gather_columns",
                "model_ops.split",
                "engram.hadamard32",
                "engram.gate",
            ]:
                continue
            if d["parameters"].get("_row_capacity") != (256 if local else 2048):
                continue
            if not any(".engram." in n for n in step["buffers"]):
                continue
            if d["op"] in result:
                continue
            binary = path.parent / d["library"]["file"]
            assert file_hash(binary) == d["library"]["sha256"]
            lib = ctypes.CDLL(str(binary.resolve()))
            libraries.append(lib)
            fn = lib.orinfer_ascend_kernel_v2
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int32
            result[d["op"]] = fn
            metadata.append(dict(path=str(path), sha256=file_hash(path)))
        assert len(result) == 4, result.keys()
        return result

    old, new = stages(args.baseline, False), stages(args.plan, True)

    def call(entries, name, tensors, rows):
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
        status = entries[name](
            pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_
        )
        if status:
            raise RuntimeError(f"{name}: {status}")

    cap, width = 2048, 5120
    hidden = torch.randn(cap, 4, width, device="npu").bfloat16()
    shards = torch.randn(8, cap, 3200, device="npu").bfloat16()
    mask = torch.ones(cap, dtype=torch.int8, device="npu")
    mask[::7] = 0
    channel = torch.randn(4, width, device="npu") * 0.1
    rotation = torch.tensor(
        [
            [(-1.0 if (i & j).bit_count() % 2 else 1.0) / (32**0.5) for j in range(32)]
            for i in range(32)
        ],
        device="npu",
    ).flatten()

    def allocate(rows):
        return dict(
            received=torch.empty(8 * rows * 3200, dtype=torch.bfloat16, device="npu"),
            projected=torch.empty(rows, 25600, dtype=torch.bfloat16, device="npu"),
            key=torch.empty(rows, 4, width, dtype=torch.bfloat16, device="npu"),
            value=torch.empty(rows, width, dtype=torch.bfloat16, device="npu"),
            original=torch.empty(rows, 4, width, device="npu"),
            output=torch.empty(rows, 4, width, dtype=torch.bfloat16, device="npu"),
        )

    a, b = allocate(cap), allocate(cap // 8)
    local_hidden = torch.empty(cap // 8, 4, width, dtype=torch.bfloat16, device="npu")
    local_mask = torch.empty(cap // 8, dtype=torch.int8, device="npu")

    def pipeline(entries, buf, h, m, n):
        call(entries, "memory.gather_columns", [buf["received"], buf["projected"]], n)
        call(entries, "model_ops.split", [buf["projected"], buf["key"], buf["value"]], n)
        call(entries, "engram.hadamard32", [h, rotation, buf["original"]], n)
        call(
            entries,
            "engram.gate",
            [h, buf["original"], buf["key"], buf["value"], channel, m, buf["output"]],
            n,
        )

    reports = []
    for rows in [513, 2048]:
        count = (rows + 7) // 8
        for rank in [0, 7]:
            first = rank * count

            def upload():
                packed = shards[:, :rows, :].contiguous().flatten()
                a["received"][: packed.numel()].copy_(packed)
                # Simulate the pure-copy AlltoAllV. Its source padding is zero.
                transfer = shards[:, first : first + count, :].clone()
                if first + count > rows:
                    transfer[:, max(0, rows - first) :, :] = 0
                b["received"][: transfer.numel()].copy_(transfer.flatten())
                local_hidden[:count].copy_(hidden[first : first + count])
                local_mask[:count].copy_(mask[first : first + count])

            upload()
            pipeline(old, a, hidden, mask, rows)
            pipeline(new, b, local_hidden, local_mask, count)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                pipeline(old, a, hidden, mask, rows)
                pipeline(new, b, local_hidden, local_mask, count)
            for phase in range(2):
                hidden.neg_()
                shards.neg_()
                upload()
                graph.replay()
                torch.npu.synchronize()
                live = min(count, max(0, rows - first))
                for field in ["projected", "key", "value", "original", "output"]:
                    expected = a[field][first : first + live].cpu()
                    actual = b[field][:live].cpu()
                    assert torch.equal(actual, expected), (rows, rank, phase, field)
                reports.append(
                    dict(rows=rows, rank=rank, phase=phase, exact=True, changed_graph=True)
                )
                print(reports[-1], flush=True)
            graph.reset()
    atomic_json(
        args.output,
        dict(
            manifests=metadata,
            validation=reports,
            scope="Existing math and new sequence exchange layout; native TP8 validates actual HCCL transport.",
        ),
    )


if __name__ == "__main__":
    main()

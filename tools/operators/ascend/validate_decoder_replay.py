"""Validate compiled decoder row compaction, boundaries and masked softmax."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    args = p.parse_args()
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    model = json.loads(args.plan.read_text())
    kernels = {}
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"]:
            if step["op"] != "kernel" or step["manifest"] in kernels:
                continue
            path = args.plan.parent / step["manifest"]
            spec = json.loads(path.read_text())
            if (
                spec["op"].startswith("decoder_replay.")
                or spec["op"] == "attention_prefill.replay_softmax"
            ):
                kernels[step["manifest"]] = (path, spec)
    if not kernels:
        raise ValueError("No replay kernels were selected")
    kinds = dict(bf16=torch.bfloat16, f32=torch.float32, i32=torch.int32)
    reports = []
    for path, spec in kernels.values():
        parameters = spec["parameters"]
        capacity = parameters["_row_capacity"]
        tensors = [
            torch.empty(o["shape"], dtype=kinds[o["dtype"]], device="npu") for o in spec["operands"]
        ]
        library = ctypes.CDLL(str(path.parent / "kernel.so"))
        fn = library.orinfer_ascend_kernel_v2
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])

        def launch(rows):
            status = fn(pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"Replay launch failed: {status}")

        name = spec["op"]
        if name == "decoder_replay.compact_suffix":
            keep, width = parameters["keep"], parameters["width"]
            host = (
                (torch.arange(capacity * width, dtype=torch.int32) % 251 - 125)
                .reshape(capacity, width)
                .to(tensors[0].dtype)
            )
            tensors[0].copy_(host.npu())
            samples = sorted({1, keep - 1, keep, min(capacity, keep + 1), capacity})
            for rows in samples:
                count = min(rows, keep)
                tensors[1].fill_(17)
                launch(rows)
                torch.npu.synchronize()
                expected = torch.full((keep, width), 17, dtype=host.dtype)
                expected[:count] = host[rows - count : rows]
                assert torch.equal(tensors[1].cpu(), expected), (name, rows)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    launch(rows)
                tensors[0].neg_()
                tensors[1].fill_(17)
                graph.replay()
                torch.npu.synchronize()
                expected[:count].neg_()
                assert torch.equal(tensors[1].cpu(), expected), (name, rows, "Graph")
                tensors[0].copy_(host.npu())
                del graph
            reports.append(dict(op=name, parameters=parameters, rows=samples, changed_graph=True))
        elif name == "decoder_replay.replay_start":
            keep = parameters["keep"]
            positions = torch.arange(capacity, dtype=torch.int32) + 129
            tensors[0].copy_(positions.npu())
            for rows in [1, keep, min(capacity, keep + 1), capacity]:
                launch(rows)
                torch.npu.synchronize()
                assert int(tensors[1].cpu()[0]) == (
                    int(positions[rows - keep]) if rows > keep else 0
                )
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    launch(rows)
                tensors[0].add_(31)
                graph.replay()
                torch.npu.synchronize()
                assert int(tensors[1].cpu()[0]) == (
                    int(positions[rows - keep]) + 31 if rows > keep else 0
                )
                tensors[0].copy_(positions.npu())
                del graph
            reports.append(dict(op=name, parameters=parameters, changed_graph=True))
        else:
            c, bucket = parameters["capacity"], parameters["bucket"]
            active = [2048, 4096, c][bucket]
            first = [1300, 3000, min(8000, c - capacity)][bucket]
            positions = torch.arange(capacity, dtype=torch.int32) + first
            scores = torch.randn(tensors[0].shape) * 2
            selected = torch.zeros((capacity, c), dtype=torch.int32)
            for row in range(capacity):
                selected[row, : int(positions[row]) + 1 : 3] = 1
            sink = torch.linspace(-3, 2, 8)
            for tensor, host in zip(tensors[:4], [scores, positions, selected, sink], strict=True):
                tensor.copy_(host.npu())
            tensors[4].fill_(17)
            tensors[5].fill_(first)
            launch(capacity)
            torch.npu.synchronize()

            def check(lower):
                got = tensors[4].cpu().float().numpy()
                for row in sorted({0, capacity // 2, capacity - 1}):
                    position = int(positions[row])
                    start = max(0, (int(positions[row // 16 * 16]) - 127) // 128) * 128
                    short = (
                        scores[row * 8 : (row + 1) * 8, start : start + 384].double().numpy()
                        * 512**-0.5
                    )
                    mask = (np.arange(start, start + 384) < max(position - 127, lower)) | (
                        np.arange(start, start + 384) > position
                    )
                    short[:, mask] = -np.inf
                    long = (
                        scores[row * 8 : (row + 1) * 8, c : c + active].double().numpy() * 512**-0.5
                    )
                    long[:, selected[row, :active].numpy() == 0] = -np.inf
                    full = np.concatenate([short, long, sink.double().numpy()[:, None]], axis=1)
                    maximum = full.max(-1, keepdims=True)
                    e = np.exp(full - maximum)
                    expected = e / e.sum(-1, keepdims=True)
                    np.testing.assert_allclose(
                        got[row * 8 : (row + 1) * 8, :384], expected[:, :384], atol=1e-5, rtol=0.008
                    )
                    np.testing.assert_allclose(
                        got[row * 8 : (row + 1) * 8, 384 : 384 + active],
                        expected[:, 384:-1],
                        atol=1e-5,
                        rtol=0.008,
                    )

            check(first)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                launch(capacity)
            tensors[5].add_(17)
            graph.replay()
            torch.npu.synchronize()
            check(first + 17)
            reports.append(dict(op=name, parameters=parameters, changed_graph=True, fp64=True))
            del graph
        print(reports[-1], flush=True)
    atomic_json(args.output, dict(seed=20261002, checks=reports))


if __name__ == "__main__":
    main()

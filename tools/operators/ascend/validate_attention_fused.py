"""Validate the compiled A2 fused attention against independent FP64 attention."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 17, 129, 513, 2048])
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    spec = json.loads(args.manifest.read_text())
    p = spec["parameters"]
    maximum, capacity, ratio = p["_row_capacity"], p["capacity"], p["ratio"]
    kinds = dict(bf16=torch.bfloat16, i32=torch.int32, f32=torch.float32)
    tensors = [
        torch.empty(v["shape"], dtype=kinds[v["dtype"]], device="npu") for v in spec["operands"]
    ]
    lib = ctypes.CDLL(str(args.manifest.parent / "kernel.so"))
    entry = lib.orinfer_ascend_kernel_v2
    entry.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    entry.restype = ctypes.c_int32
    pointers = (ctypes.c_void_p * len(tensors))(*[v.data_ptr() for v in tensors])

    def launch(rows):
        code = entry(pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_)
        if code:
            raise RuntimeError(f"Fused attention launch failed: {code}")

    query = (torch.randn(maximum * 8, 512) * 0.7).bfloat16()
    swa = (torch.randn(capacity, 512) * 0.5).bfloat16()
    long = (torch.randn(capacity // ratio, 512) * 0.5).bfloat16()
    sink = torch.linspace(-3, 2, 8)
    for target, value in zip(tensors[:3], [query, swa, long], strict=True):
        target.copy_(value.npu())
    tensors[5].copy_(sink.npu())
    reports = []
    for rows in sorted({r for r in args.rows if 1 <= r <= maximum}):
        for start in sorted({0, max(0, capacity - rows - 1)}):
            positions = torch.arange(start, start + maximum, dtype=torch.int32)
            selected = torch.zeros(maximum, capacity // ratio, dtype=torch.int32)
            for row in range(rows):
                end = (start + row + 1) // ratio
                # Deterministic sparse holes and causal, sorted selections.
                chosen = torch.arange(end, dtype=torch.int32)
                chosen = chosen[chosen % 3 != 1][-512:]
                selected[row, chosen.long()] = 1
            tensors[3].copy_(positions.npu())
            tensors[4].copy_(selected.npu())
            tensors[6].fill_(17)
            launch(rows)
            torch.npu.synchronize()
            samples = sorted({0, rows // 2, rows - 1})

            def check(q, short, compressed):
                actual = tensors[6].cpu()
                assert bool(torch.isfinite(actual[: rows * 8]).all()), "Nonfinite attention output"
                assert bool((actual[rows * 8 :] == 17).all()), "Fused attention wrote the tail"
                worst, squared, elements = 0.0, 0.0, 0
                for row in samples:
                    pos = int(positions[row])
                    values = short[max(0, pos - 127) : pos + 1]
                    if p["compressed"]:
                        values = torch.cat([values, compressed[selected[row].bool()]])
                    keys = values.float().numpy().astype(np.float64)
                    scores = q[row * 8 : (row + 1) * 8].float().numpy().astype(np.float64) @ keys.T
                    scores *= 512**-0.5
                    sinks = sink.numpy().astype(np.float64)[:, None]
                    m = np.maximum(scores.max(-1, keepdims=True), sinks)
                    e = np.exp(scores - m)
                    probability = e / (e.sum(-1, keepdims=True) + np.exp(sinks - m))
                    expected = probability @ keys
                    got = actual[row * 8 : (row + 1) * 8].float().numpy()
                    delta = np.abs(got - expected)
                    worst = max(worst, float(delta.max()))
                    squared += float((delta * delta).sum())
                    elements += delta.size
                    np.testing.assert_allclose(got, expected, atol=0.004, rtol=0.025)
                return dict(max_abs=worst, rmse=(squared / elements) ** 0.5)

            error = check(query, swa, long)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                launch(rows)
            tensors[0].neg_()
            tensors[1].neg_()
            tensors[2].neg_()
            positions.add_(1)
            tensors[3].copy_(positions.npu())
            tensors[6].fill_(17)
            graph.replay()
            torch.npu.synchronize()
            changed = check(-query, -swa, -long)
            tensors[0].copy_(query.npu())
            tensors[1].copy_(swa.npu())
            tensors[2].copy_(long.npu())
            reports.append(dict(rows=rows, start=start, fp64=error, changed_graph=changed))
            print(reports[-1], flush=True)
            del graph
    atomic_json(args.output, dict(seed=20261002, manifest=str(args.manifest), checks=reports))


if __name__ == "__main__":
    main()

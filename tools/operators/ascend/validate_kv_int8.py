"""Check INT8 KV encoding, selected reads, prefix tails and Graph/state reuse."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    base = args.plan.parent
    models = [json.loads(p.read_text()) for p in (args.plan, args.reference)]
    specs, entries = {}, {}
    kinds = {}
    for model in models:
        for packet in model["ranks"][0]["packets"]:
            for step in packet["body"]:
                if step["op"] != "kernel":
                    continue
                path = base / step["manifest"]
                if path not in specs:
                    specs[path] = json.loads(path.read_text())
                spec = specs[path]
                if spec["op"].startswith("kv_int8."):
                    key = json.dumps([spec["op"], spec["parameters"]], sort_keys=True)
                    kinds[key] = (path, spec)
    dtype = dict(
        i8=torch.int8, i32=torch.int32, f16=torch.float16, bf16=torch.bfloat16, f32=torch.float32
    )

    def entry(path, spec, tensors, rows):
        if path not in entries:
            lib = ctypes.CDLL(str(path.parent / "kernel.so"))
            dynamic = "_row_capacity" in spec["parameters"]
            fn = lib.orinfer_ascend_kernel_v2 if dynamic else lib.orinfer_ascend_kernel_v1
            fn.argtypes = (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64]
                + ([ctypes.c_uint64] if dynamic else [])
                + [ctypes.c_void_p]
            )
            fn.restype = ctypes.c_int32
            entries[path] = (lib, fn, dynamic)
        _, fn, dynamic = entries[path]
        pointers = (ctypes.c_void_p * len(tensors))(*[x.data_ptr() for x in tensors])
        status = fn(
            pointers,
            len(tensors),
            *([rows] if dynamic else []),
            torch.npu.current_stream()._as_parameter_,
        )
        if status:
            raise RuntimeError(f"{path}: status {status}")

    def oracle(x):
        v = x.float().numpy().astype(np.float64).reshape(-1, 8, 64)
        amax = np.abs(v).max(-1)
        scale = np.clip(amax / 127, 2.0**-24, 65504).astype(np.float16)
        q = np.clip(np.rint(v / scale.astype(np.float64)[..., None]), -127, 127).astype(np.int8)
        decoded = torch.from_numpy(
            q.astype(np.float32) * scale.astype(np.float32)[..., None]
        ).bfloat16()
        return torch.from_numpy(q), torch.from_numpy(scale), decoded

    reports = []
    for path, spec in kinds.values():
        p = spec["parameters"]
        capacity = p.get("_row_capacity", p["rows"])
        name = spec["op"]
        if name == "kv_int8.scatter":
            table_rows = p["table_rows"]
            host = (torch.randn(capacity, 8, 64) * 3).bfloat16()
            if capacity > 1:
                host[0].zero_()
            if capacity > 1:
                host[1, :, :8] = torch.tensor(
                    [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127, -127]
                ).bfloat16()
                host[1, :, 8:] = 0
            if capacity > 2:
                host[2] = (torch.randn(8, 64) * 2.0**-24).bfloat16()
            indices = torch.arange(capacity, dtype=torch.int32) + 11
            indices %= table_rows
            if capacity > 4:
                indices[3:6] = torch.tensor([-1, -2, table_rows], dtype=torch.int32)
            source, ids = host.npu(), indices.npu()
            expected_q, expected_scale, expected_decoded = oracle(host)
            table_dtype = torch.bfloat16 if p["reference_bf16"] else torch.int8
            table = torch.empty((table_rows, 8, 64), dtype=table_dtype, device="npu")
            scales = torch.empty((table_rows, 8), dtype=torch.float16, device="npu")
            errors = torch.empty(capacity, dtype=torch.int32, device="npu")
            tensors = [source, ids, table, scales, errors]
            samples = sorted({1, min(17, capacity), min(513, capacity), capacity})
            for rows in samples:
                table.fill_(17)
                scales.fill_(19)
                errors.fill_(23)
                entry(path, spec, tensors, rows)
                torch.npu.synchronize()
                qref = expected_decoded if p["reference_bf16"] else expected_q
                want = torch.full((table_rows, 8, 64), 17, dtype=table_dtype)
                want_scale = torch.full((table_rows, 8), 19, dtype=torch.float16)
                want_errors = torch.full((capacity,), 23, dtype=torch.int32)
                for row in range(rows):
                    index = int(indices[row])
                    want_errors[row] = int(index < -1 or index >= table_rows)
                    if 0 <= index < table_rows:
                        want[index] = qref[row]
                        want_scale[index] = expected_scale[row]
                assert torch.equal(table.cpu(), want), (name, rows, "data")
                assert torch.equal(scales.cpu(), want_scale), (name, rows, "scale")
                assert torch.equal(errors.cpu(), want_errors), (name, rows, "errors/tail")
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    entry(path, spec, tensors, rows)
                # Mutate both content and destination ordering on stable addresses.
                source.neg_()
                ids.copy_(torch.flip(indices, (0,)).npu())
                table.fill_(17)
                scales.fill_(19)
                errors.fill_(23)
                graph.replay()
                torch.npu.synchronize()
                neg_q, neg_scale, neg_decoded = oracle(-host)
                qref = neg_decoded if p["reference_bf16"] else neg_q
                want.fill_(17)
                want_scale.fill_(19)
                want_errors.fill_(23)
                flipped = torch.flip(indices, (0,))
                for row in range(rows):
                    index = int(flipped[row])
                    want_errors[row] = int(index < -1 or index >= table_rows)
                    if 0 <= index < table_rows:
                        want[index] = qref[row]
                        want_scale[index] = neg_scale[row]
                assert torch.equal(table.cpu(), want), (name, rows, "changed Graph data")
                assert torch.equal(scales.cpu(), want_scale)
                assert torch.equal(errors.cpu(), want_errors)
                source.copy_(host.npu())
                ids.copy_(indices.npu())
                del graph
            reports.append(dict(op=name, parameters=p, samples=samples, graph=True))
        else:
            tensors = [
                torch.empty(o["shape"], dtype=dtype[o["dtype"]], device="npu")
                for o in spec["operands"]
            ]
            for tensor in tensors:
                if tensor.dtype in (torch.int8, torch.int32):
                    tensor.zero_()
                else:
                    tensor.fill_(17)
            if name == "kv_int8.decode_prefix":
                c = p["capacity"]
                codes = torch.randint(-127, 128, (c, 8, 64), dtype=torch.int8)
                scales = (torch.rand(c, 8) * 0.02 + 0.0001).half()
                decoded = (codes.float() * scales.float()[..., None]).bfloat16()
                tensors[0].copy_(codes.npu())
                tensors[1].copy_(scales.npu())
                for rows in sorted({1, min(17, capacity), min(513, capacity), capacity}):
                    for start in (0, max(0, c * p["ratio"] - rows)):
                        positions = torch.arange(start, start + capacity, dtype=torch.int32)
                        tensors[2].copy_(positions.npu())
                        tensors[3].fill_(17)
                        entry(path, spec, tensors, rows)
                        torch.npu.synchronize()
                        begin = max(0, (start - 127) // 128) * 128 if p["windowed"] else 0
                        end = min(c, ((start + rows) // p["ratio"] + 127) // 128 * 128)
                        want = torch.full((c, 8, 64), 17, dtype=torch.bfloat16)
                        want[begin:end] = decoded[begin:end]
                        assert torch.equal(tensors[3].cpu(), want), (name, rows, start)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    entry(path, spec, tensors, rows)
                tensors[0].neg_()
                tensors[2].copy_(torch.arange(capacity, dtype=torch.int32).npu())
                tensors[3].fill_(17)
                graph.replay()
                torch.npu.synchronize()
                end = min(c, ((rows // p["ratio"] + 127) // 128) * 128)
                want.fill_(17)
                want[:end] = -decoded[:end]
                assert torch.equal(tensors[3].cpu(), want), (name, "changed Graph")
                del graph
            elif name == "kv_int8.gather_attention":
                c, l = p["capacity"], p["compressed_capacity"]
                selected = 640 if p["compressed"] else 128
                banks = []
                for slot, n in [(0, c), (2, l)]:
                    codes = torch.randint(-127, 128, (n, 8, 64), dtype=torch.int8)
                    scales = (torch.rand(n, 8) * 0.01 + 0.0001).half()
                    tensors[slot].copy_(codes.npu())
                    tensors[slot + 1].copy_(scales.npu())
                    banks.append((codes.float() * scales.float()[..., None]).bfloat16())
                si = (
                    torch.arange(capacity * 128, dtype=torch.int32).reshape(capacity, 128) % (c + 2)
                    - 1
                )
                li = (
                    torch.arange(capacity * 512, dtype=torch.int32).reshape(capacity, 512) % (l + 2)
                    - 1
                )
                si[0, -2:] = torch.tensor([c, c + 1], dtype=torch.int32)
                li[0, -2:] = torch.tensor([l, l + 1], dtype=torch.int32)
                tensors[4].copy_(si.npu())
                tensors[5].copy_(li.npu())
                for changed in (False, True):
                    if changed:
                        tensors[0].neg_()
                        tensors[2].neg_()
                        si = torch.flip(si, (1,))
                        li = torch.flip(li, (1,))
                        tensors[4].copy_(si.npu())
                        tensors[5].copy_(li.npu())
                    if not changed:
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            entry(path, spec, tensors, capacity)
                    tensors[6].fill_(17)
                    tensors[7].fill_(23)
                    graph.replay()
                    torch.npu.synchronize()
                    want = torch.zeros(capacity, selected, 8, 64, dtype=torch.bfloat16)
                    flags = torch.full((capacity, selected), -1, dtype=torch.int32)
                    for indices, bank, offset in [(si, banks[0], 0)] + (
                        [(li, banks[1], 128)] if p["compressed"] else []
                    ):
                        valid = (indices >= 0) & (indices < len(bank))
                        values = bank[indices.clamp(0, len(bank) - 1)]
                        if changed:
                            values = -values
                        want[:, offset : offset + indices.shape[1]][valid] = values[valid]
                        flags[:, offset : offset + indices.shape[1]][valid] = 0
                    assert torch.equal(tensors[6].cpu().reshape(want.shape), want), (
                        name,
                        "changed",
                        changed,
                    )
                    assert torch.equal(tensors[7].cpu(), flags)
                del graph
            reports.append(dict(op=name, parameters=p, graph=True))
        torch.npu.synchronize()
    atomic_json(args.output, dict(seed=20261002, checks=reports))
    print(json.dumps(dict(kernels=len(reports), all_passed=True)), flush=True)


if __name__ == "__main__":
    main()

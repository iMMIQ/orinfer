"""Check fused draft Gumbel draws against the target RNG and greedy reference."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json
from tools.operators.ascend.validate_sampling import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(7)
    spec = json.loads(args.manifest.read_text())
    cap, vocab, stride = (spec["parameters"][k] for k in ("_row_capacity", "vocab", "block_size"))
    parts = (vocab + 2047) // 2048
    base = torch.randn((cap * stride, vocab), device="npu")
    bias = torch.randn((cap, vocab), device="npu")
    raw = torch.full_like(base, 12345.0)
    packed = torch.empty((parts * cap * 2,), device="npu")
    step_index = torch.zeros((1,), dtype=torch.int32, device="npu")
    controls = torch.zeros((cap * stride, 4), dtype=torch.int32, device="npu")
    lib = ctypes.CDLL(str((args.manifest.parent / spec["library"]["file"]).resolve()))
    entry = lib.orinfer_ascend_kernel_v2
    entry.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    entry.restype = ctypes.c_int32
    buffers = [base, bias, raw, packed, step_index, controls]
    pointers = (ctypes.c_void_p * len(buffers))(*[b.data_ptr() for b in buffers])

    def run(rows):
        assert entry(pointers, len(buffers), rows, torch.npu.current_stream()._as_parameter_) == 0

    checks = []
    for rows in [1, 17, cap]:
        run(rows)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run(rows)
        for phase in range(3):
            step = [0, 3, stride - 1][phase]
            x, b = torch.randn((rows * stride, vocab)), torch.randn((rows, vocab))
            ctl = torch.randint(-(1 << 31), (1 << 31) - 1, (rows * stride, 4), dtype=torch.int32)
            ctl[:, 2] = (7 if phase & 1 else 0) * vocab
            inv = torch.ones(rows * stride)
            inv[::3] = 0  # Mixed greedy/host-fallback owners keep exact argmax.
            inv[1::3] = 1 / 0.7
            ctl[:, 3] = inv.view(torch.int32)
            if phase == 2:
                x = x.reshape(rows, stride, vocab).flip(0).reshape(rows * stride, vocab)
                ctl = ctl.reshape(rows, stride, 4).flip(0).reshape(rows * stride, 4)
                inv = inv.reshape(rows, stride).flip(0).flatten()
                b = b.flip(0)
            base[: rows * stride].copy_(x)
            bias[:rows].copy_(b)
            controls[: rows * stride].copy_(ctl)
            raw.fill_(12345.0)
            step_index.fill_(step)
            graph.replay()
            torch.npu.synchronize()
            positions = torch.arange(rows) * stride + step
            scores = x[positions] + b
            torch.testing.assert_close(raw[positions].cpu(), scores, rtol=0, atol=0)
            untouched = torch.ones(cap * stride, dtype=torch.bool)
            untouched[positions] = False
            assert bool((raw.cpu()[untouched] == 12345.0).all())
            u = reference(torch, ctl[positions], vocab)
            noisy = scores * inv[positions, None] - torch.log(-torch.log(u))
            scores = torch.where(inv[positions, None] > 0, noisy, scores)
            actual = packed[: parts * rows * 2].cpu().reshape(parts, rows, 2)
            # Compare every local tile, including the vocabulary's nonaligned tail.
            for part in range(parts):
                first, end = part * 2048, min((part + 1) * 2048, vocab)
                values, ids = scores[:, first:end].max(dim=1)
                assert torch.equal(actual[part, :, 1].long(), ids + first)
                torch.testing.assert_close(actual[part, :, 0], values, atol=3e-5, rtol=2e-6)
            checks.append(dict(rows=rows, step=step, exact_tokens=True, graph_changed_input=True))
    atomic_json(args.output, dict(checks=checks, target_rng_reference=True, untouched_rows=True))
    print(json.dumps(dict(checks=len(checks), passed=True)), flush=True)


if __name__ == "__main__":
    main()

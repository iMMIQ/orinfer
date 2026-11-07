"""Validate device categorical law, sharding, active tails and changed Graph inputs."""

import argparse
import ctypes
import json
import math
from pathlib import Path

from orinfer_build.common.files import atomic_json


def reference(torch, controls, vocab):
    mask = (1 << 32) - 1
    origin = controls[:, 2].to(torch.int64)
    c = torch.arange(vocab, dtype=torch.int64)[None, :] + origin[:, None]
    d = (controls[:, 1].to(torch.int64) & mask)[:, None].expand_as(c).clone()
    key = (controls[:, 0].to(torch.int64) & mask)[:, None]
    for _ in range(10):
        product = c * 0xD256D193
        c, d = ((product >> 32) ^ key ^ d) & mask, product & mask
        key = (key + 0x9E3779B9) & mask
    return ((c >> 9).to(torch.float32) + 0.5) * (2.0**-23)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--merge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    spec = json.loads(args.local.read_text())
    cap = spec["parameters"]["_row_capacity"]
    vocab = spec["parameters"]["vocab"]
    parts = (vocab + 1023) // 1024
    logits = torch.empty((cap, vocab), device="npu")
    controls = torch.empty((cap, 4), dtype=torch.int32, device="npu")
    active = torch.zeros(1, dtype=torch.int32, device="npu")
    partial = torch.empty((cap, parts, 2), device="npu")
    output = torch.full((cap, 2), 12345.0, device="npu")
    libraries = []

    def bind(path, tensors):
        description = json.loads(path.read_text())
        lib = ctypes.CDLL(str((path.parent / description["library"]["file"]).resolve()))
        libraries.append(lib)
        entry = lib.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * len(tensors))(*[x.data_ptr() for x in tensors])

        def call():
            if entry(pointers, len(tensors), cap, torch.npu.current_stream()._as_parameter_):
                raise RuntimeError("Categorical kernel launch failed")

        return call

    first = bind(args.local, [logits, controls, active, partial])
    second = bind(args.merge, [partial, active, output])

    def run():
        first()
        second()

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    checks = []
    for phase, rows in enumerate([1, 17, 96, 32]):
        x = torch.randn((rows, vocab))
        ctl = torch.randint(-(1 << 31), (1 << 31) - 1, (rows, 4), dtype=torch.int32)
        ctl[:, 2] = (7 if phase & 1 else 0) * vocab
        inverse = torch.ones(rows)
        inverse[::3] = 0.0  # Mixed greedy and stochastic owners.
        inverse[1::3] = 1.0 / 0.7
        ctl[:, 3] = inverse.view(torch.int32)
        if phase == 3:
            x, ctl, inverse = x.flip(0), ctl.flip(0), inverse.flip(0)
        logits[:rows].copy_(x)
        controls[:rows].copy_(ctl)
        active.fill_(rows)
        output.fill_(12345.0)
        graph.replay()
        torch.npu.synchronize()
        u = reference(torch, ctl, vocab)
        noisy = x * inverse[:, None] - torch.log(-torch.log(u))
        scores = torch.where(inverse[:, None] > 0, noisy, x)
        expected = scores.argmax(dim=1)
        actual = output[:rows].cpu()
        assert torch.equal(actual[:, 1].to(torch.int64), expected), (phase, actual[:, 1], expected)
        torch.testing.assert_close(
            actual[:, 0], scores.gather(1, expected[:, None])[:, 0], atol=3e-5, rtol=2e-6
        )
        assert bool((output[rows:] == 12345.0).all()), "Sampling wrote inactive output rows"
        checks.append(dict(rows=rows, phase=phase, exact_selected=True, changed_input_graph=True))
    # A nonfinite real logit must reach the host as an invalid candidate.
    active.fill_(1)
    logits[0, 22] = float("nan")
    graph.replay()
    torch.npu.synchronize()
    assert bool(torch.isposinf(output[0, 0]))
    # Categorical frequencies over three candidates in different vocabulary tiles.
    logits.fill_(-10000.0)
    choices = [2, 1025, vocab - 1]
    probabilities = [0.2, 0.3, 0.5]
    for index, probability in zip(choices, probabilities, strict=True):
        logits[:, index] = math.log(probability)
    ctl = torch.zeros((cap, 4), dtype=torch.int32)
    ctl[:, 3] = torch.ones(cap).view(torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    active.fill_(cap)
    for batch in range(32):
        ctl[:, 0] = torch.arange(cap, dtype=torch.int32) + batch * cap
        ctl[:, 1] = 20261002
        controls.copy_(ctl)
        graph.replay()
        torch.npu.synchronize()
        tokens = output[:, 1].cpu().to(torch.int64)
        assert bool(sum(tokens == index for index in choices).all())
        for i, index in enumerate(choices):
            counts[i] += (tokens == index).sum()
    n = int(counts.sum())
    frequencies = (counts / n).tolist()
    for observed, probability in zip(frequencies, probabilities, strict=True):
        assert (
            abs(observed - probability) < 6 * math.sqrt(probability * (1 - probability) / n) + 0.003
        )
    timings = []
    for rows in [1, 16, 96]:
        active.fill_(rows)
        graph.replay()
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(16):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(dict(rows=rows, graph_ms=start.elapsed_time(end) / 16))
    graph.reset()
    report = dict(
        checks=checks,
        invalid_logit_rejected=True,
        draws=n,
        frequencies=frequencies,
        timings=timings,
    )
    atomic_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

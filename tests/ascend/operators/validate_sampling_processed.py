"""Check masked target draws, compact probabilities and changed-input Graph replay."""

import argparse
import ctypes
import json
from pathlib import Path

from orinfer_build.common.files import atomic_json
from tests.ascend.operators.validate_sampling import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from orinfer_build.common.shared_json import decode
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(7)
    model = decode(json.loads(args.plan.read_text()))
    sampling = model["sampling"]
    cap, vocab = sampling["capacity"], sampling["vocab"]
    parts, width = (vocab + 1023) // 1024, 48
    stride = ((vocab + 255) // 256) * 32
    logits = torch.empty((cap, vocab), device="npu")
    controls = torch.empty((cap, 4), dtype=torch.int32, device="npu")
    active = torch.zeros(1, dtype=torch.int32, device="npu")
    masks = torch.empty((cap, stride), dtype=torch.uint8, device="npu")
    options = torch.empty((cap, 2), dtype=torch.int32, device="npu")
    partial = torch.empty((cap, parts, width), device="npu")
    output = torch.empty((cap, width), device="npu")
    libraries = []

    def bind(name, buffers):
        path = args.plan.parent / sampling[name]
        spec = json.loads(path.read_text())
        lib = ctypes.CDLL(str((path.parent / spec["library"]["file"]).resolve()))
        libraries.append(lib)
        entry = lib.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * len(buffers))(*[b.data_ptr() for b in buffers])

        def run():
            assert (
                entry(pointers, len(buffers), cap, torch.npu.current_stream()._as_parameter_) == 0
            )

        return run

    local = bind("processed_local", [logits, controls, active, masks, options, partial])
    merge = bind("processed_merge", [partial, active, options, output])

    def run():
        local()
        merge()

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    checks = []
    for phase, rows in enumerate([1, 17, 96, 192, cap]):
        x = torch.randn((rows, vocab))
        allowed = torch.rand(rows, vocab) < 0.03
        allowed[:, 3] = True
        allowed[0] = False  # One rank may have no legal tokens.
        enabled = torch.ones(rows, dtype=torch.bool)
        enabled[1::3] = False
        allowed[~enabled] = True
        ctl = torch.randint(-(1 << 31), (1 << 31) - 1, (rows, 4), dtype=torch.int32)
        ctl[:, 2] = (7 if phase & 1 else 0) * vocab
        inverse = torch.ones(rows)
        inverse[::3] = 0.0
        inverse[1::3] = 1.0 / 0.7
        ctl[:, 3] = inverse.view(torch.int32)
        mode = torch.tensor([0, 1, 2, 21], dtype=torch.int32)[torch.arange(rows) % 4]
        if phase == 4:
            # Equal scores across vocabulary tiles must sort by token ID.
            x[2::3] = 0.0
        packed = torch.zeros((rows, stride * 8), dtype=torch.int64)
        packed[:, :vocab] = allowed
        packed = (packed.reshape(rows, stride, 8) * (1 << torch.arange(8))).sum(-1).to(torch.uint8)
        logits[:rows].copy_(x)
        controls[:rows].copy_(ctl)
        masks[:rows].copy_(packed)
        options[:rows].copy_(torch.stack([enabled.int(), mode], -1))
        active.fill_(rows)
        output.fill_(12345.0)
        graph.replay()
        torch.npu.synchronize()
        actual = output[:rows].cpu()
        scores = x * torch.where(inverse > 0, inverse, 1)[:, None]
        scores.masked_fill_(~allowed, float("-inf"))
        noise = -torch.log(-torch.log(reference(torch, ctl, vocab)))
        draw = torch.where(inverse[:, None] > 0, scores + noise, scores)
        expected = draw.argmax(-1)
        assert torch.equal(actual[:, 1].long(), expected), (phase, actual[:, 1], expected)
        torch.testing.assert_close(
            actual[:, 0], draw.gather(1, expected[:, None])[:, 0], atol=3e-5, rtol=2e-6
        )
        torch.testing.assert_close(
            actual[:, 2], scores.gather(1, expected[:, None])[:, 0], atol=2e-6, rtol=2e-6
        )
        for row in range(rows):
            if mode[row] == 0:
                continue
            maximum = scores[row].max()
            torch.testing.assert_close(actual[row, 3], maximum, atol=2e-6, rtol=2e-6)
            if not bool(allowed[row].any()):
                assert actual[row, 4] == 0
                continue
            mass = (scores[row] - maximum).exp().sum()
            torch.testing.assert_close(actual[row, 4], mass, atol=3e-5, rtol=3e-5)
            probability = (
                actual[row, 2].double() - actual[row, 3].double() - actual[row, 4].double().log()
            )
            expected_lp = scores[row].double().log_softmax(0)[expected[row]]
            torch.testing.assert_close(probability, expected_lp, atol=5e-5, rtol=2e-6)
            count = min(int(mode[row]) - 1, int(allowed[row].sum()))
            top = torch.argsort(scores[row], descending=True, stable=True)[:count]
            assert torch.equal(actual[row, 6 : 6 + 2 * count : 2].long(), top), (
                phase,
                row,
                actual[row],
                top,
            )
            torch.testing.assert_close(
                actual[row, 5 : 5 + 2 * count : 2], scores[row, top], atol=2e-6, rtol=2e-6
            )
        assert bool((output[rows:] == 12345.0).all()), "Inactive output overwritten"
        assert torch.equal(logits[:rows].cpu(), x), "Sampler modified target logits"
        checks.append(
            dict(rows=rows, masked_draws=True, compact_reports=True, changed_graph_inputs=True)
        )
    active.fill_(1)
    logits[0, 22] = float("nan")
    graph.replay()
    torch.npu.synchronize()
    assert bool(torch.isposinf(output[0, 0]))
    atomic_json(args.output, dict(passed=True, checks=checks, invalid_logits_detected=True))
    print(json.dumps(dict(passed=True, checks=checks)), flush=True)


if __name__ == "__main__":
    main()

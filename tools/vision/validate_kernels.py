"""Offline numerical and changed-input CUDA-graph tests for vision kernels."""

import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from tools.operators.common import configure, error, export_kernel, write_json
from kernels.vision.encoder import linear, layer_norm, add, position, qkv_rope, attention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    configure()
    report = []
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def check(name, kernel, inputs, outputs, reference, tolerance=0.003, atol=0.004):
        kernel.adapter.func(*inputs, *outputs, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        errors = [error(a, b) for a, b in zip(outputs, reference)]
        assert all(
            e["finite"] and e["relative_l2"] < tolerance and e["max_abs"] < atol for e in errors
        ), (name, errors)
        export_kernel(kernel, args.output / name)
        saved = [o.clone() for o in outputs]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel.adapter.func(*inputs, *outputs, stream=torch.cuda.current_stream().cuda_stream)
        original = inputs[0].clone()
        inputs[0].mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        assert any(not torch.equal(o, s) for o, s in zip(outputs, saved)), name
        inputs[0].copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(o, s) for o, s in zip(outputs, saved)), name
        report.append({"kernel": name, "error": errors, "changed_input_graph_replay": True})
        print(name, errors, flush=True)

    for rows, n, k, act in [
        (65, 1152, 1536, "none"),
        (65, 4304, 1152, "gelu_tanh"),
        (17, 1152, 4304, "none"),
        (17, 4608, 4608, "gelu"),
    ]:
        x = torch.randn((rows, k), device="cuda", dtype=torch.float16) * 0.1
        w = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
        b = torch.randn(n, device="cuda", dtype=torch.float16) * 0.01
        y = torch.empty((rows, n), device="cuda", dtype=torch.float16)
        reference = F.linear(x, w, b)
        if act != "none":
            reference = F.gelu(reference, approximate="tanh" if act == "gelu_tanh" else "none")
        check(f"linear_{rows}_{n}_{k}_{act}", linear(n, k, act), [x, w, b], [y], [reference])
    # The patch projection is Conv3D: its native-dtype bias addition rounds
    # separately from the dot product, unlike the following Linear modules.
    x = torch.randn((65, 1536), device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn((1152, 1536), device="cuda", dtype=torch.float16) * 0.02
    b = torch.randn(1152, device="cuda", dtype=torch.float16) * 0.1
    y = torch.empty((65, 1152), device="cuda", dtype=torch.float16)
    expected = F.conv3d(
        x.reshape(-1, 3, 2, 16, 16), w.reshape(1152, 3, 2, 16, 16), b, stride=(2, 16, 16)
    ).reshape(65, 1152)
    check("patch_conv_bias", linear(1152, 1536, separate_bias=True), [x, w, b], [y], [expected])
    rows = 68
    h = 1152
    x = torch.randn((rows, h), device="cuda", dtype=torch.float16)
    w = torch.randn(h, device="cuda", dtype=torch.float16) * 0.05 + 1
    b = torch.randn_like(w) * 0.05
    y = torch.empty_like(x)
    check(
        "layernorm", layer_norm(h), [x, w, b], [y], [F.layer_norm(x, (h,), w, b, 1e-6)], atol=0.008
    )
    r = torch.randn_like(x)
    check("residual_add", add(h), [x, r], [y], [x + r], tolerance=1e-6, atol=1e-6)
    # 6x10 = 60 real patches inside the 68-row workspace, merge-order coordinates.
    gh, gw, length = 6, 10, 60
    grid = torch.tensor([gh, gw], device="cuda", dtype=torch.int32)
    ln = torch.tensor([length], device="cuda", dtype=torch.int32)
    coords = torch.tensor(
        [
            (br * 2 + ir, bc * 2 + ic)
            for br in range(gh // 2)
            for bc in range(gw // 2)
            for ir in range(2)
            for ic in range(2)
        ],
        device="cuda",
    )
    table = torch.randn((2304, h), device="cuda", dtype=torch.float16) * 0.02
    hf = coords[:, 0].float() * 47 / (gh - 1)
    wf = coords[:, 1].float() * 47 / (gw - 1)
    h0 = hf.long()
    w0 = wf.long()
    h1 = (h0 + 1).clamp_max(47)
    w1 = (w0 + 1).clamp_max(47)
    dh = hf - h0
    dw = wf - w0
    a = table[h0 * 48 + w0] * ((1 - dh) * (1 - dw)).half()[:, None]
    b = table[h0 * 48 + w1] * ((1 - dh) * dw).half()[:, None]
    c = table[h1 * 48 + w0] * (dh * (1 - dw)).half()[:, None]
    d = table[h1 * 48 + w1] * (dh * dw).half()[:, None]
    expected = torch.zeros_like(x)
    expected[:length] = x[:length] + (((a + b) + c) + d)
    check("position", position(h), [x, table, grid, ln], [y], [expected], atol=0.008)
    packed = torch.randn((rows, 3 * h), device="cuda", dtype=torch.float16) * 0.2
    q, k, v = [torch.empty((rows, h), device="cuda", dtype=torch.float16) for _ in range(3)]
    q0, k0, v0 = packed[:length].reshape(length, 3, 16, 72).unbind(1)
    freq = 1 / (10000 ** (torch.arange(0, 36, 2, device="cuda").float() / 36))
    angles = (coords[:, :, None].float() * freq[None, None, :]).flatten(1)
    angle = torch.cat((angles, angles), -1)[:, None, :]

    def rotate(z):
        return torch.cat((-z[..., 36:], z[..., :36]), -1)

    reference = []
    for z in (q0, k0):
        dst = torch.zeros_like(q)
        dst[:length] = (z.float() * angle.cos() + rotate(z).float() * angle.sin()).half().flatten(1)
        reference.append(dst)
    dst = torch.zeros_like(v)
    dst[:length] = v0.flatten(1)
    reference.append(dst)
    check("qkv_rope", qkv_rope(), [packed, grid, ln], [q, k, v], reference, atol=0.002)
    expected = torch.zeros_like(q)
    qh = q[:length].reshape(length, 16, 72).transpose(0, 1).float()
    kh = k[:length].reshape(length, 16, 72).transpose(0, 1).float()
    vh = v[:length].reshape(length, 16, 72).transpose(0, 1).float()
    expected[:length] = (
        (torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(72), -1) @ vh)
        .transpose(0, 1)
        .reshape(length, h)
        .half()
    )
    check("attention_tail", attention(), [q, k, v, ln], [y], [expected], atol=0.002)
    write_json(args.output / "result.json", {"status": "passed", "checks": report})
    print(json.dumps({"status": "passed", "checks": len(report)}), flush=True)


if __name__ == "__main__":
    main()

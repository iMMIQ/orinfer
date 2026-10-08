"""Shared real-checkpoint input/reference helpers; never load the full model."""

import hashlib
import json
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS
from tools.reference import CHECKPOINT
from tools.reference import checkpoint_sha256 as checkpoint_sha256
import statistics
import time
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
MODEL = CHECKPOINT / "model.safetensors"
ACTIVATIONS = REFERENCE_ACTIVATIONS
PARTS = {
    "gate_up": ["mlp.gate_proj", "mlp.up_proj"],
    "down": ["mlp.down_proj"],
    "gdn_qkvz": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
    "gdn_out": ["linear_attn.out_proj"],
}
MATCH = {
    "gate_up": "gate_up_proj",
    "down": "down_proj",
    "gdn_qkvz": "in_proj_qkvz",
    "gdn_out": "out_proj",
}


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def raw_weights(case):
    identity = []
    fields = {key: [] for key in ("weight_packed", "weight_scale", "weight_zero_point")}
    start = time.perf_counter()
    with safe_open(str(MODEL), framework="pt", device="cpu") as f:
        for part in PARTS[case]:
            for field in fields:
                name = "model.language_model.layers.0." + part + "." + field
                t = f.get_tensor(name)
                identity.append(
                    dict(name=name, shape=list(t.shape), dtype=str(t.dtype), sha256=sha(t))
                )
                fields[field].append(t)
    r = {k: torch.cat(v).contiguous() for k, v in fields.items()}
    scale = r["weight_scale"].half()
    assert torch.equal(scale.bfloat16(), r["weight_scale"])
    r["weight_scale"] = scale
    return r, identity, time.perf_counter() - start


def logical(raw):
    start = time.perf_counter()
    r = raw["weight_packed"]
    n, k8 = r.shape
    k = k8 * 8
    shifts = torch.arange(8, dtype=torch.int32) * 4
    q = ((r[:, :, None] >> shifts) & 15).to(torch.uint8).reshape(n, k)
    zero = (
        ((raw["weight_zero_point"][:, None, :] >> shifts[None, :, None]) & 15)
        .to(torch.int8)
        .reshape(n, k // 128)
    )
    p = (q[:, ::2] | (q[:, 1::2] << 4)).contiguous()
    return p, raw["weight_scale"], zero, q, time.perf_counter() - start


def dequant(q, s, z):
    n, k = q.shape
    b = torch.empty((n, k), device="cuda", dtype=torch.float16)
    for begin in range(0, n, 1024):
        code = q[begin : begin + 1024].cuda()
        scale = s[begin : begin + 1024].cuda()
        zero = z[begin : begin + 1024].cuda()
        b[begin : begin + 1024] = (
            (
                (code.reshape(-1, k // 128, 128).float() - zero[:, :, None].float())
                * scale[:, :, None].float()
            )
            .reshape(-1, k)
            .half()
        )
    return b


def inputs(case):
    found = []
    for f in ACTIVATIONS.glob("*.json"):
        m = json.loads(f.read_text())
        if m["mode"] == "decode" and "layers.0." in m["kind"] and m["kind"].endswith(MATCH[case]):
            path = ACTIVATIONS / m["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == m["file_sha256"]
            found.append(
                (
                    m["computed_tokens_before"],
                    torch.load(path, map_location="cpu", weights_only=True),
                    m,
                )
            )
    found.sort(key=lambda x: x[0])
    assert len(found) == 8
    cat = torch.cat([t for _, t, _ in found])
    assert cat.dtype == torch.float16
    return [
        (
            m,
            cat[:m].contiguous(),
            dict(
                origin="stack of consecutive real M1 decode activations; not a simultaneous model batch",
                metadata_files=[
                    str(ACTIVATIONS / x[2]["file"]).replace(".pt", ".json") for x in found[:m]
                ],
                positions=[x[0] for x in found[:m]],
                tensor_sha256=sha(cat[:m]),
            ),
        )
        for m in (1, 2, 3, 4, 5, 7, 8)
    ]


def error(ref, out):
    d = out.float() - ref.float()
    return dict(
        relative_l2=float(d.norm() / ref.norm()),
        max_abs=float(d.abs().max()),
        finite=bool(torch.isfinite(out).all()),
    )


def measure(run, out, a):
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            run()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    saved = a.clone()
    a.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert bool((out == 0).all())
    a.copy_(saved)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    times = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(4):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 20)
    return dict(
        median_ms=statistics.median(times),
        trials_ms=times,
        graph_output_poison_and_changed_input_verified=True,
    )

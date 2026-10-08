"""Offline op15 synthetic FP32 cross-chunk state validation and measurement."""

import argparse
import json
import time
from pathlib import Path

import torch

from abi import parse_host
from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from gdn_reference import chunk_matrices, triangle_transform, wy, chunk_scan, recurrent
from kernels.operators.op15_gdn_chunk_state import gdn_chunk_state, launch

ROOT = Path(__file__).resolve().parents[2]


def make_inputs(b, tokens, bt, mode="random"):
    c = (tokens + bt - 1) // bt
    k = torch.nn.functional.normalize(torch.randn(b, 16, c, bt, 128, device="cuda"), dim=-1).half()
    v = torch.randn(b, 48, c, bt, 128, device="cuda").half()
    beta = torch.rand(b, 48, c, bt, device="cuda")
    g = -torch.rand_like(beta) * 0.025
    sin = torch.randn(b, 48, 128, 128, device="cuda") * 0.03
    if mode == "g0":
        g.zero_()
    if mode == "beta0":
        beta.zero_()
    if mode == "strong_decay":
        g.fill_(-20.0)
    if tokens % bt:
        valid = tokens % bt
        k[:, :, -1, valid:] = 0
        v[:, :, -1, valid:] = 0
        beta[:, :, -1, valid:] = 0
        g[:, :, -1, valid:] = 0
    gc = g.cumsum(-1)
    system, _ = chunk_matrices(k, k, gc, beta)
    transform = triangle_transform(system)
    w, u = wy(transform, k, v, gc, beta)
    return [k, gc, w, u, sin], (v, g, beta)


def outputs(inputs):
    k, g, w, u, sin = inputs
    b, h, c, bt = g.shape
    return [
        torch.empty((b, h, c, 128, 128), device="cuda"),
        torch.empty_like(u),
        torch.empty_like(sin),
    ]


def invoke(kernel, inputs, out):
    launch(kernel, *inputs, *out, stream=torch.cuda.current_stream().cuda_stream)


def check(out, ref, long=False):
    result = {
        name: error(value, expected)
        for name, value, expected in zip(("entering_state", "residual", "final_state"), out, ref)
    }
    for metric in result.values():
        assert metric["finite"] and metric["relative_l2"] <= (0.005 if long else 0.001), result
    return result


def resume_and_isolation(kernel, inputs, out):
    k, g, w, u, sin = inputs
    c = g.shape[2]
    split = max(1, c // 2)
    first = [x[:, :, :split].contiguous() for x in inputs[:4]] + [sin]
    first_out = outputs(first)
    invoke(kernel, first, first_out)
    second = [x[:, :, split:].contiguous() for x in inputs[:4]] + [first_out[-1].clone()]
    second_out = outputs(second)
    invoke(kernel, second, second_out)
    torch.cuda.synchronize()
    assert torch.equal(first_out[0], out[0][:, :, :split])
    assert torch.equal(first_out[1], out[1][:, :, :split])
    assert torch.equal(second_out[0], out[0][:, :, split:])
    assert torch.equal(second_out[1], out[1][:, :, split:])
    assert torch.equal(second_out[2], out[2])
    for b in range(sin.shape[0]):
        isolated = [x[b : b + 1].contiguous() for x in inputs]
        isolated_out = outputs(isolated)
        invoke(kernel, isolated, isolated_out)
        torch.cuda.synchronize()
        assert all(torch.equal(a, z[b : b + 1]) for a, z in zip(isolated_out, out))
    checkpoint = second[-1].clone()
    branch = [x.clone() for x in second]
    branch[3].add_(0.013)
    branch_out = outputs(branch)
    invoke(kernel, branch, branch_out)
    check(branch_out, chunk_scan(*branch))
    assert torch.equal(second[-1], checkpoint)
    assert not torch.equal(branch_out[-1], second_out[-1])
    # Reuse the same saved checkpoint after another branch, bit for bit.
    invoke(kernel, second, second_out)
    torch.cuda.synchronize()
    assert torch.equal(second_out[-1], out[-1])
    native_vk = sin.transpose(-1, -2).contiguous()
    assert torch.equal(native_vk.transpose(-1, -2).contiguous(), sin)
    return {
        "resume_chunk_bitwise": True,
        "request_isolation_bitwise": True,
        "branch_checkpoint_immutable": True,
        "checkpoint_restore_bitwise": True,
        "native_vk_explicit_transpose_roundtrip": True,
    }


def graph_validation(kernel, inputs, out, graph, ref):
    names = ("K", "G", "W", "U", "Sin")
    result = {}
    for index, name in enumerate(names):
        original = inputs[index].clone()
        if name in ("K", "G", "W"):
            inputs[index].mul_(0.75)
        else:
            inputs[index].add_(0.017)
        expected = chunk_scan(*inputs)
        for value in out:
            value.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        result[name] = check(out, expected)
        # Senter[0] cannot depend on K/G/W/U; final state must respond.
        assert not torch.equal(out[-1], ref[-1])
        inputs[index].copy_(original)
        for value in out:
            value.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(out, ref)
    result["restored_all_outputs"] = True
    return result


def recurrent_check(inputs, auxiliaries, final):
    # Three value heads sharing one key head, real D=128, full token chain.
    k, gc, w, u, sin = inputs
    v, g, beta = auxiliaries
    b, _, c, bt, d = k.shape
    kt = k[:, :1].reshape(b, 1, c * bt, d)
    vt = v[:, :3].reshape(b, 3, c * bt, d)
    gt, betat = g[:, :3].reshape(b, 3, c * bt), beta[:, :3].reshape(b, 3, c * bt)
    _, expected = recurrent(kt, kt, vt, gt, betat, sin[:, :3])
    metric = error(final[:, :3], expected)
    assert metric["finite"] and metric["relative_l2"] < (0.005 if c * bt >= 8192 else 0.001), metric
    return metric


def partition_check(kernels, tile):
    """The same 129 valid tokens with BT16/32/64, including padded tails."""
    tokens = 129
    inputs, auxiliary = make_inputs(1, tokens, 64)
    k, _, _, _, sin = inputs
    v, g, beta = auxiliary
    final = None
    result = {}

    def repartition(tensor, bt):
        b, h = tensor.shape[:2]
        remaining = tuple(tensor.shape[4:])
        flat = tensor.reshape(b, h, -1, *remaining)[:, :, :tokens]
        physical = ((tokens + bt - 1) // bt) * bt
        padded = torch.zeros((b, h, physical, *remaining), device="cuda", dtype=tensor.dtype)
        padded[:, :, :tokens] = flat
        return padded.reshape(b, h, physical // bt, bt, *remaining)

    for bt in (64, 32, 16):
        if (bt, tile) not in kernels:
            kernels[bt, tile] = gdn_chunk_state(bt, tile)
        kt, vt, gt, betat = [repartition(x, bt) for x in (k, v, g, beta)]
        gc = gt.cumsum(-1)
        system, _ = chunk_matrices(kt, kt, gc, betat)
        w, u = wy(triangle_transform(system), kt, vt, gc, betat)
        candidate = [kt, gc, w, u, sin]
        out = outputs(candidate)
        invoke(kernels[bt, tile], candidate, out)
        torch.cuda.synchronize()
        metrics = {
            "scan": check(out, chunk_scan(*candidate)),
            "recurrent": recurrent_check(candidate, (vt, gt, betat), out[-1]),
        }
        if final is None:
            final = out[-1].clone()
        else:
            metrics["versus_BT64_final"] = error(out[-1], final)
            assert metrics["versus_BT64_final"]["relative_l2"] < 0.001
        result[str(bt)] = metrics
    return {
        "same_valid_tokens": tokens,
        "layouts": result,
        "partition_bitwise_expected": False,
        "note": "FP32 reduction grouping changes with BT; compare numerically, resume within BT is bitwise",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--value-tile", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    dest = Path(args.output)
    configure()
    env = environment()
    report = {
        "environment": env,
        "source": identity(ROOT / "kernels/operators/op15_gdn_chunk_state.py"),
        "reference_source": identity(ROOT / "tools/operators/gdn_reference.py"),
        "input_provenance": "synthetic normalized FP16 K/V, FP32 gates and shared analytical W/U; no model trace",
        "cases": [],
        "tuning": [],
        "workspace_bytes": 0,
        "resident_parameter_bytes": 0,
        "budget_ms_B1": {"512": 0.400, "2048": 1.600, "8192": 6.400},
    }
    kernels = {}
    for tile in (16, 32) if args.tune else (args.value_tile,):
        started = time.perf_counter()
        kernel = gdn_chunk_state(64, tile)
        preparation = time.perf_counter() - started
        kernels[64, tile] = kernel
        inputs, _ = make_inputs(1, 512, 64)
        out = outputs(inputs)
        started = time.perf_counter()
        invoke(kernel, inputs, out)
        torch.cuda.synchronize()
        first = (time.perf_counter() - started) * 1000
        accuracy = check(out, chunk_scan(*inputs))
        timing, graph = benchmark(lambda: invoke(kernel, inputs, out), repetitions=10)
        del graph
        report["tuning"].append(
            {
                "value_tile": tile,
                "threads": 128,
                "prepare_s": preparation,
                "first_use_ms": first,
                "accuracy": accuracy,
                "timing": timing,
            }
        )
        write_json(dest / "progress.json", report)
        print(json.dumps(report["tuning"][-1]), flush=True)
    tile = min(report["tuning"], key=lambda x: x["timing"]["median_ms"])["value_tile"]
    report["selected_value_tile"] = tile
    specs = [(1, 512, 64, "random")]
    if not args.quick:
        specs = [(1, t, 64, "random") for t in (511, 512, 513, 2048, 8192)]
        specs += [(b, 129, 64, "random") for b in (2, 3, 4, 5, 7, 8)]
        specs += [(b, 2 * bt + 1, bt, "random") for bt in (16, 32) for b in (1, 2, 3, 4, 5, 7, 8)]
        specs += [
            (1, 2 * bt + 1, bt, mode)
            for bt in (16, 32, 64)
            for mode in ("g0", "beta0", "strong_decay")
        ]
    for b, tokens, bt, mode in specs:
        if (bt, tile) not in kernels:
            started = time.perf_counter()
            kernels[bt, tile] = gdn_chunk_state(bt, tile)
            report.setdefault("compilation", []).append(
                {"BT": bt, "prepare_s": time.perf_counter() - started}
            )
        kernel = kernels[bt, tile]
        inputs, auxiliary = make_inputs(b, tokens, bt, mode)
        initial = inputs[-1].clone()
        out = outputs(inputs)
        invoke(kernel, inputs, out)
        torch.cuda.synchronize()
        ref = chunk_scan(*inputs)
        case = {
            "B": b,
            "T": tokens,
            "BT": bt,
            "C": inputs[1].shape[2],
            "mode": mode,
            "accuracy": check(out, ref, tokens >= 8192),
            "input_bytes": sum(x.numel() * x.element_size() for x in inputs),
            "output_bytes": sum(x.numel() * x.element_size() for x in out),
        }
        assert torch.equal(inputs[-1], initial)
        case["Sin_immutable"] = True
        if tokens % bt:
            assert bool((out[1][:, :, -1, tokens % bt :] == 0).all())
            case["invalid_tail_residual_exact_zero"] = True
        case["timing"], graph = benchmark(lambda: invoke(kernel, inputs, out), repetitions=8)
        if tokens == 512 and b == 1:
            case["graph"] = graph_validation(kernel, inputs, out, graph, ref)
            case["recurrent"] = recurrent_check(inputs, auxiliary, out[-1])
        if b in (1, 3) and mode == "random" and tokens <= 513:
            case["state_semantics"] = resume_and_isolation(kernel, inputs, out)
        if tokens in (2048, 8192) and b == 1:
            case["recurrent"] = recurrent_check(inputs, auxiliary, out[-1])
        del graph
        del ref
        del initial
        inputs = None
        del auxiliary
        out = None
        report["cases"].append(case)
        write_json(dest / "progress.json", report)
        print(
            json.dumps(
                {
                    "case": [b, tokens, bt, mode],
                    "ms": case["timing"]["median_ms"],
                    "state_l2": case["accuracy"]["final_state"]["relative_l2"],
                }
            ),
            flush=True,
        )
    report["same_chain_chunk_partition"] = partition_check(kernels, tile)
    for (bt, candidate), kernel in kernels.items():
        if candidate != tile:
            continue
        folder = dest / f"bt{bt}-v{tile}"
        artifacts = export_kernel(kernel, folder)
        write_json(
            folder / "abi.json",
            {
                "operator": "op15_gdn_chunk_state",
                "BT": bt,
                "value_tile": tile,
                "threads": 128,
                "sm": 87,
                "toolchain": env,
                "logical_parameters": [
                    "K_fp16[B,16,C,BT,128]",
                    "G_fp32[B,48,C,BT]",
                    "W_fp32[B,48,C,BT,128]",
                    "U_fp32[B,48,C,BT,128]",
                    "Sin_fp32[B,48,128,128]",
                    "Senter_fp32[B,48,C,128,128]",
                    "R_fp32[B,48,C,BT,128]",
                    "Sfinal_fp32[B,48,128,128]",
                ],
                "layout": "contiguous row-major, logical state [K,V]; native [V,K] requires explicit transpose",
                "actual_generated_launches": parse_host((folder / "host.txt").read_text()),
                "cooperative_launch": False,
                "workspace_bytes": 0,
                "resident_parameter_bytes": 0,
                "alias_policy": "all allocations disjoint, inputs immutable",
                "stream": "explicit caller stream",
                "tail_policy": "K/W/U invalid rows zero, G repeats final valid cumulative gate",
                "rounding": "FP32 SIMT multiply/add compiler FMA, K FP16 to FP32; no TF32 or tensor cores",
                "artifacts": artifacts,
            },
        )
    report["peak_torch_validation_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "passed"
    write_json(dest / "results.json", report)
    print(
        json.dumps({"status": "passed", "cases": len(report["cases"]), "selected": tile}),
        flush=True,
    )


if __name__ == "__main__":
    main()

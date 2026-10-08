"""TileLang GDN prefill composition; synthetic QKV/Z, real layer0 weights."""

import argparse
import gc
import shutil
from pathlib import Path
import torch
import tilelang.language as T
from safetensors import safe_open
from common import configure, error, benchmark, identity, write_json, environment, export_kernel
from abi import parse_host
from op07_gdn_ab import MODEL, input_rows
from op08_gdn_conv_prep import reference as conv_reference
from op17_gdn_gated_norm import reference as norm_reference
from gdn_reference import chunked
from kernels.operators import op07_gdn_ab as proj
from kernels.operators import op08_gdn_conv_prep as conv
from kernels.operators import op09_gdn_gates as gates
from kernels.operators import op11_gdn_chunk_cumsum as scan
from kernels.operators import op12_gdn_chunk_matrices as matrices
from kernels.operators import op13_gdn_chunk_solve as solve
from kernels.operators import op14_gdn_chunk_wy as wy
from kernels.operators import op15_gdn_chunk_state as state
from kernels.operators import op16_gdn_chunk_output as output
from kernels.operators import op17_gdn_gated_norm as norm

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    frozen = out / "dependencies"
    frozen.mkdir()
    report = {
        "environment": environment(),
        "sources": [],
        "cases": [],
        "scope": "GDN components before mixer_out; real layer0 weights; QKV/Z and state synthetic, not model TPS",
    }
    names = (
        "07_gdn_ab",
        "08_gdn_conv_prep",
        "09_gdn_gates",
        "11_gdn_chunk_cumsum",
        "12_gdn_chunk_matrices",
        "13_gdn_chunk_solve",
        "14_gdn_chunk_wy",
        "15_gdn_chunk_state",
        "16_gdn_chunk_output",
        "17_gdn_gated_norm",
    )
    paths = [ROOT / f"kernels/operators/op{name}.py" for name in names]
    paths += [
        ROOT / "tools/operators" / name
        for name in (
            "op07_gdn_ab.py",
            "op08_gdn_conv_prep.py",
            "op17_gdn_gated_norm.py",
            "gdn_reference.py",
        )
    ]
    for path in paths:
        dest = frozen / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        report["sources"].append(identity(dest))
    with safe_open(str(MODEL), framework="pt", device="cpu") as model:
        prefix = "model.language_model.layers.0.linear_attn."
        wab = (
            torch.cat([model.get_tensor(prefix + f"in_proj_{part}.weight") for part in ("a", "b")])
            .half()
            .cuda()
        )
        wc = model.get_tensor(prefix + "conv1d.weight").reshape(10240, 4).half().cuda()
        al, dt = [model.get_tensor(prefix + name).float().cuda() for name in ("A_log", "dt_bias")]
        wn = model.get_tensor(prefix + "norm.weight").half().cuda()
    bt = 64
    pk = proj.gdn_ab_tensorcore(T.dynamic("M"), BM=32, BK=128, output_dtype="float16")
    gk = gates.gdn_gates(dtype="float16", packed_ab=True, beta_round_fp16=True)
    ck = conv.gdn_conv_prep(tile_tokens=16)
    sk = scan.gdn_pack_chunk_cumsum(bt=bt)
    mk = matrices.gdn_chunk_matrices(q_scale=128**-0.5, bt=bt)
    ak = solve.gdn_chunk_solve(bt=bt)
    wk = wy.gdn_chunk_wy(bt=bt)
    stk = state.gdn_chunk_state(bt=bt, value_tile=32)
    ok = output.gdn_chunk_output(
        q_scale=128**-0.5, bt=bt, token_tile=8, value_tile=16, output_layout="tokenmajor"
    )
    nk = norm.gdn_gated_norm(x_dtype="float16")
    report["exports"] = []
    stage_kernels = (
        ("07_gdn_ab", pk),
        ("09_gdn_gates", gk),
        ("08_gdn_conv_prep", ck),
        ("11_gdn_chunk_cumsum", sk),
        ("12_gdn_chunk_matrices", mk),
        ("13_gdn_chunk_solve", ak),
        ("14_gdn_chunk_wy", wk),
        ("15_gdn_chunk_state", stk),
        ("16_gdn_chunk_output", ok),
        ("17_gdn_gated_norm", nk),
    )
    for stage_name, kernel in stage_kernels:
        dest = out / "aot" / stage_name
        exported = export_kernel(kernel, dest)
        report["exports"].append(
            {
                "stage": stage_name,
                "artifacts": exported,
                "abi": parse_host((dest / "host.txt").read_text()),
            }
        )
    mode = {
        "normalize_round_fp16": True,
        "q_scale": 1.0,
        "qk_output_dtype": "float16",
        "conv_product_round_fp16": True,
    }
    for batch, valid in ((1, 511), (1, 512), (1, 513), (2, 129), (3, 65), (1, 2048), (1, 8192)):
        physical = ((valid + bt - 1) // bt) * bt
        chunks = physical // bt
        cpu, origin = input_rows(0, min(valid, 512))
        hidden = (
            cpu.repeat((batch * physical + len(cpu) - 1) // len(cpu), 1)[: batch * physical]
            .contiguous()
            .cuda()
        )
        x = torch.randn((batch, physical, 10240), device="cuda", dtype=torch.float16) * 0.5
        z = torch.randn((batch, physical, 48, 128), device="cuda", dtype=torch.float16)
        hi = torch.randn((batch, 3, 10240), device="cuda", dtype=torch.float16)
        pos = torch.arange(batch, device="cuda", dtype=torch.int32) * 8
        lengths = torch.tensor([valid - i for i in range(batch)], device="cuda", dtype=torch.int32)
        for b in range(batch):
            x[b, int(lengths[b]) :] = float("nan")
            z[b, int(lengths[b]) :] = 0
        q = torch.empty((batch, 16, physical, 128), device="cuda", dtype=torch.float16)
        k = torch.empty_like(q)
        v = torch.empty((batch, 48, physical, 128), device="cuda", dtype=torch.float16)
        ho = torch.empty_like(hi)
        po = torch.empty_like(pos)
        ab = torch.empty((batch * physical, 96), device="cuda", dtype=torch.float16)
        g = torch.empty((batch * physical, 48), device="cuda")
        beta = torch.empty_like(g)
        cumulative = torch.empty((batch, 48, chunks, bt), device="cuda")
        padded_beta = torch.empty_like(cumulative)
        system = torch.empty((batch, 48, chunks, bt, bt), device="cuda")
        qk = torch.empty_like(system)
        transform = torch.empty_like(system)
        w = torch.empty((batch, 48, chunks, bt, 128), device="cuda")
        u = torch.empty_like(w)
        r = torch.empty_like(w)
        si = torch.randn((batch, 48, 128, 128), device="cuda") * 0.05
        enters = torch.empty((batch, 48, chunks, 128, 128), device="cuda")
        sf = torch.empty_like(si)
        y = torch.empty((batch, physical, 48, 128), device="cuda", dtype=torch.float16)
        normalized = torch.empty_like(y)
        qc, kc, vc = [tensor.view(batch, tensor.shape[1], chunks, bt, 128) for tensor in (q, k, v)]

        def run():
            stream = torch.cuda.current_stream().cuda_stream
            pk(hidden, wab, ab, stream=stream)
            gates.launch(gk, ab, ab, al, dt, g, beta, stream=stream)
            conv.launch(ck, x, wc, hi, lengths, pos, q, k, v, ho, po, stream=stream)
            scan.launch_pack(
                sk,
                g.view(batch, physical, 48),
                beta.view(batch, physical, 48),
                lengths,
                cumulative,
                padded_beta,
                stream=stream,
            )
            matrices.launch(mk, qc, kc, cumulative, padded_beta, system, qk, stream=stream)
            solve.launch(ak, system, transform, stream=stream)
            wy.launch(wk, transform, kc, vc, cumulative, padded_beta, w, u, stream=stream)
            state.launch(stk, kc, cumulative, w, u, si, enters, r, sf, stream=stream)
            output.launch(ok, qc, cumulative, qk, enters, r, y, stream=stream)
            norm.launch(
                nk,
                y.view(batch * physical, 48, 128),
                z.view(batch * physical, 48, 128),
                wn,
                normalized.view(batch * physical, 48, 128),
                stream=stream,
            )

        def reference():
            rq, rk, rv, rh, rp, _ = conv_reference(x, wc, hi, lengths, pos, mode)
            rab = (hidden.float() @ wab.float().T).half().float()
            rg = -al.exp() * torch.nn.functional.softplus(rab[:, :48] + dt)
            rb = torch.sigmoid(rab[:, 48:]).half().float()
            rg = rg.view(batch, physical, 48).permute(0, 2, 1).contiguous()
            rb = rb.view(batch, physical, 48).permute(0, 2, 1).contiguous()
            for b in range(batch):
                rg[b, :, int(lengths[b]) :] = 0
                rb[b, :, int(lengths[b]) :] = 0
            tensor_chunks = [
                tensor.reshape(batch, tensor.shape[1], chunks, bt, 128) for tensor in (rq, rk, rv)
            ]
            ro, rs, _ = chunked(
                *tensor_chunks,
                rg.reshape(batch, 48, chunks, bt),
                rb.reshape(batch, 48, chunks, bt),
                si,
                q_scale=128**-0.5,
            )
            ro = ro.reshape(batch, 48, physical, 128).permute(0, 2, 1, 3).contiguous().half()
            ry = norm_reference(ro, z, wn).half()
            return ro, ry, rs, rh, rp

        def check():
            assert torch.equal(beta, beta.half().float()), (
                "native beta rounding must survive lowering"
            )
            ry, rn, rs, rh, rp = reference()
            errors = {
                "output": error(y, ry),
                "normalized": error(normalized, rn),
                "state": error(sf, rs),
            }
            for key, e in errors.items():
                assert e["finite"] and e["relative_l2"] < (0.005 if key == "state" else 0.002), (
                    errors
                )
            assert torch.equal(ho, rh) and torch.equal(po, rp)
            for b in range(batch):
                assert torch.count_nonzero(y[b, int(lengths[b]) :]) == 0
                assert torch.count_nonzero(normalized[b, int(lengths[b]) :]) == 0
            return errors

        run()
        original = check()
        timing, graph = benchmark(run, repetitions=3, calls_per_replay=1)
        saved = [tensor.clone() for tensor in (hidden, x, z, hi, pos, si, lengths)]
        hidden.mul_(0.8)
        x.mul_(0.5)
        z.mul_(0.75)
        hi.mul_(0.9)
        pos.add_(1)
        si.add_(0.02)
        lengths.sub_(1)
        for tensor in (
            normalized,
            y,
            sf,
            enters,
            r,
            w,
            u,
            system,
            qk,
            transform,
            cumulative,
            padded_beta,
            ho,
        ):
            tensor.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        changed = check()
        for tensor, value in zip((hidden, x, z, hi, pos, si, lengths), saved):
            tensor.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        restored = check()
        report["cases"].append(
            {
                "batch": batch,
                "valid_tokens": valid,
                "physical_tokens": physical,
                "hidden_input": origin,
                "hidden_repeat_policy": "Captured <=512 rows repeated, not an independent long-context trace",
                "original": original,
                "graph_changed": changed,
                "graph_restored": restored,
                "timing": timing,
            }
        )
        write_json(out / "progress.json", report)
        del graph
        # Scoped tensors and captured closures are replaced next iteration.
        gc.collect()
    for path in paths:
        frozen_path = frozen / path.relative_to(ROOT)
        assert identity(path)["sha256"] == identity(frozen_path)["sha256"], (
            f"Source changed during measurement: {path}"
        )
    report["status"] = "passed"
    write_json(out / "results.json", report)


if __name__ == "__main__":
    main()

"""op20 -> op27 paged gather -> op21, with independent native-math reference."""

import argparse
import shutil
from pathlib import Path

import torch
from common import (
    ROOT,
    benchmark,
    configure,
    environment,
    error,
    export_kernel,
    identity,
    write_json,
)
from abi import parse_host
from kernels.operators.op20_full_prepare import full_prepare, validate_host_metadata
from kernels.operators.op27_state_lifecycle import paged_kv_gather, validate_paged_metadata
from kernels.operators.op21_attention_prefill import attention_prefill
from tools.operators.op20_full_prepare import binding, math_reference, B, MP, NP, BS, MAXPOS
from tools.operators.op21_attention_prefill import reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    configure()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dependencies = [
        ROOT / p
        for p in (
            "kernels/operators/op20_full_prepare.py",
            "kernels/operators/op27_state_lifecycle.py",
            "kernels/operators/op21_attention_prefill.py",
            "tools/operators/op20_full_prepare.py",
            "tools/operators/op21_attention_prefill.py",
            "tools/operators/common.py",
            "tools/operators/abi.py",
        )
    ]
    before = [identity(p) for p in dependencies]
    for src in dependencies:
        dest = out / "measurement-source" / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    weights, rotary, _, info = binding(out)
    wq, wk = weights[3]
    prepare = full_prepare(B, MP, NP, max_position=MAXPOS)
    gather = paged_kv_gather()
    result = {
        "status": "in_progress",
        "environment": environment(),
        "source": before,
        "binding": info,
        "cases": [],
        "exports": [],
        "scope": "Real QK norm weights, synthetic projection/cache; snapshot replay, no model TPS",
    }

    def export(name, kernel):
        dest = out / "aot" / name
        files = export_kernel(kernel, dest)
        result["exports"].append(
            {
                "name": name,
                "files": files,
                "actual_abi": parse_host((dest / "host.txt").read_text()),
            }
        )

    export("prepare", prepare)
    export("gather", gather)
    for b, tq, tkv in ((1, 512, 512), (3, 65, 513), (2, 64, 8448)):
        attention = attention_prefill(None, tq, tkv, kv_layout="token_major")
        export(f"attention_q{tq}_kv{tkv}", attention)
        x = torch.randn((b * tq, 14336), device="cuda", dtype=torch.float16)
        req = torch.arange(b, device="cuda", dtype=torch.int32).repeat_interleave(tq)
        pos = (torch.arange(tq, device="cuda", dtype=torch.int32) + tkv - tq).repeat(b)
        pages = torch.randperm(B * MP, device="cuda").reshape(B, MP).int().contiguous()
        if b > 1:
            # This shared page is immutable: all current writes are in the suffix.
            pages[:b, 0] = pages[0, 0]
        lengths = torch.full((b,), tkv, device="cuda", dtype=torch.int32)
        base_k = torch.randn((NP, BS, 4, 256), device="cuda", dtype=torch.float16) * 0.5
        base_v = torch.randn_like(base_k)
        kp, vp = torch.empty_like(base_k), torch.empty_like(base_v)
        q = torch.empty((b * tq, 24, 256), device="cuda", dtype=torch.float16)
        gate = torch.empty_like(q)
        ko = torch.empty((b, tkv, 4, 256), device="cuda", dtype=torch.float16)
        vo = torch.empty_like(ko)
        y = torch.empty((b, tq, 24, 256), device="cuda", dtype=torch.float16)
        status = torch.zeros(1, device="cuda", dtype=torch.int32)

        def validate():
            validate_host_metadata(
                req.cpu().tolist(), pos.cpu().tolist(), pages.cpu().tolist(), NP, BS, MAXPOS
            )
            validate_paged_metadata(pages[:b].cpu().tolist(), lengths.cpu().tolist(), NP, tkv, BS)

        def reset_inputs():
            kp.copy_(base_k)
            vp.copy_(base_v)

        def run():
            stream = torch.cuda.current_stream().cuda_stream
            prepare(x, wq, wk, rotary, req, pos, pages, status, q, gate, kp, vp, stream=stream)
            gather(
                kp.view(torch.int32).reshape(NP, BS, 512),
                vp.view(torch.int32).reshape(NP, BS, 512),
                pages[:b],
                lengths,
                ko.view(torch.int32).reshape(b, tkv, 512),
                vo.view(torch.int32).reshape(b, tkv, 512),
                stream=stream,
            )
            attention(
                q.view(b, tq, 24, 256),
                ko,
                vo,
                gate.view(b, tq, 24, 256),
                pos.view(b, tq),
                lengths,
                y,
                stream=stream,
            )

        def expected():
            qr, gr, kr, vr, _, _ = math_reference(x, pos, wq, wk, rotary)
            rk, rv = base_k.clone(), base_v.clone()
            physical = pages[req.long(), pos.long() // BS].long()
            rk[physical, pos.long() % BS] = kr
            rv[physical, pos.long() % BS] = vr
            tokens = torch.arange(tkv, device="cuda")
            rko = torch.zeros_like(ko)
            rvo = torch.zeros_like(vo)
            for r in range(b):
                valid = tokens < lengths[r]
                tok = tokens[valid]
                pp = pages[r, tok // BS].long()
                rko[r, valid] = rk[pp, tok % BS]
                rvo[r, valid] = rv[pp, tok % BS]
            return reference(
                qr.view(b, tq, 24, 256),
                rko,
                rvo,
                gr.view(b, tq, 24, 256),
                pos.view(b, tq),
                lengths,
                kv_layout="token_major",
            )

        def check():
            observed = error(y, expected())
            assert observed["finite"] and observed["relative_l2"] < 0.002, observed
            return observed

        validate()
        reset_inputs()
        run()
        torch.cuda.synchronize()
        observed = check()
        baseline = y.clone()
        hot, graph = benchmark(run, repetitions=args.repetitions)
        mutations = []
        if b == 2:
            for name, target in (
                ("X", x),
                ("pages", pages),
                ("prefix_K", base_k),
                ("prefix_V", base_v),
                ("positions_lengths", pos),
            ):
                saved = target.clone()
                saved_lengths = lengths.clone()
                if name == "pages":
                    target.copy_(target.flip(1))
                elif name == "positions_lengths":
                    target.sub_(1)
                    lengths.sub_(1)
                else:
                    target.mul_(-0.75).add_(0.25)
                validate()
                reset_inputs()
                for output in (q, gate, ko, vo, y):
                    output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                changed = check()
                assert not torch.equal(y, baseline), name
                target.copy_(saved)
                lengths.copy_(saved_lengths)
                reset_inputs()
                for output in (q, gate, ko, vo, y):
                    output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(y, baseline), name
                mutations.append({"input": name, "changed_error": changed, "restored_exact": True})
        result["cases"].append(
            {
                "B": b,
                "Tq": tq,
                "Tkv": tkv,
                "error": observed,
                "hot": hot,
                "graph_mutations": mutations,
                "contiguous_KV_bytes": b * tkv * 4 * 256 * 4,
            }
        )
        write_json(out / "results.json", result)
        print(
            f"prepare+gather+prefill B{b} q{tq} kv{tkv}: {hot['median_ms']:.6f} ms, L2 {observed['relative_l2']:.6g}",
            flush=True,
        )
    assert before == [identity(p) for p in dependencies], "Sources changed during measurement"
    result["status"] = "passed"
    write_json(out / "results.json", result)


if __name__ == "__main__":
    main()

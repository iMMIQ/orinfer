"""Optional KV Q8 pack and REAL gather -> op21 attention baseline runner."""

import argparse
import re
import shutil
import time
from pathlib import Path
from tools.reference import CHECKPOINT, SOURCE
import torch
from common import (
    ROOT,
    configure,
    environment,
    error,
    export_kernel,
    benchmark,
    identity,
    write_json,
)
from abi import parse_host
from kernels.operators.op33_kv_quantization import (
    kv_quantize_pack,
    kv_page_gather,
    validate_write_metadata,
    validate_disjoint_buffers,
    validate_finite_inputs,
)
from kernels.operators.op21_attention_prefill import attention_prefill
from kernels.operators.op22_attention_decode import validate_host_metadata

MP, NP, BS = 68, 544, 128
CAP = MP * BS


def quantize_reference(x):
    # Independent CPU IEEE FP32 divide + round-even, not reciprocal multiply.
    f = x.cpu().float()
    amax = f.abs().amax(-1)
    scale = torch.where(amax > 0, amax / 127, torch.ones_like(amax))
    return torch.round(f / scale[:, :, None]).clamp(-127, 127).to(torch.int8), scale


def compare_pack(xk, xv, req, pos, pages, qk, qv, sk, sv):
    for start in range(0, len(req), 2048):
        stop = min(start + 2048, len(req))
        r = req[start:stop].long()
        p = pos[start:stop].long()
        physical = pages[r, p // BS].long()
        for x, q, s in ((xk, qk, sk), (xv, qv, sv)):
            eq, es = quantize_reference(x[start:stop])
            assert torch.equal(q[physical, p % BS].cpu(), eq)
            assert torch.equal(s[physical, p % BS].cpu().view(torch.int32), es.view(torch.int32))
    return {
        "codes_exact": True,
        "FP32_scale_bits_exact": True,
        "reference": "CPU FP32 division and torch.round ties-even",
    }


def attention_reference(
    q, k, v, gate, pages, lengths, positions, quantized=False, sk=None, sv=None
):
    outputs = []
    for b in range(len(lengths)):
        n = min(int(lengths[b]), int(positions[b]) + 1)
        if n <= 0:
            outputs.append(torch.zeros_like(q[b]))
            continue
        t = torch.arange(n, device=q.device)
        p = pages[b, t // BS].long()
        kr = k[p, t % BS]
        vr = v[p, t % BS]
        if quantized:
            kr = (kr.float() * sk[p, t % BS, :, None]).half()
            vr = (vr.float() * sv[p, t % BS, :, None]).half()
        kr = kr.float().permute(1, 0, 2)
        vr = vr.float().permute(1, 0, 2)
        score = (q[b, 0].float().reshape(4, 6, 256) @ kr.transpose(1, 2)) * 0.0625
        attn = (torch.softmax(score, dim=-1) @ vr).reshape(1, 24, 256)
        outputs.append((attn.half() * torch.sigmoid(gate[b]).half()).half())
    return torch.stack(outputs)


def export(kernel, out, name, api, layouts):
    dest = out / "aot" / name
    data = export_kernel(kernel, dest)
    source = (dest / "kernel.cu").read_text()
    abi = {
        "operator": "op33_kv_quantization",
        "variant": name,
        "sm": 87,
        "tensor_api_order": api,
        "layouts": layouts,
        "actual_launches": parse_host((dest / "host.txt").read_text()),
        "cuda_declarations": [
            decl
            for decl in re.findall(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", source)
            if decl[0] in data["symbols"]
        ],
        "cooperative_launch": False,
        "explicit_capture_current_stream": True,
        "shape_parameters": {
            "B": "dynamic int32",
            "N": "dynamic pack rows",
            "MP": MP,
            "P": NP,
            "BS": BS,
        },
        "toolchain": environment(),
        "files": data["files"],
    }
    write_json(dest / "abi.json", abi)
    return abi


def rejections():
    cases = [
        ([[0]], [0, 0], [0, 0], ()),
        ([[0]], [1], [0], ()),
        ([[0]], [0], [128], ()),
        ([[-1]], [0], [0], ()),
        ([[NP]], [0], [0], ()),
        ([[0]], [0], [0], (0,)),
        ([[0], [0]], [0], [0], ()),
        ([[0]], [True], [0], ()),
        ([[0]], [0], [-1], ()),
    ]
    result = []
    for pages, req, pos, ro in cases:
        try:
            validate_write_metadata(pages, req, pos, NP, ro)
        except ValueError as exc:
            result.append(str(exc))
        else:
            raise AssertionError("illegal write accepted")
    tensor = torch.zeros(16)
    for tensors in ((tensor, tensor), (tensor[:12], tensor[8:])):
        try:
            validate_disjoint_buffers(*tensors)
        except ValueError as exc:
            result.append(str(exc))
        else:
            raise AssertionError("alias accepted")
    for value in (float("nan"), float("inf"), -float("inf")):
        try:
            validate_finite_inputs(torch.tensor([value]))
        except ValueError as exc:
            result.append(str(exc))
        else:
            raise AssertionError("nonfinite accepted")
    assert validate_write_metadata([[0], [1]], [0, 1], [0, 0], NP)
    return result


def pack_case(n, kernel, pages, qk, qv, sk, sv, reps, graph_test=False, batch=None):
    prep = time.perf_counter()
    if batch is not None:
        pages = pages[:batch].contiguous()
    k = torch.randn((n, 4, 256), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    k[0] = 0
    v[0] = 0
    if n > 1:
        k[1] = 0
        k[1, :, 0] = 127
        k[1, :, 1:7] = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device="cuda")
        v[1] = torch.finfo(torch.float16).smallest_normal
    if n > 2:
        k[2] = 65504
        v[2] = 2**-24
    req = torch.arange(n, device="cuda", dtype=torch.int32) // CAP
    pos = torch.arange(n, device="cuda", dtype=torch.int32) % CAP
    if batch is not None:
        req = torch.arange(n, device="cuda", dtype=torch.int32) % batch
        pos = torch.arange(n, device="cuda", dtype=torch.int32) // batch
    validate_write_metadata(pages.cpu().tolist(), req.cpu().tolist(), pos.cpu().tolist(), NP)
    validate_disjoint_buffers(k, v, req, pos, pages, qk, qv, sk, sv)
    validate_finite_inputs(k, v)

    def run():
        kernel(
            k, v, req, pos, pages, qk, qv, sk, sv, stream=torch.cuda.current_stream().cuda_stream
        )

    prepare_s = time.perf_counter() - prep
    start = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    exact = compare_pack(k, v, req, pos, pages, qk, qv, sk, sv)
    timing, graph = benchmark(run, repetitions=reps, calls_per_replay=16 if n <= 8 else 1)
    replay = None
    if graph_test:
        targets = (k, v, req, pos, pages)
        saves = [x.clone() for x in targets]
        k.mul_(-0.5)
        v.add_(0.25)
        req.fill_(1)
        pos.add_(1)
        pages.copy_((pages + 17) % NP)
        validate_write_metadata(pages.cpu().tolist(), req.cpu().tolist(), pos.cpu().tolist(), NP)
        qk.fill_(-128)
        qv.fill_(-128)
        sk.fill_(float("nan"))
        sv.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        compare_pack(k, v, req, pos, pages, qk, qv, sk, sv)
        for target, saved in zip(targets, saves):
            target.copy_(saved)
        qk.fill_(-128)
        qv.fill_(-128)
        sk.fill_(float("nan"))
        sv.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        compare_pack(k, v, req, pos, pages, qk, qv, sk, sv)
        replay = {"changed": "K,V,Req,Pos,Pages", "poisoned": "QK,QV,SK,SV", "restored": True}
    return {
        "N": n,
        "B": len(pages),
        "prepare_validation_allocation_s": prepare_s,
        "first_launch_s": first,
        "hot": timing,
        "exact": exact,
        "graph": replay,
    }


def state_cases(pack, pages, qk, qv, sk, sv):
    """Real kernel writes; scheduler copies/metadata changes stay explicit."""
    k = torch.randn((3, 4, 256), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    req = torch.zeros(3, device="cuda", dtype=torch.int32)
    pos = torch.tensor([127, 128, 129], device="cuda", dtype=torch.int32)

    def execute(first, last, pt=pages, r=req, p=pos):
        validate_write_metadata(
            pt.cpu().tolist(), r[first:last].cpu().tolist(), p[first:last].cpu().tolist(), NP
        )
        pack(
            k[first:last],
            v[first:last],
            r[first:last],
            p[first:last],
            pt,
            qk,
            qv,
            sk,
            sv,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    execute(0, 3)
    torch.cuda.synchronize()
    saved = [x[:2].clone() for x in (qk, qv, sk, sv)]
    # Different chunk partition must produce byte-identical physical cache.
    for x in (qk, qv):
        x[:2].fill_(-128)
    for x in (sk, sv):
        x[:2].fill_(float("nan"))
    execute(0, 1)
    execute(1, 2)
    execute(2, 3)
    torch.cuda.synchronize()
    for x, s in zip((qk, qv, sk, sv), saved):
        # Only slots127,128,129 were defined: ignore intentionally poisoned others.
        assert torch.equal(x[0, 127].view(torch.uint8), s[0, 127].view(torch.uint8))
        assert torch.equal(x[1, :2].view(torch.uint8), s[1, :2].view(torch.uint8))
    pt = pages[:2].clone()
    pt[:, 0] = 0
    prefix = [x[0].clone() for x in (qk, qv, sk, sv)]
    r = torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32)
    p = torch.tensor([128, 128, 129], device="cuda", dtype=torch.int32)
    execute(0, 3, pt, r, p)
    torch.cuda.synchronize()
    for x, s in zip((qk, qv, sk, sv), prefix):
        assert torch.equal(x[0].view(torch.uint8), s.view(torch.uint8))
    # Caller COW of a shared partial page: source physical3 remains immutable.
    pt[:, 0] = 3
    try:
        validate_write_metadata(pt.cpu().tolist(), [1], [127], NP)
    except ValueError:
        pass
    else:
        raise AssertionError("shared partial write was accepted")
    source = [x[3].clone() for x in (qk, qv, sk, sv)]
    for x in (qk, qv, sk, sv):
        x[NP - 1].copy_(x[3])
    pt[1, 0] = NP - 1
    p[0] = 127
    r[0] = 1
    execute(0, 1, pt, r, p)
    torch.cuda.synchronize()
    for x, s in zip((qk, qv, sk, sv), source):
        assert torch.equal(x[3].view(torch.uint8), s.view(torch.uint8))
    compare_pack(k[:1], v[:1], r[:1], p[:1], pt, qk, qv, sk, sv)
    return {
        "sequential_vs_three_token_chunk_slots_exact": True,
        "shared_complete_prefix_private_tail_write_isolation_exact": True,
        "shared_partial_page_rejected_before_COW": True,
        "caller_COW_then_write_source_preserved": True,
        "caller_COW_copy_bytes": 2 * BS * 4 * 256 + 2 * BS * 4 * 4,
        "scope": "synthetic cache/scheduler fixture, not integrated allocator or prefix model quality",
    }


def read_case(batch, context, origk, origv, qk, qv, sk, sv, gather, attn, reps, graph_test=False):
    prep = time.perf_counter()
    pages = torch.arange(NP, device="cuda", dtype=torch.int32).reshape(8, MP)[:batch].clone()
    if batch > 1 and context - (batch - 1) * 3 >= 128:
        pages[:, 0] = pages[0, 0]  # only complete shared prefix; private partial pages
    lengths = torch.tensor(
        [max(0, context - b * 3) for b in range(batch)], device="cuda", dtype=torch.int32
    )
    positions = (lengths - 1).reshape(batch, 1).contiguous()
    for b in range(batch):
        pages[b, (int(lengths[b]) + 127) // 128 :] = -999
    validate_host_metadata(
        pages.cpu().tolist(), lengths.cpu().tolist(), positions[:, 0].cpu().tolist(), NP
    )
    q = torch.randn((batch, 1, 24, 256), device="cuda", dtype=torch.float16)
    gate = torch.randn_like(q) * 3
    y = torch.empty_like(q)
    kt = torch.empty((batch, CAP, 4, 256), device="cuda", dtype=torch.float16)
    vt = torch.empty_like(kt)
    validate_disjoint_buffers(qk, qv, sk, sv, pages, lengths, kt, vt, q, gate, y, positions)
    # Poison unused scale/payload slots in final page. Last prefix page shared
    # only when all request lengths exceed128, so no poisoned valid reader.
    tail_saves = []
    for b in range(batch):
        n = int(lengths[b])
        tail = n % BS
        if tail:
            p = int(pages[b, n // BS])
            tail_saves.append((p, tail, sk[p, tail:].clone(), sv[p, tail:].clone()))
            sk[p, tail:] = float("nan")
            sv[p, tail:] = float("nan")

    def run_gather():
        gather(
            qk, qv, sk, sv, pages, lengths, kt, vt, stream=torch.cuda.current_stream().cuda_stream
        )

    def run_attn():
        attn(q, kt, vt, gate, positions, lengths, y, stream=torch.cuda.current_stream().cuda_stream)

    def run():
        run_gather()
        run_attn()

    prepare_s = time.perf_counter() - prep
    start = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    expected = attention_reference(q, qk, qv, gate, pages, lengths, positions[:, 0], True, sk, sv)
    baseline = attention_reference(q, origk, origv, gate, pages, lengths, positions[:, 0])
    same = error(y, expected)
    assert same["finite"] and same["relative_l2"] <= 0.002
    for b in range(batch):
        n = int(lengths[b])
        assert (kt[b, n:] == 0).all() and (vt[b, n:] == 0).all()
        t = torch.arange(n, device="cuda")
        p = pages[b, t // BS].long()
        assert torch.equal(kt[b, :n], (qk[p, t % BS].float() * sk[p, t % BS, :, None]).half())
        assert torch.equal(vt[b, :n], (qv[p, t % BS].float() * sv[p, t % BS, :, None]).half())
    gather_t, _ = benchmark(run_gather, repetitions=reps)
    attn_t, _ = benchmark(run_attn, repetitions=reps)
    chain_t, graph = benchmark(run, repetitions=reps)
    replay = None
    if graph_test:
        targets = (q, gate, qk, qv, sk, sv, pages, lengths, positions)
        saves = [x.clone() for x in targets]
        q.mul_(-0.5)
        gate.add_(0.5)
        qk.neg_()
        qv.neg_()
        sk.mul_(0.75)
        sv.mul_(1.25)
        # Reverse request page mappings while keeping valid tails, reducing
        # lengths to a full page keeps previously poisoned pads inaccessible.
        pages.copy_(pages.flip(0))
        lengths.fill_(128)
        positions.fill_(63)
        kt.fill_(float("nan"))
        vt.fill_(float("nan"))
        y.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        changed = attention_reference(
            q, qk, qv, gate, pages, lengths, positions[:, 0], True, sk, sv
        )
        assert error(y, changed)["relative_l2"] <= 0.002
        for target, saved in zip(targets, saves):
            target.copy_(saved)
        kt.fill_(float("nan"))
        vt.fill_(float("nan"))
        y.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert error(y, expected)["relative_l2"] <= 0.002
        replay = {
            "changed": "Q,gate,QK,QV,SK,SV,Pages,lengths,positions",
            "poisoned": "Ktmp,Vtmp,Y",
            "restored": True,
        }
    for p, t, s1, s2 in tail_saves:
        sk[p, t:] = s1
        sv[p, t:] = s2
    return {
        "B": batch,
        "context": context,
        "ragged_lengths": lengths.cpu().tolist(),
        "prepare_validation_allocation_s": prepare_s,
        "first_chain_launch_s": first,
        "same_quantized_math_error": same,
        "new_quantization_loss_vs_FP16_attention": error(expected, baseline),
        "gather_hot": gather_t,
        "attention_hot": attn_t,
        "complete_readside_attention_hot": chain_t,
        "shared_prefix_readonly": batch > 1 and context - (batch - 1) * 3 >= 128,
        "unused_pad_scales_NaN_not_read": True,
        "gather_valid_bits_exact_and_padzero": True,
        "graph": replay,
        "workspace_FP16_copy_bytes": kt.numel() * kt.element_size() * 2,
        "logical_gather_read_bytes": int(lengths.sum()) * (2048 + 32)
        + pages.numel() * 4
        + lengths.numel() * 4,
        "logical_gather_write_bytes": kt.numel() * kt.element_size() * 2,
        "attention_budget_ms": 0.35 if batch == 1 else None,
        "complete_chain_within_attention_budget": chain_t["median_ms"] <= 0.35
        if batch == 1
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    for f in (
        "kernels/operators/op21_attention_prefill.py",
        "kernels/operators/op22_attention_decode.py",
        "tools/operators/abi.py",
    ):
        dest = out / "measurement-source" / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f, dest)
    config_path = CHECKPOINT / "config.json"
    import json

    config = json.loads(config_path.read_text())["text_config"]
    assert (config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]) == (
        24,
        4,
        256,
    )
    native = SOURCE / "model_executor/models/qwen3_next.py"
    shutil.copyfile(native, out / "qwen3_next.py")
    assert "gate = torch.sigmoid(gate)" in native.read_text()
    result = {
        "status": "in_progress",
        "environment": environment(),
        "source": "synthetic original Q/K/V, no frozen actual full-attention QKV trace",
        "native_source": identity(out / "qwen3_next.py"),
        "model_config": identity(config_path),
        "scope": "optional Q8 exact pack; non-inline gather to continuous FP16 -> actual op21 Tq1 attention",
        "model_quality": "BF16 and long-context model suite NOT passed; optional format disabled by default",
        "cpu_rejections": rejections(),
        "exports": [],
        "pack_cases": [],
        "read_cases": [],
    }
    start = time.perf_counter()
    pack = kv_quantize_pack(MP, NP)
    gather = kv_page_gather(MP, NP)
    attn = attention_prefill(None, 1, CAP, kv_layout="token_major", block_m=16, block_n=64)
    result["compile_prepare_s"] = time.perf_counter() - start
    for kernel, name, api, layouts in (
        (
            pack,
            "pack",
            ["K", "V", "Req", "Pos", "Pages", "QK", "QV", "SK", "SV"],
            {"K/V": "[N,4,256] FP16", "QK/QV": "[544,128,4,256] int8", "SK/SV": "[544,128,4] FP32"},
        ),
        (
            gather,
            "gather",
            ["QK", "QV", "SK", "SV", "Pages", "Lengths", "Ktmp", "Vtmp"],
            {"Ktmp/Vtmp": "[B,8704,4,256] FP16", "Pages": "[B,68] int32"},
        ),
        (
            attn,
            "attention_op21",
            ["Q", "K", "V", "Gate", "Positions", "Lengths", "Y"],
            {"Q/Gate/Y": "[B,1,24,256] FP16", "K/V": "[B,8704,4,256] FP16"},
        ),
    ):
        result["exports"].append(export(kernel, out, name, api, layouts))
    src = (out / "aot/pack/kernel.cu").read_text()
    assert "__fdiv_rn" in src and "nearbyintf" in src
    assert "AllReduce<tl::MaxOp, 256" in src and "float values[1]" in src
    assert "if (((int)threadIdx.x) == 0)" in src
    qk = torch.empty((NP, BS, 4, 256), device="cuda", dtype=torch.int8)
    qv = torch.empty_like(qk)
    sk = torch.empty((NP, BS, 4), device="cuda")
    sv = torch.empty_like(sk)
    pages = torch.arange(NP, device="cuda", dtype=torch.int32).reshape(8, MP)
    for n in (
        (1, 3, 513)
        if args.smoke
        else (1, 2, 3, 4, 5, 7, 8, 127, 128, 129, 511, 512, 513, 2048, 8192, 8448)
    ):
        case = pack_case(n, pack, pages, qk, qv, sk, sv, args.repetitions, n == 513)
        result["pack_cases"].append(case)
        write_json(out / "results.json", result)
        print(f"pack N{n} exact {case['hot']['median_ms']:.6f}ms", flush=True)
    for b in (1, 2, 3, 4, 5, 7, 8):
        case = pack_case(b, pack, pages, qk, qv, sk, sv, args.repetitions, batch=b)
        result["pack_cases"].append(case)
        write_json(out / "results.json", result)
        print(f"pack dynamic B{b} N{b} exact {case['hot']['median_ms']:.6f}ms", flush=True)
    result["state_cases"] = state_cases(pack, pages, qk, qv, sk, sv)
    # Initialize entire physical cache through actual pack, then preserve
    # original FP16 for a SEPARATE added-quantization-loss reference.
    origk = torch.randn((NP, BS, 4, 256), device="cuda", dtype=torch.float16) * 0.5
    origv = torch.randn_like(origk)
    req = torch.arange(NP * BS, device="cuda", dtype=torch.int32) // CAP
    pos = torch.arange(NP * BS, device="cuda", dtype=torch.int32) % CAP
    pack(
        origk.reshape(-1, 4, 256),
        origv.reshape(-1, 4, 256),
        req,
        pos,
        pages,
        qk,
        qv,
        sk,
        sv,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    result["full_physical_cache_initialization_exact"] = compare_pack(
        origk.reshape(-1, 4, 256), origv.reshape(-1, 4, 256), req, pos, pages, qk, qv, sk, sv
    )
    cases = (
        [(1, 512), (2, 8192)]
        if args.smoke
        else [(b, c) for b in (1, 2, 3, 4, 5, 7, 8) for c in (512, 2048, 8192, 8448)]
    )
    if not args.smoke:
        cases += [(3, c) for c in (127, 128, 129, 511, 513)]
    for b, c in cases:
        case = read_case(
            b, c, origk, origv, qk, qv, sk, sv, gather, attn, args.repetitions, b == 3 and c == 513
        )
        result["read_cases"].append(case)
        write_json(out / "results.json", result)
        print(
            f"read B{b} ctx{c}: full {case['complete_readside_attention_hot']['median_ms']:.5f}ms L2 {case['same_quantized_math_error']['relative_l2']:.4g}",
            flush=True,
        )
    result["memory"] = {
        "KV_code_bytes": qk.numel() * 2,
        "KV_scale_bytes": sk.numel() * 4 * 2,
        "FP16_baseline_KV_bytes": origk.numel() * 2 * 2,
        "page_table_bytes_at_B8": 8 * MP * 4,
        "pack_metadata_bytes_per_token": 8,
        "read_metadata_bytes_per_request": MP * 4 + 8,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    result["implementation_identity"] = [
        identity(ROOT / "kernels/operators/op33_kv_quantization.py"),
        identity(Path(__file__)),
    ]
    result["status"] = (
        "passed standalone exact pack and real readside+attention; model/Rust/scheduler integration pending"
    )
    write_json(out / "results.json", result)
    print("op33 complete; wrapper releases GPU lock", flush=True)


if __name__ == "__main__":
    main()

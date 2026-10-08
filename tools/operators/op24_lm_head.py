"""Full 248320-row real untied head: streamed W4 screening and AOT evidence."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS
from tools.reference import CHECKPOINT, checkpoint_sha256

import torch
import tilelang.language as T
from safetensors import safe_open
from common import (
    ROOT,
    configure,
    environment,
    error,
    export_kernel,
    identity,
    write_json,
    benchmark,
)
from abi import parse_host, evaluate
from kernels.operators.op24_lm_head import lm_head, VOCAB, HIDDEN
from kernels.projections.candidates import fp16_gemm
from kernels.operators import op25_token_selection as select
from kernels.operators import op26_probability_score as score

MODEL = CHECKPOINT / "model.safetensors"
ACTIVATIONS = REFERENCE_ACTIVATIONS
SHAPES = (1, 2, 3, 4, 5, 7, 8)


def activations():
    rows = []
    for path in ACTIVATIONS.glob("*-lm_head.json"):
        meta = json.loads(path.read_text())
        source = ACTIVATIONS / meta["file"]
        assert identity(source)["sha256"] == meta["file_sha256"]
        a = torch.load(source, map_location="cpu", weights_only=True)
        assert a.dtype == torch.float16 and a.shape[-1] == HIDDEN
        rows.append(
            (meta["mode"], meta["computed_tokens_before"], a[-1:], dict(metadata=str(path), **meta))
        )
    decode = sorted((r for r in rows if r[0] == "decode"), key=lambda r: r[1])[:8]
    prefill = next(r for r in rows if r[0] == "prefill")
    assert len(decode) == 8, "eight real final-hidden captures required"
    cpu = torch.cat([r[2] for r in decode] + [prefill[2]]).contiguous()
    assert cpu.shape == (9, HIDDEN)
    return cpu, [r[3] for r in decode] + [prefill[3]]


def export(kernel, folder, tensors, config):
    files = export_kernel(kernel, folder)
    host = (folder / "host.txt").read_text()
    manifest = dict(
        operator="op24_lm_head",
        actual_abi=parse_host(host),
        actual_cuda_declarations=re.findall(
            r"__global__\s+void\s+\w+\s*\([^)]*\)", (folder / "kernel.cu").read_text()
        ),
        tensors=tensors,
        config=config,
        target="sm_87",
        cooperative=False,
        explicit_stream=True,
        toolchain=environment(),
        artifact=files,
    )
    for name, flag in [("resources.txt", "--dump-resource-usage")]:
        r = subprocess.run(
            ["/usr/local/cuda/bin/cuobjdump", flag, str(folder / "kernel.cubin")],
            capture_output=True,
            text=True,
        )
        (folder / name).write_text(r.stdout + r.stderr)
        manifest[name] = dict(exit_code=r.returncode, **identity(folder / name))
    write_json(folder / "abi.json", manifest)
    return manifest


def binary(path, tensor):
    path.write_bytes(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return dict(file=path.name, **identity(path))


def stream_weights(output, a):
    """Only 2048 source rows/FP32 rows at a time; no full FP32 weight copy."""
    started = time.perf_counter()
    p = torch.empty((VOCAB, HIDDEN // 2), device="cuda", dtype=torch.uint8)
    s = torch.empty((VOCAB, HIDDEN // 128), device="cuda", dtype=torch.float16)
    z = torch.empty((VOCAB, HIDDEN // 128), device="cuda", dtype=torch.int8)
    # Solely a separately accounted BF16->FP16 baseline; destroyed before W4 timings.
    baseline = torch.empty((VOCAB, HIDDEN), device="cuda", dtype=torch.float16)
    refs = torch.empty((a.shape[0], VOCAB), device="cuda", dtype=torch.float32)
    bfrefs = torch.empty_like(refs)
    packed_folder = output / "packed-weight"
    packed_folder.mkdir()
    files = {name: (packed_folder / (name + ".bin")).open("wb") for name in ("P", "S", "Z")}
    hashes = {name: hashlib.sha256() for name in ("source", "P", "S", "Z")}
    stat = dict(
        cast_changed_values=0,
        cast_underflow_to_zero=0,
        cast_squared_error=0.0,
        cast_max_abs=0.0,
        quant_squared_error=0.0,
        source_squared_norm=0.0,
        quant_max_abs=0.0,
    )
    with safe_open(str(MODEL), framework="pt", device="cpu") as checkpoint:
        view = checkpoint.get_slice("lm_head.weight")
        assert view.get_shape() == [VOCAB, HIDDEN]
        for begin in range(0, VOCAB, 2048):
            end = min(begin + 2048, VOCAB)
            raw = view[begin:end, :]
            assert raw.dtype == torch.bfloat16
            hashes["source"].update(raw.contiguous().view(torch.uint8).numpy().tobytes())
            w = raw.cuda().float()
            assert bool(torch.isfinite(w).all())
            half = w.half()
            stat["cast_changed_values"] += int((half.float() != w).sum())
            stat["cast_underflow_to_zero"] += int(((half == 0) & (w != 0)).sum())
            cast_difference = half.float() - w
            stat["cast_squared_error"] += float(cast_difference.square().sum())
            stat["cast_max_abs"] = max(stat["cast_max_abs"], float(cast_difference.abs().max()))
            baseline[begin:end].copy_(half)
            grouped = w.reshape(end - begin, HIDDEN // 128, 128)
            low = grouped.amin(-1).clamp_max(0)
            high = grouped.amax(-1).clamp_min(0)
            scale = ((high - low) / 15).clamp_min(2**-24).half()
            zero = torch.round(-low / scale.float()).clamp(0, 15).to(torch.int8)
            q = (
                torch.round(grouped / scale.float()[:, :, None] + zero.float()[:, :, None])
                .clamp(0, 15)
                .to(torch.uint8)
                .reshape(end - begin, HIDDEN)
            )
            packed = (q[:, ::2] | (q[:, 1::2] << 4)).contiguous()
            deq = (
                (
                    (q.reshape(end - begin, HIDDEN // 128, 128).float() - zero.float()[:, :, None])
                    * scale.float()[:, :, None]
                )
                .reshape(end - begin, HIDDEN)
                .half()
            )
            p[begin:end].copy_(packed)
            s[begin:end].copy_(scale)
            z[begin:end].copy_(zero)
            refs[:, begin:end] = a.float() @ deq.float().T
            bfrefs[:, begin:end] = a.float() @ w.T
            difference = deq.float() - w
            stat["quant_squared_error"] += float(difference.square().sum())
            stat["source_squared_norm"] += float(w.square().sum())
            stat["quant_max_abs"] = max(stat["quant_max_abs"], float(difference.abs().max()))
            for name, tensor in (("P", packed), ("S", scale), ("Z", zero)):
                data = tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()
                hashes[name].update(data)
                files[name].write(data)
            if begin % 32768 == 0:
                print(
                    json.dumps(
                        dict(stage="streaming-full-vocabulary", rows_done=end, rows_total=VOCAB)
                    ),
                    flush=True,
                )
    for f in files.values():
        f.close()
    torch.cuda.synchronize()
    stat["quant_weight_relative_l2"] = (
        stat["quant_squared_error"] / stat["source_squared_norm"]
    ) ** 0.5
    stat["source_to_f16_lossless"] = stat["cast_changed_values"] == 0
    stat["cast_weight_relative_l2"] = (
        stat["cast_squared_error"] / stat["source_squared_norm"]
    ) ** 0.5
    stat.update(
        stream_rows=2048,
        streamed_read_quant_reference_export_s=time.perf_counter() - started,
        source_tensor=dict(
            name="lm_head.weight",
            shape=[VOCAB, HIDDEN],
            dtype="BF16",
            sha256=hashes["source"].hexdigest(),
            bytes=2 * VOCAB * HIDDEN,
        ),
        packed_tensors={
            name: dict(
                path=str(packed_folder / (name + ".bin")),
                sha256=hashes[name].hexdigest(),
                bytes=(packed_folder / (name + ".bin")).stat().st_size,
            )
            for name in ("P", "S", "Z")
        },
        quantization="uncalibrated weight-only asymmetric group128; include-zero min/max; FP16 scale; RNE zero/code; no official quantization quality acceptance",
    )
    return p, s, z, baseline, refs, bfrefs, stat


def graph_validate(kernel, a, p, s, z, out, reference):
    def run():
        kernel(a, p, s, z, out, stream=torch.cuda.current_stream().cuda_stream)

    run()
    torch.cuda.synchronize()
    original = out.clone()
    saved_a = a.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, original), "poison replay"
    a.neg_()
    graph.replay()
    torch.cuda.synchronize()
    changed = error(out, -reference)
    assert changed["finite"] and changed["relative_l2"] <= 0.002, changed
    a.copy_(saved_a)
    details = []
    # Change the actual registered P/S/Z addresses after graph capture. Only
    # output token0's first group changes; all remaining vocab must stay exact.
    for name, target in (("P", p), ("S", s), ("Z", z)):
        saved = target[0].clone()
        old_q = torch.stack((p[0, :64] & 15, p[0, :64] >> 4), -1).reshape(128).float()
        old_w = ((old_q - z[0, 0].float()) * s[0, 0].float()).half().float()
        if name == "P":
            target[0, :64].bitwise_xor_(255)
        elif name == "S":
            target[0, 0].mul_(2)
        else:
            target[0, 0] = (int(target[0, 0]) + 1) % 16
        new_q = torch.stack((p[0, :64] & 15, p[0, :64] >> 4), -1).reshape(128).float()
        new_w = ((new_q - z[0, 0].float()) * s[0, 0].float()).half().float()
        expected = reference[:, 0] + a[:, :128].float() @ (new_w - old_w)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:, 1:], original[:, 1:]), name
        assert torch.allclose(out[:, 0], expected, rtol=0.002, atol=0.01), (
            name,
            out[:, 0],
            expected,
        )
        assert not torch.equal(out[:, 0], original[:, 0]), f"{name}: mutation did not affect output"
        details.append(
            dict(
                buffer=name,
                actual_changed_token0=out[:, 0].cpu().tolist(),
                expected_token0=expected.cpu().tolist(),
            )
        )
        target[0].copy_(saved)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, original), "restore all input/weights"
    return dict(poison_restore=True, changed_hidden=True, changed_P_S_Z=details)


def top_buffers(m):
    blocks = (VOCAB + 4095) // 4096
    return [
        torch.empty((m, blocks, 3), device="cuda"),
        torch.empty((m, blocks, 3), device="cuda", dtype=torch.int32),
        torch.empty((m, blocks), device="cuda", dtype=torch.int32),
        torch.empty((m, 3), device="cuda"),
        torch.empty((m, 3), device="cuda", dtype=torch.int32),
        torch.empty(m, device="cuda", dtype=torch.int32),
        torch.empty(m, device="cuda", dtype=torch.int32),
    ]


def probability_buffers(m, q):
    return [
        torch.empty((m, (VOCAB + 4095) // 4096, 3), device="cuda"),
        torch.empty(m, device="cuda"),
        torch.empty((m, q), device="cuda"),
        torch.empty((m, q), device="cuda"),
        torch.empty(m, device="cuda", dtype=torch.int32),
        torch.empty((m, q), device="cuda", dtype=torch.int32),
    ]


def quality_bridge(output, candidate, baseline, metadata):
    """Actual op24 FP32 logits -> op25 greedy/top3 -> op26 full-vocab scores."""
    m = candidate.shape[0]
    started = time.perf_counter()
    selection = (select.topk_partials(dtype="float32"), select.topk_merge())
    scoring = (
        score.probability_partials(dtype="float32"),
        score.probability_merge(dtype="float32"),
    )
    compiled = []
    for name, kernel in zip(
        ("top3-partials", "top3-merge", "prob-partials", "prob-merge"), selection + scoring
    ):
        compiled.append(
            export(
                kernel, output / "compiled" / "bridge" / name, {}, dict(dtype="FP32", vocab=VOCAB)
            )
        )
    bv, cv = top_buffers(m), top_buffers(m)
    stream = torch.cuda.current_stream().cuda_stream
    select.launch(*selection, baseline, *bv, stream=stream)
    select.launch(*selection, candidate, *cv, stream=stream)
    assert not bool(bv[-1].any() or cv[-1].any())
    # Target is last vocab row as a fixed independent queried token, not a
    # claimed ground-truth next token (capture lacks generated token IDs).
    ids = torch.cat(
        (bv[4], torch.full((m, 1), VOCAB - 1, device="cuda", dtype=torch.int32)), 1
    ).contiguous()
    bp, cp = probability_buffers(m, 4), probability_buffers(m, 4)
    score.launch(*scoring, baseline, ids, *bp, stream=stream)
    score.launch(*scoring, candidate, ids, *cp, stream=stream)
    torch.cuda.synchronize()
    assert not any(bool(x.any()) for x in (bp[-1], bp[-2], cp[-1], cp[-2]))
    for logits, v, prob in ((baseline, bv, bp), (candidate, cv, cp)):
        ref_ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :3].int()
        assert torch.equal(v[4], ref_ids)
        logp = logits.double().log_softmax(-1).gather(1, ids.long()).float()
        assert torch.allclose(prob[2], logp, atol=0.001, rtol=0)
    rows = []
    for i in range(m):
        bi, ci = bv[4][i].cpu().tolist(), cv[4][i].cpu().tolist()
        rows.append(
            dict(
                case_id=f"captured-head-{i}",
                capture_metadata=metadata[i],
                position=metadata[i]["computed_tokens_before"],
                seed=20261002,
                execution_mode="paired identical real final hidden; LM-head-only diagnostic",
                normalization_vocab=VOCAB,
                baseline_top3_token_ids=bi,
                candidate_top3_token_ids=ci,
                baseline_selected_token_id=int(bv[5][i]),
                candidate_selected_token_id=int(cv[5][i]),
                baseline_top3_probabilities=bp[3][i, :3].cpu().tolist(),
                candidate_probabilities_on_baseline_top3=cp[3][i, :3].cpu().tolist(),
                candidate_logprobs_on_baseline_top3=cp[2][i, :3].cpu().tolist(),
                missing_baseline_ids_explicitly_queried=[token for token in bi if token not in ci],
                extra_queried_token_id=VOCAB - 1,
                extra_query_semantics="fixed diagnostic vocab-tail ID; no real next-token ID or target NLL available",
                baseline_extra_query_logprob=float(bp[2][i, 3]),
                candidate_extra_query_logprob=float(cp[2][i, 3]),
                baseline_extra_query_probability=float(bp[3][i, 3]),
                candidate_extra_query_probability=float(cp[3][i, 3]),
                top1_same=bi[0] == ci[0],
                top3_overlap=len(set(bi) & set(ci)) / 3,
            )
        )
    write_json(
        output / "quality-diagnostic.json",
        dict(
            rows=rows,
            bridge_artifacts=compiled,
            measured_s=time.perf_counter() - started,
            scope="no full-model state replay; AWQ-text-body final hiddens with exact original BF16 untied head baseline; not official BF16 full-model quality benchmark",
            historical_token_identity="prompt token IDs available; generated teacher-forced token IDs/positions absent in old capture; exact hidden file hashes bind paired history",
        ),
    )
    return dict(
        rows=m,
        top1_matches=sum(r["top1_same"] for r in rows),
        mean_top3_overlap=sum(r["top3_overlap"] for r in rows) / m,
        missing_baseline_ids=sum(len(r["missing_baseline_ids_explicitly_queried"]) for r in rows),
        fullvocab_bridge=True,
        file=str(output / "quality-diagnostic.json"),
    )


def fixture(output, folder, abi, a, p, s, z, ref):
    root = output / "rust-fixture-M1"
    root.mkdir()
    for name in ("kernel.cubin", "kernel.cu", "host.txt"):
        shutil.copyfile(folder / name, root / name)
    binary(root / "A.bin", a[:1])
    binary(root / "reference-f32.bin", ref[:1])

    def fi(path):
        return dict(file=str(path.relative_to(root)), sha256=identity(path)["sha256"])

    buffers = []
    for name, tensor, dtype, layout in (
        ("A", a[:1], "f16", "row_major"),
        ("P", p, "u8", "nk-packed-low-high-u4"),
        ("S", s, "f16", "ng"),
        ("Z", z, "i8", "ng"),
    ):
        path = root / "A.bin" if name == "A" else root / (name + ".bin")
        if name != "A":
            path.hardlink_to(output / "packed-weight" / (name + ".bin"))
        buffers.append(
            dict(
                name=name,
                dtype=dtype,
                shape=list(tensor.shape),
                layout=layout,
                alignment=16,
                access="read",
                data=fi(path),
            )
        )
    buffers.append(
        dict(
            name="Logits",
            dtype="f32",
            shape=[1, VOCAB],
            layout="row_major",
            alignment=16,
            access="write",
        )
    )
    call = abi["actual_abi"][0]
    conf = call["launch_expressions"]
    arguments = []
    for item in call["ordered_arguments"]:
        value = item["value"]
        if value.endswith(".data_ptr()"):
            arguments.append(dict(kind="buffer", name=value[:-11]))
        elif value == "M":
            arguments.append(dict(kind="i32", value=1))
        else:
            raise ValueError(value)
    manifest = dict(
        schema_version=1,
        target="sm_87",
        toolchain={k: str(environment()[k]) for k in ("torch", "tilelang", "cuda")},
        buffers=buffers,
        kernels=[
            dict(
                name="op24",
                module=fi(root / "kernel.cubin"),
                source=fi(root / "kernel.cu"),
                host_abi=fi(root / "host.txt"),
                symbol=call["symbol"],
                grid=[evaluate(conf["gridDim" + i], {"M": 1}) for i in ("X", "Y", "Z")],
                block=[evaluate(conf["blockDim" + i], {"M": 1}) for i in ("X", "Y", "Z")],
                shared_memory_bytes=evaluate(conf["sharedMemBytes"], {}),
                cooperative=False,
                args=arguments,
            )
        ],
        validation=dict(
            output="Logits",
            reference=fi(root / "reference-f32.bin"),
            relative_l2_tolerance=0.002,
            zero_input="A",
            repetitions=3,
        ),
    )
    write_json(root / "manifest.json", manifest)
    return str(root / "manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=6)
    args = parser.parse_args()
    output = args.output
    configure()
    # Freeze every imported implementation actually used, beyond wrapper defaults.
    for source in (
        "tools/operators/op24_lm_head.py",
        "kernels/operators/op24_lm_head.py",
        "kernels/operators/op03_ffn_gate_up.py",
        "kernels/projections/candidates.py",
        "kernels/operators/op25_token_selection.py",
        "kernels/operators/op26_probability_score.py",
        "tools/operators/abi.py",
        "configs/quick-quality.json",
    ):
        destination = output / "measurement-source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    report = dict(
        environment=environment(),
        checkpoint=dict(
            path=str(MODEL),
            full_sha256=checkpoint_sha256(MODEL),
            full_file_hash_policy="Supplied checkpoint hashed once; streamed head tensor SHA separately computed",
        ),
        identities={
            n: identity(MODEL.parent / n)
            for n in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        },
        failures=[],
        candidates=[],
        budget_ms=4.0,
        scope="isolated untied full vocabulary head; no full-model throughput or quality acceptance",
    )

    def save():
        write_json(output / "results.json", report)

    cpu, meta = activations()
    report["activations"] = meta
    report["activation_batch_semantics"] = (
        "consecutive actual M1 captures stacked for shape checks; not concurrent model state execution"
    )
    a9 = cpu.cuda()
    # Full-vocab extreme/tail references computed in the same streamed pass.
    extra = torch.cat((torch.zeros_like(a9[:1]), -8 * a9[:1], a9[:1] * 1e-5), 0).contiguous()
    all_a = torch.cat((a9, extra), 0)
    p, s, z, b, refs, bfrefs, stat = stream_weights(output, all_a)
    report["weights"] = stat
    report["resident_weight_bytes"] = sum(t.numel() * t.element_size() for t in (p, s, z))
    report["bits_per_parameter"] = 8 * report["resident_weight_bytes"] / (VOCAB * HIDDEN)
    report["workspace_bytes"] = 0
    report["bf16_f16_baseline_weight_bytes"] = 2 * VOCAB * HIDDEN
    report["source_frozen"] = [
        identity(path) for path in (output / "measurement-source").rglob("*") if path.is_file()
    ]
    save()
    # FP16 source weights are independently retained for this benchmark only.
    started = time.perf_counter()
    baseline_kernel = fp16_gemm(T.dynamic("M"), VOCAB, HIDDEN, BM=16)
    baseline_prepare = time.perf_counter() - started
    baseout = torch.empty((8, VOCAB), device="cuda", dtype=torch.float16)
    baseline_abi = export(
        baseline_kernel,
        output / "compiled" / "bf16-cast-f16-baseline",
        {},
        dict(BM=16, N=VOCAB, K=HIDDEN, baseline=True),
    )
    baseline_shapes = []
    for m in (1, 8):
        aa = a9[:m]
        oo = baseout[:m]

        def run():
            return baseline_kernel(aa, b, oo, stream=torch.cuda.current_stream().cuda_stream)

        run()
        torch.cuda.synchronize()
        err = error(oo, bfrefs[:m])
        assert err["finite"] and err["relative_l2"] <= 0.002, err
        measured, _ = benchmark(run, repetitions=args.repetitions)
        baseline_shapes.append(dict(M=m, error=err, **measured))
    report["baseline"] = dict(
        prepare_s=baseline_prepare,
        abi=baseline_abi,
        shapes=baseline_shapes,
        weights="original BF16 cast FP16 with measured tiny cast error; reference-only expanded 2.543GB, released before W4 timing",
    )
    del run
    baseline_kernel = None
    del baseout
    b = None
    torch.cuda.synchronize()
    report["production_weight_cuda_allocated_after_baseline_release"] = (
        torch.cuda.memory_allocated()
    )
    save()
    tensors = dict(
        A=["M", HIDDEN, "f16", "row-major"],
        P=[VOCAB, HIDDEN // 2, "u8", "adjacent low/high U4"],
        S=[VOCAB, HIDDEN // 128, "f16", "NG"],
        Z=[VOCAB, HIDDEN // 128, "i8 numeric 0..15", "NG"],
        Logits=["M", VOCAB, "f32", "row-major"],
    )
    compiled = {}
    probe = []
    # A finite two-route screen, no unbounded autotuning.
    for name in ("register", "shared"):
        started = time.perf_counter()
        kernel = lm_head(T.dynamic("M"), implementation=name)
        folder = output / "compiled" / name
        row = dict(
            implementation=name,
            prepare_s=time.perf_counter() - started,
            abi=export(kernel, folder, tensors, dict(BM=16, BN=64, BK=128, stages=2, threads=128)),
            shapes=[],
        )
        report["candidates"].append(row)
        compiled[name] = (kernel, folder, row)
        for m in (1, 8):
            aa = a9[:m]
            out = torch.empty((m, VOCAB), device="cuda")

            def run():
                return kernel(aa, p, s, z, out, stream=torch.cuda.current_stream().cuda_stream)

            begin = time.perf_counter()
            run()
            torch.cuda.synchronize()
            first = time.perf_counter() - begin
            err = error(out, refs[:m])
            assert err["finite"] and err["relative_l2"] <= 0.002, err
            timing, _ = benchmark(run, repetitions=args.repetitions)
            result = dict(
                M=m,
                same_quant_error=err,
                source_bf16_quantization_error=error(out, bfrefs[:m]),
                first_launch_host_s=first,
                full_vocab=VOCAB,
                output_bytes=4 * m * VOCAB,
                **timing,
            )
            row["shapes"].append(result)
            probe.append((name, m, timing["median_ms"]))
            print(json.dumps(dict(implementation=name, **result)), flush=True)
            save()
    chosen = min((item for item in probe if item[1] == 1), key=lambda item: item[2])[0]
    kernel, folder, row = compiled[chosen]
    report["selected"] = chosen
    for m in (2, 3, 4, 5, 7):
        aa = a9[:m]
        out = torch.empty((m, VOCAB), device="cuda")

        def run():
            return kernel(aa, p, s, z, out, stream=torch.cuda.current_stream().cuda_stream)

        run()
        torch.cuda.synchronize()
        err = error(out, refs[:m])
        assert err["finite"] and err["relative_l2"] <= 0.002, err
        timing, _ = benchmark(run, repetitions=args.repetitions)
        row["shapes"].append(
            dict(
                M=m,
                same_quant_error=err,
                source_bf16_quantization_error=error(out, bfrefs[:m]),
                full_vocab=VOCAB,
                output_bytes=4 * m * VOCAB,
                **timing,
            )
        )
        save()
    out = torch.empty((8, VOCAB), device="cuda")
    report["graph"] = graph_validate(kernel, a9[:8], p, s, z, out, refs[:8])
    report["extremes"] = []
    for i, name in enumerate(("zero", "sign-reversed-times8", "tiny-times1e-5"), start=9):
        aa = all_a[i : i + 1]
        oo = torch.empty((1, VOCAB), device="cuda")
        kernel(aa, p, s, z, oo, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        err = error(oo, refs[i : i + 1])
        assert err["finite"] and err["relative_l2"] <= 0.002, err
        report["extremes"].append(dict(case=name, full_vocab=VOCAB, same_quant_error=err))
    # Generation prefill scores only final hidden; scoring paths use a few rows.
    aa = a9[8:9]
    oo = torch.empty((1, VOCAB), device="cuda")

    def run():
        return kernel(aa, p, s, z, oo, stream=torch.cuda.current_stream().cuda_stream)

    run()
    torch.cuda.synchronize()
    err = error(oo, refs[8:9])
    assert err["finite"] and err["relative_l2"] <= 0.002, err
    timing, _ = benchmark(run, repetitions=args.repetitions)
    report["prefill_generation"] = dict(
        prompt_tokens=512,
        scored_final_hidden_rows=1,
        error=err,
        **timing,
        limit="same LM-head cost for 2K/8K final-row use is not measured real 2K/8K model execution",
    )
    quality_out = torch.empty((8, VOCAB), device="cuda")
    kernel(a9[:8], p, s, z, quality_out, stream=torch.cuda.current_stream().cuda_stream)
    report["quality_bridge"] = quality_bridge(output, quality_out, bfrefs[:8], meta[:8])
    report["rust_fixture"] = fixture(output, folder, row["abi"], a9, p, s, z, refs)
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["budget_met_M1"] = (
        next(item["median_ms"] for item in row["shapes"] if item["M"] == 1) <= 4
    )
    save()
    print(
        json.dumps(
            dict(
                selected=chosen,
                budget_met=report["budget_met_M1"],
                quality=report["quality_bridge"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

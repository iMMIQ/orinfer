"""Screen staged attention with actual model Q/K/V, gates and positions.

Model and payload fingerprints are checked. Each candidate must also handle
empty and nonaligned KV lengths, changed graph inputs and output guards.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model.screen_prefill_ffn import model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from kernels.model.attention_prefill_staged import attention_prefill_staged
    from tools.operators.common import configure, benchmark, error, export_kernel, write_json

    configure()
    torch.empty(1, device="cuda")
    capture = json.loads((args.activations / "capture.json").read_text())
    if capture["fingerprint"] != model_identity(args.model) or capture["seed"] != 20261002:
        raise ValueError("Capture identity mismatch")
    rows, context = capture["rows"], capture["context"]
    capacity = json.loads((args.model / "cache/model.json").read_text())["metadata"]["max_context"]
    if capacity < context or capacity % 64:
        raise ValueError("Invalid model KV capacity")
    if context % 64 or rows % 128:
        raise ValueError("Screen expects complete 128-query and 64-KV tiles")

    def tensor(name, dtype, shape):
        entry = capture["inputs"][name]
        raw = (args.activations / entry["file"]).read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Capture payload changed: " + name)
        return torch.from_numpy(np.frombuffer(raw, dtype=dtype).copy().reshape(shape)).cuda()

    q = tensor("FullQ", np.float16, (1, rows, 6144))
    gate = tensor("FullGate", np.float16, (1, rows, 6144))
    k = tensor("PrefillK", np.float16, (1, context, 1024))
    v = tensor("PrefillV", np.float16, (1, context, 1024))
    positions = tensor("Positions", np.int32, (1, rows))
    lengths = tensor("SeqLength", np.int32, (1,))
    if int(lengths.item()) != context or not torch.equal(
        positions[0], torch.arange(context - rows, context, device="cuda", dtype=torch.int32)
    ):
        raise ValueError("Unexpected absolute attention positions")
    original = [x.clone() for x in (q, k, v, positions, lengths)]
    reference = torch.empty((1, rows, 24, 256), device="cuda", dtype=torch.float16)
    baseline = attention_prefill_staged(
        1,
        rows,
        capacity,
        kv_layout="token_major",
        block_m=64,
        block_n=32,
        num_stages=1,
        interior_mask=True,
        contiguous_queries=True,
    )
    baseline.adapter.func(
        q, k, v, gate, positions, lengths, reference, stream=torch.cuda.current_stream().cuda_stream
    )
    torch.cuda.synchronize()
    report = dict(
        status="running",
        seed=20261002,
        fingerprint=capture["fingerprint"],
        rows=rows,
        context=context,
        capacity=capacity,
        cases=[],
        scope="Real captured full-attention inputs; candidate numeric/replay screening, not model TPS.",
    )
    choices = [
        (64, 32, 128, 1, "precise"),
        (64, 32, 128, 1, "fast"),
        (128, 32, 256, 1, "precise"),
        (128, 32, 256, 1, "fast"),
        (64, 64, 256, 1, "precise"),
        (64, 64, 256, 1, "fast"),
        (128, 64, 256, 1, "fast"),
        (64, 32, 256, 2, "fast"),
    ]
    for bm, bn, threads, stages, exponent in choices:
        for dst, src in zip((q, k, v, positions, lengths), original):
            dst.copy_(src)
        key = f"m{bm}-n{bn}-t{threads}-s{stages}-{exponent}"
        print("compile", key, flush=True)
        kernel = attention_prefill_staged(
            1,
            rows,
            capacity,
            kv_layout="token_major",
            block_m=bm,
            block_n=bn,
            threads=threads,
            num_stages=stages,
            exp_mode=exponent,
            interior_mask=True,
            contiguous_queries=True,
        )
        guarded = torch.full((rows * 6144 + 256,), 123.0, device="cuda", dtype=torch.float16)
        out = guarded[:-256].view(1, rows, 24, 256)

        def run():
            kernel.adapter.func(
                q,
                k,
                v,
                gate,
                positions,
                lengths,
                out,
                stream=torch.cuda.current_stream().cuda_stream,
            )

        timing, graph = benchmark(run, repetitions=8)
        observed = error(out, reference)
        assert observed["finite"] and observed["relative_l2"] < 0.002, (key, observed)
        assert bool((guarded[-256:] == 123.0).all())
        checks = []
        for length in [0, 1, 145, 1025, context]:
            q.copy_(original[0]).mul_(0.97)
            k.copy_(original[1])
            v.copy_(original[2])
            k[:, length:].fill_(float("nan"))
            v[:, length:].fill_(float("nan"))
            k[:, length : min(context, ((length + bn - 1) // bn) * bn)].zero_()
            v[:, length : min(context, ((length + bn - 1) // bn) * bn)].zero_()
            lengths.fill_(length)
            positions.copy_(
                torch.arange(
                    max(0, length - rows),
                    max(0, length - rows) + rows,
                    device="cuda",
                    dtype=torch.int32,
                )[None, :]
            )
            positions[0, 0] = -1
            out.fill_(float("nan"))
            graph.replay()
            # The synchronous baseline masks invalid KV rather than requiring
            # zero padding, so it independently checks candidate padding reads.
            safe_baseline = attention_prefill_staged(
                1, rows, capacity, kv_layout="token_major", block_m=64, block_n=32, num_stages=0
            )
            safe_baseline.adapter.func(
                q,
                k,
                v,
                gate,
                positions,
                lengths,
                reference,
                stream=torch.cuda.current_stream().cuda_stream,
            )
            torch.cuda.synchronize()
            e = error(out, reference)
            assert e["finite"] and e["relative_l2"] < 0.002, (key, length, e)
            assert bool((out[0, 0] == 0).all()) and bool((guarded[-256:] == 123.0).all())
            checks.append(dict(length=length, error=e, empty_query=True, guard=True))
        for dst, src in zip((q, k, v, positions, lengths), original):
            dst.copy_(src)
        baseline.adapter.func(
            q,
            k,
            v,
            gate,
            positions,
            lengths,
            reference,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        graph.replay()
        torch.cuda.synchronize()
        restored = error(out, reference)
        assert restored["finite"] and restored["relative_l2"] < 0.002
        export_kernel(kernel, args.output / key)
        record = dict(
            bm=bm,
            bn=bn,
            threads=threads,
            stages=stages,
            exp_mode=exponent,
            timing=timing,
            error=observed,
            metadata_checks=checks,
            graph_restore_error=restored,
            export=key,
        )
        report["cases"].append(record)
        write_json(args.output / "result.json", report)
        print(key, round(timing["median_ms"], 4), observed, flush=True)
        kernel = None
        del graph
        del guarded
        out = None
        del safe_baseline
    finalists = sorted(report["cases"], key=lambda r: r["timing"]["median_ms"])[:3]
    for record in finalists:
        kernel = attention_prefill_staged(
            1,
            rows,
            capacity,
            kv_layout="token_major",
            block_m=record["bm"],
            block_n=record["bn"],
            threads=record["threads"],
            num_stages=record["stages"],
            exp_mode=record["exp_mode"],
            interior_mask=True,
            contiguous_queries=True,
        )
        out = torch.empty_like(reference)

        def run():
            kernel.adapter.func(
                q,
                k,
                v,
                gate,
                positions,
                lengths,
                out,
                stream=torch.cuda.current_stream().cuda_stream,
            )

        record["screen_timing"] = record["timing"]
        record["timing"], graph = benchmark(run, repetitions=32)
        e = error(out, reference)
        assert e["finite"] and e["relative_l2"] < 0.002
        record["finalist_rechecked"] = True
        del graph
        kernel = None
        out = None
    min(finalists, key=lambda r: r["timing"]["median_ms"])["selected"] = True
    report["status"] = "passed"
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()

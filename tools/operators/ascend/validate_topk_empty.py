"""Compare published local TopK buffers and changed-input Graph replay exactly."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.tune import npu_benchmark
from tools.model.tuning import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--length", type=int, nargs="+", help="Check only these score widths")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)

    def kernels(plan):
        result, seen = {}, set()
        model = json.loads(plan.read_text())
        for rank in model["ranks"]:
            for packet in rank["packets"]:
                for step in packet["body"] + packet["head"]:
                    if step["op"] != "kernel":
                        continue
                    path = plan.parent / step["manifest"]
                    if path in seen:
                        continue
                    seen.add(path)
                    spec = json.loads(path.read_text())
                    if spec["op"] == "topk.local":
                        p = spec["parameters"]
                        if args.length and p["length"] not in args.length:
                            continue
                        key = (p["_row_capacity"], p["length"], p.get("selected", 512))
                        result[key] = (path, spec)
        return result

    before, after = kernels(args.baseline), kernels(args.plan)
    assert before.keys() == after.keys() and before
    libraries, reports = [], []

    def bind(description, tensors, rows):
        path, spec = description
        binary = path.parent / spec["library"]["file"]
        assert file_hash(binary) == spec["library"]["sha256"]
        library = ctypes.CDLL(str(binary.resolve()))
        libraries.append(library)
        entry = library.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])

        def run():
            if entry(pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_):
                raise RuntimeError("Local TopK launch failed")

        return run

    for capacity, length, selected in before:
        block = 4096 if selected <= 512 else 2048
        chunks = (length + block - 1) // block
        scores = torch.randn((capacity, length), device="npu")
        counts = torch.empty(capacity, dtype=torch.int32, device="npu")
        old = torch.empty((capacity + 1, chunks, 2 * selected), device="npu")
        new = torch.empty_like(old)
        for rows in sorted({1, min(17, capacity), capacity}):
            original = bind(before[(capacity, length, selected)], [scores, counts, old], rows)
            candidate = bind(after[(capacity, length, selected)], [scores, counts, new], rows)

            def run():
                original()
                candidate()

            counts.fill_(0)
            run()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run()
            for phase in range(3):
                scores.neg_()
                if rows > 1:
                    scores[1].zero_()  # Equal-score ordering must also match.
                choices = [0, 1, 511, 512, 513, 2049, 4096, 4097, length - 1, length]
                if rows == 1:
                    lengths = [choices[(0, 7, 9)[phase]]]
                else:
                    lengths = [choices[(r + phase * 3) % len(choices)] for r in range(rows)]
                counts[:rows].copy_(torch.tensor(lengths, dtype=torch.int32))
                old.fill_(12345)
                new.fill_(12345)
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(new[:rows], old[:rows], atol=0, rtol=0)
                assert bool((new[rows:] == 12345).all()), "TopK wrote the row tail"
                # Independently verify every empty chunk's canonical pair lanes.
                for row, count in enumerate(lengths):
                    start = (count + block - 1) // block
                    if start < chunks:
                        assert bool(torch.isneginf(new[row, start:, 0::2]).all())
                        assert bool((new[row, start:, 1::2] == -1).all())
            graph.reset()
            reports.append(
                dict(
                    rows=rows,
                    capacity=capacity,
                    length=length,
                    exact_original=True,
                    changed_input_graph=True,
                    empty_sentinels=True,
                )
            )
        timings = []
        for context in sorted({min(2048, length), min(32768, length), length}):
            counts.fill_(context)
            times = []
            for description, output in (
                (before[(capacity, length, selected)], old),
                (after[(capacity, length, selected)], new),
            ):
                call = bind(description, [scores, counts, output], capacity)

                def repeated():
                    for _ in range(8):
                        call()

                times.append(
                    measure(repeated, budget_ms=8, max_repetitions=16, benchmark=npu_benchmark) / 8
                )
            timings.append(dict(context=context, baseline_ms=times[0], candidate_ms=times[1]))
        reports[-1]["timings"] = timings
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, reports)
        print(json.dumps(dict(capacity=capacity, length=length, timings=timings)), flush=True)
    print("Exact local TopK, sentinels, row tails and modified Graph inputs passed", flush=True)


if __name__ == "__main__":
    main()

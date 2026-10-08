"""Validate GPU history penalties against f64 total-order greedy semantics."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import numpy as np
    import torch
    from kernels.model.greedy_sampling import history_counts, penalized_partials, penalized_merge
    from tools.operators.common import configure, benchmark, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    vocab, context = 248320, 262144
    blocks = (vocab + 4095) // 4096
    count = history_counts(vocab, context)
    partials = penalized_partials(vocab)
    merge = penalized_merge(vocab)
    x = torch.empty(vocab, device="cuda", dtype=torch.float32)
    history = torch.zeros(context, device="cuda", dtype=torch.int32)
    length = torch.zeros(1, device="cuda", dtype=torch.int32)
    counts = torch.zeros(vocab, device="cuda", dtype=torch.int32)
    parameters = torch.empty(3, device="cuda", dtype=torch.uint64)
    keys = torch.empty(blocks, device="cuda", dtype=torch.uint64)
    ids, bad = [torch.empty(blocks, device="cuda", dtype=torch.int32) for _ in range(2)]
    token, status = [torch.empty(1, device="cuda", dtype=torch.int32) for _ in range(2)]

    def run():
        counts.zero_()
        count.torch_function(history, length, counts)
        partials.torch_function(x, counts, parameters, keys, ids, bad)
        merge.torch_function(keys, ids, bad, token, status)

    values = np.random.default_rng(20261002).normal(size=vocab).astype(np.float32)
    rng = np.random.default_rng(20261002)
    cases = []
    for n in (0, 17, 8193, context):
        hist = rng.integers(0, vocab, size=n, dtype=np.int32)
        for params in [(1.05, 0.0, 0.0), (0.5, 2.0, -2.0), (2.0, -2.0, 2.0)]:
            cases.append((values.copy(), hist, params))
    for v in (np.zeros(vocab, dtype=np.float32), np.full(vocab, -2.0, dtype=np.float32)):
        v[0], v[1] = -0.0, 0.0
        cases.append((v, np.array([1, 1, vocab - 1], dtype=np.int32), (1.0, 0.0, 0.0)))
    for special in (np.nan, np.inf, -np.inf):
        v = values.copy()
        v[-1] = special
        cases.append((v, np.array([vocab - 1], dtype=np.int32), (1.05, 0.0, 0.0)))
    cases.append(
        (
            np.full(vocab, -2.0, dtype=np.float32),
            np.array([0], dtype=np.int32),
            (np.finfo(np.float64).max, 0.0, 0.0),
        )
    )
    report = dict(seed=20261002, cases=[], status="running")
    graph = None
    for index, (v, hist, params) in enumerate(cases):
        x.copy_(torch.from_numpy(v))
        history[: len(hist)].copy_(torch.from_numpy(hist))
        length.fill_(len(hist))
        parameters.copy_(torch.from_numpy(np.array(params, dtype=np.float64).view(np.uint64)))
        with np.errstate(over="ignore", invalid="ignore"):
            counts_cpu = np.bincount(hist, minlength=vocab)
            score = v.astype(np.float64)
            seen = counts_cpu > 0
            score[seen] = np.where(
                score[seen] < 0.0, score[seen] * params[0], score[seen] / params[0]
            )
            score[seen] -= params[1]
            score -= params[2] * counts_cpu
        invalid = not bool(np.isfinite(score).all())
        bits = score.view(np.uint64)
        ordered = np.where((bits >> 63) != 0, ~bits, bits ^ np.uint64(1 << 63))
        expected = -1 if invalid else int(ordered.argmax())
        if graph is None:
            timings, graph = benchmark(run, repetitions=8)
            report["gpu_graph_timing"] = timings
        else:
            graph.replay()
            torch.cuda.synchronize()
        assert int(status.item()) == int(invalid), (index, "status")
        assert int(token.item()) == expected, (index, int(token.item()), expected)
        assert np.array_equal(counts.cpu().numpy(), counts_cpu), (index, "history counts")
        report["cases"].append(
            dict(
                history=len(hist),
                parameters=list(params),
                token=expected,
                exact_equal=True,
                changed_input_graph_replay=True,
            )
        )
        write_json(args.output / "result.json", report)
        print("greedy", index, "passed", flush=True)
    report["status"] = "passed"
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()

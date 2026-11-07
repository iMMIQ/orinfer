"""GPU sampling contract, tails, deterministic radix and changed-input replay."""

import argparse
import json
from pathlib import Path
import torch
from orinfer_build.compiler.cuda.common import benchmark, configure, export_kernel, environment
from kernels.cuda.model import sampling as s
from orinfer_build.models.sampling import geometry, batch_layout, FUNCTIONS


class BatchedPipeline:
    """Exercise the production independent_rows wrapper against scalar kernels."""

    def __init__(self, pipes):
        from orinfer_build.compiler.compile import instantiate, compile_function
        from kernels.cuda.model.independent_rows import independent_rows

        self.pipes = pipes
        vocab, context = pipes[0].vocab, pipes[0].context
        self.shapes = geometry(vocab, context)
        self.offsets, self.stride = batch_layout(self.shapes)
        self.scratch = torch.empty((len(pipes), self.stride), dtype=torch.uint8, device="cuda")
        self.pointers = torch.zeros((128, 2), dtype=torch.uint64, device="cuda")
        self.output = torch.full((128, 2), -91, dtype=torch.int32, device="cuda")
        self.kernels = {}
        for name, names in FUNCTIONS.items():
            fn = instantiate(getattr(s, name), vocab, *([context] if name == "history" else []))
            roles = {n: ("private", "Logits" if n == "Logits" else "Scratch") for n in names}
            if "Output" in names:
                roles["Output"] = ("row", "Output")
            fn, arguments = independent_rows(
                fn,
                128,
                roles,
                {"Logits": 0, "Scratch": 1},
                dynamic=True,
                offsets={n: self.offsets[n] for n in names if n in self.offsets},
            )
            self.kernels[name] = compile_function(fn), arguments

    def inputs(self, owners):
        import math
        import re

        table = torch.zeros((128, 2), dtype=torch.uint64)
        for row, owner in enumerate(owners):
            if owner is None:
                continue
            pipe = self.pipes[owner]
            table[row, 0] = pipe.logits.data_ptr()
            table[row, 1] = self.scratch[row].data_ptr()
            for name in ("History", "BiasIds", "BiasValues", "Control", "Parameters", "Mask"):
                attr = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
                source = getattr(pipe, attr)
                dtype, shape = self.shapes[name]
                target = self.scratch[
                    row,
                    self.offsets[name] : self.offsets[name]
                    + math.prod(shape) * source.element_size(),
                ]
                target.copy_(source.reshape(-1).view(torch.uint8))
        self.pointers.copy_(table)

    def run(self, rows):
        stream = torch.cuda.current_stream().cuda_stream

        def launch(name, **scalars):
            kernel, arguments = self.kernels[name]
            values = {None: self.pointers, "Output": self.output, "__rows__": rows, **scalars}
            kernel.adapter.func(*(values[name] for name in arguments), stream=stream)

        for name in ("initialize", "history", "scores", "masses"):
            launch(name)
        for mode in range(3):
            for digit in range(12):
                launch("radix_histogram", digit=digit, mode=mode)
                launch("radix_choose", digit=digit, mode=mode)


def check_batched(sanitizer=False):
    reports = []
    for vocab, maximum in ((1057, 17), (248320, 3)) if sanitizer else ((1057, 128), (248320, 8)):
        pipes = [Pipeline(vocab, context=32) for _ in range(maximum)]
        batch = BatchedPipeline(pipes)

        def prepare(owner, pipe):
            logits = torch.randn(vocab, generator=torch.Generator().manual_seed(20261002 + owner))
            pipe.inputs(
                logits,
                (0, 1, 7, 40)[owner % 4],
                (1.0, 0.9, 0.8)[owner % 3],
                (owner + 0.37) / maximum,
                penalties=(0.2, 0.13, 1.1),
                history=[0, 0, 2, vocab - 1],
                bias={3: 0.7, vocab - 1: -0.2},
                mask=[i % 3 != owner % 3 for i in range(vocab)],
            )

        for owner, pipe in enumerate(pipes):
            prepare(owner, pipe)
            pipe.run()
        expected = [p.output.cpu().tolist() for p in pipes]
        cases = []
        for rows in (n for n in (1, 3, 8, 17, 128) if n <= maximum):
            owners = list(reversed(range(rows)))
            batch.inputs(owners)
            batch.run(rows)
            assert batch.output[:rows].cpu().tolist() == [expected[i] for i in owners]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                batch.run(rows)
            # Change membership and every live row's logits/parameters. Reuse
            # the same captured addresses, including an inactive interior row.
            owners.reverse()
            for owner in owners:
                pipes[owner].inputs(torch.arange(vocab, dtype=torch.float32), 1)
            if rows > 1:
                owners[rows // 2] = None
            batch.output.fill_(-91)
            batch.inputs(owners)
            graph.replay()
            actual = batch.output[:rows].cpu().tolist()
            assert actual == [[vocab - 1, 0] if i is not None else [-91, -91] for i in owners]
            bad_row = next(i for i, owner in enumerate(owners) if owner is not None)
            for invalid in ("mask", "nan"):
                logits = torch.zeros(vocab)
                if invalid == "nan":
                    logits[-1] = float("nan")
                pipes[owners[bad_row]].inputs(
                    logits, mask=[False] * vocab if invalid == "mask" else None
                )
                batch.inputs(owners)
                graph.replay()
                actual = batch.output[:rows].cpu().tolist()
                assert actual == [
                    [0, 1]
                    if i == bad_row
                    else ([vocab - 1, 0] if owner is not None else [-91, -91])
                    for i, owner in enumerate(owners)
                ]
            cases.append(
                dict(rows=rows, exact=True, changed_membership_replay=True, errors_isolated=True)
            )
            # Restore the heterogeneous distributions for the next row count.
            for owner, pipe in enumerate(pipes):
                prepare(owner, pipe)
        reports.append(dict(vocab=vocab, cases=cases))
    return reports


class Pipeline:
    def __init__(self, vocab, context=1024):
        self.vocab, self.context = vocab, context
        blocks = (vocab + s.TILE - 1) // s.TILE
        self.kernels = {
            name: getattr(s, name)(vocab, *([context] if name == "history" else []))
            for name in (
                "initialize",
                "history",
                "scores",
                "masses",
                "radix_histogram",
                "radix_choose",
            )
        }

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device="cuda")

        self.counts = empty((vocab,), torch.int32)
        self.bias = empty((vocab,), torch.float64)
        self.history = empty((context,), torch.int32)
        self.bias_ids = empty((vocab,), torch.int32)
        self.bias_values = empty((vocab,), torch.float64)
        self.control = empty((4,), torch.int32)
        self.parameters = empty((6,), torch.float64)
        self.mask = empty(((vocab + 31) // 32,), torch.uint32)
        self.scores = empty((vocab,), torch.float64)
        self.keys = empty((vocab,), torch.uint64)
        self.maximum = empty((blocks,), torch.float64)
        self.invalid = empty((blocks,), torch.int32)
        self.mass = empty((vocab,), torch.uint64)
        self.state = empty((7,), torch.uint64)
        self.hist = empty((blocks, 256), torch.uint64)
        self.output = empty((2,), torch.int32)
        self.logits = empty((vocab,), torch.float32)

    def run(self):
        k = self.kernels
        k["initialize"](self.counts, self.bias)
        k["history"](
            self.history, self.bias_ids, self.bias_values, self.control, self.counts, self.bias
        )
        k["scores"](
            self.logits,
            self.counts,
            self.bias,
            self.mask,
            self.control,
            self.parameters,
            self.scores,
            self.keys,
            self.maximum,
            self.invalid,
        )
        k["masses"](
            self.scores,
            self.keys,
            self.maximum,
            self.invalid,
            self.parameters,
            self.mass,
            self.output,
        )
        for mode in range(3):
            for digit in range(12):
                k["radix_histogram"](
                    self.keys,
                    self.mass,
                    self.state,
                    self.control,
                    self.parameters,
                    self.hist,
                    digit,
                    mode,
                )
                k["radix_choose"](
                    self.hist, self.state, self.control, self.parameters, self.output, digit, mode
                )

    def inputs(
        self,
        logits,
        top_k=0,
        top_p=1.0,
        uniform=0.37,
        temperature=0.8,
        penalties=(0.0, 0.0, 1.0),
        history=(),
        bias=None,
        mask=None,
    ):
        self.logits.copy_(logits)
        self.control.copy_(
            torch.tensor(
                [len(history), len(bias or {}), int(mask is not None), top_k], dtype=torch.int32
            )
        )
        self.parameters.copy_(
            torch.tensor([temperature, top_p, *penalties, uniform], dtype=torch.float64)
        )
        if history:
            self.history[: len(history)].copy_(torch.tensor(history, dtype=torch.int32))
        if bias:
            self.bias_ids[: len(bias)].copy_(torch.tensor(list(bias), dtype=torch.int32))
            self.bias_values[: len(bias)].copy_(
                torch.tensor(list(bias.values()), dtype=torch.float64)
            )
        if mask is not None:
            words = [
                sum(int(mask[i]) << (i % 32) for i in range(j, min(j + 32, self.vocab)))
                for j in range(0, self.vocab, 32)
            ]
            self.mask.copy_(torch.tensor(words, dtype=torch.uint32))


def reference(logits, k, p, u, temp, penalties, history, bias, mask):
    values = logits.double().cpu()
    counts = torch.bincount(torch.tensor(history, dtype=torch.int64), minlength=len(values))
    presence, frequency, repetition = penalties
    values = torch.where(
        counts > 0,
        torch.where(values < 0, values * repetition, values / repetition) - presence,
        values,
    )
    values -= counts * frequency
    for token, value in bias.items():
        values[token] += value
    ids = [i for i in range(len(values)) if mask is None or mask[i]]
    ids.sort(key=lambda i: (-values[i].item(), i))
    if k:
        ids = ids[:k]
    mass = torch.exp((values[ids] - values[ids[0]]) / temp)
    cutoff = int(torch.searchsorted(mass.cumsum(0), mass.sum() * p))
    ids, mass = ids[: cutoff + 1], mass[: cutoff + 1]
    index = min(int(torch.searchsorted((mass / mass.sum()).cumsum(0), u, right=True)), len(ids) - 1)
    return ids[index]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sanitizer", action="store_true", help="Skip instrumented timing repetitions")
    p.add_argument("--batch-only", action="store_true")
    a = p.parse_args()
    configure()
    # Establish the primary context before NVRTC/cuda-python adapter discovery.
    torch.empty((), device="cuda")
    if a.batch_only:
        reports = check_batched(a.sanitizer)
        (a.output / "results.json").write_text(
            json.dumps(dict(environment=environment(), batched=reports), indent=2)
        )
        return
    reports = []
    for vocab in (35, 1057, 248320):
        pipe = Pipeline(vocab)
        cases = 0
        gen = torch.Generator().manual_seed(20261002)
        settings = (
            ((0, 1.0), (7, 0.8))
            if a.sanitizer
            else ((0, 1.0), (0, 0.9), (1, 0.4), (7, 0.8), (vocab, 0.95))
        )
        for k, nucleus in settings:
            for uniform in (0.17,) if a.sanitizer else (0.0, 0.17, 0.73, 0.999999):
                logits = torch.randn(vocab, generator=gen) * 3
                logits[:4] = 0  # ties, including when k cuts a tied group
                penalties = (0.2, 0.13, 1.1)
                history = [0, 0, 2, vocab - 1]
                bias = {3: 0.7, vocab - 1: -0.2}
                mask = [i % 3 != 1 for i in range(vocab)]
                pipe.inputs(
                    logits,
                    k,
                    nucleus,
                    uniform,
                    penalties=penalties,
                    history=history,
                    bias=bias,
                    mask=mask,
                )
                pipe.run()
                actual = pipe.output.cpu().tolist()
                expected = reference(
                    logits, k, nucleus, uniform, 0.8, penalties, history, bias, mask
                )
                assert actual == [expected, 0], (vocab, k, nucleus, uniform, actual, expected)
                cases += 1
        # A tied vocabulary checks exact token-ID ordering at both cutoffs.
        logits = torch.zeros(vocab)
        pipe.inputs(logits, 7, 0.5, 0.73)
        pipe.run()
        assert pipe.output.cpu().tolist() == [2, 0]
        if a.sanitizer:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pipe.run()
            latency = None
        else:
            latency, graph = benchmark(pipe.run, repetitions=10)
        pipe.inputs(torch.arange(vocab, dtype=torch.float32), 1)
        graph.replay()
        assert pipe.output.cpu().tolist() == [vocab - 1, 0]
        bad = torch.zeros(vocab)
        bad[-1] = float("nan")
        pipe.inputs(bad)
        graph.replay()
        assert pipe.output.cpu().tolist()[1] == 1
        pipe.inputs(torch.zeros(vocab), mask=[False] * vocab)
        graph.replay()
        assert pipe.output.cpu().tolist()[1] == 1
        for name, kernel in pipe.kernels.items():
            export_kernel(kernel, a.output / str(vocab) / name)
        reports.append(dict(vocab=vocab, cases=cases, changed_input_replay=True, timing=latency))
        print(json.dumps(reports[-1]), flush=True)
    (a.output / "results.json").write_text(
        json.dumps(dict(environment=environment(), cases=reports), indent=2)
    )


if __name__ == "__main__":
    main()

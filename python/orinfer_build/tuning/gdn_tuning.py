"""Select a complete request-state strategy over measured decode cohorts."""

import inspect
import math

from orinfer_build.tuning.row_tuning import launch
from orinfer_build.contracts.row_selection import aggregate_winner


def select(tuner, dense):
    import torch
    import tilelang
    from kernels.cuda.model.gdn_sequence import _gdn_sequence
    from kernels.cuda.model.gdn_deferred import recurrent
    from kernels.cuda.model.independent_rows import independent_rows
    from orinfer_build.models.qwen35.parallel import geometry

    hk, hv = tuner.hk, tuner.hv
    configs = [dict(deferred=False)] + [
        dict(deferred=True, capacity=capacity, value_tile=vt, threads=threads)
        for capacity in (2, 4, 8, 16)
        for vt, threads in [(16, 64), (32, 128), (64, 256)]
    ]
    cycles = math.lcm(*(config.get("capacity", 1) for config in configs))

    def compile_function(fn, prefetch=False):
        private = {"State", "LogK", "LogD", "LogG", "Count"}
        names = {b.name for b in fn.buffer_map.values()} & private
        columns = {name: i for i, name in enumerate(sorted(names))}
        roles = {
            b.name: ("private", b.name) if b.name in private else ("row", b.name)
            for b in fn.buffer_map.values()
        }
        fn, inputs = independent_rows(fn, 128, roles, columns, dynamic=True)
        passes = (
            {"tl.disable_thread_storage_sync": True, "tl.disable_shared_memory_reuse": True}
            if prefetch
            else {}
        )
        kernel = tilelang.compile(
            fn,
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
            pass_configs=passes,
        )
        return kernel, inputs, columns

    def build(config):
        if not config["deferred"]:
            return None
        return compile_function(
            recurrent(
                config["capacity"],
                key_heads=hk,
                value_heads=hv,
                value_tile=config["value_tile"],
                threads=config["threads"],
            )
        )

    def inputs(rows):
        config = geometry(dense, rows)
        reference = compile_function(
            inspect.unwrap(_gdn_sequence)(1, in_place=True, key_heads=hk, value_heads=hv, **config),
            config["prefetch_qk"],
        )
        q = torch.nn.functional.normalize(torch.randn(rows, hk, 1, 128, device="cuda"), dim=-1)
        values = dict(
            Q=q.half(),
            K=torch.nn.functional.normalize(torch.randn_like(q), dim=-1).half(),
            V=torch.randn(rows, hv, 1, 128, device="cuda").half() * 0.1,
            G=-torch.rand(rows, 1, hv, device="cuda") * 0.1,
            Beta=torch.rand(rows, 1, hv, device="cuda"),
        )
        initial = torch.randn(rows, hv, 128, 128, device="cuda") * 0.01
        expected = torch.empty(rows, 1, hv * 128, device="cuda", dtype=torch.float16)
        expected_state = torch.empty_like(initial)

        def runner(compiled, state, out, logs, traces=None):
            kernel, names, columns = compiled
            private = dict(State=state, **logs)
            table = torch.tensor(
                [[private[name][i].data_ptr() for name in columns] for i in range(rows)],
                device="cuda",
                dtype=torch.uint64,
            )
            args = [
                table
                if name is None
                else rows
                if name == "__rows__"
                else out
                if name == "Out"
                else values[name]
                for name in names
            ]

            def run():
                state[:rows].copy_(initial)
                if logs:
                    logs["Count"][:rows].zero_()
                # Equal complete cycles, including the dense merge. Reset and
                # state copies are charged identically to both strategies.
                for step in range(cycles):
                    launch(kernel, *args)
                    if traces is not None:
                        traces[step].copy_(out[:rows])

            return run

        run_reference = runner(reference, expected_state, expected, {})
        expected_traces = torch.empty(
            (cycles, rows, 1, hv * 128), device="cuda", dtype=torch.float16
        )
        reference_trace = runner(reference, expected_state, expected, {}, expected_traces)

        def fixture(compiled, config):
            state = torch.full((rows + 1, hv, 128, 128), 91.0, device="cuda")
            out = torch.full((rows + 1, 1, hv * 128), 91, device="cuda", dtype=torch.float16)
            logs = {}
            if config["deferred"]:
                for name, shape, dtype in [
                    ("LogK", [config["capacity"], hk, 128], torch.float16),
                    ("LogD", [config["capacity"], hv, 128], torch.float32),
                    ("LogG", [config["capacity"], hv], torch.float32),
                    ("Count", [hv, 128], torch.int32),
                ]:
                    logs[name] = torch.full((rows + 1, *shape), 91, device="cuda", dtype=dtype)
            run = runner(compiled or reference, state, out, logs)
            traces = torch.empty_like(expected_traces)
            trace = runner(compiled or reference, state, out, logs, traces)

            def check():
                run_reference()
                assert torch.equal(state[:rows], expected_state)
                assert torch.equal(out[:rows], expected)
                assert bool(torch.all(state[rows:] == 91)) and bool(torch.all(out[rows:] == 91))
                assert all(bool(torch.all(value[rows:] == 91)) for value in logs.values())
                # Validate every pending-log phase outside the timed work.
                reference_trace()
                trace()
                assert torch.equal(traces, expected_traces)
                assert torch.equal(state[:rows], expected_state)

            return run, check, lambda: values["V"].neg_()

        yield fixture

    result = tuner.select(
        dict(kind="gdn_state_strategy", key_heads=hk, value_heads=hv, cycles=cycles, dense=dense),
        configs,
        build,
        inputs,
        measure_rows=[1, 2, 3, 4, 8, 17, 32, 64, 128],
    )
    # Logs cannot change representation when batch membership changes. Select
    # one whole strategy by relative geometric mean, retaining the measured
    # dense policy if no log strategy wins across these cohorts.
    chosen = configs[int(aggregate_winner(result["measurements"]))]
    if not chosen["deferred"]:
        return dense
    return dict(configs=[chosen], default="0", selections=[])


def chunked(tuner):
    """Choose one dynamic prefill schedule; validate its FP32 state separately."""
    import torch
    from kernels.cuda.model import gdn_chunk
    from kernels.cuda.model.gdn_sequence import _gdn_sequence
    from orinfer_build.compiler.compile import primitive

    spec = dict(
        kind="gdn_chunk",
        key_heads=tuner.hk,
        value_heads=tuner.hv,
        state="fp32",
        operands="fp16",
        version=1,
    )
    key = str(spec)
    if key in tuner.records:
        return tuner.records[key]
    configs = [dict(C=c, BV=bv) for c in (16, 32) for bv in (32, 64)] + [
        dict(C=c, BV=bv, threads=threads)
        for c, bv, threads in [(16, 64, 256), (32, 128, 256), (32, 128, 512)]
    ]
    reference = primitive(
        _gdn_sequence,
        4096,
        in_place=True,
        value_tile=16,
        threads=32,
        prefetch_qk=True,
        token_unroll=8,
        dynamic_rows=True,
        key_heads=tuner.hk,
        value_heads=tuner.hv,
    )

    def build(config):
        return (
            primitive(
                gdn_chunk.prepare, 4096, HK=tuner.hk, HV=tuner.hv, C=config["C"], dynamic_rows=True
            ),
            primitive(
                gdn_chunk.recurrent, 4096, HK=tuner.hk, HV=tuner.hv, **config, dynamic_rows=True
            ),
        )

    def inputs(rows, validation=False):
        q = torch.nn.functional.normalize(
            torch.randn(tuner.hk, rows, 128, device="cuda"), dim=-1
        ).half()
        k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).half()
        v = torch.randn(tuner.hv, rows, 128, device="cuda").half() * 0.1
        g = -torch.rand(rows, tuner.hv, device="cuda") * (0.1 if not validation else 10)
        beta = torch.rand_like(g)
        if validation:
            k.copy_(k[:, :1].expand_as(k))
            beta.fill_(1)
            g[:, ::2] = 0
        initial = torch.randn(tuner.hv, 128, 128, device="cuda") * 0.01
        expected_state = torch.empty_like(initial)
        expected = torch.empty(rows, tuner.hv * 128, device="cuda", dtype=torch.float16)
        scratch = torch.empty(1, device="cuda")

        def fixture(kernels, config):
            c = config["C"]
            chunks = (rows + c - 1) // c
            w = torch.empty(tuner.hv, chunks, c, 128, device="cuda", dtype=torch.float16)
            u = torch.empty_like(w)
            d = torch.empty(tuner.hv, chunks, c, c, device="cuda", dtype=torch.float16)
            gp = torch.empty(tuner.hv, chunks, c, 2, device="cuda")
            state = torch.empty_like(initial)
            backing = torch.full((rows + 1, tuner.hv * 128), 91, device="cuda", dtype=torch.float16)
            out = backing[:rows]

            def run():
                state.copy_(initial)
                launch(kernels[0], q, k, v, g, beta, w, u, d, gp, rows)
                launch(kernels[1], q, k, w, u, d, gp, state, out, rows)

            def check():
                expected_state.copy_(initial)
                launch(reference, q, k, v, g, beta, expected_state, scratch, expected, rows)
                torch.testing.assert_close(state, expected_state, atol=5e-4, rtol=0.005)
                torch.testing.assert_close(out, expected, atol=6e-5, rtol=0.002)
                assert bool(torch.all(backing[rows:] == 91))

            return run, check, lambda: v.neg_()

        yield fixture

    result = tuner.select(
        spec,
        configs,
        build,
        inputs,
        measure_rows=[1, 17, 128, 512, 4096],
        validation_inputs=lambda rows: inputs(rows, True),
    )
    # One schedule covers all prefill lengths. These are candidates, not
    # additional runtime dispatch boundaries or per-length implementations.
    costs = {
        str(i): sum(row["times"][str(i)] for row in result["measurements"])
        for i in range(len(configs))
    }
    selected = dict(result, default=min(costs, key=costs.get), selections=[])
    tuner.records[key] = selected
    return selected

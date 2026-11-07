"""Offline selection of shared dynamic-row kernel pipelines.

The published ranges reuse Flash Next's measured row selection machinery.
Every candidate receives the same data, tail guards and changed-input graph
check. Only selected tiles are emitted; no tuning runs in the Rust runtime.
"""

import math
from itertools import chain
from pathlib import Path
from orinfer_build.compiler.compile import primitive
from orinfer_build.contracts.row_selection import aggregate_winner, ranges
from orinfer_build.common.files import atomic_json
from orinfer_build.tuning.tuning import compile_candidates, measure, read_record, record_path

ROTATION_TILES = (256, 512, 1024)
ATTENTION_PIPELINES = [
    dict(head_group=1, block_m=64, block_n=64, threads=256, stages=s) for s in (0, 1)
]
ATTENTION_TARGET = 16
ATTENTION_SPLITS = [8, 16]
GDN_AB_TILES = [
    dict(BM=16, BN=n, BK=k, threads=t)
    for n, k, t in ((8, 512, 32), (16, 256, 64), (16, 512, 64), (32, 256, 128), (64, 64, 128))
] + [dict(BM=m, BN=64, BK=128, threads=128) for m in (16, 64)]


def launch(kernel, *args):
    import torch

    kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)


class RowTuner:
    def __init__(self, output, identity, text=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.records = {}
        text = text or {}
        self.h = text.get("hidden_size", 5120)
        self.hk = text.get("linear_num_key_heads", 16)
        self.hv = text.get("linear_num_value_heads", 48)
        self.qh = text.get("num_attention_heads", 24)
        self.kvh = text.get("num_key_value_heads", 4)
        self.dimension = text.get("head_dim", 256)
        self.gated = text.get("attn_output_gate", True)

    def select(
        self,
        spec,
        configs,
        build,
        inputs,
        *,
        measure_rows=None,
        relative_tolerance=0.0,
        validation_inputs=None,
    ):
        from orinfer_build.compiler.cuda.common import SEED
        import torch

        measured = [1, 2, 3, 4, 8, 17, 64, 129, 512, 4096] if measure_rows is None else measure_rows
        if not measured or any(type(r) is not int or not 1 <= r <= 4096 for r in measured):
            raise ValueError("Invalid row tuning measurements")
        measured = sorted(set(measured))
        if not 0 <= relative_tolerance <= 0.1:
            raise ValueError("Invalid row selection tolerance")
        torch.manual_seed(SEED)
        identity = dict(
            toolchain=self.identity, spec=spec, configs=configs, seed=SEED, samples=measured
        )
        if relative_tolerance:
            identity["selection_tolerance"] = relative_tolerance
        path = record_path(self.output, identity)
        key = path.stem
        cached = read_record(path, identity)
        if cached is not None:
            return cached
        kernels, _, _ = compile_candidates(configs, build)
        records = []
        for rows in measured:
            times = {str(i): [] for i in range(len(configs))}
            fixtures = chain(
                ((True, fixture) for fixture in inputs(rows)),
                (
                    ((False, fixture) for fixture in validation_inputs(rows))
                    if validation_inputs is not None
                    else ()
                ),
            )
            for measured_case, fixture in fixtures:
                for index, pair in enumerate(kernels):
                    run, check, change = fixture(pair, configs[index])
                    run()
                    check()
                    latency = measure(
                        run,
                        budget_ms=0.25 if measured_case else 0,
                        max_repetitions=32,
                        change=change,
                        check=check,
                    )
                    if measured_case:
                        times[str(index)].append(latency)
            record = dict(
                rows=rows,
                times={k: math.exp(sum(map(math.log, v)) / len(v)) for k, v in times.items()},
            )
            records.append(record)
            # The cache record retains the full fixture identity, including
            # measured routing histograms. Keep progress output bounded.
            print("ROW TUNING", spec.get("kind", "pipeline"), key, record, flush=True)
        decisions = records
        if relative_tolerance:
            preferred = aggregate_winner(records)
            # Prefer the aggregate winner when differences are within noise.
            # Keep original measurements intact in the published tuning report.
            decisions = [
                dict(
                    rows=r["rows"],
                    times={
                        name: value * (1 if name == preferred else 1 + relative_tolerance)
                        for name, value in r["times"].items()
                    },
                )
                for r in records
            ]
        chosen = ranges(decisions, capacity=4096)
        result = dict(identity=identity, configs=configs, measurements=records, **chosen)
        atomic_json(path, result)
        return result

    def activation_rotation(self, k, swiglu, *, multiplier=1):
        """Choose one exact row transform across the full prefill extent."""
        import torch
        from kernels.cuda.model.rotation_a8 import rotate_activation_a8

        if type(multiplier) is not int or multiplier <= 0:
            raise ValueError("Invalid activation row multiplier")
        spec = dict(kind="activation_rotation", k=k, swiglu=swiglu, multiplier=multiplier)
        key = str(spec)
        if key in self.records:
            return self.records[key]
        configs = [dict(registers=value) for value in (False, True)]

        def build(config):
            return primitive(
                rotate_activation_a8,
                4096 * multiplier,
                k,
                swiglu=swiglu,
                dynamic_rows=True,
                **config,
            )

        reference = build(configs[0])

        def inputs(rows):
            live = rows * multiplier
            x = torch.randn((live, k * (2 if swiglu else 1)), device="cuda", dtype=torch.float16)
            signs = torch.randint(0, 2, (k,), device="cuda", dtype=torch.int8) * 2 - 1
            eq = torch.empty((live, k), device="cuda", dtype=torch.int8)
            es = torch.empty(live, device="cuda", dtype=torch.float16)

            def fixture(kernel, config):
                storage = torch.full((live + 1, k), 91, device="cuda", dtype=torch.int8)
                scales = torch.full((live + 1,), 91, device="cuda", dtype=torch.float16)
                q, s = storage[:live], scales[:live]

                def run():
                    launch(kernel, x, signs, q, s, live)

                def check():
                    launch(reference, x, signs, eq, es, live)
                    assert torch.equal(q, eq) and torch.equal(s, es)
                    assert bool(torch.all(storage[live:] == 91)) and bool(
                        torch.all(scales[live:] == 91)
                    )

                return run, check, lambda: x.neg_()

            yield fixture

        result = self.select(
            spec, configs, build, inputs, measure_rows=[1, 17, 128, 512, 2048, 4096]
        )
        default = min(
            result["used"],
            key=lambda name: sum(row["times"][name] for row in result["measurements"]),
        )
        result = dict(result, default=default, selections=[], used=[default])
        self.records[key] = result
        return result

    def rotation(self, k, swiglu, *, plane_rows=0):
        from kernels.cuda.model.rotation_a8 import (
            rotate_activation_tiles,
            quantize_rotation_tiles,
            rotate_activation_a8,
        )
        import torch

        spec = dict(kind="rotation", k=k, swiglu=swiglu, plane_rows=plane_rows)
        if str(spec) in self.records:
            return self.records[str(spec)]
        configs = [dict(block_k=b) for b in ROTATION_TILES] + [
            dict(block_k=b, registers=True, threads=t)
            for b, t in ((512, 128), (1024, 128), (1024, 256), (2048, 256))
        ]
        reference = primitive(rotate_activation_a8, 4096, k, swiglu=swiglu, dynamic_rows=True)

        def build(c):
            b = c["block_k"]
            return (
                primitive(
                    rotate_activation_tiles,
                    4096,
                    k,
                    b,
                    swiglu,
                    dynamic_rows=True,
                    plane_rows=plane_rows,
                    **{key: value for key, value in c.items() if key != "block_k"},
                ),
                primitive(quantize_rotation_tiles, 4096, k, b, dynamic_rows=True),
            )

        def inputs(rows):
            shape = (plane_rows + rows, k) if plane_rows else (rows, k * (2 if swiglu else 1))
            x = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1
            signs = torch.randint(0, 2, (k,), device="cuda", dtype=torch.int8) * 2 - 1
            eq = torch.empty((rows, k), device="cuda", dtype=torch.int8)
            es = torch.empty(rows, device="cuda", dtype=torch.float16)

            def fixture(pair, c):
                first, second = pair
                b = c["block_k"]
                rotated = torch.empty((rows, k), device="cuda", dtype=torch.float16)
                maxima = torch.empty((rows, (k + b - 1) // b), device="cuda")
                storage = torch.full((rows + 1, k), 91, device="cuda", dtype=torch.int8)
                scales = torch.full((rows + 1,), 91, device="cuda", dtype=torch.float16)
                q, s = storage[:rows], scales[:rows]

                def run():
                    launch(first, x, signs, rotated, maxima, rows)
                    launch(second, rotated, maxima, q, s, rows)

                def check():
                    original = (
                        torch.cat((x[:rows], x[plane_rows : plane_rows + rows]), dim=1)
                        if plane_rows
                        else x
                    )
                    launch(reference, original, signs, eq, es, rows)
                    assert torch.equal(q, eq) and torch.equal(s, es), (
                        "Rotation pipeline changed codes/scales"
                    )
                    assert bool(torch.all(storage[rows:] == 91)) and bool(
                        torch.all(scales[rows:] == 91)
                    )

                return run, check, lambda: x.neg_()

            yield fixture

        result = self.select(spec, configs, build, inputs)
        self.records[str(spec)] = result
        return result

    def attention(self, context):
        spec = dict(
            kind="attention",
            context=context,
            query_heads=self.qh,
            kv_heads=self.kvh,
            dimension=self.dimension,
            gated=self.gated,
            target_tiles=ATTENTION_TARGET,
            max_splits=ATTENTION_SPLITS,
        )
        if str(spec) in self.records:
            return self.records[str(spec)]
        configs = [
            dict(head_group=g, block_m=m, block_n=n, max_splits=s)
            for s in ATTENTION_SPLITS
            for g in (1, self.qh // self.kvh)
            for m in (16, 32, 64)
            for n in (32, 64)
        ] + [c | dict(max_splits=s) for s in ATTENTION_SPLITS for c in ATTENTION_PIPELINES]
        # Include the unsplit reference geometry in the same computation body.
        # This removes split bookkeeping at compile time, without a row cutoff.
        configs.append(dict(head_group=1, block_m=32, block_n=32, max_splits=1))
        result = self._measure_attention(context, spec, configs)
        self.records[str(spec)] = result
        return result

    def causal_attention(self, context):
        from orinfer_build.models.qwen35.attention import select
        from orinfer_build.models.qwen35.parallel import geometry

        key = str(("independent_query_attention", context))
        if key not in self.records:
            self.records[key] = select(self, context, geometry(self.attention(context)))
        return self.records[key]

    def staged_attention(self, context, selection):
        """Measure complete INT8 and temporary-FP16 KV pipelines at every row extent."""
        from orinfer_build.contracts.row_selection import samples

        names = dict.fromkeys(
            [
                selection["default"],
                *(choice["implementation"] for choice in selection["selections"]),
            ]
        )
        configs = [selection["configs"][int(name)] for name in names]
        configs += [
            dict(
                block_m=m,
                block_n=64,
                head_group=g,
                threads=t,
                stages=1,
                max_splits=1,
                decoded_kv=True,
            )
            for m, g, t in [(64, 1, 256), (128, 2, 512)]
            if (self.qh // self.kvh) % g == 0
        ]
        spec = dict(
            kind="staged_attention",
            context=context,
            query_heads=self.qh,
            kv_heads=self.kvh,
            dimension=self.dimension,
            gated=self.gated,
            candidates=configs,
        )
        key = str(spec)
        if key not in self.records:
            # Decoder launch, attention and merge are timed together. Small
            # prefill tails compete with large extents; no row cutoff is assumed.
            self.records[key] = self._measure_attention(
                context, spec, configs, measure_rows=samples(4096)
            )
        return self.records[key]

    def _measure_attention(self, context, spec, configs, *, measure_rows=None):
        from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
        from kernels.cuda.model.kv_int8 import dequantize_attention_kv
        from orinfer_build.validation.attention_reference import attention_prefill_int8
        import torch

        reference = primitive(
            attention_prefill_int8,
            1,
            4096,
            context,
            dynamic_rows=True,
            query_heads=self.qh,
            kv_heads=self.kvh,
            dimension=self.dimension,
            gated=self.gated,
        )

        def build(c):
            return (
                primitive(
                    causal_attention_int8,
                    4096,
                    context,
                    c["block_m"],
                    c["block_n"],
                    ATTENTION_TARGET,
                    c["max_splits"],
                    head_group=c["head_group"],
                    query_heads=self.qh,
                    kv_heads=self.kvh,
                    dimension=self.dimension,
                    gated=self.gated,
                    threads=c.get("threads", 128),
                    stages=c.get("stages", 0),
                    decoded_kv=c.get("decoded_kv", False),
                ),
                primitive(
                    merge_attention_int8,
                    4096,
                    c["block_m"],
                    ATTENTION_TARGET,
                    c["max_splits"],
                    head_group=c["head_group"],
                    query_heads=self.qh,
                    kv_heads=self.kvh,
                    dimension=self.dimension,
                    gated=self.gated,
                ),
                primitive(
                    dequantize_attention_kv,
                    4096,
                    context,
                    kv_heads=self.kvh,
                    dimension=self.dimension,
                )
                if c.get("decoded_kv")
                else None,
            )

        def inputs(rows):
            # Autoregressive positions usually end inside a tile. Exact tile
            # multiples alone hide loading/masking and instruction-cache costs.
            for length in sorted({max(rows, 536), max(rows, 8216)}):
                q = (
                    torch.randn((rows, self.qh, self.dimension), device="cuda", dtype=torch.float16)
                    * 0.1
                )
                gate = torch.randn_like(q)
                kv = [
                    torch.randint(
                        -127,
                        128,
                        (1, context, self.kvh, self.dimension),
                        device="cuda",
                        dtype=torch.int8,
                    )
                    for _ in range(2)
                ]
                scales = [
                    torch.full(
                        (1, context, self.kvh, self.dimension // 64),
                        0.001,
                        device="cuda",
                        dtype=torch.float16,
                    )
                    for _ in range(2)
                ]
                pos = torch.arange(length - rows, length, device="cuda", dtype=torch.int32)
                lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
                expected = torch.empty_like(q)

                def fixture(
                    pair,
                    c,
                    q=q,
                    gate=gate,
                    kv=kv,
                    scales=scales,
                    pos=pos,
                    lengths=lengths,
                    expected=expected,
                ):
                    first, second, decoder = pair
                    bm = c["block_m"]
                    heads = self.qh // c["head_group"]
                    partial = torch.empty(
                        (heads, 2 * ATTENTION_TARGET, bm, self.dimension), device="cuda"
                    )
                    maximum = torch.empty((heads, 2 * ATTENTION_TARGET, bm), device="cuda")
                    denom = torch.empty_like(maximum)
                    storage = torch.full(
                        (rows + 1, self.qh, self.dimension), 91, device="cuda", dtype=torch.float16
                    )
                    out = storage[:rows]
                    decoded = (
                        [
                            torch.empty(
                                (1, context, self.kvh * self.dimension),
                                device="cuda",
                                dtype=torch.float16,
                            )
                            for _ in range(2)
                        ]
                        if decoder
                        else []
                    )

                    def run():
                        if decoder:
                            launch(decoder, *kv, *scales, lengths, *decoded, rows)
                        launch(
                            first,
                            q,
                            *(decoded if decoder else kv),
                            *scales,
                            gate,
                            pos,
                            lengths,
                            out,
                            partial,
                            maximum,
                            denom,
                            rows,
                        )
                        launch(second, partial, maximum, denom, gate, out, rows)

                    def check():
                        launch(reference, q, *kv, gate, pos, lengths, expected, *scales, rows)
                        torch.testing.assert_close(out, expected, atol=4e-5, rtol=0.005)
                        assert bool(torch.all(storage[rows:] == 91))

                    def change():
                        q.neg_()
                        kv[1].neg_()

                    return run, check, change

                yield fixture

        return self.select(spec, configs, build, inputs, measure_rows=measure_rows)

    def gdn_ab(self):
        """Tune CTA coverage while preserving the FP16 MMA K accumulation order."""
        import tilelang.language as T
        import torch
        from kernels.cuda.operators.op07_gdn_ab import gdn_ab_tensorcore

        spec = dict(kind="gdn_ab", k=self.h, n=2 * self.hv)
        if str(spec) in self.records:
            return self.records[str(spec)]
        reference = primitive(gdn_ab_tensorcore, T.dynamic("m"), N=2 * self.hv, K=self.h)

        def build(config):
            return primitive(gdn_ab_tensorcore, T.dynamic("m"), N=2 * self.hv, K=self.h, **config)

        def inputs(rows):
            x = torch.randn((rows, self.h), device="cuda", dtype=torch.float16) * 0.1
            w = torch.randn((2 * self.hv, self.h), device="cuda", dtype=torch.float16) * 0.1
            expected = torch.empty((rows, 2 * self.hv), device="cuda", dtype=torch.float16)

            def fixture(kernel, _config):
                storage = torch.full(
                    (rows + 1, 2 * self.hv), 91, device="cuda", dtype=torch.float16
                )
                out = storage[:rows]

                def run():
                    launch(kernel, x, w, out)

                def check():
                    launch(reference, x, w, expected)
                    assert torch.equal(out, expected), "GDN A/B tile changed the FP16 result"
                    assert bool(torch.all(storage[rows:] == 91))

                return run, check, lambda: x.neg_()

            yield fixture

        result = self.select(spec, GDN_AB_TILES, build, inputs)
        self.records[str(spec)] = result
        return result

    def gdn_convolution(self):
        """Choose time tiles while preserving the native per-row reduction."""
        import torch
        from kernels.cuda.operators.op08_gdn_conv_prep import gdn_conv_prep
        from orinfer_build.contracts.projection_policy import SPECIALIZATIONS

        spec = dict(kind="gdn_convolution", key_heads=self.hk, value_heads=self.hv)
        if str(spec) in self.records:
            return self.records[str(spec)]
        configs = [dict(tile_tokens=t) for t in (1, 2, 4, 8, 16)]
        channels = (2 * self.hk + self.hv) * 128

        def build(config):
            return primitive(
                gdn_conv_prep,
                1,
                4096,
                dynamic_tokens=True,
                key_heads=self.hk,
                value_heads=self.hv,
                **config,
            )

        reference = build(configs[-1])

        def inputs(rows):
            x = torch.randn((1, rows, channels), device="cuda", dtype=torch.float16)
            w = torch.randn((channels, 4), device="cuda", dtype=torch.float16) * 0.1
            history = torch.randn((1, 3, channels), device="cuda", dtype=torch.float16)
            length = torch.full((1,), rows, device="cuda", dtype=torch.int32)
            position = torch.full_like(length, 512)

            def outputs():
                return [
                    torch.empty((1, h, rows, 128), device="cuda", dtype=torch.float16)
                    for h in (self.hk, self.hk, self.hv)
                ] + [torch.empty_like(history), torch.empty_like(position)]

            expected = outputs()

            def fixture(kernel, _config):
                shapes = [tuple(value.shape) for value in expected]
                storage = [
                    torch.full((value.numel() + 128,), 91, device="cuda", dtype=value.dtype)
                    for value in expected
                ]
                actual = [
                    s[: v.numel()].view(shape) for s, v, shape in zip(storage, expected, shapes)
                ]

                def run():
                    launch(kernel, x, w, history, length, position, *actual)

                def check():
                    launch(reference, x, w, history, length, position, *expected)
                    for value, wanted, allocated in zip(actual, expected, storage):
                        assert torch.equal(value, wanted), "Convolution tile changed output/history"
                        assert bool(torch.all(allocated[value.numel() :] == 91))

                return run, check, lambda: x.neg_()

            yield fixture

        # Include the exact AOT rows: a tail at 17 can favour a smaller tile,
        # while an aligned 32-row chunk needs its own measured cost.
        measured = sorted(set(SPECIALIZATIONS) | {17, 64, 129, 512, 4096})
        result = self.select(spec, configs, build, inputs, measure_rows=measured)
        self.records[str(spec)] = result
        return result

    def gdn_sequence(self):
        """Select loop unrolling without changing the sequential FP32 recurrence."""
        import torch
        from kernels.cuda.model.gdn_sequence import _gdn_sequence

        spec = dict(
            kind="gdn_sequence",
            key_heads=self.hk,
            value_heads=self.hv,
            value_tile=16,
            threads=32,
            prefetch_qk=True,
        )
        if str(spec) in self.records:
            return self.records[str(spec)]
        configs = [dict(token_unroll=u) for u in (1, 2, 4, 8)]

        def build(config):
            return primitive(
                _gdn_sequence,
                4096,
                in_place=True,
                value_tile=16,
                threads=32,
                prefetch_qk=True,
                dynamic_rows=True,
                key_heads=self.hk,
                value_heads=self.hv,
                **config,
            )

        reference = build(configs[0])

        def inputs(rows):
            q = torch.nn.functional.normalize(
                torch.randn((self.hk, rows, 128), device="cuda"), dim=-1
            ).half()
            k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).half()
            v = torch.randn((self.hv, rows, 128), device="cuda", dtype=torch.float16) * 0.1
            g = -torch.rand((rows, self.hv), device="cuda") * 0.1
            beta = torch.rand_like(g)
            initial = torch.randn((self.hv, 128, 128), device="cuda") * 0.01
            expected_state = torch.empty_like(initial)
            expected = torch.empty((rows, self.hv * 128), device="cuda", dtype=torch.float16)
            scratch = torch.empty(1, device="cuda")

            def fixture(kernel, _config):
                state = torch.empty_like(initial)
                storage = torch.full(
                    (rows + 1, self.hv * 128), 91, device="cuda", dtype=torch.float16
                )
                out = storage[:rows]

                def run():
                    state.copy_(initial)
                    launch(kernel, q, k, v, g, beta, state, scratch, out, rows)

                def check():
                    expected_state.copy_(initial)
                    launch(reference, q, k, v, g, beta, expected_state, scratch, expected, rows)
                    assert torch.equal(state, expected_state) and torch.equal(out, expected), (
                        "GDN unrolling changed the FP32 state or FP16 output"
                    )
                    assert bool(torch.all(storage[rows:] == 91))

                return run, check, lambda: v.neg_()

            yield fixture

        result = self.select(spec, configs, build, inputs)
        self.records[str(spec)] = result
        return result

    def private_gdn(self):
        from orinfer_build.tuning.gdn_tuning import select

        return select(self, self._private_gdn_dense())

    def _private_gdn_dense(self):
        """Measure request-parallel ownership with the same FP32 sum tree."""
        import torch
        import tilelang
        from kernels.cuda.model.gdn_sequence import _gdn_sequence
        from kernels.cuda.model.independent_rows import independent_rows

        spec = dict(kind="private_gdn", key_heads=self.hk, value_heads=self.hv)
        if str(spec) in self.records:
            return self.records[str(spec)]
        configs = [
            dict(value_tile=16, threads=32, prefetch_qk=True),
            dict(value_tile=32, threads=128, prefetch_qk=False),
            dict(value_tile=16, threads=64, prefetch_qk=False, shared_state=True),
            dict(value_tile=64, threads=256, prefetch_qk=False, shared_state=True),
        ]

        def build(config):
            fn = _gdn_sequence(1, in_place=True, key_heads=self.hk, value_heads=self.hv, **config)
            roles = {
                b.name: ("private", "State") if b.name == "State" else ("row", b.name)
                for b in fn.buffer_map.values()
            }
            fn, names = independent_rows(fn, 128, roles, {"State": 0}, dynamic=True)
            kernel = tilelang.compile(
                fn,
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
                pass_configs={
                    "tl.disable_thread_storage_sync": True,
                    "tl.disable_shared_memory_reuse": True,
                }
                if config["prefetch_qk"]
                else {},
            )
            return kernel, names

        reference = build(configs[0])

        def inputs(rows):
            q = torch.nn.functional.normalize(
                torch.randn((rows, self.hk, 1, 128), device="cuda"), dim=-1
            ).half()
            values = dict(
                Q=q,
                K=torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).half(),
                V=torch.randn((rows, self.hv, 1, 128), device="cuda", dtype=torch.float16) * 0.1,
                G=-torch.rand((rows, 1, self.hv), device="cuda") * 0.1,
                Beta=torch.rand((rows, 1, self.hv), device="cuda"),
            )
            initial = torch.randn((rows, self.hv, 128, 128), device="cuda") * 0.01

            def fixture(pair, _config):
                state = torch.full((rows + 1, self.hv, 128, 128), 91.0, device="cuda")
                out = torch.full(
                    (rows + 1, 1, self.hv * 128), 91.0, device="cuda", dtype=torch.float16
                )
                expected_state = torch.empty_like(initial)
                expected = torch.empty_like(out[:rows])

                def invoke(compiled, current, output):
                    kernel, names = compiled
                    pointers = torch.tensor(
                        [[current[i].data_ptr()] for i in range(rows)],
                        dtype=torch.uint64,
                        device="cuda",
                    )
                    return kernel, [
                        pointers
                        if name is None
                        else rows
                        if name == "__rows__"
                        else output
                        if name == "Out"
                        else values[name]
                        for name in names
                    ]

                kernel, args = invoke(pair, state, out)
                original, ref_args = invoke(reference, expected_state, expected)

                def run():
                    state[:rows].copy_(initial)
                    launch(kernel, *args)

                def check():
                    expected_state.copy_(initial)
                    launch(original, *ref_args)
                    assert torch.equal(state[:rows], expected_state) and torch.equal(
                        out[:rows], expected
                    ), "Private GDN changed the FP32 state or FP16 output"
                    assert bool(torch.all(state[rows:] == 91)) and bool(torch.all(out[rows:] == 91))

                return run, check, lambda: values["V"].neg_()

            yield fixture

        result = self.select(
            spec,
            configs,
            build,
            inputs,
            measure_rows=[1, 2, 3, 4, 8, 17, 32, 64, 128],
            relative_tolerance=0.03,
        )
        self.records[str(spec)] = result
        return result

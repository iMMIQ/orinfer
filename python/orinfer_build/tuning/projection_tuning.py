"""Offline TileLang autotuning with fixed inputs and exact execution checks.

Only integer dot-product tiles are searched. Scales, output rounding and the
model's floating-point reductions are not tuning parameters. No Torch or
autotuner dependency enters the published Rust runtime.
"""

from orinfer_build.common.paths import ROOT


import json
import math
from pathlib import Path

from orinfer_build.common.files import atomic_json, source_hash
from orinfer_build.tuning.tuning import compile_candidates, measure, read_record, record_path

from orinfer_build.contracts.projection_policy import (
    q4_configs,
    e8p_q4_configs,
    w8_configs,
    swiglu_configs,
    candidate_scores,
)
from orinfer_build.tuning.expert_tuning import (
    EXACT_EXPERT_PASSES as EXACT_PASSES,
    expert_configs,
    expert_projection,
)


TARGET = {"kind": "cuda", "arch": "sm_87"}


def profile_rows(capacity):
    """Geometric coverage of a dynamic allocation, including non-AOT tails."""
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Invalid dynamic tuning capacity")
    return sorted(
        {1, capacity, min(capacity, 5), min(capacity, 17)}
        | {round(math.exp(math.log(capacity) * i / 5)) for i in range(1, 5)}
    )


def measurement_rows(capacity, dynamic, requested):
    if requested is None:
        return profile_rows(capacity) if dynamic else [capacity]
    if (
        not dynamic
        or not requested
        or any(type(r) is not int or not 1 <= r <= capacity for r in requested)
    ):
        raise ValueError("Explicit measurement rows require valid dynamic extents")
    return sorted(set(requested))


def make_prim(spec, config):
    from kernels.shared.rows import explicit_rows

    rows, n, k = spec["rows"], spec["outputs"], spec["inputs"]
    dynamic = spec["dynamic"]
    if spec["kind"] in ("q4", "e8p_q4"):
        options = {}
        if spec["kind"] == "e8p_q4":
            from kernels.cuda.model.e8p_q4 import projection as q4_decode

            q4_decode = q4_decode.__wrapped__
            options = {k: spec[k] for k in ("packed_k", "basis_nonzero") if k in spec}
        else:
            from kernels.cuda.model.q4_projection import q4_decode

        fn = q4_decode(
            rows,
            n,
            k,
            output_dtype=spec["dtype"],
            dynamic_rows=dynamic,
            output_stride=spec.get("output_stride"),
            **config,
            **options,
        )
    elif spec["kind"] == "w8":
        from kernels.cuda.model.int8_projection import int8_projection, int8_small_projection

        if config.get("algorithm") == "dp4a":
            fn = int8_small_projection.__wrapped__(
                rows,
                n,
                k,
                spec["dtype"],
                dynamic,
                block_m=config["block_m"],
                block_n=config["block_n"],
                output_stride=spec.get("output_stride"),
            )
        else:
            knobs = {key: value for key, value in config.items() if key != "algorithm"}
            fn = int8_projection.__wrapped__(
                rows,
                n,
                k,
                spec["dtype"],
                dynamic_rows=dynamic,
                output_stride=spec.get("output_stride"),
                **knobs,
            )
    elif spec["kind"] == "swiglu":
        from kernels.cuda.model.int8_swiglu import int8_swiglu

        fn = int8_swiglu.__wrapped__(
            rows,
            n,
            k,
            config["block_n"],
            dynamic,
            algorithm=config["algorithm"],
            block_m=config["block_m"],
        )
    elif spec["kind"] == "expert":
        fn = expert_projection(
            rows,
            n,
            k,
            spec["experts"],
            10,
            codec="e8p_q2",
            config=config,
            dynamic_rows=dynamic,
        )
    else:
        raise ValueError("Unsupported decode tuning operator")
    return explicit_rows(fn) if dynamic else fn


def factory(spec):
    # TileLang includes primitive closure values in its compiled/tuning cache
    # key. Avoid capturing mutable tensors or factories in the kernel closure.
    encoded = json.dumps(spec, sort_keys=True)

    def build(
        algorithm="tensor_core",
        block_m=16,
        block_n=64,
        block_k=128,
        stages=2,
        split_k=1,
        group_m=0,
        tail_m=0,
        cache_scales=False,
    ):
        decoded = json.loads(encoded)
        cfg = dict(block_m=block_m)
        if decoded["kind"] != "expert":
            cfg.update(algorithm=algorithm, block_n=block_n, block_k=block_k)
        if decoded["kind"] in ("q4", "e8p_q4"):
            cfg["stages"] = stages
        if decoded["kind"] == "e8p_q4" and algorithm == "int4_direct":
            cfg["split_k"] = split_k
        if decoded["kind"] in ("q4", "e8p_q4", "w8") and algorithm == "tensor_core":
            cfg.update({k: v for k, v in dict(group_m=group_m, tail_m=tail_m).items() if v})
        if decoded["kind"] in ("w8", "e8p_q4") and algorithm == "tensor_core":
            if cache_scales:
                cfg["cache_scales"] = True
        return make_prim(decoded, cfg)

    return build


def profile_stream(kernel):
    """Use the capture stream even when an NVRTC adapter defaults to stream 0."""
    import torch

    launch = kernel.adapter.func

    def run(*inputs, stream=None):
        return launch(
            *inputs, stream=torch.cuda.current_stream().cuda_stream if stream is None else stream
        )

    kernel.adapter.func = run
    return kernel


class ProjectionTuner:
    def __init__(self, output: Path, cache: Path | None = None):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = cache or output / "tuning-cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def _tune(
        self,
        kind,
        rows,
        outputs,
        inputs,
        dtype="float16",
        *,
        dynamic=False,
        experts=512,
        block_m=None,
        weights=None,
        weight_identity=None,
        measure_rows=None,
        output_stride=None,
        packed_k=0,
        objective="relative_geomean",
        candidate_configs=None,
    ):
        import tilelang
        import torch
        from tilelang.autotuner import AutoTuner
        from orinfer_build.compiler.cuda.common import SEED

        if objective not in ("relative_geomean", "total_latency"):
            raise ValueError("Unknown dynamic tuning objective")
        spec = dict(
            kind=kind,
            rows=rows,
            outputs=outputs,
            inputs=inputs,
            dtype=dtype,
            dynamic=dynamic,
            objective=objective,
        )
        if output_stride is not None:
            if (
                kind not in ("q4", "e8p_q4", "w8")
                or type(output_stride) is not int
                or output_stride < outputs
            ):
                raise ValueError("Only Q4/W8 projections support a larger output row stride")
            spec["output_stride"] = output_stride
        if kind == "expert":
            spec["experts"] = experts
        if kind == "e8p_q4":
            from orinfer_build.quantization.e8p import basis
            from orinfer_build.quantization.vq import (
                e8p_selector_table,
                e8p_selector_supports_parity_xor,
            )

            book = (
                weights["Book"].cpu().numpy()
                if weights is not None
                else e8p_selector_table(basis())
            )
            spec.update(packed_k=packed_k, basis_nonzero=e8p_selector_supports_parity_xor(book))
            configs = e8p_q4_configs(rows, direct=bool(packed_k and spec["basis_nonzero"]))
            configs = [c for c in configs if inputs % (c["block_k"] * c.get("split_k", 1)) == 0]
        elif kind == "q4":
            configs = q4_configs(rows)
        elif kind == "w8":
            configs = w8_configs(rows, outputs)
        elif kind == "swiglu":
            configs = swiglu_configs(rows)
        elif kind == "expert":
            configs = expert_configs("e8p_q2", block_m=block_m)
        else:
            raise ValueError("Unsupported decode tuning operator")
        if candidate_configs is not None:
            if not candidate_configs or any(c not in configs for c in candidate_configs):
                raise ValueError("Candidate subset must contain legal projection tiles")
            configs = list(candidate_configs)
        samples = measurement_rows(rows, dynamic, measure_rows)
        sample = samples[0]
        root = ROOT
        # Include inlined components as well as public wrappers. A change to a
        # shared loader/mainloop/epilogue must invalidate every consumer's tile.
        from orinfer_build.common.sources import source_closure

        sources = [
            *source_closure(["orinfer_build.tuning.projection_tuning"], root),
            root / "tools/environments/cuda/uv.lock",
        ]
        identity = dict(
            spec=spec,
            configs=configs,
            samples=samples,
            seed=SEED,
            source=source_hash(root, sources),
            weights=weight_identity,
            tilelang=tilelang.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
        )
        cached = record_path(self.cache, identity)
        key = cached.stem
        passes = EXACT_PASSES if kind == "expert" else {}
        record = read_record(
            cached, identity, lambda r: r["config"] in configs and r["checked_rows"]
        )
        if record is not None:
            kernel = tilelang.compile(
                make_prim(spec, record["config"]),
                out_idx=[],
                execution_backend="nvrtc",
                target=TARGET,
                pass_configs=passes,
            )
        elif dynamic:
            candidates, legal, rejected = compile_candidates(
                configs,
                lambda cfg: tilelang.compile(
                    make_prim(spec, cfg),
                    out_idx=[],
                    execution_backend="nvrtc",
                    target=TARGET,
                    pass_configs=passes,
                ),
                tolerate_failures=True,
            )
            invalid = set()
            timings = []
            for live in samples:
                supply, reset, check = self._inputs(spec, live, weights)
                times = []
                for index, candidate in enumerate(candidates):
                    if index in invalid:
                        times.append(None)
                        continue
                    args = supply(candidate.get_profiler()._get_params(with_output=False))

                    def run(candidate=candidate, args=args):
                        candidate.adapter.func(
                            *args, stream=torch.cuda.current_stream().cuda_stream
                        )

                    reset()
                    run()
                    torch.cuda.synchronize()
                    try:
                        check()
                    except AssertionError as error:
                        invalid.add(index)
                        rejected.append(
                            dict(
                                config=legal[index],
                                stage="validation",
                                rows=live,
                                reason=str(error),
                            )
                        )
                        times.append(None)
                        continue
                    # Large dynamic tiles can take tens of milliseconds. Spend
                    # the same minimum timing budget on every candidate rather
                    # than replaying the slowest fallback eight times per trial.
                    times.append(measure(run, budget_ms=1.0, max_repetitions=8))
                timings.append(dict(rows=live, times_ms=times))
            scores = candidate_scores(timings, invalid, objective)
            if len(invalid) == len(legal):
                raise ValueError(f"All dynamic tile candidates failed numerical checks: {rejected}")
            selected = min(range(len(legal)), key=scores.__getitem__)
            kernel = candidates[selected]
            record = dict(
                identity=identity,
                config=legal[selected],
                candidates=legal,
                rejected=rejected,
                profiles=timings,
                relative_cost=scores[selected],
            )
            checked = sorted(set(samples) | set(profile_rows(rows)))
            for live in checked:
                self._validate(kernel, spec, record["config"], live, weights)
            record["checked_rows"] = checked
            atomic_json(cached, record)
        else:
            supply, reference, check = self._inputs(spec, sample, weights)
            tuner = AutoTuner.from_kernel(factory(dict(spec, tuning_identity=key)), configs)
            tuner.set_compile_args(
                out_idx=[], target=TARGET, execution_backend="nvrtc", pass_configs=passes
            )
            tuner.set_profile_args(
                supply_prog=supply,
                ref_prog=reference,
                manual_check_prog=check,
                cache_input_tensors=False,
                backend="cudagraph",
            )

            def compile_candidate(**config):
                return profile_stream(tuner._default_compile(**config))

            tuner.jit_compile = compile_candidate
            result = tuner.run(warmup=3, rep=20, timeout=120)
            kernel = profile_stream(result.kernel)
            record = dict(identity=identity, config=result.config, latency_ms=result.latency)
            # The native tuner has validated its profiling inputs; validate
            # additional runtime tails and replay using the selected geometry.
            checked = []
            for live in sorted(set([sample, 1, 5, 7, 9, 33, 127, rows])) if dynamic else [rows]:
                if live > rows:
                    continue
                self._validate(kernel, spec, record["config"], live, weights)
                checked.append(live)
            record["checked_rows"] = checked
            atomic_json(cached, record)
        self.results[key] = record
        atomic_json(self.output / "tuning.json", list(self.results.values()))
        return kernel, record

    def compile(self, *args, **kwargs):
        kernel, record = self._tune(*args, **kwargs)
        return kernel, record["config"]

    def compile_choices(self, *args, fixed=(), **kwargs):
        """Bounded AOT row choices from the same dynamic projection body.

        The caller supplies the bounded allocation and retains its fallback.
        Every emitted interval comes from measured, numerically checked
        candidates, rather than a hand-written batch threshold.
        """
        import tilelang
        from orinfer_build.contracts.row_selection import choices, ranges

        if not kwargs.get("dynamic"):
            raise ValueError("Bounded choices require runtime row arguments")
        _, record = self._tune(*args, **kwargs)
        spec = record["identity"]["spec"]
        capacity = spec["rows"]
        measurements = [
            dict(
                rows=p["rows"],
                times={
                    str(i): value
                    for i, value in enumerate(p["times_ms"])
                    if value is not None
                    and all(q["times_ms"][i] is not None for q in record["profiles"])
                },
            )
            for p in record["profiles"]
        ]
        selection = dict(ranges(measurements, capacity), configs=record["candidates"])
        compiled = {}
        for suffix, config in choices(selection, capacity, fixed, bounded=True):
            key = json.dumps(config, sort_keys=True)
            if key not in compiled:
                kernel = tilelang.compile(
                    make_prim(spec, config),
                    out_idx=[],
                    execution_backend="nvrtc",
                    target=TARGET,
                )
                for live in record["checked_rows"]:
                    self._validate(kernel, spec, config, live, kwargs.get("weights"))
                compiled[key] = kernel
            yield suffix, config, compiled[key]

    @staticmethod
    def _inputs(spec, live, weights=None):
        """Same mathematical inputs for all candidates; expert maps match tiles."""
        import torch
        from kernels.cuda.model.q4_projection import q4_gemv
        from kernels.cuda.model.e8p_exact import split_a8, e8p_exact_grouped
        from kernels.cuda.model.int8_projection import int8_projection
        from orinfer_build.compiler.cuda.common import SEED

        torch.manual_seed(SEED)
        n, k = spec["outputs"], spec["inputs"]
        device = "cuda"
        a = torch.randint(-128, 128, (live, k), device=device, dtype=torch.int8)
        scales = torch.full((live,), 0.001, device=device, dtype=torch.float16)
        dtype = getattr(torch, spec["dtype"])
        stride = spec.get("output_stride", n)
        storage = torch.full((live + 2, stride), 91, device=device, dtype=dtype)
        out = storage[:live, :n]
        expected = torch.empty_like(out)
        kind = spec["kind"]
        if kind == "q4":
            weight = (
                weights["W"]
                if weights is not None
                else torch.randint(0, 65536, (n, k // 4), device=device, dtype=torch.int32).to(
                    torch.uint16
                )
            )
            ws = (
                weights["WS"]
                if weights is not None
                else torch.full((n,), 0.001, device=device, dtype=torch.float16)
            )
            base = q4_gemv(live, n, k, output_dtype=spec["dtype"], block_m=min(live, 8))
            base.adapter.func(
                a, weight, ws, scales, expected, stream=torch.cuda.current_stream().cuda_stream
            )
            args = [a, weight, ws, scales, out]
        elif kind == "e8p_q4":
            from kernels.cuda.model.e8p_q4 import unpack
            from orinfer_build.quantization.e8p import basis
            from orinfer_build.quantization.vq import e8p_selector_table

            weight = (
                weights["W"]
                if weights is not None
                else torch.randint(0, 2**32, (n, k // 8), device=device, dtype=torch.int64).to(
                    torch.uint32
                )
            )
            ws = (
                weights["WS"]
                if weights is not None
                else torch.full((n,), 0.001, device=device, dtype=torch.float16)
            )
            book = (
                weights["Book"]
                if weights is not None
                else torch.from_numpy(e8p_selector_table(basis())).cuda()
            )
            packed_k = spec.get("packed_k", 0)
            if weights is None and packed_k:
                weight = (
                    weight.reshape(n, k // packed_k, packed_k // 8).permute(1, 0, 2).contiguous()
                )
            expected_shape = (k // packed_k, n, packed_k // 8) if packed_k else (n, k // 8)
            if tuple(weight.shape) != expected_shape:
                raise ValueError("E8P tuning weights differ from the compressed layout")
            decoded = torch.empty((n, k), device=device, dtype=torch.int8)
            stream = torch.cuda.current_stream().cuda_stream
            unpack(n, k, packed_k=packed_k).adapter.func(weight, book, decoded, stream=stream)
            # Keep the independent full-K reference dynamic too: profiling
            # many row extents must not compile a reference cubin per length.
            from orinfer_build.compiler.compile import primitive

            reference = primitive(
                int8_projection,
                spec["rows"] if spec["dynamic"] else live,
                n,
                k,
                output_dtype=spec["dtype"],
                dynamic_rows=spec["dynamic"],
            )
            reference.adapter.func(
                a,
                decoded,
                ws,
                scales,
                expected,
                *([live] if spec["dynamic"] else []),
                stream=stream,
            )
            args = [a, weight, book, ws, scales, out]
        elif kind == "w8":
            from orinfer_build.compiler.compile import primitive

            weight = (
                weights["W"]
                if weights is not None
                else torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
            )
            ws = (
                weights["WS"]
                if weights is not None
                else torch.full((n,), 0.001, device=device, dtype=torch.float16)
            )
            base = primitive(
                int8_projection,
                spec["rows"] if spec["dynamic"] else live,
                n,
                k,
                output_dtype=spec["dtype"],
                dynamic_rows=spec["dynamic"],
            )
            base.adapter.func(
                a,
                weight,
                ws,
                scales,
                expected,
                *([live] if spec["dynamic"] else []),
                stream=torch.cuda.current_stream().cuda_stream,
            )
            args = [a, weight, ws, scales, out]
        elif kind == "swiglu":
            from orinfer_build.compiler.compile import primitive
            from kernels.cuda.model.flash_next import swiglu

            gate = torch.empty_like(expected)
            up = torch.empty_like(expected)
            wg = (
                weights["Gate"]
                if weights is not None
                else torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
            )
            wu = (
                weights["Up"]
                if weights is not None
                else torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
            )
            gs = (
                weights["GateScale"]
                if weights is not None
                else torch.full((n,), 0.001, device=device, dtype=torch.float16)
            )
            us = (
                weights["UpScale"]
                if weights is not None
                else torch.full((n,), 0.0007, device=device, dtype=torch.float16)
            )
            base = primitive(
                int8_projection,
                spec["rows"] if spec["dynamic"] else live,
                n,
                k,
                dynamic_rows=spec["dynamic"],
            )
            stream = torch.cuda.current_stream().cuda_stream
            live_args = [live] if spec["dynamic"] else []
            base.adapter.func(a, wg, gs, scales, gate, *live_args, stream=stream)
            base.adapter.func(a, wu, us, scales, up, *live_args, stream=stream)
            activation = primitive(
                swiglu,
                spec["rows"] if spec["dynamic"] else live,
                n,
                dynamic_rows=spec["dynamic"],
            )
            activation.adapter.func(gate, up, expected, *live_args, stream=stream)
            args = [a, wg, wu, gs, us, scales, out]
        else:
            e = spec["experts"]
            assignments = live * 10
            a = torch.randint(-128, 128, (assignments, k), device=device, dtype=torch.int8)
            scales = torch.full((assignments,), 0.001, device=device, dtype=torch.float16)
            storage = torch.full((assignments + 2, n), 91, device=device, dtype=dtype)
            out, expected = storage[:assignments], torch.empty_like(storage[:assignments])
            low = torch.empty((assignments, k // 8), device=device, dtype=torch.uint32)
            high, sums = (
                torch.empty_like(low),
                torch.empty(assignments, device=device, dtype=torch.int32),
            )
            weight = (
                weights["P"]
                if weights is not None
                else torch.randint(
                    0, 65536, (e, k // 64, n, 8), device=device, dtype=torch.int32
                ).to(torch.uint16)
            )
            book = (
                weights["Book"]
                if weights is not None
                else torch.randint(0, 2**32, (1, 256, 1), device=device, dtype=torch.int64).to(
                    torch.uint32
                )
            )
            ws = (
                weights["WS"]
                if weights is not None
                else torch.full((e, n), 0.001, device=device, dtype=torch.float16)
            )
            # Mix spread routing with a hot expert to exercise both tile
            # padding and repeated rows; the ordered activations stay fixed.
            counts = [0] * e
            counts[0] = live
            for j in range(assignments - counts[0]):
                counts[1 + j % (e - 1)] += 1
            offsets, first = [], 0
            for count in counts:
                offsets.append(first)
                first += count
            count_t = torch.tensor(counts, device=device, dtype=torch.int32)
            offset_t = torch.tensor(offsets, device=device, dtype=torch.int32)
            split_a8(assignments, k).adapter.func(
                a, low, high, sums, stream=torch.cuda.current_stream().cuda_stream
            )

            def maps(bm, capacity):
                te, tr = [], []
                for expert, count in enumerate(counts):
                    for row in range(0, count, bm):
                        te.append(expert)
                        tr.append(row)
                assert len(te) <= capacity
                return [
                    torch.tensor(te + [0] * (capacity - len(te)), device=device, dtype=torch.int32),
                    torch.tensor(tr + [0] * (capacity - len(tr)), device=device, dtype=torch.int32),
                    torch.tensor([len(te)], device=device, dtype=torch.int32),
                ]

            tile_cap = (assignments + 7) // 8 + min(e, assignments)
            te, tr, tc = maps(8, tile_cap)
            args = [low, high, sums, weight, book, ws, scales, count_t, offset_t, te, tr, tc, out]
            base = e8p_exact_grouped(assignments, e, tile_cap, n, k, block_m=8)
            base.adapter.func(*args[:-1], expected, stream=torch.cuda.current_stream().cuda_stream)

        torch.cuda.synchronize()
        scalar = live * 10 if kind == "expert" else live
        if spec["dynamic"]:
            args.append(scalar)

        def supply(params):
            supplied = list(args)
            if kind == "expert":
                # TE's static allocation reflects the compiled BM. The host
                # builds matching tile maps; these are never profiling inputs
                # for a differently sized expert tile.
                capacity = int(params[9].shape[0])
                assignments = spec["rows"] * 10
                bm = next(
                    m
                    for m in (8, 16)
                    if (assignments + m - 1) // m + min(spec["experts"], assignments) == capacity
                )
                supplied[9:12] = maps(bm, capacity)
            return supplied

        def reference(*_):
            storage.fill_(91)
            return []

        def check(*_):
            if (
                not torch.equal(out, expected)
                or not bool(torch.all(storage[scalar:] == 91))
                or not bool(torch.all(storage[:scalar, n:] == 91))
            ):
                raise AssertionError(
                    "Decode tile changed integer/scaling output or wrote beyond live rows"
                )

        return supply, reference, check

    @staticmethod
    def _validate(kernel, spec, config, live, weights=None):
        import torch

        supply, _, check = ProjectionTuner._inputs(spec, live, weights)
        args = supply(kernel.get_profiler()._get_params(with_output=False))
        kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)
        # Scratch outputs change after capture; replay must actually run.
        output_index = {"expert": 12, "swiglu": 6, "e8p_q4": 5}.get(spec["kind"], 4)
        out = args[output_index]
        out.fill_(91)
        graph.replay()
        torch.cuda.synchronize()
        check()
        # Compare replay after changing the live input with a direct launch.
        # The independent arithmetic reference above has already checked the
        # candidate; this additionally proves capture observes updated storage.
        args[0].zero_()
        kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        changed = out.clone()
        out.fill_(91)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(out, changed):
            raise AssertionError("Decode graph replay ignored the changed live input")

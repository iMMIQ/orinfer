"""Shared expert geometry, kernel construction and offline pipeline selection.

Model adapters supply routing/operand fixtures. Candidate and publication paths
use the same factory, so tile capacity, codec, tail policy and compiler passes
cannot diverge between measurement and the emitted execution package.
"""

from orinfer_build.tuning.row_tuning import RowTuner, launch
from orinfer_build.compiler.compile import primitive


EXACT_EXPERT_PASSES = {
    "tl.disable_thread_storage_sync": True,
    "tl.disable_shared_memory_reuse": True,
    "tl.if_stmt_binding_inline_replayable_binds": False,
}


def tile_capacity(rows, experts, top_k, block_m):
    if any(type(n) is not int or n <= 0 for n in (rows, experts, top_k, block_m)):
        raise ValueError("Invalid routed projection geometry")
    slots = rows * top_k
    return (slots + block_m - 1) // block_m + min(experts, slots)


def cold_expert_config():
    return dict(block_m=32, expand=False, folded=True, minimum_blocks=4, sparse_rows=True)


def expert_configs(codec, *, block_m=None, direct=False):
    if codec == "e8p_q2":
        tiles = (8, 16) if block_m is None else (block_m,)
        return [dict(block_m=m, stages=s) for m in tiles for s in (2, 4)]
    if codec != "e8p_q4" or block_m is not None:
        raise ValueError("Unsupported routed projection search")
    configs = [
        dict(block_m=16, block_n=64, block_k=128, stages=2, tail_m=0, expand=False),
        dict(block_m=32, block_n=64, block_k=128, stages=1, tail_m=8, expand=False),
        dict(block_m=64, block_n=64, block_k=128, stages=1, tail_m=0, expand=False),
        dict(
            block_m=64,
            block_n=64,
            block_k=128,
            stages=1,
            tail_m=0,
            expand=False,
            minimum_blocks=6,
        ),
        dict(block_m=128, block_n=32, block_k=128, stages=1, tail_m=0, expand=False),
        dict(
            block_m=128,
            block_n=64,
            block_k=128,
            stages=1,
            tail_m=0,
            expand=False,
            minimum_blocks=4,
        ),
        dict(
            block_m=256,
            block_n=64,
            block_k=128,
            stages=1,
            tail_m=64,
            expand=False,
            minimum_blocks=3,
            reuse_shared=True,
        ),
        dict(block_m=32, block_n=128, block_k=128, stages=1, tail_m=8, expand=False),
        dict(block_m=32, block_n=64, block_k=128, stages=1, tail_m=0, expand=True),
        dict(block_m=16, block_n=64, block_k=128, stages=2, tail_m=0, expand=True),
        dict(
            block_m=32, block_n=64, block_k=128, stages=1, tail_m=8, expand=False, reuse_shared=True
        ),
    ]
    if direct:
        configs += [
            dict(block_m=m, expand=False, folded=True, minimum_blocks=blocks, sparse_rows=sparse)
            for m, blocks in ((16, 6), (32, 4))
            for sparse in (False, True)
        ]
        configs += [dict(configs[6], cold_rows=n) for n in (16, 32, 64, 128)]
    return configs


def expert_passes(codec, config):
    if codec == "e8p_q2" or config.get("folded"):
        return EXACT_EXPERT_PASSES
    return {"tl.enable_aggressive_shared_memory_merge": True} if config.get("reuse_shared") else {}


def expert_projection(rows, outputs, inputs, experts, top_k, *, codec, config, dynamic_rows=False):
    import inspect

    tile = config["block_m"]
    tiles = tile_capacity(rows, experts, top_k, tile)
    if codec == "e8p_q2":
        from kernels.cuda.model.e8p_exact import e8p_exact_grouped

        return inspect.unwrap(e8p_exact_grouped)(
            rows * top_k,
            experts,
            tiles,
            outputs,
            inputs,
            block_m=tile,
            stages=config.get("stages", 2),
            dynamic_rows=dynamic_rows,
        )
    if codec not in ("e8p_q4", "w8"):
        raise ValueError("Unsupported expert codec")
    if config.get("folded"):
        from kernels.cuda.model.e8p_prefill import e8p_prefill_grouped

        return inspect.unwrap(e8p_prefill_grouped)(
            rows * top_k,
            experts,
            tiles,
            outputs,
            inputs,
            dynamic_rows=dynamic_rows,
            sparse_rows=config["sparse_rows"],
            codec=codec,
            minimum_blocks=config["minimum_blocks"],
            block_m=tile,
        )
    from kernels.cuda.model.moe import integer_experts

    options = {
        k: v
        for k, v in config.items()
        if k not in ("expand", "reuse_shared", "folded", "cold_rows")
    }
    return inspect.unwrap(integer_experts)(
        rows,
        outputs,
        inputs,
        experts,
        top_k,
        codec="w8" if config.get("expand") else codec,
        dynamic_rows=dynamic_rows,
        **options,
    )


class MoeTuner(RowTuner):
    def topk(self, experts, top_k):
        """Measure exact block/warp selection, including tie and tail fixtures."""
        import torch
        from kernels.cuda.model import moe
        from orinfer_build.contracts.row_selection import samples

        spec = dict(kind="router_topk", experts=experts, top_k=top_k)
        key = str(spec)
        if key in self.records:
            return self.records[key]
        configs = [dict(algorithm="block")] + [dict(algorithm="warp", warps=w) for w in (1, 2, 4)]

        def build(config):
            if config["algorithm"] == "block":
                return primitive(moe.router_topk, 4096, experts, top_k, dynamic_rows=True)
            return primitive(
                moe.router_topk_warp, 4096, experts, top_k, dynamic_rows=True, warps=config["warps"]
            )

        baseline = build(configs[0])

        def inputs(rows):
            scores = torch.randn(rows, experts, device="cuda")
            scores[0].zero_()
            if rows > 1:
                scores[1].copy_(torch.arange(experts, device="cuda") % 3)
            expected = [
                torch.full((rows + 2, top_k), 42, dtype=t, device="cuda")
                for t in (torch.int32, torch.float32)
            ]
            launch(baseline, scores, *expected, rows)
            torch.cuda.synchronize()

            def fixture(kernel, config):
                actual = [torch.full_like(t, 42) for t in expected]

                def run():
                    launch(kernel, scores, *actual, rows)

                def check():
                    torch.cuda.synchronize()
                    if not all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)):
                        raise ValueError("Router selection changes scores, ties or tail guards")

                def change():
                    scores.copy_(scores.flip(1))
                    launch(baseline, scores, *expected, rows)

                return run, check, change

            yield fixture

        result = self.select(
            spec, configs, build, inputs, measure_rows=samples(4096), relative_tolerance=0.03
        )
        self.records[key] = result
        return result

    def router(self, hidden, experts):
        """Tune the FP32 router result without changing its K16 MMA reduction."""
        import torch
        from kernels.cuda.model.moe import router_projection
        from orinfer_build.contracts.row_selection import prefill_fallback

        spec = dict(kind="router", hidden=hidden, experts=experts)
        key = str(spec)
        if key in self.records:
            return self.records[key]
        configs = [dict(block_m=16, block_n=32, block_k=128, stages=2)] + [
            dict(block_m=m, block_n=n, block_k=128, stages=s)
            for m, n, s in [(32, 64, 2), (64, 64, 2), (64, 128, 2), (128, 64, 1), (64, 128, 1)]
        ]

        def build(config):
            return primitive(router_projection, 4096, hidden, experts, dynamic_rows=True, **config)

        reference = build(configs[0])

        def inputs(rows):
            x = torch.randn((rows, hidden), device="cuda", dtype=torch.float16) * 0.1
            weights = torch.randn((experts + 1, hidden), device="cuda", dtype=torch.float16) * 0.01
            expected = torch.empty((rows, experts), device="cuda")
            gate_expected = torch.empty(rows, device="cuda", dtype=torch.float16)

            def fixture(kernel, config):
                logits = torch.full((rows + 1, experts), 91, device="cuda", dtype=torch.float32)
                gate = torch.full((rows + 1,), 91, device="cuda", dtype=torch.float16)

                def run():
                    launch(kernel, x, weights, logits[:rows], gate[:rows], rows)

                def check():
                    launch(reference, x, weights, expected, gate_expected, rows)
                    assert torch.equal(logits[:rows], expected) and torch.equal(
                        gate[:rows], gate_expected
                    )
                    assert bool(torch.all(logits[rows:] == 91)) and bool(
                        torch.all(gate[rows:] == 91)
                    )

                return run, check, lambda: x.neg_()

            yield fixture

        result = self.select(
            spec, configs, build, inputs, measure_rows=[1, 17, 128, 512, 2048, 4096]
        )
        costs = {
            str(i): sum(row["times"][str(i)] for row in result["measurements"])
            for i in range(len(configs))
        }
        result = prefill_fallback(result, min(costs, key=costs.get))
        self.records[key] = result
        return result

    def routed(
        self,
        hidden,
        intermediate,
        experts,
        top_k,
        weights,
        weight_identity,
        *,
        routing_samples=None,
    ):
        import torch
        from kernels.cuda.model import moe
        from kernels.cuda.model.e8p_prefill import split_prefill_a8, split_routed_prefill_a8
        from kernels.cuda.model import expert_partition
        from orinfer_build.quantization.vq import e8p_selector_supports_parity_xor

        spec = dict(
            kind="routed_q4",
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            top_k=top_k,
            weights=weight_identity,
            timed_routing=["independent"],
            validation_routing=["four", "hot"],
        )
        routing_samples = routing_samples or {}
        for rows, samples in routing_samples.items():
            if (
                type(rows) is not int
                or not 1 <= rows <= 4096
                or not samples
                or any(
                    len(counts) != experts
                    or sum(counts) != rows * top_k
                    or any(type(n) is not int or n < 0 for n in counts)
                    for counts in samples
                )
            ):
                raise ValueError("Invalid measured MoE routing histogram")
        if routing_samples:
            spec["routing_samples"] = {
                str(rows): samples for rows, samples in sorted(routing_samples.items())
            }
        configs = expert_configs(
            "e8p_q4", direct=e8p_selector_supports_parity_xor(weights["book"].cpu().numpy())
        )
        dimensions = [(2 * intermediate, hidden), (hidden, intermediate)]

        def build(config):
            result = dict(
                offsets=primitive(moe.expert_offsets, experts, config["block_m"]),
                tiles=primitive(
                    moe.expert_tiles, 4096, experts, top_k, config["block_m"], dynamic_rows=True
                ),
            )
            if config.get("cold_rows"):
                cold = cold_expert_config()
                args = (experts, config["block_m"], cold["block_m"], config["cold_rows"])
                result["offsets"] = primitive(expert_partition.offsets, *args)
                result["tiles"] = primitive(
                    expert_partition.tiles,
                    4096,
                    experts,
                    top_k,
                    config["block_m"],
                    cold["block_m"],
                    config["cold_rows"],
                    dynamic_rows=True,
                )
                result["cold_packs"] = [
                    primitive(
                        split_routed_prefill_a8,
                        4096 * top_k,
                        k,
                        experts,
                        cold["block_m"],
                        dynamic_rows=True,
                    )
                    for _, k in dimensions
                ]
                result["cold_products"] = [
                    primitive(
                        expert_projection,
                        4096,
                        n,
                        k,
                        experts,
                        top_k,
                        codec="e8p_q4",
                        config=cold,
                        dynamic_rows=True,
                        compiler_options=expert_passes("e8p_q4", cold),
                    )
                    for n, k in dimensions
                ]
            if config.get("folded"):
                slots = 4096 * top_k
                result["packs"] = [
                    primitive(split_prefill_a8, slots, k, dynamic_rows=True) for _, k in dimensions
                ]
            result["products"] = [
                primitive(
                    expert_projection,
                    4096,
                    n,
                    k,
                    experts,
                    top_k,
                    codec="e8p_q4",
                    config=config,
                    dynamic_rows=True,
                    compiler_options=expert_passes("e8p_q4", config),
                )
                for n, k in dimensions
            ]
            if config["expand"]:
                result["unpack"] = [
                    primitive(moe.expand_experts, n, k, experts) for n, k in dimensions
                ]
            return result

        reference = build(configs[0])

        def routing_counts(rows, patterns):
            for pattern in patterns:
                if pattern == "measured":
                    for counts in routing_samples[rows]:
                        yield torch.tensor(counts, device="cuda", dtype=torch.int32)
                    continue
                choices = (
                    rows if pattern == "independent" else min(rows, 4 if pattern == "four" else 1)
                )
                ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(choices)])
                ids = ids[torch.arange(rows) % choices].flatten()
                yield torch.bincount(ids, minlength=experts).int().cuda()

        def inputs(rows, patterns=("independent",)):
            for counts in routing_counts(rows, patterns):
                data = [
                    torch.randint(-100, 101, (rows * top_k, k), device="cuda", dtype=torch.int8)
                    for _, k in dimensions
                ]
                scale = torch.full((rows * top_k,), 0.001, device="cuda", dtype=torch.float16)
                expected = [
                    torch.empty((rows * top_k, n), device="cuda", dtype=torch.float16)
                    for n, _ in dimensions
                ]

                def runner(kernels, config, outputs):
                    tile = config["block_m"]
                    capacity = (4096 * top_k + tile - 1) // tile + experts
                    offsets = torch.empty(experts, device="cuda", dtype=torch.int32)
                    tile_offsets = torch.empty_like(offsets)
                    tile_count = torch.empty(1, device="cuda", dtype=torch.int32)
                    tile_expert = torch.empty(capacity, device="cuda", dtype=torch.int32)
                    tile_row = torch.empty_like(tile_expert)
                    expanded = torch.empty(
                        experts * max(n * k for n, k in dimensions) if config["expand"] else 1,
                        device="cuda",
                        dtype=torch.int8,
                    )
                    planes = (
                        [
                            torch.empty(
                                rows * top_k * max(hidden, intermediate) // 2,
                                device="cuda",
                                dtype=torch.int8,
                            )
                            for _ in range(2)
                        ]
                        if config.get("folded") or config.get("cold_rows")
                        else []
                    )
                    sums = torch.empty(rows * top_k, device="cuda", dtype=torch.int32)
                    cold = None
                    if config.get("cold_rows"):
                        capacity = tile_capacity(
                            4096, experts, top_k, cold_expert_config()["block_m"]
                        )
                        cold = dict(
                            offsets=torch.empty_like(offsets),
                            count=torch.empty_like(tile_count),
                            expert=torch.empty(capacity, device="cuda", dtype=torch.int32),
                            row=torch.empty(capacity, device="cuda", dtype=torch.int32),
                        )

                    def run():
                        if cold is not None:
                            launch(
                                kernels["offsets"],
                                counts,
                                offsets,
                                tile_offsets,
                                tile_count,
                                cold["offsets"],
                                cold["count"],
                            )
                            launch(
                                kernels["tiles"],
                                counts,
                                tile_offsets,
                                tile_expert,
                                tile_row,
                                cold["offsets"],
                                cold["expert"],
                                cold["row"],
                                rows,
                            )
                        else:
                            launch(kernels["offsets"], counts, offsets, tile_offsets, tile_count)
                            launch(
                                kernels["tiles"], counts, tile_offsets, tile_expert, tile_row, rows
                            )
                        for i, product in enumerate(kernels["products"]):
                            w = weights["banks"][i][0]
                            if cold is not None:
                                launch(
                                    kernels["cold_packs"][i],
                                    data[i],
                                    counts,
                                    offsets,
                                    cold["expert"],
                                    cold["row"],
                                    cold["count"],
                                    *planes,
                                    sums,
                                    rows * top_k,
                                )
                                launch(
                                    kernels["cold_products"][i],
                                    *planes,
                                    sums,
                                    w,
                                    weights["book"],
                                    weights["banks"][i][1],
                                    scale,
                                    counts,
                                    offsets,
                                    cold["expert"],
                                    cold["row"],
                                    cold["count"],
                                    outputs[i],
                                    rows * top_k,
                                )
                            if config.get("folded"):
                                launch(kernels["packs"][i], data[i], *planes, sums, rows * top_k)
                                launch(
                                    product,
                                    *planes,
                                    sums,
                                    w,
                                    weights["book"],
                                    weights["banks"][i][1],
                                    scale,
                                    counts,
                                    offsets,
                                    tile_expert,
                                    tile_row,
                                    tile_count,
                                    outputs[i],
                                    rows * top_k,
                                )
                                continue
                            if config["expand"]:
                                launch(kernels["unpack"][i], w, weights["book"], counts, expanded)
                                w = expanded
                            launch(
                                product,
                                data[i],
                                w,
                                weights["book"],
                                weights["banks"][i][1],
                                scale,
                                counts,
                                offsets,
                                tile_expert,
                                tile_row,
                                tile_count,
                                outputs[i],
                                rows,
                            )

                    return run

                run_reference = runner(reference, configs[0], expected)

                def fixture(kernels, config):
                    storage = [
                        torch.full((rows * top_k + 1, n), 91, device="cuda", dtype=torch.float16)
                        for n, _ in dimensions
                    ]
                    actual = [value[:-1] for value in storage]
                    run = runner(kernels, config, actual)

                    def check():
                        run_reference()
                        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), (
                            "Routed pipeline changed integer projection outputs"
                        )
                        assert all(bool(torch.all(value[-1] == 91)) for value in storage)

                    def change():
                        for value in data:
                            value.neg_()

                    return run, check, change

                yield fixture

        result = self.select(
            spec,
            configs,
            build,
            lambda rows: inputs(
                rows, ("measured",) if rows in routing_samples else ("independent",)
            ),
            measure_rows=sorted(
                {1, 2, 3, 4, 8, 17, 32, 64, 128, 512, 2048, 4096} | routing_samples.keys()
            ),
            # Captured routes determine cost at supplied extents; otherwise
            # use independent gating. Duplicated-route extremes still
            # undergo exact/tail and changed-input graph checks, but are not
            # assumed to occupy two thirds of real decode iterations.
            validation_inputs=lambda rows: inputs(rows, ("four", "hot")),
        )
        from orinfer_build.contracts.row_selection import prefill_fallback

        costs = {
            str(i): sum(row["times"][str(i)] for row in result["measurements"] if row["rows"] > 128)
            for i in range(len(configs))
        }
        return prefill_fallback(result, min(costs, key=costs.get))

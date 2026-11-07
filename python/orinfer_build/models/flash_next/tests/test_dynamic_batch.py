"""CPU compiler checks for the affine-shape host ABI regression."""

import unittest
from tilelang import tvm

from kernels.shared.rows import explicit_rows
from orinfer_build.tuning.projection_tuning import factory as projection_factory, make_prim
from kernels.cuda.model import moe, hyperconnection, flash_next, rotation_a8
from kernels.cuda.model.int8_swiglu import int8_swiglu
from kernels.cuda.model.int8_projection import int8_small_projection
from kernels.cuda.model.q4_projection import q4_projection
from kernels.cuda.model.bf16_single import bf16_single
from kernels.cuda.model.greedy import greedy_merge

TEST_ENVIRONMENT = "cuda"

Analyzer = tvm.arith.Analyzer
tir = tvm.tirx


class DynamicRowsTests(unittest.TestCase):
    def test_shared_expert_factory_keeps_token_and_assignment_extents(self):
        from orinfer_build.tuning.expert_tuning import expert_projection

        for codec, config in [
            ("w8", dict(block_m=16, stages=1, expand=False)),
            ("e8p_q4", dict(block_m=16, stages=2, expand=False)),
            ("e8p_q2", dict(block_m=8)),
            ("e8p_q4", dict(block_m=16, folded=True, minimum_blocks=6, sparse_rows=True)),
        ]:
            fn = explicit_rows(
                expert_projection(
                    512,
                    129,
                    256,
                    4,
                    2,
                    codec=codec,
                    config=config,
                    dynamic_rows=True,
                )
            )
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            output = next(b for b in fn.buffer_map.values() if b.name == "Out")
            multiplier = 1 if codec == "e8p_q2" or config.get("folded") else 2
            tvm.ir.assert_structural_equal(Analyzer().simplify(output.shape[0]), rows * multiplier)
            self.assertEqual(int(output.shape[1]), 129)

    def test_sparse_expert_tail_reuses_one_decoder_mainloop(self):
        from kernels.cuda.model.e8p_prefill import e8p_prefill_grouped

        counts = []
        for sparse in (False, True):
            fn = e8p_prefill_grouped.__wrapped__(
                1024,
                256,
                320,
                129,
                256,
                codec="e8p_q4",
                dynamic_rows=True,
                sparse_rows=sparse,
            )
            decoders = []

            def visit(node):
                if isinstance(node, tir.Call) and node.op.name == "tirx.call_pure_extern":
                    if node.args[0].value == "nib_fast":
                        decoders.append(node)

            tir.stmt_functor.post_order_visit(fn.body, visit)
            counts.append(len(decoders))
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[0], counts[1], "Tail masking duplicated the expert mainloop")

    def test_integer_tail_and_grouping_keep_one_runtime_extent(self):
        for kind, tile in [("e8p_q4", 256), ("q4", 128), ("w8", 128)]:
            spec = dict(
                kind=kind, rows=4096, outputs=128, inputs=256, dtype="float16", dynamic=True
            )
            config = dict(
                algorithm="tensor_core",
                block_m=tile,
                block_n=64,
                block_k=128,
                **({"stages": 1} if kind != "w8" else {}),
                tail_m=16,
                group_m=2,
            )
            fn = make_prim(spec, config)
            tvm.ir.assert_structural_equal(fn, projection_factory(spec)(**config))
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            self.assertFalse(
                tir.analysis.undefined_vars(
                    fn.body, list(fn.params) + [b.data for b in fn.buffer_map.values()]
                )
            )
            extents = []

            def visit(node):
                if (
                    isinstance(node, tir.For)
                    and node.thread_binding is not None
                    and node.thread_binding.thread_tag == "blockIdx.x"
                ):
                    extents.append(node.extent)

            tir.stmt_functor.post_order_visit(fn.body, visit)
            self.assertEqual(len(extents), 1)
            # Partial last CTAs keep the same host grid and row scalar. Smaller
            # arithmetic tiles are selected inside the compiled projection.
            for live in [1, 17, 127, 129, 255, 256, 257, 512, 513, 2049, 4095, 4096]:
                actual = Analyzer().simplify(tir.stmt_functor.substitute(extents[0], {rows: live}))
                self.assertEqual(int(actual), ((live + tile - 1) // tile) * 2)

    def test_two_stage_q4_keeps_one_dynamic_row_abi(self):
        for algorithm, tile, columns in [("dp4a", 4, 16), ("tensor_core", 16, 64)]:
            for dynamic in [False, True]:
                fn = make_prim(
                    dict(
                        kind="e8p_q4",
                        rows=4096,
                        outputs=65,
                        inputs=256,
                        dtype="float32",
                        dynamic=dynamic,
                        output_stride=68,
                    ),
                    dict(algorithm=algorithm, block_m=tile, block_n=columns, block_k=128, stages=2),
                )
                buffers = {b.name: b for b in fn.buffer_map.values()}
                self.assertEqual(str(buffers["W"].dtype), "uint32")
                self.assertEqual([int(v) for v in buffers["W"].shape], [65, 32])
                self.assertEqual([int(v) for v in buffers["Book"].shape], [256, 1])
                self.assertEqual(int(buffers["Out"].shape[1]), 68)
                scalars = [p for p in fn.params if str(p.dtype) == "int32"]
                self.assertEqual(len(scalars), int(dynamic))
                if dynamic:
                    self.assertTrue(buffers["Out"].shape[0].same_as(scalars[0]))

    def test_k_major_e8p_shares_the_dynamic_abi_across_compute_schedules(self):
        from kernels.cuda.model.e8p_q4 import projection, unpack

        for algorithm, bm, bn, bk in [
            ("tensor_core", 16, 64, 128),
            ("dp4a", 4, 16, 128),
            ("int4_direct", 8, 64, 128),
        ]:
            fn = make_prim(
                dict(
                    kind="e8p_q4",
                    rows=4096,
                    outputs=65,
                    inputs=256,
                    dtype="float32",
                    dynamic=True,
                    output_stride=68,
                    packed_k=128,
                    basis_nonzero=True,
                ),
                dict(algorithm=algorithm, block_m=bm, block_n=bn, block_k=bk, stages=2),
            )
            buffers = {b.name: b for b in fn.buffer_map.values()}
            self.assertEqual([int(v) for v in buffers["W"].shape], [2, 65, 16])
            scalars = [p for p in fn.params if str(p.dtype) == "int32"]
            self.assertEqual(len(scalars), 1)
            self.assertTrue(buffers["Out"].shape[0].same_as(scalars[0]))
        for options in [
            dict(packed_k=0, basis_nonzero=True),
            dict(packed_k=128, basis_nonzero=False),
        ]:
            with self.assertRaisesRegex(ValueError, "verified nonzero basis"):
                projection.__wrapped__(1, 65, 256, algorithm="int4_direct", block_k=128, **options)
        for packed_k in (True, -1, 256, 96):
            with self.assertRaisesRegex(ValueError, "compressed K layout"):
                unpack.__wrapped__(65, 256, packed_k=packed_k)
        for split in (2, 4):
            fn = make_prim(
                dict(
                    kind="e8p_q4",
                    rows=129,
                    outputs=65,
                    inputs=512,
                    dtype="float16",
                    dynamic=True,
                    packed_k=128,
                    basis_nonzero=True,
                ),
                dict(
                    algorithm="int4_direct",
                    block_m=8,
                    block_n=64,
                    block_k=128,
                    stages=2,
                    split_k=split,
                ),
            )
            self.assertEqual(len([p for p in fn.params if str(p.dtype) == "int32"]), 1)
        for options in (
            dict(split_k=True),
            dict(split_k=0),
            dict(split_k=3),
            dict(split_k=4, block_n=128),
        ):
            with self.assertRaises(ValueError):
                projection.__wrapped__(
                    1, 65, 512, algorithm="int4_direct", packed_k=128, basis_nonzero=True, **options
                )

    def test_planar_swiglu_keeps_affine_input_extent_and_one_live_row_argument(self):
        from kernels.cuda.model.rotation_a8 import rotate_activation_tiles

        fn = explicit_rows(
            rotate_activation_tiles.__wrapped__(
                4096, 17408, swiglu=True, dynamic_rows=True, plane_rows=4096
            )
        )
        (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
        buffers = {b.name: b for b in fn.buffer_map.values()}
        self.assertEqual(int(Analyzer().simplify(buffers["X"].shape[0] - rows)), 4096)
        self.assertEqual(int(buffers["X"].shape[1]), 17408)
        self.assertTrue(buffers["Rotated"].shape[0].same_as(rows))
        for options in (
            dict(plane_rows=True, swiglu=True),
            dict(plane_rows=4, swiglu=True),
            dict(plane_rows=4096, swiglu=False),
        ):
            with self.assertRaisesRegex(ValueError, "plane capacity"):
                rotate_activation_tiles.__wrapped__(8, 256, **options)

    def test_metadata_scalars_survive_independent_state_wrapping(self):
        from kernels.cuda.model import qsa
        from kernels.cuda.model.hc_bf16_cache import hc_up13_mix
        from kernels.cuda.model.independent_rows import independent_rows

        hc = explicit_rows(
            hc_up13_mix.__wrapped__(1024, 2560, 320, base_exp=None, dynamic_rows=True)
        )
        self.assertEqual({p.name for p in hc.params if p.dtype == "int32"}, {"m", "base_exp"})
        for primitive in (qsa.radix_histogram, qsa.radix_choose):
            fn = primitive.__wrapped__(1, 262144, None)
            roles = {b.name: ("private", "Scratch") for b in fn.buffer_map.values()}
            fn, identities = independent_rows(fn, 128, roles, dict(Scratch=0), dynamic=True)
            self.assertIn("shift", identities)
            self.assertEqual({p.name for p in fn.params if p.dtype == "int32"}, {"m", "shift"})
            self.assertFalse(
                tir.analysis.undefined_vars(
                    fn.body, list(fn.params) + [b.data for b in fn.buffer_map.values()]
                )
            )

    def test_independent_attention_split_occupancy_uses_live_owner_count(self):
        from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
        from kernels.cuda.model.independent_rows import independent_rows
        from kernels.shared.rows import static_rows

        for primitive in (causal_attention_int8, merge_attention_int8):
            options = dict(
                block_m=16,
                target_tiles=16,
                max_splits=8,
                query_heads=16,
                kv_heads=2,
                head_group=8,
                parallel_instances=True,
            )
            if primitive is causal_attention_int8:
                options["context"] = 8192
            fn = static_rows(primitive.__wrapped__(1, **options))
            roles = {b.name: ("private", "Scratch") for b in fn.buffer_map.values()}
            wrapped, identities = independent_rows(
                fn,
                128,
                roles,
                dict(Scratch=0),
                dynamic=True,
                batch_parameters=("independent_instances",),
            )
            self.assertNotIn("independent_instances", identities)
            (rows,) = [p for p in wrapped.params if p.dtype == "int32"]
            self.assertEqual(rows.name, "m")
            self.assertFalse(
                tir.analysis.undefined_vars(
                    wrapped.body,
                    list(wrapped.params) + [b.data for b in wrapped.buffer_map.values()],
                )
            )
            split_calls, split_grids = [], []

            def visit(node):
                if isinstance(node, tir.Call) and str(node.args[0]) == '"attn_split_count"':
                    split_calls.append(node)
                if isinstance(node, tir.For) and node.thread_binding is not None:
                    if node.thread_binding.thread_tag == "blockIdx.z":
                        split_grids.append(node.extent)

            tir.stmt_functor.post_order_visit(wrapped.body, visit)
            self.assertEqual(len(split_calls), 1)
            self.assertTrue(split_calls[0].args[-1].same_as(rows))
            if primitive is causal_attention_int8:
                self.assertEqual(len(split_grids), 1)
                for owners, expected in [(1, 8), (2, 8), (3, 6), (4, 4), (8, 2), (16, 1), (128, 1)]:
                    actual = Analyzer().simplify(
                        tir.stmt_functor.substitute(split_grids[0], {rows: owners})
                    )
                    self.assertEqual(int(actual), expected)
            with self.assertRaisesRegex(ValueError, "scheduling scalar"):
                independent_rows(fn, 128, roles, dict(Scratch=0), batch_parameters=("missing",))

    def test_moe_combine_bound_tracks_live_assignments_beyond_initial_capacity(self):
        for capacity in [8, 16, 128]:
            fn = explicit_rows(
                moe.moe_combine.__wrapped__(capacity, 2560, capacity * 10, 10, dynamic_rows=True)
            )
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            buffers = {b.name: b for b in fn.buffer_map.values()}
            bound = buffers["Expert"].shape[0]
            for live in [1, 9, 17, 33, 128]:
                actual = Analyzer().simplify(tir.stmt_functor.substitute(bound, {rows: live}))
                self.assertEqual(int(actual), live * 10)

    def test_fused_shared_ffn_launch_covers_multiple_row_tiles(self):
        fn = explicit_rows(int8_swiglu.__wrapped__(128, 640, 2560, dynamic_rows=True))
        (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
        buffers = {b.name: b for b in fn.buffer_map.values()}
        self.assertTrue(buffers["A"].shape[0].same_as(rows))
        self.assertTrue(buffers["Output"].shape[0].same_as(rows))
        extents = []

        def visit(node):
            if isinstance(node, tir.For) and node.thread_binding is not None:
                if node.thread_binding.thread_tag == "blockIdx.y":
                    extents.append(node.extent)

        tir.stmt_functor.post_order_visit(fn.body, visit)
        self.assertEqual(len(extents), 1)
        for live in [1, 8, 9, 16, 17, 33, 127, 128]:
            extent = Analyzer().simplify(tir.stmt_functor.substitute(extents[0], {rows: live}))
            self.assertEqual(int(extent), (live + 15) // 16)

    def test_decode_factory_uses_one_row_contract_for_both_integer_algorithms(self):
        for dynamic in [False, True]:
            for rows in [1, 3, 16, 32, 128]:
                for algorithm, tile in [("dp4a", min(rows, 8)), ("tensor_core", 16)]:
                    fn = make_prim(
                        dict(
                            kind="q4",
                            rows=rows,
                            outputs=128,
                            inputs=256,
                            dtype="float32",
                            dynamic=dynamic,
                        ),
                        dict(
                            algorithm=algorithm,
                            block_m=tile,
                            block_n=16 if algorithm == "dp4a" else 64,
                        ),
                    )
                    buffers = {b.name: b for b in fn.buffer_map.values()}
                    scalars = [p for p in fn.params if str(p.dtype) == "int32"]
                    self.assertEqual(len(scalars), int(dynamic))
                    for name in ["A", "AS", "Out"]:
                        if dynamic:
                            self.assertTrue(buffers[name].shape[0].same_as(scalars[0]))
                        else:
                            self.assertEqual(int(buffers[name].shape[0]), rows)

    def test_large_w8_dp4a_uses_configured_tiles_and_live_rows(self):
        for rows, dynamic in [(1, False), (8, False), (128, True)]:
            for columns in [4, 8, 16]:
                fn = make_prim(
                    dict(
                        kind="w8",
                        rows=rows,
                        outputs=640,
                        inputs=2560,
                        dtype="float16",
                        dynamic=dynamic,
                    ),
                    dict(algorithm="dp4a", block_m=min(rows, 4), block_n=columns, block_k=128),
                )
                buffers = {b.name: b for b in fn.buffer_map.values()}
                self.assertEqual(int(buffers["Weight"].shape[0]), 640)
                scalars = [p for p in fn.params if str(p.dtype) == "int32"]
                self.assertEqual(len(scalars), int(dynamic))
                if dynamic:
                    self.assertTrue(buffers["Output"].shape[0].same_as(scalars[0]))

    def test_fused_ffn_algorithms_share_live_rows_and_exact_packed_bytes(self):
        for dynamic, rows in [(False, 1), (False, 4), (True, 128)]:
            for algorithm, bm, bn in [("dp4a", min(rows, 4), 8), ("tensor_core", 16, 64)]:
                fn = make_prim(
                    dict(
                        kind="swiglu",
                        rows=rows,
                        outputs=640,
                        inputs=2560,
                        dtype="float16",
                        dynamic=dynamic,
                    ),
                    dict(algorithm=algorithm, block_m=bm, block_n=bn, block_k=128),
                )
                buffers = {b.name: b for b in fn.buffer_map.values()}
                scalars = [p for p in fn.params if str(p.dtype) == "int32"]
                self.assertEqual(len(scalars), int(dynamic))
                if dynamic:
                    self.assertTrue(buffers["Output"].shape[0].same_as(scalars[0]))
                words = int(buffers["Gate"].shape[1])
                self.assertEqual(words * (4 if algorithm == "dp4a" else 1), 2560)
                self.assertEqual(str(buffers["Output"].dtype), "float16")

    def test_fused_route_scans_live_assignments_and_bounds_all_tiles(self):
        from kernels.cuda.model.moe import route_metadata

        for bm in (8, 16):
            fn = explicit_rows(route_metadata.__wrapped__(128, block_m=bm, dynamic_rows=True))
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            buffers = {b.name: b for b in fn.buffer_map.values()}
            self.assertTrue(buffers["RelativeSlot"].shape[0].same_as(rows))
            self.assertEqual(int(buffers["TileExpert"].shape[0]), (1280 + bm - 1) // bm + 512)
            loops = []

            def visit(node):
                if isinstance(node, tir.For) and Analyzer().can_prove_equal(node.extent, rows * 10):
                    loops.append(node)

            tir.stmt_functor.post_order_visit(fn.body, visit)
            self.assertEqual(len(loops), 1)

    def test_lowered_q4_retains_explicit_live_rows(self):
        fn = explicit_rows(q4_projection.__wrapped__(4, 6144, 2560, dynamic_rows=True))
        (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
        buffers = {b.name: b for b in fn.buffer_map.values()}
        for name in ("A", "AS", "Out"):
            self.assertTrue(buffers[name].shape[0].same_as(rows))
        self.assertEqual(int(buffers["W"].shape[1]), 640)

    def test_lowered_head_keeps_fp32_logits(self):
        fn = explicit_rows(
            q4_projection.__wrapped__(4, 248320, 2560, output_dtype="float32", dynamic_rows=True)
        )
        buffers = {b.name: b for b in fn.buffer_map.values()}
        self.assertEqual(str(buffers["Out"].dtype), "float32")

    def test_direct_bf16_rows_do_not_change_private_split_layout(self):
        fn = explicit_rows(bf16_single.__wrapped__(320, 2560, 4, rows=4, dynamic_rows=True))
        (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
        buffers = {b.name: b for b in fn.buffer_map.values()}
        self.assertTrue(buffers["A"].shape[0].same_as(rows))
        self.assertTrue(buffers["Output"].shape[1].same_as(rows))
        self.assertEqual(int(buffers["Output"].shape[0]), 4)

    def test_large_route_histograms_keep_live_segment_strides_and_assignment_rows(self):
        for primitive in (moe.expert_histogram_local, moe.expert_histogram_finish):
            fn = explicit_rows(primitive.__wrapped__(1024, dynamic_rows=True))
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            buffers = {b.name: b for b in fn.buffer_map.values()}
            self.assertTrue(buffers["IDs"].shape[0].same_as(rows))
            self.assertTrue(buffers["Slots"].shape[0].same_as(rows))
            partial = buffers.get("Partial", buffers.get("Offsets"))
            self.assertEqual(int(partial.shape[0]), 512)
            for live in [1, 512, 1024]:
                stride = tir.stmt_functor.substitute(
                    partial.shape[1], {rows: tir.IntImm("int32", live)}
                )
                self.assertEqual(int(Analyzer().simplify(stride)), (live * 10 + 1023) // 1024)

    def test_affine_greedy_output_has_an_explicit_scalar_parameter(self):
        fn = explicit_rows(greedy_merge.__wrapped__(248320, 4, dynamic_rows=True))
        (scalar,) = [p for p in fn.params if str(p.dtype) == "int32"]
        self.assertEqual(scalar.name, "m")
        self.assertTrue(explicit_rows(fn).same_as(fn))

    def test_expert_rotation_uses_assignment_rows_without_implicit_affine_inference(self):
        fn = explicit_rows(
            rotation_a8.rotate_activation_a8.__wrapped__(80, 640, swiglu=True, dynamic_rows=True)
        )
        (scalar,) = [p for p in fn.params if str(p.dtype) == "int32"]
        for buffer in fn.buffer_map.values():
            if buffer.name in ("X", "Q", "S"):
                self.assertTrue(buffer.shape[0].same_as(scalar))

    def test_single_output_gate_keeps_the_dynamic_row_argument(self):
        for capacity in (8, 128):
            with self.subTest(capacity=capacity):
                fn = explicit_rows(
                    flash_next.scalar_bf16_projection.__wrapped__(capacity, 2560, dynamic_rows=True)
                )
                (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
                buffers = {b.name: b for b in fn.buffer_map.values()}
                self.assertTrue(buffers["A"].shape[0].same_as(rows))
                self.assertTrue(buffers["Output"].shape[0].same_as(rows))
                self.assertEqual(int(buffers["Weight"].shape[0]), 1)
                self.assertEqual(str(buffers["Weight"].dtype), "bfloat16")

    def test_hc_down_cache_and_streaming_keep_symbolic_partial_rows(self):
        for capacity in (8, 128):
            with self.subTest(capacity=capacity):
                fn = explicit_rows(
                    hyperconnection.hc_down_partial.__wrapped__(
                        capacity, 320, 10240, 4, dynamic_rows=True
                    )
                )
                (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
                buffers = {b.name: b for b in fn.buffer_map.values()}
                self.assertTrue(buffers["A"].shape[0].same_as(rows))
                self.assertTrue(buffers["Partials"].shape[1].same_as(rows))
                self.assertEqual(int(buffers["Partials"].shape[0]), 4)
                self.assertEqual(str(buffers["Partials"].dtype), "float32")

    def test_small_int8_projection_keeps_dynamic_rows_and_packed_operand_width(self):
        for capacity in (8, 128):
            with self.subTest(capacity=capacity):
                fn = explicit_rows(
                    int8_small_projection.__wrapped__(
                        capacity, 48, 2560, "float16", dynamic_rows=True
                    )
                )
                (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
                buffers = {b.name: b for b in fn.buffer_map.values()}
                for name in ("A", "Output", "TokenScale"):
                    self.assertTrue(buffers[name].shape[0].same_as(rows))
                self.assertEqual(int(buffers["A"].shape[1]), 640)
                self.assertEqual(str(buffers["A"].dtype), "int32")
                self.assertEqual(int(buffers["Weight"].shape[0]), 48)


if __name__ == "__main__":
    unittest.main()

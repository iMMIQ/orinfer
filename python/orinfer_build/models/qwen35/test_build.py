"""CPU contracts for native Qwen Q4A8 construction and lossless shard adoption."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from safetensors.numpy import save_file
from orinfer_build.models.qwen35.build import Builder, CAPACITY, configuration
from orinfer_build.models.qwen35.storage import Weights


def config():
    text = dict(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        vocab_size=248320,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        max_position_embeddings=262144,
        layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)],
        rope_parameters=dict(
            rope_type="default",
            rope_theta=10000000,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        ),
    )
    return dict(
        model_type="qwen3_5",
        text_config=text,
        quantization_config=dict(
            quant_method="orinfer_q4_int8",
            version=1,
            rotation="signed-block128",
            scale="fp16-output-row",
            compute_dtype="int8_quality",
        ),
    )


class BuildTests(unittest.TestCase):
    def test_joined_integer_banks_preserve_output_rows_scales_and_layout(self):
        from orinfer_build.common.files import file_hash
        from orinfer_build.weights.safetensors_source import local_header

        for codec in ("int8", "packed"):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                values = {}
                for i, rows in enumerate([4, 2, 2]):
                    dtype = np.int8 if codec == "int8" else np.uint32
                    values[f"p{i}.{codec}"] = np.arange(
                        rows * (256 if codec == "int8" else 32), dtype=dtype
                    ).reshape(rows, -1)
                    values[f"p{i}.scale"] = np.full(rows, (i + 1) * 0.01, np.float16)
                shard = source / "weights.safetensors"
                save_file(
                    values,
                    str(shard),
                    metadata={
                        f"orin.layout.{n}": "e8p-q4-row-u32"
                        if n.endswith("packed")
                        else "native_contiguous"
                        for n in values
                    },
                )
                (source / "model.safetensors.index.json").write_text(
                    json.dumps(
                        dict(
                            metadata={"orinfer.sha256": {shard.name: file_hash(shard)}},
                            weight_map={n: shard.name for n in values},
                        )
                    )
                )
                weights = Weights(source, root / "model", packed_k=128)
                weights.combined_projection("joined", ["p0", "p1", "p2"])
                actual = weights.array(f"joined.{codec}")
                if codec == "packed":
                    actual = actual.transpose(1, 0, 2).reshape(8, 32)
                np.testing.assert_array_equal(
                    actual, np.concatenate([values[f"p{i}.{codec}"] for i in range(3)])
                )
                np.testing.assert_array_equal(
                    weights.array("joined.scale"),
                    np.concatenate([values[f"p{i}.scale"] for i in range(3)]),
                )
                weights.finish()
                _, header = local_header(root / "model/cache/weights/runtime.safetensors")
                self.assertEqual(
                    set(header) - {"__metadata__"}, {f"joined.{codec}", "joined.scale"}
                )
                self.assertEqual(
                    header["__metadata__"][f"orin.layout.joined.{codec}"],
                    "e8p-q4-k128-u32" if codec == "packed" else "native_contiguous",
                )

    def test_rotation_projects_quantization_choices_without_register_parameters(self):
        def rotate(*args, **kwargs):
            return args, kwargs

        def quantize(*args, **kwargs):
            return args, kwargs

        selection = dict(
            configs=[
                dict(block_k=1024, registers=True, threads=128),
                dict(block_k=1024, registers=True, threads=256),
            ],
            default="0",
            selections=[dict(min_rows=1, max_rows=7, implementation="1")],
        )
        builder = SimpleNamespace(
            lock=None,
            signs={6144: "Signs"},
            row_tuner=SimpleNamespace(rotation=lambda *a, **kw: selection),
            row_choices=Builder.row_choices,
        )
        compiled = []
        builder.emit = lambda name, key, build, *args: compiled.append((name, key, build()))
        with (
            patch.dict(
                "sys.modules",
                {
                    "kernels.cuda.model.rotation_a8": SimpleNamespace(
                        rotate_activation_a8=rotate,
                        rotate_activation_tiles=rotate,
                        quantize_rotation_tiles=quantize,
                    )
                },
            ),
            patch(
                "orinfer_build.models.qwen35.build.primitive",
                side_effect=lambda fn, *a, **kw: (fn, kw),
            ),
        ):
            Builder.rotation(
                builder, "layer0/post/swiglu_rotate", 6144, "GateUp", swiglu=True, plane_rows=4096
            )
            first = list(compiled)
            compiled.clear()
            Builder.rotation(builder, "layer1/post/swiglu_rotate", 6144, "GateUp", swiglu=True)
        transforms = [entry for entry in first if "swiglu_rotate" in entry[0]]
        quantizations = [entry for entry in first if "swiglu_quantize" in entry[0]]
        self.assertEqual(len(transforms), 2)
        self.assertEqual(len(quantizations), 1)
        self.assertEqual({entry[2][1]["threads"] for entry in transforms}, {128, 256})
        self.assertTrue(all(entry[2][1]["registers"] for entry in transforms))
        self.assertEqual(quantizations[0][2], (quantize, dict(dynamic_rows=True, block_k=1024)))
        quantized = [entry for entry in compiled if "swiglu_quantize" in entry[0]]
        self.assertEqual(quantized[0][1:], quantizations[0][1:])

    def test_expert_schedules_follow_selected_rows_and_assignment_multiplier(self):
        from orinfer_build.models.qwen35 import moe

        def kernel(*args, **kwargs):
            return args, kwargs

        buffers = {}

        def fixed(name, dtype, shape):
            buffers[name] = dict(dtype=dtype, shape=shape)

        builder = SimpleNamespace(
            weights=SimpleNamespace(buffers=buffers, fixed=fixed),
            h=2048,
            f=512,
            book="book",
            text=dict(num_experts=256, num_experts_per_tok=8),
            moe_selection=dict(
                configs=[
                    dict(block_m=256, expand=False, cold_rows=32),
                    dict(block_m=16, expand=False, folded=True, minimum_blocks=6, sparse_rows=True),
                ],
                default="0",
                selections=[dict(min_rows=2, max_rows=128, implementation="1")],
            ),
            row_choices=Builder.row_choices,
        )
        launches = []
        builder.emit = lambda name, key, build, bindings, rows, **options: launches.append(
            (name, build(), bindings, rows, options)
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "kernels.cuda.model.moe": SimpleNamespace(
                        expert_offsets=kernel, expert_tiles=kernel, integer_experts=kernel
                    ),
                    "kernels.cuda.model.e8p_prefill": SimpleNamespace(
                        split_prefill_a8=kernel,
                        split_routed_prefill_a8=kernel,
                        e8p_prefill_grouped=kernel,
                    ),
                    "kernels.cuda.model.expert_partition": SimpleNamespace(
                        offsets=kernel, tiles=kernel
                    ),
                    "kernels.cuda.operators.op30_activation_quantization": SimpleNamespace(
                        _compile=lambda *args, **kwargs: None
                    ),
                },
            ),
            patch.object(moe, "primitive", side_effect=lambda fn, *args, **kw: (args, kw)),
        ):
            moe.projections(builder, "post", ("uw", "us"), ("dw", "ds"))
        self.assertEqual(buffers["MoeLow"]["shape"], [32768 * 2048 // 2])
        self.assertEqual(buffers["MoeSum"]["shape"], [32768])
        folded = [x for x in launches if x[0] == "post/expert_up/rows/2-128"]
        self.assertEqual(len(folded), 1)
        _, (dimensions, compile_options), bindings, rows, options = folded[0]
        self.assertEqual(dimensions, (4096, 1024, 2048, 256, 8))
        from orinfer_build.tuning.expert_tuning import tile_capacity

        self.assertEqual(tile_capacity(dimensions[0], dimensions[3], dimensions[4], 16), 2304)
        self.assertEqual(rows, 32768)
        self.assertEqual(options["row_multiplier"], 8)
        self.assertTrue(compile_options["dynamic_rows"])
        self.assertEqual(bindings["P"], "uw")
        self.assertEqual(bindings["Sum"], "MoeSum")
        names = {x[0] for x in launches}
        for role in (
            "expert_up_cold_planes",
            "expert_up_cold",
            "expert_down_cold_planes",
            "expert_down_cold",
        ):
            self.assertIn("post/" + role, names)
            self.assertIn("post/" + role + "/skip/2-128", names)
        self.assertIn("MoeColdTileExpert", buffers)
        for role in ("expert_up_planes", "expert_down_planes"):
            self.assertIn("post/" + role, names)
            self.assertIn("post/" + role + "/skip/1-1", names)
            self.assertIn("post/" + role + "/skip/129-4096", names)

    def test_four_billion_requires_tied_head_and_matching_attention(self):
        candidate = config()
        t = candidate["text_config"]
        t.update(
            hidden_size=2560,
            intermediate_size=9216,
            num_hidden_layers=32,
            linear_num_value_heads=32,
            num_attention_heads=16,
            tie_word_embeddings=True,
            layer_types=t["layer_types"][:32],
        )
        candidate["quantization_config"].update(
            quant_method="orinfer_e8p_q4_int8",
            coefficients=[4, 1],
            basis_tensor="orinfer.e8p_q4.basis",
            layout="u32-low-coarse-high-residual",
            embedding="e8p_q4",
        )
        self.assertEqual(configuration(candidate, 262144)["hidden_size"], 2560)
        for key, value in (("tie_word_embeddings", False), ("num_key_value_heads", 2)):
            with self.subTest(key=key), patch.dict(t, {key: value}):
                with self.assertRaises(ValueError):
                    configuration(candidate, 262144)

    def test_expert_repack_and_assembled_banks_do_not_duplicate_unbound_weights(self):
        from orinfer_build.weights.safetensors_source import local_header

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            codes = np.arange(2 * 3 * 32, dtype=np.uint32).reshape(2, 3, 32)
            shard = source / "experts.safetensors"
            save_file(
                {"Experts.packed": codes, "Unbound": np.ones((4, 4), np.float16)},
                str(shard),
                metadata={"orin.layout.Experts.packed": "e8p-q4-row-u32"},
            )
            before = hashlib.sha256(shard.read_bytes()).hexdigest()
            index = dict(
                metadata={"orinfer.sha256": {shard.name: before}},
                weight_map={"Experts.packed": shard.name, "Unbound": shard.name},
            )
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            weights = Weights(source, root / "model", packed_k=128)
            weights.immutable("Experts.packed")
            bank = weights.array("Experts.packed")
            self.assertEqual(bank.shape, (2, 2, 3, 16))
            np.testing.assert_array_equal(bank.transpose(0, 2, 1, 3).reshape(codes.shape), codes)
            weights.constant("Assembled", np.ones((2, 4, 4), np.int8))
            weights.finish()
            _, header = local_header(root / "model/cache/weights" / shard.name)
            self.assertEqual(set(header) - {"__metadata__"}, {"Experts.packed"})
            self.assertEqual(hashlib.sha256(shard.read_bytes()).hexdigest(), before)

    def test_moe_draft_accepts_a_jit_factory_without_a_name_attribute(self):
        from orinfer_build.models.qwen35 import moe

        def quantize(*args, **options):
            return args, options

        wrapped = SimpleNamespace(func=quantize)
        names = (
            "router_projection",
            "router_topk",
            "expert_histogram_local",
            "expert_histogram_offsets",
            "expert_histogram_finish",
            "expert_offsets",
            "expert_tiles",
            "expert_dispatch",
            "integer_experts",
            "moe_combine",
        )
        kernels = SimpleNamespace(**{name: quantize for name in names})
        builder = SimpleNamespace(
            weights=SimpleNamespace(combined=lambda name, weights: name),
            text=dict(num_experts=256, num_experts_per_tok=8),
            h=2048,
            f=512,
            book="book",
            projection=lambda *args, **kw: None,
            lock=None,
            moe_router_tuner=SimpleNamespace(
                router=lambda *_: dict(configs=[{}], default="0", selections=[]),
                topk=lambda *_: dict(configs=[dict(algorithm="block")], default="0", selections=[]),
            ),
            row_choices=lambda selection, *_: [("", selection["configs"][0])],
        )
        launches = []

        def emit(name, key, build, bindings, rows, **options):
            launches.append((name, key, build(), rows, options))

        builder.emit = emit
        with (
            patch.dict(
                "sys.modules",
                {
                    "kernels.cuda.model.moe": kernels,
                    "kernels.cuda.model.rotation_a8": SimpleNamespace(),
                    "kernels.cuda.operators.op30_activation_quantization": SimpleNamespace(
                        _compile=wrapped
                    ),
                },
            ),
            patch.object(moe, "workspace"),
            patch.object(moe, "banks", return_value=[("up", "us"), ("down", "ds")]),
            patch.object(moe, "primitive", side_effect=lambda fn, *a, **kw: (fn, a, kw)),
        ):
            moe.build(builder, "mtp", "mtp.layers.0.mlp", rows=512, draft=True)
        quantizations = [
            item for item in launches if item[0] in ("mtp/down_quant", "mtp/expert_rotate")
        ]
        self.assertEqual(len(quantizations), 2)
        self.assertEqual([item[3] for item in quantizations], [512, 4096])
        for _, key, (factory, args, options), _, _ in quantizations:
            self.assertTrue(key.startswith("moe-quantize"))
            self.assertIs(factory, wrapped)
            self.assertEqual(args, (512, 512, False, True, 256))
            self.assertNotIn("dynamic_rows", options)

    def test_moe_geometry_keeps_shared_experts_and_router_explicit(self):
        candidate = config()
        candidate["model_type"] = "qwen3_5_moe"
        t = candidate["text_config"]
        t.update(
            hidden_size=2048,
            intermediate_size=512,
            num_hidden_layers=40,
            linear_num_value_heads=32,
            num_attention_heads=16,
            num_key_value_heads=2,
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
            mlp_only_layers=[],
            layer_types=t["layer_types"][:40],
        )
        self.assertEqual(configuration(candidate, 262144)["num_experts"], 256)
        for key, value in (
            ("num_experts_per_tok", 10),
            ("shared_expert_intermediate_size", 640),
            ("tie_word_embeddings", True),
            ("mlp_only_layers", [0]),
        ):
            with self.subTest(key=key), patch.dict(t, {key: value}):
                with self.assertRaises(ValueError):
                    configuration(candidate, 262144)

    def test_medium_dense_geometry_requires_matching_gdn_and_gqa(self):
        candidate = config()
        t = candidate["text_config"]
        t.update(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=32,
            linear_num_value_heads=32,
            num_attention_heads=16,
            layer_types=t["layer_types"][:32],
        )
        self.assertEqual(configuration(candidate, 262144)["hidden_size"], 4096)
        for key, value in (("linear_num_value_heads", 48), ("num_key_value_heads", 2)):
            with self.subTest(key=key), patch.dict(t, {key: value}):
                with self.assertRaises(ValueError):
                    configuration(candidate, 262144)

    def test_native_shard_layout_is_explicit_without_requantization(self):
        from orinfer_build.common.files import file_hash
        from orinfer_build.weights.safetensors_source import local_header

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            shard = source / "weights.safetensors"
            values = {"W.int8": np.arange(-16, 16, dtype=np.int8).reshape(4, 8)}
            save_file(values, shard)
            original = file_hash(shard)
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    dict(
                        metadata={"orinfer.sha256": {shard.name: original}},
                        weight_map={"W.int8": shard.name},
                    )
                )
            )
            weights = Weights(source, root / "model")
            weights.immutable("W.int8")
            np.testing.assert_array_equal(weights.array("W.int8"), values["W.int8"])
            _, header = local_header(root / "model/cache/weights" / shard.name)
            self.assertEqual(header["__metadata__"]["orin.layout.W.int8"], "native_contiguous")
            self.assertEqual(file_hash(shard), original)

    def test_small_dense_geometry_and_tied_head(self):
        candidate = config()
        t = candidate["text_config"]
        t.update(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=24,
            linear_num_value_heads=16,
            num_attention_heads=8,
            num_key_value_heads=2,
            tie_word_embeddings=True,
            layer_types=t["layer_types"][:24],
        )
        candidate["quantization_config"].update(
            quant_method="orinfer_e8p_q4_int8",
            coefficients=[4, 1],
            basis_tensor="orinfer.e8p_q4.basis",
            layout="u32-low-coarse-high-residual",
            embedding="e8p_q4",
            vision="e8p_q4",
        )
        self.assertEqual(configuration(candidate, 262144)["hidden_size"], 2048)
        t["tie_word_embeddings"] = False
        with self.assertRaises(ValueError):
            configuration(candidate, 262144)

    def test_tied_head_binds_the_shared_compressed_weight(self):
        builder = Builder.__new__(Builder)
        builder.weights = SimpleNamespace(
            immutable=lambda n: n, packed_k=128, index={"weight_map": {}}, buffers={}
        )
        builder.q4_codec, builder.book = "e8p_q4", "book"
        launches = []
        builder.emit = lambda name, key, build, bindings, rows, **options: launches.append(bindings)
        builder.projection(
            "head/project",
            "embedding",
            248320,
            2048,
            "HeadLogits",
            head=True,
            rows=1,
            dynamic=False,
        )
        self.assertEqual(
            launches,
            [
                dict(
                    A="HeadA8",
                    W="embedding.packed",
                    S="embedding.scale",
                    AS="HeadAS",
                    Out="HeadLogits",
                    Book="book",
                )
            ],
        )

    def test_prefetch_views_bind_the_owning_kv_and_exported_abi(self):
        first, merge = (lambda: None), (lambda: None)
        first.__name__, merge.__name__ = "attention", "merge"
        selection = dict(
            configs=[
                dict(block_m=64, block_n=64, head_group=1, max_splits=8, threads=256, stages=1)
            ],
            default="0",
            selections=[],
        )
        builder = Builder.__new__(Builder)
        builder.context, builder.lock = 262144, None
        builder.qh, builder.kvh = 24, 4
        builder.dimension, builder.standard, builder.kv_width = 256, False, 1024
        builder.row_tuner = SimpleNamespace(
            attention=lambda _: selection,
            causal_attention=lambda _: dict(selection["configs"][0], block_m=16, head_group=6),
        )
        launches = []
        builder.emit = lambda name, key, build, bindings, rows: launches.append(
            (name, build(), bindings, rows)
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "kernels.cuda.model.attention_rows": SimpleNamespace(
                        causal_attention_int8=first, merge_attention_int8=merge
                    )
                },
            ),
            patch("orinfer_build.models.qwen35.build.primitive", side_effect=lambda fn, *a, **k: k),
        ):
            builder.attention(
                "draft_attention",
                q="query",
                gate="gate",
                position="pos",
                length="length",
                output="output",
                prefix="Draft",
                rows=7,
            )
        _, options, bindings, rows = launches[0]
        self.assertEqual((options["threads"], options["stages"], rows), (256, 1, 7))
        for name, expected in (
            ("k", "KPages"),
            ("v", "VPages"),
            ("ks", "KPagesScale"),
            ("vs", "VPagesScale"),
        ):
            self.assertEqual(bindings[f"live_{name}"], "Draft" + expected)
        self.assertEqual(bindings["Lengths"], "length")
        self.assertTrue(options["independent_queries"])
        merge = next(x for x in launches if x[0] == "draft_attention_merge")
        self.assertNotIn("stages", merge[1])
        self.assertNotIn("live_k", merge[2])

    def test_staged_attention_preserves_private_kv_and_obeys_measured_decoder_ranges(self):
        first, merge, decoder = (lambda: None), (lambda: None), (lambda: None)
        first.__name__, merge.__name__, decoder.__name__ = "attention", "merge", "decode"
        baseline = dict(block_m=64, block_n=64, head_group=1, max_splits=8, threads=256, stages=1)
        staged = dict(
            block_m=128,
            block_n=64,
            head_group=2,
            max_splits=1,
            threads=512,
            stages=1,
            decoded_kv=True,
        )
        selection = dict(
            configs=[baseline, staged],
            default="1",
            selections=[dict(min_rows=1, max_rows=73, multiple_of=1, implementation="0")],
        )
        builder = Builder.__new__(Builder)
        builder.context, builder.lock, builder.qh, builder.kvh = 262144, None, 16, 2
        builder.dimension, builder.standard, builder.kv_width = 256, False, 512
        builder.row_tuner = SimpleNamespace(
            attention=lambda _: selection,
            staged_attention=lambda _, selection: selection,
            causal_attention=lambda _: dict(baseline, block_m=16, head_group=8),
        )
        buffers = {}
        builder.weights = SimpleNamespace(
            buffers=buffers, fixed=lambda name, dtype, shape: buffers.update({name: (dtype, shape)})
        )
        launches = {}
        builder.emit = lambda name, key, build, bindings, rows: launches.update(
            {name: (build(), bindings, rows)}
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "kernels.cuda.model.attention_rows": SimpleNamespace(
                        causal_attention_int8=first, merge_attention_int8=merge
                    ),
                    "kernels.cuda.model.kv_int8": SimpleNamespace(dequantize_attention_kv=decoder),
                },
            ),
            patch("orinfer_build.models.qwen35.build.primitive", side_effect=lambda fn, *a, **k: k),
        ):
            builder.attention(
                "layer3/private/attention",
                q="query",
                gate="gate",
                position="pos",
                length="length",
                output="output",
                prefix="L3_",
                staging=True,
            )
        role = "layer3/private/attention"
        self.assertIn(role + "_decode/skip/1-8", launches)
        self.assertIn(role + "_decode/skip/9-73", launches)
        self.assertEqual(launches[role + "_decode"][1]["K"], "L3_KPages")
        self.assertEqual(launches[role + "_decode"][1]["DK"], "AttentionDecodedK")
        self.assertTrue(launches[role][0]["decoded_kv"])
        self.assertEqual(launches[role][1]["live_k"], "AttentionDecodedK")
        self.assertFalse(launches[role + "/rows/9-73"][0]["decoded_kv"])
        self.assertEqual(launches[role + "/rows/9-73"][1]["K"], "L3_KPages")
        self.assertTrue(launches[role + "/rows/1-8"][0]["independent_queries"])
        self.assertEqual(launches[role + "/rows/1-8"][2], 8)
        self.assertEqual(
            buffers,
            {name: ("f16", [262144, 512]) for name in ("AttentionDecodedK", "AttentionDecodedV")},
        )

    def test_projection_bounds_measured_choices_and_preserves_exact_aot(self):
        builder = Builder.__new__(Builder)
        builder.weights = SimpleNamespace(
            immutable=lambda name: name, packed_k=128, index={"weight_map": {}}, buffers={}
        )
        builder.q4_codec = "e8p_q4"
        builder.book = "selectors"
        builder.lock = None
        builder.compile_projection = lambda *args, **kw: [("/rows/5-7", {"block_m": 8}, object())]
        launches = []
        builder.emit = lambda name, key, build, bindings, rows, **options: launches.append(
            (name, rows, options.get("dynamic", True), bindings)
        )
        for kind in ("q4", "w8"):
            launches.clear()
            builder.projection(
                "projection", "weight", 65, 256, "output", stride=68, offset=3, kind=kind
            )
            self.assertEqual(
                [(name, rows, dynamic) for name, rows, dynamic, _ in launches],
                [("projection", CAPACITY, True)]
                + [(f"projection/rows/{r}-{r}", r, False) for r in (1, 2, 3, 4, 8, 16, 32)]
                + [("projection/rows/5-7", 128, True)],
            )
            output = "Out" if kind == "q4" else "Output"
            for _, _, _, bindings in launches:
                self.assertEqual(
                    bindings[output], dict(kind="buffer_slice", name="output", offset=6)
                )

    def test_equivalent_projection_strides_share_the_compiled_identity(self):
        builder = Builder.__new__(Builder)
        builder.weights = SimpleNamespace(
            immutable=lambda name: name, packed_k=128, index={"weight_map": {}}, buffers={}
        )
        builder.q4_codec, builder.book = "e8p_q4", "selectors"
        keys = []
        builder.emit = lambda name, key, *args, **kwargs: keys.append(key)
        for stride in (None, 64, 68):
            builder.projection(
                "projection", "weight", 64, 256, "output", stride=stride, rows=1, dynamic=False
            )
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])

    def test_bounded_tuning_is_shared_across_layer_weights_but_not_strides(self):
        builder = Builder.__new__(Builder)
        builder.weights = SimpleNamespace(
            immutable=lambda name: name, packed_k=128, index={"weight_map": {}}, buffers={}
        )
        builder.q4_codec, builder.book, builder.lock = "e8p_q4", "selectors", None
        builder.emit = lambda *args, **kwargs: None
        with patch.object(builder, "compile_projection", return_value=[]) as compile:
            for weight, stride in [("layer0", None), ("layer1", None), ("layer2", 68)]:
                builder.projection("projection", weight, 64, 256, "output", stride=stride)
            self.assertEqual(compile.call_count, 2)

    def test_two_stage_codec_has_an_explicit_contract(self):
        candidate = config()
        quant = candidate["quantization_config"]
        quant.update(
            quant_method="orinfer_e8p_q4_int8",
            coefficients=[4, 1],
            basis_tensor="orinfer.e8p_q4.basis",
            layout="u32-low-coarse-high-residual",
        )
        self.assertEqual(configuration(candidate, 8192)["hidden_size"], 5120)
        for key, value in [
            ("coefficients", [3, 1]),
            ("coefficients", [4.0, 1.0]),
            ("basis_tensor", "other"),
            ("layout", "high-coarse"),
        ]:
            saved = quant[key]
            quant[key] = value
            with self.assertRaises(ValueError):
                configuration(candidate, 8192)
            quant[key] = saved

    def test_compiler_rejects_wrong_math_codec_and_context(self):
        self.assertEqual(configuration(config(), 262144)["num_hidden_layers"], 64)
        for component, key, value in [
            ("text_config", "head_dim", 128),
            ("text_config", "rms_norm_eps", 1e-5),
            ("quantization_config", "rotation", "none"),
        ]:
            d = config()
            d[component][key] = value
            with self.assertRaises(ValueError):
                configuration(d, 262144)
        for context in [0, 262145]:
            with self.assertRaises(ValueError):
                configuration(config(), context)

    def test_e8p_repack_is_lossless_and_publishes_one_weight_representation(self):
        from orinfer_build.packaging.compact import write_shard
        from orinfer_build.common.files import file_hash
        from orinfer_build.weights.safetensors_source import local_header

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            raw = np.arange(3 * 64, dtype=np.uint32).reshape(3, 64) * 1234567
            values = {
                "W.packed": raw,
                "W.scale": np.array([0.1, 0.2, 0.3], np.float16),
                "Sensitive": np.array([0x8000, 0x7F80, 0x3F81, 0x0001], np.uint16),
            }
            tensors = []
            for name, value in values.items():
                path = root / (name + ".bin")
                value.tofile(path)
                tensors.append(
                    dict(
                        name=name,
                        dtype="U32"
                        if name.endswith("packed")
                        else "F16"
                        if name.endswith("scale")
                        else "BF16",
                        shape=list(value.shape),
                        segments=[(path, 0, value.nbytes)],
                    )
                )
            shard = source / "layer.safetensors"
            write_shard(shard, tensors, {"orin.layout.W.packed": "e8p-q4-row-u32"})
            source_sha = file_hash(shard)
            index = dict(
                metadata={"orinfer.sha256": {shard.name: source_sha}},
                weight_map={name: shard.name for name in values},
            )
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            for packed_k in (64, 128):
                destination = root / str(packed_k)
                weights = Weights(source, destination, packed_k=packed_k)
                for name in values:
                    weights.immutable(name)
                actual = weights.array("W.packed")
                np.testing.assert_array_equal(actual.transpose(1, 0, 2).reshape(raw.shape), raw)
                self.assertEqual(weights.buffers["W.packed"]["layout"], f"e8p-q4-k{packed_k}-u32")
                self.assertEqual(
                    weights.buffers["W.packed"]["shape"], [512 // packed_k, 3, packed_k // 8]
                )
                for name in ("W.scale", "Sensitive"):
                    payload, _ = weights.raw(name)
                    self.assertEqual(payload, values[name].tobytes())
                    self.assertEqual(
                        weights.buffers[name]["data"]["sha256"], hashlib.sha256(payload).hexdigest()
                    )
                target = destination / "cache/weights/layer.safetensors"
                self.assertNotEqual(target.stat().st_ino, shard.stat().st_ino)
                begin, header = local_header(target)
                for name in ("W.scale", "Sensitive"):
                    self.assertEqual(
                        header["__metadata__"][f"orin.layout.{name}"], "native_contiguous"
                    )
                self.assertEqual(
                    target.stat().st_size - begin, sum(v.nbytes for v in values.values())
                )
                self.assertEqual(set(header) - {"__metadata__"}, set(values))
                self.assertEqual(list(target.parent.iterdir()), [target])
                self.assertEqual(file_hash(shard), source_sha)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                index["metadata"]["orinfer.sha256"][shard.name] = "0" * 64
                (source / "model.safetensors.index.json").write_text(json.dumps(index))
                Weights(source, root / "bad", packed_k=128).immutable("W.packed")

    def test_repack_rejects_wrong_codec_or_incomplete_k_tile(self):
        for dtype, shape, layout in [
            (np.uint8, (3, 16), "e8p-q4-row-u32"),
            (np.uint32, (3, 15), "e8p-q4-row-u32"),
            (np.uint32, (3, 16), "q4-row-packed-little-nibbles"),
            (np.uint32, (1, 2, 3, 16), "e8p-q4-row-u32"),
        ]:
            with (
                self.subTest(dtype=dtype, shape=shape, layout=layout),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                shard = source / "layer.safetensors"
                save_file(
                    {"W.packed": np.zeros(shape, dtype)},
                    str(shard),
                    metadata={"orin.layout.W.packed": layout},
                )
                index = dict(
                    metadata={
                        "orinfer.sha256": {
                            shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()
                        }
                    },
                    weight_map={"W.packed": shard.name},
                )
                (source / "model.safetensors.index.json").write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError, "Invalid source E8P weight layout"):
                    Weights(source, root / "destination", packed_k=128).immutable("W.packed")

    def test_shards_are_linked_and_packed_layout_and_scale_bits_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            destination = root / "model"
            packed = np.arange(2 * 64, dtype=np.uint8).reshape(2, 64)
            scale = np.array([0.001, 0.02], np.float16)
            save_file(
                {"W.packed": packed, "W.scale": scale},
                str(source / "layer.safetensors"),
                metadata={
                    "orin.layout.W.packed": "q4-row-packed-little-nibbles",
                    "orin.layout.W.scale": "native_contiguous",
                },
            )
            index = dict(
                metadata={
                    "orinfer.sha256": {
                        "layer.safetensors": hashlib.sha256(
                            (source / "layer.safetensors").read_bytes()
                        ).hexdigest()
                    }
                },
                weight_map={"W.packed": "layer.safetensors", "W.scale": "layer.safetensors"},
            )
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            weights = Weights(source, destination)
            weights.immutable("W.packed")
            weights.immutable("W.scale")
            self.assertEqual(weights.buffers["W.packed"]["layout"], "q4-row-packed-little-nibbles")
            self.assertEqual(
                weights.buffers["W.scale"]["data"]["sha256"],
                hashlib.sha256(scale.tobytes()).hexdigest(),
            )
            target = destination / "cache/weights/layer.safetensors"
            self.assertEqual(target.stat().st_ino, (source / "layer.safetensors").stat().st_ino)
            weights.constant("Control", np.arange(8, dtype=np.int32))
            signs = np.array([-1, 1, -1, 1], np.int8)
            weights.constant("Signs", signs)
            selectors = np.arange(256, dtype=np.uint32).reshape(256, 1)
            weights.constant("Selectors", selectors)
            weights.finish()
            from safetensors import safe_open

            with safe_open(
                destination / "cache/weights/runtime.safetensors", framework="numpy"
            ) as runtime:
                np.testing.assert_array_equal(runtime.get_tensor("Signs"), signs)
                self.assertEqual(runtime.metadata()["orin.layout.Signs"], "native_contiguous")
                np.testing.assert_array_equal(runtime.get_tensor("Selectors"), selectors)
            self.assertEqual(weights.buffers["Selectors"]["dtype"], "u32")
            self.assertEqual(
                index["metadata"]["orinfer.sha256"]["layer.safetensors"],
                hashlib.sha256(target.read_bytes()).hexdigest(),
            )
            raw = bytearray((source / "layer.safetensors").read_bytes())
            raw[-1] ^= 1
            (source / "layer.safetensors").write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                Weights(source, root / "other").immutable("W.packed")

    def test_asr_standard_attention_uses_the_shared_q4_contract(self):
        candidate = config()
        candidate["model_type"] = "qwen3_asr"
        candidate["text_config"].update(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            vocab_size=151936,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            tie_word_embeddings=True,
            attn_output_gate=False,
            max_position_embeddings=65536,
            layer_types=["full_attention"] * 28,
            rope_parameters=dict(rope_type="default", rope_theta=1000000),
        )
        candidate["quantization_config"].update(
            quant_method="orinfer_e8p_q4_int8",
            coefficients=[4, 1],
            basis_tensor="orinfer.e8p_q4.basis",
            layout="u32-low-coarse-high-residual",
            embedding="e8p_q4",
            rotated_w8_layers=[0, 1, 2, 3],
        )
        self.assertEqual(configuration(candidate, 65536)["head_dim"], 128)
        candidate["text_config"]["attn_output_gate"] = True
        with self.assertRaises(ValueError):
            configuration(candidate, 65536)


if __name__ == "__main__":
    unittest.main()

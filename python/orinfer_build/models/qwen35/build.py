"""Build native dense/MoE Qwen3.5 data and a schema-1 Rust/TileLang package.

Consumes completed scalar or two-stage E8P Q4A8 checkpoints.
Weight shards are linked or losslessly K-repacked; transforms, scales and the
single packed weight copy remain fixed. All online scheduling and layer order are registered Rust code.
"""

from orinfer_build.common.paths import ROOT, gpu_lock_path, work_path


import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from orinfer_build.common.gpu import gpu_lock

import numpy as np
from orinfer_build.packaging.publication import atomic_model
from orinfer_build.common.files import file_hash, write_json
from orinfer_build.packaging.package import config_signature, execution_identity, model_library
from orinfer_build.packaging.compact import compact_package
from orinfer_build.models.qwen35.storage import Weights
from orinfer_build.quantization.row_q4 import MAGNITUDES
from orinfer_build.compiler.compile import primitive


CAPACITY = 4096
LANES = 128
HEAD_ROWS = 512


def configuration(config, context):
    if config.get("model_type") not in ("qwen3_5", "qwen3_5_moe", "qwen3_asr"):
        raise ValueError("Expected Qwen3_5 checkpoint")
    t = config["text_config"]
    expected = dict(
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
    )
    standard = config["model_type"] == "qwen3_asr"
    if standard:
        expected = dict(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            vocab_size=151936,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            tie_word_embeddings=True,
            attn_output_gate=False,
        )
    elif config["model_type"] == "qwen3_5_moe":
        expected.update(
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
        )
    elif t.get("hidden_size") == 2048:
        expected.update(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=24,
            linear_num_value_heads=16,
            num_attention_heads=8,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    elif t.get("hidden_size") == 2560:
        expected.update(
            hidden_size=2560,
            intermediate_size=9216,
            num_hidden_layers=32,
            linear_num_value_heads=32,
            num_attention_heads=16,
            tie_word_embeddings=True,
        )
    elif t.get("hidden_size") == 4096:
        expected.update(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=32,
            linear_num_value_heads=32,
            num_attention_heads=16,
        )
    layers = (
        ["full_attention"] * expected["num_hidden_layers"]
        if standard
        else [
            "full_attention" if i % 4 == 3 else "linear_attention"
            for i in range(expected["num_hidden_layers"])
        ]
    )
    if (
        any(t.get(k) != v or type(t.get(k)) is not type(v) for k, v in expected.items())
        or t.get("layer_types") != layers
    ):
        raise ValueError("Unsupported Qwen arithmetic geometry")
    rope = t["rope_parameters"]
    expected_rope = (
        dict(rope_type="default", rope_theta=1000000)
        if standard
        else dict(
            rope_type="default",
            rope_theta=10000000,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        )
    )
    if any(rope.get(k) != v for k, v in expected_rope.items()):
        raise ValueError("Unsupported dense Qwen RoPE contract")
    q = config.get("quantization_config", {})
    if any(
        q.get(k) != v or type(q.get(k)) is not type(v)
        for k, v in dict(
            version=1,
            rotation="signed-block128",
            scale="fp16-output-row",
            compute_dtype="int8_quality",
        ).items()
    ):
        raise ValueError("Expected our completed table-Q4A8 conversion")
    if standard:
        from orinfer_build.quantization.qwen_e8p import rotated_w8_layers

        if q.get("rotated_w8_layers") != rotated_w8_layers(config):
            raise ValueError("ASR requires its quality-first W8 layer policy")
    if q.get("quant_method") not in ("orinfer_q4_int8", "orinfer_e8p_q4_int8"):
        raise ValueError("Unsupported Qwen Q4 codec")
    if q["quant_method"] == "orinfer_e8p_q4_int8":
        if any(
            q.get(k) != v
            for k, v in dict(
                coefficients=[4, 1],
                basis_tensor="orinfer.e8p_q4.basis",
                layout="u32-low-coarse-high-residual",
            ).items()
        ) or any(type(v) is not int for v in q["coefficients"]):
            raise ValueError("Unsupported two-stage E8P-Q4 contract")
    if t["tie_word_embeddings"] and (
        q["quant_method"] != "orinfer_e8p_q4_int8" or q.get("embedding") != "e8p_q4"
    ):
        raise ValueError("Tied embedding/head requires the shared E8P Q4 bank")
    if any(q.get(role) not in (None, "e8p_q4") for role in ("embedding", "vision")):
        raise ValueError("Unsupported embedding or vision codec")
    if (
        any(q.get(role) for role in ("embedding", "vision"))
        and q["quant_method"] != "orinfer_e8p_q4_int8"
    ):
        raise ValueError("Embedding/vision Q4 requires the two-stage E8P codec")
    if type(context) is not int or not 1 <= context <= min(t["max_position_embeddings"], 262144):
        raise ValueError("Invalid context capacity")
    return t


class Builder:
    def __init__(self, source, root, output, cache, context, lock):
        from orinfer_build.models.qwen35.compiler import Aot
        from orinfer_build.packaging.frontend import prepare_chat_template

        self.source, self.root, self.output = source, root, output
        self.config = json.loads((source / "config.json").read_text())
        self.text = configuration(self.config, context)
        t = self.text
        self.standard = self.config["model_type"] == "qwen3_asr"
        self.dimension = t["head_dim"]
        self.tied = t.get("tie_word_embeddings", False)
        self.h, self.v = t["hidden_size"], t["vocab_size"]
        self.f = t["intermediate_size"]
        self.moe = self.config["model_type"] == "qwen3_5_moe"
        self.hk, self.hv = t.get("linear_num_key_heads", 1), t.get("linear_num_value_heads", 16)
        self.qh, self.kvh = t["num_attention_heads"], t["num_key_value_heads"]
        self.qkv, self.mixdim = (2 * self.hk + self.hv) * 128, self.hv * 128
        self.qproj = self.qh * self.dimension * (1 if self.standard else 2)
        self.kv_width = self.kvh * self.dimension
        if self.standard:
            self.mixdim = self.qh * self.dimension
        self.fullx = self.qproj + 2 * self.kv_width
        self.architecture = "qwen3_asr" if self.standard else "qwen3_5"
        self.head_weight = (
            "model.language_model.embed_tokens.weight" if self.tied else "lm_head.weight"
        )
        self.embedding_q4 = self.config["quantization_config"].get("embedding") == "e8p_q4"
        self.q4_codec = (
            "e8p_q4"
            if self.config["quantization_config"]["quant_method"] == "orinfer_e8p_q4_int8"
            else "q4"
        )
        state = json.loads((source / "progress.json").read_text())
        if not state.get("complete"):
            raise ValueError("Quantization is still running; no partial model may be published")
        self.state = state
        self.context = context
        self.lock = lock
        self.gdn_execution = (
            None if self.standard else dict(prefill="sequential", decode=dict(kind="dense"))
        )
        self.weights = Weights(source, root, packed_k=128 if self.q4_codec == "e8p_q4" else 0)
        prepare_chat_template(source, root / "chat_template.jinja")
        self.aot = Aot(
            root / "cache", output, cache, self.weights.buffers, compile_lock=lambda: gpu_lock(lock)
        )
        self.aot.identity["weights"] = hashlib.sha256(state["contract"].encode()).hexdigest()
        self.aot.identity["context"] = str(context)
        from orinfer_build.tuning.row_tuning import RowTuner

        self.row_tuner = RowTuner(output / "row-tuning", self.aot.identity, self.text)
        self.workspace()
        from orinfer_build.models.qwen35 import vision

        self.vision = None
        self.audio = None
        if self.standard:
            from orinfer_build.models.qwen_asr.audio import build as build_audio

            build_audio(self)
        else:
            vision.build(self)
        self.forward()
        self.head()
        from orinfer_build.models.qwen35 import mtp

        self.mtp = None
        if not self.standard:
            mtp.build(self)
        from orinfer_build.models.qwen35 import parallel

        parallel.build(self)
        from orinfer_build.models.qwen35 import packet

        packet.build(self)
        from orinfer_build.models.qwen35 import causal_parallel

        if not self.standard:
            causal_parallel.build(self)
        from orinfer_build.models.qwen35 import commit

        if not self.standard:
            commit.build(self)
        from orinfer_build.models.qwen35 import mtp_packets

        self.packet_spec = {}
        mtp_packets.build(self)

    def workspace(self):
        w = self.weights
        h, f, v = self.h, self.f, self.v
        c = CAPACITY
        for name, dtype, shape in [
            ("Hidden", "f16", [c, h]),
            ("R0", "f32", [c, h]),
            ("R1", "f32", [c, h]),
            ("Norm", "f16", [c, h]),
            ("QKV", "f16", [c, self.qkv]),
            ("Zout", "f16", [c, self.mixdim]),
            ("AB", "f16", [c, 2 * self.hv]),
            ("g", "f32", [c, self.hv]),
            ("Beta", "f32", [c, self.hv]),
            ("Q", "f16", [16, c, 128]),
            ("K", "f16", [16, c, 128]),
            ("V", "f16", [self.hv, c, 128]),
            ("HistoryOut", "f16", [1, 3, self.qkv]),
            ("PositionOut", "i32", [1]),
            ("GdnScratch", "f32", [1]),
            ("Y", "f16", [c, self.mixdim]),
            ("MixerIn", "f16", [c, self.mixdim]),
            ("FullX", "f16", [c, self.fullx]),
            ("FullQ", "f16", [c, self.qh, self.dimension]),
            ("FullGate", "f16", [c, self.qh, self.dimension]),
            ("Mix", "f16", [c, h]),
            ("GateUp", "f16", [2 * c, f]),
            ("RotationValues", "f16", [c, f]),
            ("RotationMaxima", "f32", [c, math.ceil(f / 256)]),
            ("AttentionPartial", "f32", [self.qh, 32, 64, self.dimension]),
            ("AttentionMaximum", "f32", [self.qh, 32, 64]),
            ("AttentionDenominator", "f32", [self.qh, 32, 64]),
            ("A8", "i8", [c * max(f, h, self.mixdim, 2 * h)]),
            ("AS", "f16", [c]),
            ("UnusedA8Mask", "u8", [max(f, h, self.mixdim, 2 * h)]),
            ("HeadHidden", "f16", [HEAD_ROWS, h]),
            ("HeadA8", "i8", [HEAD_ROWS, h]),
            ("HeadAS", "f16", [HEAD_ROWS]),
            ("HeadLogits", "f32", [HEAD_ROWS, v]),
            ("GreedyValues", "f32", [HEAD_ROWS * math.ceil(v / 1024)]),
            ("GreedyIndices", "i32", [HEAD_ROWS * math.ceil(v / 1024)]),
            ("GreedyInvalid", "i32", [HEAD_ROWS * math.ceil(v / 1024)]),
            ("Pairs", "i32", [HEAD_ROWS, 2]),
        ]:
            if not (
                self.standard
                and name
                in {
                    "QKV",
                    "Zout",
                    "AB",
                    "g",
                    "Beta",
                    "Q",
                    "K",
                    "V",
                    "HistoryOut",
                    "PositionOut",
                    "GdnScratch",
                    "Y",
                }
            ):
                w.fixed(name, dtype, shape)
        for name, dtype, shape in [
            ("Input", "i32", [c]),
            ("Token", "i32", [1]),
            ("TokenStatus", "i32", [1]),
            ("Step", "i32", [1]),
            ("Positions", "i32", [c]),
            ("SeqLength", "i32", [1]),
            ("SegmentLength", "i32", [1]),
            ("LastIndex", "i32", [1]),
            ("Requests", "i32", [c]),
            ("MRopePositions", "i32", [self.context, 3]),
            ("PrepareStatus", "i32", [1]),
            ("Logits", "f32", [v]),
        ]:
            w.fixed(name, dtype, shape, "sequence")
        self.reset = ["Step", "Token", "TokenStatus"]
        self.kv = {}
        for i, kind in enumerate(self.text["layer_types"]):
            if kind == "linear_attention":
                for suffix, dtype, shape in [
                    ("History", "f16", [1, 3, self.qkv]),
                    ("State", "f32", [self.hv, 128, 128]),
                ]:
                    name = f"L{i}_{suffix}"
                    w.fixed(name, dtype, shape, "sequence")
                    self.reset.append(name)
            else:
                for suffix, dtype, width in [
                    ("KPages", "i8", self.kv_width),
                    ("VPages", "i8", self.kv_width),
                    ("KPagesScale", "f16", self.kv_width // 64),
                    ("VPagesScale", "f16", self.kv_width // 64),
                ]:
                    name = f"L{i}_{suffix}"
                    w.fixed(name, dtype, [self.context, width], "sequence")
                    self.kv[name] = width * (2 if dtype == "f16" else 1)
                    self.reset.append(name)
        w.constant("Pages", np.arange(math.ceil(self.context / 128), dtype=np.int32)[None])
        rotary_dim = self.dimension if self.standard else 64
        theta = float(self.text["rope_parameters"]["rope_theta"])
        freq = 1 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        angles = np.arange(self.context, dtype=np.float32)[:, None] * freq[None]
        w.constant(
            "Rotary", np.concatenate([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float16)
        )
        self.signs = {}
        widths = {h, f}
        if self.config["quantization_config"].get("vision") == "e8p_q4":
            v = self.config["vision_config"]
            widths.update(
                k
                for k in (v["hidden_size"], v["intermediate_size"], 4 * v["hidden_size"])
                if k % 128 == 0
            )
        for k in sorted(widths):
            name = f"orinfer.rotation.k{k}.signs"
            signs = w.array(name)
            if signs.dtype != np.int8 or signs.shape != (k,) or not np.isin(signs, [-1, 1]).all():
                raise ValueError("Invalid rotation signs")
            self.signs[k] = w.constant(name, signs)
        if self.q4_codec == "e8p_q4":
            from orinfer_build.quantization.e8p import basis
            from orinfer_build.quantization.vq import e8p_selector_table

            table = w.array("orinfer.e8p_q4.basis")
            if table.dtype != np.int8 or not np.array_equal(table, basis()):
                raise ValueError("E8P-Q4 basis differs from compiled decoder")
            w.constant("orinfer.e8p_q4.basis", table)
            self.book = w.constant("orinfer.e8p_q4.selectors", e8p_selector_table(table))
        else:
            table = w.array("orinfer.q4.table")
            if not np.array_equal(
                table, np.array((*MAGNITUDES, *(-x for x in MAGNITUDES)), np.int8)
            ):
                raise ValueError("Q4 integer table differs from compiled decoder")
            w.constant("orinfer.q4.table", table)

    def emit(
        self,
        name,
        key,
        build,
        bindings,
        rows=CAPACITY,
        *,
        dynamic=True,
        row_divisor=1,
        row_multiplier=1,
    ):
        return self.aot.emit(
            "qwen_rows/" + name,
            key,
            build,
            bindings,
            rows,
            dynamic=dynamic,
            row_divisor=row_divisor,
            row_multiplier=row_multiplier,
        )

    def embedding(self, name, input, output, rows=CAPACITY):
        from kernels.cuda.model import qwen35, e8p_q4

        weight = "model.language_model.embed_tokens.weight"
        args = dict(IDs=input, Scale=self.weights.immutable(weight + ".scale"), Output=output)
        if self.embedding_q4:
            args.update(
                Weight=self.weights.immutable(weight + ".packed"),
                Book=self.book,
                Signs=self.signs[self.h],
            )
        else:
            args["Weight"] = self.weights.immutable(weight + ".int8")
        self.emit(
            name,
            f"embedding-{self.embedding_q4}-{rows}",
            lambda: primitive(
                e8p_q4.embedding if self.embedding_q4 else qwen35.embedding, rows, self.v, self.h
            ),
            args,
            rows,
        )

    def norm_rotation(self, section, input, residual, weight, output_residual):
        from kernels.cuda.model.rotation_a8 import residual_norm_rotate_a8

        self.emit(
            section + "/norm_rotate",
            f"norm-rotation-a8-{self.h}",
            lambda: primitive(
                residual_norm_rotate_a8,
                CAPACITY,
                self.h,
                dynamic_rows=True,
                zero_centered=not self.standard,
            ),
            dict(
                X=input,
                R=residual,
                W=self.weights.immutable(weight, cast="float16"),
                Y="Norm",
                RO=output_residual,
                Signs=self.signs[self.h],
                Q="A8",
                S="AS",
            ),
        )

    def rotation(
        self, name, k, input, *, swiglu=False, rows=CAPACITY, dynamic=True, head=False, plane_rows=0
    ):
        from kernels.cuda.model.rotation_a8 import rotate_activation_a8

        if swiglu:
            from kernels.cuda.model.rotation_a8 import (
                rotate_activation_tiles,
                quantize_rotation_tiles,
            )

            with gpu_lock(self.lock):
                selection = self.row_tuner.rotation(k, swiglu, plane_rows=plane_rows)
            first = dict(
                X=input, Signs=self.signs[k], Rotated="RotationValues", Maxima="RotationMaxima"
            )
            second = dict(Rotated="RotationValues", Maxima="RotationMaxima", Q="A8", S="AS")
            from orinfer_build.contracts.row_selection import project

            for role, fn, args, choices, options in [
                (
                    name,
                    rotate_activation_tiles,
                    first,
                    selection,
                    dict(swiglu=swiglu, plane_rows=plane_rows),
                ),
                (
                    name.removesuffix("_rotate") + "_quantize",
                    quantize_rotation_tiles,
                    second,
                    project(selection, ("block_k",)),
                    {},
                ),
            ]:
                for suffix, config in self.row_choices(choices, rows):
                    parameters = dict(dynamic_rows=True, **options, **config)
                    self.emit(
                        role + suffix,
                        f"rotation-tiles-{fn.__name__}-{k}-{parameters}",
                        lambda fn=fn, parameters=parameters: primitive(
                            fn, CAPACITY, k, **parameters
                        ),
                        args,
                        rows,
                    )
            return
        with gpu_lock(self.lock):
            selection = self.row_tuner.activation_rotation(k, swiglu)
        config = selection["configs"][int(selection["default"])]
        self.emit(
            name,
            f"rotation-{k}-{swiglu}-{rows}-{dynamic}-{config}",
            lambda: primitive(
                rotate_activation_a8, rows, k, swiglu=swiglu, dynamic_rows=dynamic, **config
            ),
            dict(
                X=input,
                Signs=self.signs[k],
                Q="HeadA8" if head else "A8",
                S="HeadAS" if head else "AS",
            ),
            rows,
            dynamic=dynamic,
        )

    @staticmethod
    def row_choices(selection, capacity, fixed=()):
        from orinfer_build.contracts.row_selection import choices

        yield from choices(selection, capacity, fixed)

    def attention(
        self, name, *, q, gate, position, length, output, prefix="", rows=CAPACITY, staging=False
    ):
        from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
        from orinfer_build.tuning.row_tuning import ATTENTION_TARGET
        from orinfer_build.models.qwen35.attention import causal_selection, CAUSAL_ROWS

        with gpu_lock(self.lock):
            selection = self.row_tuner.attention(self.context)
            if staging:
                selection = self.row_tuner.staged_attention(self.context, selection)
            selection = causal_selection(selection, self.row_tuner.causal_attention(self.context))
        partial = dict(
            Partial="AttentionPartial",
            Maximum="AttentionMaximum",
            Denominator="AttentionDenominator",
        )
        inputs = dict(
            Q=q,
            K=prefix + "KPages",
            V=prefix + "VPages",
            KS=prefix + "KPagesScale",
            VS=prefix + "VPagesScale",
            Gate=gate,
            Positions=position,
            Lengths=length,
            Y=output,
            **partial,
        )
        selected = list(self.row_choices(selection, rows))
        if any(config.get("decoded_kv") for _, config in selected):
            from kernels.cuda.model.kv_int8 import dequantize_attention_kv
            from orinfer_build.contracts.row_selection import choices

            for buffer in ("AttentionDecodedK", "AttentionDecodedV"):
                if buffer not in self.weights.buffers:
                    # Shared scratch, not request state or a prefix-cache entry.
                    self.weights.fixed(buffer, "f16", [self.context, self.kv_width])
            decode_bindings = {key: inputs[key] for key in ("K", "V", "KS", "VS", "Lengths")}
            decode_bindings.update(DK="AttentionDecodedK", DV="AttentionDecodedV")

            def emit_decode(suffix=""):
                self.emit(
                    name + "_decode" + suffix,
                    f"gqa-stage-{self.context}-{self.kvh}",
                    lambda: primitive(
                        dequantize_attention_kv,
                        CAPACITY,
                        self.context,
                        kv_heads=self.kvh,
                        dimension=self.dimension,
                    ),
                    decode_bindings,
                    rows,
                )

            emit_decode()
            for suffix, config in choices(selection, rows, bounded=True):
                if not config.get("decoded_kv"):
                    emit_decode(suffix.replace("/rows/", "/skip/"))
        for role, fn, args in [
            (name, causal_attention_int8, inputs),
            (name + "_merge", merge_attention_int8, dict(**partial, Gate=gate, Y=output)),
        ]:
            for suffix, config in selected:
                bm, bn, group = config["block_m"], config["block_n"], config["head_group"]
                threads, stages = config.get("threads", 128), config.get("stages", 0)
                options = dict(
                    block_m=bm,
                    target_tiles=ATTENTION_TARGET,
                    max_splits=config["max_splits"],
                    query_heads=self.qh,
                    kv_heads=self.kvh,
                    head_group=group,
                    dimension=self.dimension,
                    gated=not self.standard,
                    independent_queries=config.get("independent_queries", False),
                )
                if fn is causal_attention_int8:
                    options.update(
                        context=self.context,
                        block_n=bn,
                        threads=threads,
                        stages=stages,
                        decoded_kv=config.get("decoded_kv", False),
                    )
                bindings = args
                if fn is causal_attention_int8 and config.get("decoded_kv"):
                    bindings = args | dict(K="AttentionDecodedK", V="AttentionDecodedV")
                if fn is causal_attention_int8 and stages:
                    # The compiler names the shared-pointer KV views in its
                    # exported host ABI; both names refer to the same buffers.
                    bindings = bindings | {
                        f"live_{k.lower()}": bindings[k] for k in ("K", "V", "KS", "VS")
                    }
                self.emit(
                    role + suffix,
                    f"gqa-rows-{fn.__name__}-{self.context}-{bm}-{bn}-{group}-"
                    f"{config['max_splits']}-{threads}-{stages}-decoded{config.get('decoded_kv', False)}-"
                    f"independent{config.get('independent_queries', False)}",
                    lambda fn=fn, options=options, config=config: primitive(
                        fn,
                        CAUSAL_ROWS if config.get("independent_queries") else CAPACITY,
                        **options,
                    ),
                    bindings,
                    min(rows, CAUSAL_ROWS) if config.get("independent_queries") else rows,
                )

    def convolution(self, name, bindings):
        from kernels.cuda.operators.op08_gdn_conv_prep import gdn_conv_prep

        with gpu_lock(self.lock):
            selection = self.row_tuner.gdn_convolution()
        for suffix, config in self.row_choices(selection, CAPACITY):
            tile = config["tile_tokens"]
            self.emit(
                name + suffix,
                f"gdn-conv-{self.hk}-{self.hv}-tile{tile}",
                lambda tile=tile: primitive(
                    gdn_conv_prep,
                    1,
                    CAPACITY,
                    tile_tokens=tile,
                    dynamic_tokens=True,
                    key_heads=self.hk,
                    value_heads=self.hv,
                ),
                bindings,
            )

    def projection(
        self,
        name,
        weight,
        n,
        k,
        output,
        *,
        stride=None,
        offset=0,
        rows=CAPACITY,
        dynamic=True,
        head=False,
        kind="q4",
    ):
        w = self.weights
        if stride == n:
            stride = None
        if kind == "q4":
            kind = (
                "w8"
                if weight + ".int8" in w.index["weight_map"] or weight + ".int8" in w.buffers
                else self.q4_codec
            )
        compressed = kind in ("q4", "e8p_q4")
        suffix = ".packed" if compressed else ".int8"
        data = w.immutable(weight + suffix)
        scale = w.immutable(weight + ".scale")

        target = (
            dict(
                kind="buffer_slice",
                name=output,
                offset=offset * 4 if head else offset * 2,
            )
            if offset
            else output
        )
        bind = (
            dict(
                A="HeadA8" if head else "A8",
                W=data,
                S=scale,
                AS="HeadAS" if head else "AS",
                Out=target,
            )
            if compressed
            else dict(
                A="HeadA8" if head else "A8",
                Weight=data,
                WeightScale=scale,
                TokenScale="HeadAS" if head else "AS",
                Output=target,
            )
        )
        if kind == "e8p_q4":
            bind["Book"] = self.book
        from orinfer_build.contracts.projection_policy import SPECIALIZATIONS

        shapes = [(rows, dynamic, name)]
        if dynamic:
            shapes += [(r, False, f"{name}/rows/{r}-{r}") for r in SPECIALIZATIONS if r <= rows]
        for live, symbolic, binding in shapes:
            key = f"{kind}-{live}-{symbolic}-{n}-{k}-{stride}-{head}-pk{w.packed_k if kind == 'e8p_q4' else 0}"

            def build():
                return self.compile_projection(
                    weight, kind, n, k, live, symbolic, "float32" if head else "float16", stride
                )

            self.emit(binding, key, build, bind, live, dynamic=symbolic)

        if dynamic and rows > 1:
            if not hasattr(self, "bounded_projection_choices"):
                self.bounded_projection_choices = {}
            geometry = (kind, n, k, min(rows, 128), head, stride, w.packed_k)
            if geometry not in self.bounded_projection_choices:
                with gpu_lock(self.lock):
                    self.bounded_projection_choices[geometry] = list(
                        self.compile_projection(
                            weight,
                            kind,
                            n,
                            k,
                            min(rows, 128),
                            True,
                            "float32" if head else "float16",
                            stride,
                            select_rows=True,
                        )
                    )
            for suffix, config, kernel in self.bounded_projection_choices[geometry]:
                key = (
                    f"{kind}-bounded-{min(rows, 128)}-{n}-{k}-{stride}-{head}-{w.packed_k}-"
                    + json.dumps(config, sort_keys=True)
                )
                self.emit(name + suffix, key, lambda kernel=kernel: kernel, bind, min(rows, 128))

    def compile_projection(
        self, weight, kind, n, k, rows, dynamic, dtype, stride=None, *, select_rows=False
    ):
        import torch
        from orinfer_build.tuning.projection_tuning import ProjectionTuner
        from orinfer_build.contracts.row_selection import samples

        w = self.weights
        suffix = ".packed" if kind in ("q4", "e8p_q4") else ".int8"
        actual = dict(
            W=torch.from_numpy(w.array(weight + suffix).copy()).cuda(),
            WS=torch.from_numpy(w.array(weight + ".scale").copy()).cuda(),
        )
        if kind == "q4":
            actual["W"] = actual["W"].view(torch.uint16)
        if kind == "e8p_q4":
            actual["Book"] = torch.from_numpy(w.extra[self.book].copy()).cuda()
        identity = w.buffers[weight + suffix]["data"]["sha256"]
        if kind == "e8p_q4":
            identity += w.buffers[self.book]["data"]["sha256"]
        tuner = ProjectionTuner(self.output / "tuning", self.aot.cache.root / "tuning")
        compile = tuner.compile_choices if select_rows else tuner.compile
        from orinfer_build.contracts.projection_policy import SPECIALIZATIONS

        result = compile(
            kind,
            rows,
            n,
            k,
            dtype,
            dynamic=dynamic,
            weights=actual,
            weight_identity=hashlib.sha256(identity.encode()).hexdigest(),
            measure_rows=samples(rows) if dynamic and (rows >= CAPACITY or select_rows) else None,
            objective="total_latency" if dynamic and rows >= CAPACITY else "relative_geomean",
            output_stride=stride,
            packed_k=w.packed_k if kind == "e8p_q4" else 0,
            **(dict(fixed=SPECIALIZATIONS) if select_rows else {}),
        )
        return result if select_rows else result[0]

    def forward(self):
        from kernels.cuda.model import qwen35, control, kv_int8
        from kernels.cuda.model.gdn_sequence import _gdn_sequence
        from kernels.cuda.operators.op07_gdn_ab import gdn_ab_tensorcore
        from kernels.cuda.operators.op09_gdn_gates import _compile_gates
        from kernels.cuda.operators.op17_gdn_gated_norm import _compile_gated_norm
        from kernels.cuda.operators.op30_activation_quantization import _compile as a8_compile

        w = self.weights
        self.emit(
            "begin/prepare",
            "prepare",
            lambda: primitive(qwen35.prepare, CAPACITY, self.h, self.context, mrope_input=True),
            dict(
                Step="Step",
                Positions="Positions",
                Length="SeqLength",
                SegmentLength="SegmentLength",
                Last="LastIndex",
                Requests="Requests",
                MRope="MRopePositions",
                Residual="R0",
            ),
        )
        self.embedding("begin/embedding", "Input", "Hidden")
        for i, kind in enumerate(self.text["layer_types"]):
            p = f"model.language_model.layers.{i}"
            pre = f"layer{i}/pre"
            private = f"layer{i}/private"
            post = f"layer{i}/post"
            self.norm_rotation(pre, "Hidden", "R0", p + ".input_layernorm.weight", "R1")
            if kind == "linear_attention":
                base = p + ".linear_attn"
                self.projection(pre + "/qkv", base + ".in_proj_qkv.weight", self.qkv, self.h, "QKV")
                self.projection(pre + "/z", base + ".in_proj_z.weight", self.mixdim, self.h, "Zout")
                ab = w.combined(
                    f"L{i}_AB", [base + ".in_proj_a.weight", base + ".in_proj_b.weight"]
                )
                with gpu_lock(self.lock):
                    selection = self.row_tuner.gdn_ab()
                for suffix, config in self.row_choices(selection, CAPACITY):
                    self.emit(
                        pre + "/ab" + suffix,
                        "gdn-ab-" + json.dumps(config, sort_keys=True),
                        lambda config=config: primitive(
                            gdn_ab_tensorcore,
                            __import__("tilelang.language", fromlist=["dynamic"]).dynamic("m"),
                            N=2 * self.hv,
                            K=self.h,
                            **config,
                        ),
                        dict(X="Norm", W_ab=ab, Y="AB"),
                    )
                self.emit(
                    private + "/gates",
                    "gdn-gates",
                    lambda: primitive(_compile_gates, packed_ab=True, heads=self.hv),
                    dict(
                        A="AB",
                        B="AB",
                        Parameter=w.immutable(base + ".A_log", cast="float32"),
                        DtBias=w.immutable(base + ".dt_bias", cast="float32"),
                        G="g",
                        Beta="Beta",
                    ),
                )
                self.convolution(
                    private + "/conv",
                    dict(
                        X="QKV",
                        W=w.immutable(base + ".conv1d.weight", cast="float16", shape=[self.qkv, 4]),
                        HI=f"L{i}_History",
                        lengths="SegmentLength",
                        positions="Step",
                        Q="Q",
                        K="K",
                        V="V",
                        HO="HistoryOut",
                        positions_out="PositionOut",
                    ),
                )
                with gpu_lock(self.lock):
                    selection = self.row_tuner.gdn_sequence()
                recurrent_bindings = dict(
                    Q="Q",
                    K="K",
                    V="V",
                    G="g",
                    Beta="Beta",
                    State=f"L{i}_State",
                    Prefix="GdnScratch",
                    Out="Y",
                )
                for suffix, config in self.row_choices(selection, CAPACITY):
                    self.emit(
                        private + "/recurrent" + suffix,
                        "gdn-recurrent-" + str(config["token_unroll"]),
                        lambda config=config: primitive(
                            _gdn_sequence,
                            CAPACITY,
                            in_place=True,
                            value_tile=16,
                            threads=32,
                            prefetch_qk=True,
                            dynamic_rows=True,
                            key_heads=self.hk,
                            value_heads=self.hv,
                            **config,
                        ),
                        recurrent_bindings,
                    )
                from orinfer_build.models.qwen35.gdn import prefill as gdn_prefill

                gdn_prefill(self, private, i)
                self.emit(
                    private + "/gated_norm",
                    "gdn-gated-norm",
                    lambda: primitive(_compile_gated_norm, None, heads=self.hv),
                    dict(
                        X="Y",
                        Z="Zout",
                        W=w.immutable(base + ".norm.weight", cast="float16"),
                        Y="MixerIn",
                    ),
                )
            else:
                base = p + ".self_attn"
                projections = [
                    ("q", self.qproj, 0),
                    ("k", self.kv_width, self.qproj),
                    ("v", self.kv_width, self.qproj + self.kv_width),
                ]
                if self.standard:
                    combined = w.combined_projection(
                        base + ".qkv_proj.weight",
                        [base + f".{role}_proj.weight" for role, _, _ in projections],
                    )
                    self.projection(pre + "/qkv", combined, self.fullx, self.h, "FullX")
                for role, n, offset in [] if self.standard else projections:
                    self.projection(
                        pre + "/" + role,
                        base + f".{role}_proj.weight",
                        n,
                        self.h,
                        "FullX",
                        stride=self.fullx,
                        offset=offset,
                    )
                pages = math.ceil(self.context / 128)
                self.emit(
                    private + "/prepare",
                    f"kv-prepare-{self.context}",
                    lambda: primitive(
                        kv_int8.full_prepare_mrope_int8,
                        pages,
                        self.context,
                        (11, 11, 10),
                        max_position=self.context,
                        query_heads=self.qh,
                        kv_heads=self.kvh,
                        dimension=self.dimension,
                        rotary_dim=self.dimension if self.standard else 64,
                        gated=not self.standard,
                        zero_centered=not self.standard,
                    ),
                    dict(
                        X="FullX",
                        WQ=w.immutable(base + ".q_norm.weight", cast="float16"),
                        WK=w.immutable(base + ".k_norm.weight", cast="float16"),
                        Cache="Rotary",
                        Req="Requests",
                        Pos="Positions",
                        Pages="Pages",
                        Status="PrepareStatus",
                        MRope="MRopePositions",
                        Q="FullQ",
                        Gate="FullGate",
                        K=f"L{i}_KPages",
                        V=f"L{i}_VPages",
                        KS=f"L{i}_KPagesScale",
                        VS=f"L{i}_VPagesScale",
                    ),
                )
                self.attention(
                    private + "/attention",
                    q="FullQ",
                    gate="FullGate",
                    position="Positions",
                    length="SeqLength",
                    output="MixerIn",
                    prefix=f"L{i}_",
                    staging=True,
                )
            self.emit(
                post + "/quantize",
                "a8-self.mixdim",
                lambda: primitive(a8_compile, self.mixdim, self.mixdim, False, False, 256),
                dict(X="MixerIn", Mask="UnusedA8Mask", Q="A8", S="AS"),
            )
            self.projection(
                post + "/out",
                base + (".out_proj.weight" if kind == "linear_attention" else ".o_proj.weight"),
                self.h,
                self.mixdim,
                "Mix",
                kind="w8",
            )
            self.norm_rotation(post, "Mix", "R1", p + ".post_attention_layernorm.weight", "R0")
            if self.moe:
                from orinfer_build.models.qwen35.moe import build as moe_ffn

                moe_ffn(self, post, p + ".mlp")
            else:
                if self.standard:
                    combined = w.combined_projection(
                        p + ".mlp.gate_up_proj.weight",
                        [p + f".mlp.{role}_proj.weight" for role in ["gate", "up"]],
                    )
                    self.projection(post + "/gate_up", combined, 2 * self.f, self.h, "GateUp")
                for role, offset in (
                    [] if self.standard else [("gate", 0), ("up", CAPACITY * self.f)]
                ):
                    self.projection(
                        post + "/" + role,
                        p + f".mlp.{role}_proj.weight",
                        self.f,
                        self.h,
                        "GateUp",
                        stride=self.f,
                        offset=offset,
                    )
                self.rotation(
                    post + "/swiglu_rotate",
                    self.f,
                    "GateUp",
                    swiglu=True,
                    plane_rows=0 if self.standard else CAPACITY,
                )
                self.projection(
                    post + "/down", p + ".mlp.down_proj.weight", self.h, self.f, "Hidden"
                )
            print("BOUND layer", i, flush=True)
        self.emit(
            "end/advance",
            "advance",
            lambda: primitive(control.advance, CAPACITY, dynamic_rows=True),
            dict(Step="Step"),
        )

    def head(self):
        from kernels.cuda.operators.op23_final_norm import final_norm
        from kernels.cuda.model.greedy import greedy_partials, greedy_merge
        from kernels.cuda.model.flash_control import selection

        w = self.weights
        self.emit(
            "gather/norm",
            "final-norm",
            lambda: primitive(final_norm, None, 1, hidden=self.h, zero_centered=not self.standard),
            dict(
                X="Hidden",
                R="R0",
                I="LastIndex",
                W=w.immutable("model.language_model.norm.weight", cast="float16"),
                Y="HeadHidden",
            ),
        )
        for name, rows, dynamic in [("head", HEAD_ROWS, True), ("head_last", 1, False)]:
            self.rotation(
                name + "/rotate", self.h, "HeadHidden", rows=rows, dynamic=dynamic, head=True
            )
            self.projection(
                name + "/project",
                self.head_weight,
                self.v,
                self.h,
                "HeadLogits",
                rows=rows,
                dynamic=dynamic,
                head=True,
            )
            self.emit(
                name + "/partials",
                f"greedy-partials-{rows}-{dynamic}",
                lambda: primitive(greedy_partials, self.v, rows, dynamic_rows=dynamic),
                dict(
                    Logits="HeadLogits",
                    Values="GreedyValues",
                    Indices="GreedyIndices",
                    Invalid="GreedyInvalid",
                ),
                rows,
                dynamic=dynamic,
            )
            self.emit(
                name + "/merge",
                f"greedy-merge-{rows}-{dynamic}",
                lambda: primitive(greedy_merge, self.v, rows, dynamic_rows=dynamic),
                dict(
                    Values="GreedyValues",
                    Indices="GreedyIndices",
                    Invalid="GreedyInvalid",
                    Output="Pairs",
                ),
                rows,
                dynamic=dynamic,
            )
        self.emit(
            "select/selection",
            "selection",
            lambda: primitive(selection),
            dict(Pair="Pairs", Token="Token", Status="TokenStatus"),
            1,
            dynamic=False,
        )

    def publish(self, engine):
        from orinfer_build.packaging.frontend import prepare_generation_config, prepare_tokenizer

        w = self.weights
        total = w.finish()
        for name in [
            "LICENSE",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "merges.txt",
            "vocab.json",
        ]:
            if (self.source / name).is_file():
                shutil.copyfile(self.source / name, self.root / name)
        prepare_tokenizer(self.source, self.root / "tokenizer.json")
        prepare_generation_config(self.source, self.root / "generation_config.json")
        metadata = dict(
            schema_version=2,
            target="sm_87",
            model=f"{self.config.get('quantization_config', {}).get('source', 'Qwen3.5')}-{self.q4_codec}-A8",
            chunk_tokens=CAPACITY,
            max_context=self.context,
            vocab=self.v,
            toolchain=self.aot.identity,
            buffers=list(w.buffers.values()),
            reset_buffers=self.reset,
            input="Input",
            token="Token",
            status="TokenStatus",
            logits="Logits",
            position="Step",
            weight_bytes=total,
            weight_parameters=self.state["parameters"],
            weight_scope=f"Original BF16 -> signed-block128 {self.q4_codec} Q4A8; embedding/vision precision from quantization policy, sensitive output/MTP projections W8; one packed resident copy; no permanent W8 expansion",
            kv_cache=dict(
                direct_prefill=True, demand_mapping=True, buffers=self.kv, prefill_workspace={}
            ),
            vision=self.vision,
            **(dict(audio=self.audio) if self.audio else {}),
            mtp=self.mtp,
            packet_table=self.packet_spec,
            gdn_execution=self.gdn_execution,
            batch_layout=dict(
                layers=self.text["layer_types"],
                row_strides={},
                profiles={},
                hidden=self.h,
                state_columns=self.state_columns,
            ),
        )
        package = dict(
            schema_version=1,
            runtime_abi=1,
            target="sm_87",
            architecture=self.architecture,
            compute_policy="int8_quality",
            config_signature=config_signature(self.config),
            prefill_profiles=[dict(tokens=CAPACITY, min_tokens=1, kind="qwen_q4_rows")],
            kernels=self.aot.kernels,
            dynamic_batch_kernels=self.aot.dynamic,
            buffer_contracts=[
                {k: v for k, v in b.items() if k != "data"} for b in w.buffers.values()
            ],
            toolchain=self.aot.identity,
            execution=execution_identity(
                model_library(), engine, self.architecture, "int8_quality"
            ),
            plan_kernel_contracts=[
                k["name"]
                for k in self.aot.kernels
                if "/rows/" in k["name"]
                or "/skip/" in k["name"]
                or k["name"].startswith("qwen_rows/packet/")
            ],
        )
        from orinfer_build.models.sampling import append as append_sampling

        append_sampling(
            dict(metadata=metadata, buffer_scopes=w.scopes),
            package,
            self.root / "cache",
            self.output / "sampling",
        )
        digest = compact_package(self.root / "cache", package, self.root / "cache/packages")
        data = dict(
            schema_version=1,
            architecture=self.architecture,
            compute_policy="int8_quality",
            execution_package=digest,
            buffer_scopes=w.scopes,
            frontend_assets={
                name: file_hash(self.root / name)
                for name in ["tokenizer.json", "chat_template.jinja", "generation_config.json"]
            },
            metadata=metadata,
        )
        write_json(self.root / "cache/model.json", data)
        result = subprocess.run(
            [str(engine), "plan-model", str(self.root)], capture_output=True, text=True
        )
        if result.returncode:
            raise ValueError(f"Native plan validation failed: {result.stderr.strip()}")
        plan = json.loads(result.stdout)["manifest"]
        if len(plan["kernels"]) != len(package["kernels"]) or not plan.get("sampling"):
            raise ValueError("Native execution bindings were lost during publication")
        shutil.rmtree(self.root / "cache/shared-rows")
        write_json(
            self.output / "result.json",
            dict(
                execution_package=digest,
                weight_bytes=total,
                kernels=len(plan["kernels"]),
                programs=len(plan["programs"]),
            ),
        )
        return digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "model-output", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cache", type=Path, default=work_path("cache", "build", "qwen35"))
    p.add_argument("--context", type=int, default=262144)
    p.add_argument("--gpu-lock", type=Path, default=gpu_lock_path())
    p.add_argument(
        "--gpu-lock-held",
        action="store_true",
        help="The operator wrapper already holds the shared GPU lock",
    )
    p.add_argument("--engine", type=Path, default=ROOT / "target/release/orinfer")
    a = p.parse_args()
    if a.source.resolve() == a.model_output.resolve():
        p.error("Model output must differ from quantized checkpoint")
    a.output.mkdir(parents=True, exist_ok=True)
    with atomic_model(a.model_output) as staging:
        b = Builder(
            a.source, staging, a.output, a.cache, a.context, None if a.gpu_lock_held else a.gpu_lock
        )
        digest = b.publish(a.engine.resolve(strict=True))
    print("COMPLETE", digest, flush=True)


if __name__ == "__main__":
    main()

"""Build a complete-sequence causal Qwen3-Reranker E8P execution package."""

from orinfer_build.common.paths import ROOT, gpu_lock_path, work_path


import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

from orinfer_build.common.files import file_hash, write_json
from orinfer_build.common.gpu import gpu_lock
from orinfer_build.compiler.compile import primitive
from orinfer_build.models.encoder import Projections
from orinfer_build.packaging.compact import compact_package
from orinfer_build.packaging.frontend import prepare_tokenizer
from orinfer_build.packaging.package import execution_identity, model_library
from orinfer_build.packaging.publication import atomic_model
from orinfer_build.models.qwen35.compiler import Aot
from orinfer_build.models.qwen35.storage import Weights
from orinfer_build.models.qwen_reranker.config import configuration, signature


class Builder(Projections):
    kernel_prefix = "qwen_reranker"
    norm_epsilon = 1e-6

    def __init__(self, args, root):
        self.args, self.root = args, root
        self.config = json.loads((args.source / "config.json").read_text())
        configuration(self.config, quantized=True)
        self.state = json.loads((args.source / "progress.json").read_text())
        if not self.state.get("complete"):
            raise ValueError("Quantization is incomplete")
        self.capacity, self.patches = args.context, 0
        self.w = Weights(args.source, root, packed_k=128)
        self.lock = None if args.gpu_lock_held else args.gpu_lock
        self.aot = Aot(
            root / "cache",
            args.output,
            args.cache,
            self.w.buffers,
            compile_lock=lambda: gpu_lock(self.lock),
        )
        self.aot.identity["weights"] = hashlib.sha256(self.state["contract"].encode()).hexdigest()
        self.aot.identity["reranker_capacity"] = str(self.capacity)
        self.tuned = {}
        self.signs = {
            k: self.w.immutable(f"orinfer.rotation.k{k}.signs") for k in (1024, 2048, 3072)
        }
        from orinfer_build.quantization.e8p import basis
        from orinfer_build.quantization.vq import e8p_selector_table

        original = self.w.array("orinfer.e8p_q4.basis")
        if not np.array_equal(original, basis()):
            raise ValueError("Unexpected E8P basis")
        self.book = self.w.constant("E8PBook", e8p_selector_table(original))
        self.row_workspace()
        self.workspace()
        self.text()

    def workspace(self):
        for name, dtype, shape in [
            ("Input", "i32", [self.capacity]),
            ("Length", "i32", [1]),
            ("Control", "i32", [1]),
            ("ClassLogits", "f32", [128, 2]),
            ("THidden", "f16", [self.capacity, 1024]),
            ("TNorm", "f16", [self.capacity, 1024]),
            ("TQKV", "f16", [self.capacity, 4096]),
            ("TQ", "f16", [self.capacity, 2048]),
            ("TK", "f16", [self.capacity, 1024]),
            ("TValue", "f16", [self.capacity, 1024]),
            ("TAttention", "f16", [self.capacity, 2048]),
            ("TProduct", "f16", [self.capacity, 1024]),
            ("TGateUp", "f16", [self.capacity, 6144]),
            ("TFFN", "f16", [self.capacity, 3072]),
            ("A8", "i8", [self.capacity * 3072]),
            ("AS", "f16", [self.capacity]),
        ]:
            self.w.fixed(name, dtype, shape)
        import torch
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        cfg = Qwen3Config(**{k: v for k, v in self.config.items() if k != "quantization_config"})
        cos, sin = Qwen3RotaryEmbedding(cfg)(
            torch.empty(1, dtype=torch.float16), torch.arange(self.capacity).reshape(1, -1)
        )
        for name, data in [("TextCos", cos), ("TextSin", sin)]:
            self.w.constant(name, data[0].numpy())

    def text(self):
        from kernels.cuda.model import e8p_q4, embedding as e, reranking as r, encoder_rows

        self.emit(
            "embedding",
            "embedding",
            lambda: primitive(e8p_q4.embedding, self.capacity, 151669, 1024),
            dict(
                IDs="Input",
                Weight=self.w.immutable("model.embed_tokens.weight.packed"),
                Scale=self.w.immutable("model.embed_tokens.weight.scale"),
                Signs=self.signs[1024],
                Book=self.book,
                Output="THidden",
            ),
        )
        for layer in range(28):
            base = f"layer{layer}"
            root = f"model.layers.{layer}."
            attn = root + "self_attn."
            mlp = root + "mlp."
            self.norm(base + "/norm1", "THidden", "TNorm", root + "input_layernorm.weight", 1024)
            self.activation_input(base + "/rotate_qkv", "TNorm", 1024)
            fused = self.w.combined_projection(
                attn + "qkv_proj.weight", [attn + role + "_proj.weight" for role in ("q", "k", "v")]
            )
            self.projection(base + "/qkv", fused, 4096, 1024, "TQKV")
            self.emit(
                base + "/qk_rope",
                "qk-rope-heads16",
                lambda: primitive(r.qk_rope, self.capacity, head_group=16),
                dict(
                    X="TQKV",
                    Ranges="RowRanges",
                    QNorm=self.w.immutable(attn + "q_norm.weight", cast="float16"),
                    KNorm=self.w.immutable(attn + "k_norm.weight", cast="float16"),
                    Cos="TextCos",
                    Sin="TextSin",
                    Q="TQ",
                    K="TK",
                    V="TValue",
                ),
            )
            self.attention(
                base + "/attention",
                2048,
                16,
                8,
                dict(Q="TQ", K="TK", V="TValue", Length="RowRanges", Y="TAttention"),
                causal=True,
            )
            self.activation_input(base + "/quantize_o", "TAttention", 2048, compressed=False)
            self.projection(base + "/o", attn + "o_proj.weight", 1024, 2048, "TProduct")
            self.emit(
                base + "/add1",
                "add-1024-tile1024",
                lambda: primitive(e.add, self.capacity, 1024, block_size=1024),
                dict(X="TProduct", Residual="THidden", Y="THidden"),
            )
            self.norm(
                base + "/norm2", "THidden", "TNorm", root + "post_attention_layernorm.weight", 1024
            )
            self.activation_input(base + "/rotate_gateup", "TNorm", 1024)
            fused = self.w.combined_projection(
                mlp + "gate_up_proj.weight",
                [mlp + role + "_proj.weight" for role in ("gate", "up")],
            )
            self.projection(base + "/gate_up", fused, 6144, 1024, "TGateUp")
            self.emit(
                base + "/swiglu",
                "swiglu-3072-tile1024",
                lambda: primitive(e.activation, self.capacity, 3072, block_size=1024),
                dict(X="TGateUp", Y="TFFN"),
            )
            self.activation_input(base + "/rotate_down", "TFFN", 3072)
            self.projection(base + "/down", mlp + "down_proj.weight", 1024, 3072, "TProduct")
            self.emit(
                base + "/add2",
                "add-1024-tile1024",
                lambda: primitive(e.add, self.capacity, 1024, block_size=1024),
                dict(X="TProduct", Residual="THidden", Y="THidden"),
            )
        self.norm("final_norm", "THidden", "TNorm", "model.norm.weight", 1024)
        self.emit(
            "classifier",
            "two-class-head",
            lambda: primitive(encoder_rows.classifier),
            dict(
                X="TNorm",
                W=self.w.immutable("orinfer.reranker.classifier.weight"),
                Y="ClassLogits",
                Segments="Segments",
                Count="SequenceCount",
            ),
        )

    def publish(self):
        total = self.w.finish()
        names = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
            "README.md",
            "LICENSE",
        ]
        for name in names:
            if (self.args.source / name).is_file():
                shutil.copyfile(self.args.source / name, self.root / name)
        prepare_tokenizer(self.args.source, self.root / "tokenizer.json")
        ids = self.config["quantization_config"]["classifier_token_ids"]
        spec = dict(
            rows=self.row_spec(),
            program="rerank",
            length="Length",
            output="ClassLogits",
            false_token_id=ids[0],
            true_token_id=ids[1],
        )
        metadata = dict(
            schema_version=2,
            target="sm_87",
            model="Qwen3-Reranker-0.6B-E8P-Q4A8",
            chunk_tokens=self.capacity,
            max_context=self.capacity,
            vocab=151669,
            toolchain=self.aot.identity,
            buffers=list(self.w.buffers.values()),
            reset_buffers=[],
            input="Input",
            token="Control",
            status="Control",
            logits="ClassLogits",
            position="Control",
            weight_bytes=total,
            weight_parameters=self.state["parameters"],
            weight_scope="Two-stage E8P Q4A8; attention output and protected text blocks W8; norms and original two-class head FP16; one packed embedding bank",
            reranking=spec,
        )
        package = dict(
            schema_version=1,
            runtime_abi=1,
            target="sm_87",
            architecture="qwen3_reranker",
            compute_policy="int8_quality",
            config_signature=signature(self.config),
            prefill_profiles=[],
            kernels=self.aot.kernels,
            dynamic_batch_kernels=self.aot.dynamic,
            buffer_contracts=[
                {k: v for k, v in b.items() if k != "data"} for b in self.w.buffers.values()
            ],
            toolchain=self.aot.identity,
            execution=execution_identity(
                model_library(), self.args.engine, "qwen3_reranker", "int8_quality"
            ),
        )
        digest = compact_package(self.root / "cache", package, self.root / "cache/packages")
        write_json(
            self.root / "cache/model.json",
            dict(
                schema_version=1,
                architecture="qwen3_reranker",
                compute_policy="int8_quality",
                execution_package=digest,
                buffer_scopes=self.w.scopes,
                metadata=metadata,
                frontend_assets={
                    name: file_hash(self.root / name)
                    for name in [
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                        "generation_config.json",
                    ]
                },
            ),
        )
        subprocess.run([str(self.args.engine), "validate-model", str(self.root)], check=True)
        self.validate_published_rows()
        write_json(
            self.args.output / "result.json",
            dict(
                weight_bytes=total,
                parameters=self.state["parameters"],
                effective_weight_bits=total * 8 / self.state["parameters"],
                execution_package=digest,
            ),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "model-output", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--cache", type=Path, default=work_path("cache", "build", "qwen-reranker"))
    p.add_argument("--engine", type=Path, default=ROOT / "target/release/orinfer")
    p.add_argument("--gpu-lock", type=Path, default=gpu_lock_path())
    p.add_argument("--gpu-lock-held", action="store_true")
    args = p.parse_args()
    if not 512 <= args.context <= 32768 or args.context % 32:
        p.error("Context must be a multiple of 32 in 512..32768")
    args.engine = args.engine.resolve(strict=True)
    with atomic_model(args.model_output) as root:
        Builder(args, root).publish()


if __name__ == "__main__":
    main()

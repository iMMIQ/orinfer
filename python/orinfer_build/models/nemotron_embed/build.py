"""Build the native E8P Q4A8 Nemotron-3.5-Embed-1B execution package."""

import numpy as np

from orinfer_build.compiler.compile import primitive
from orinfer_build.models.encoder import Projections, encoder_parser
from orinfer_build.models.nemotron_embed.config import configuration, signature
from orinfer_build.packaging.publication import atomic_model


class Builder(Projections):
    kernel_prefix = "nemotron"

    def __init__(self, args, root):
        self.init_encoder(
            args,
            root,
            validate_config=configuration,
            capacity=args.context,
            patches=args.max_patches,
            identity_key="embedding_capacity",
            identity_value=(args.context, args.max_patches),
            sign_widths=(1024, 2048, 3072, 4096, 6144),
        )
        self.row_workspace()
        self.workspace()
        self.text()
        self.vision()

    def workspace(self):
        w = self.w
        for name, dtype, shape in [
            ("Input", "i32", [self.capacity]),
            ("Length", "i32", [1]),
            ("Control", "i32", [1]),
            ("FeatureIndex", "i32", [self.capacity]),
            ("Mean", "f32", [128, 2048]),
            ("Embedding", "f32", [128, 2048]),
            ("Pixels", "f16", [self.patches, 588]),
            ("Grid", "i32", [2]),
            ("VisionLength", "i32", [1]),
            ("Features", "f16", [self.capacity, 2048]),
            ("VisionOutput", "f16", [self.patches // 4, 2048]),
            ("Merged", "f16", [self.patches // 4, 4096]),
            ("ProjectorInput", "f16", [self.patches // 4, 1024]),
            ("ProjectorHidden", "f16", [self.patches // 4, 2048]),
        ]:
            w.fixed(name, dtype, shape)
        for p, rows, h, f, q, kv in [
            ("T", self.capacity, 2048, 6144, 3072, 1024),
            ("V", self.patches, 1024, 4096, 1024, 1024),
        ]:
            for role, width in [
                ("Hidden", h),
                ("Norm", h),
                ("QKV", q + 2 * kv),
                ("Q", q),
                ("K", kv),
                ("Value", kv),
                ("Attention", q),
                ("Product", h),
                ("GateUp", 2 * f),
                ("FFN", f),
            ]:
                w.fixed(p + role, "f16", [rows, width])
        w.fixed("A8", "i8", [max(self.capacity, self.patches) * 6144])
        w.fixed("AS", "f16", [max(self.capacity, self.patches)])
        # Produce tables with the checkpoint's reference implementation, offline.
        import torch
        from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
        from transformers.models.ministral3.modeling_ministral3 import (
            Ministral3RotaryEmbedding,
            get_llama_4_attn_scale,
        )

        cfg = Ministral3Config(**self.config["text_config"])
        positions = torch.arange(self.capacity).reshape(1, -1)
        cos, sin = Ministral3RotaryEmbedding(cfg)(torch.empty(1, dtype=torch.float16), positions)
        for name, data in [("TextCos", cos), ("TextSin", sin)]:
            w.constant(name, data[0].numpy())
        scale = get_llama_4_attn_scale(positions, 0.1, 16384).reshape(-1).half()
        w.constant("QueryScale", scale.numpy())

    def transformer(self, *, vision=False):
        from kernels.cuda.model import embedding as e

        p, root, layers, h, f, q, kv, heads, kv_heads = (
            ("V", "vision_tower.transformer.layers", 24, 1024, 4096, 1024, 1024, 16, 16)
            if vision
            else ("T", "language_model.layers", 16, 2048, 6144, 3072, 1024, 24, 8)
        )
        section = "vision" if vision else "text"
        for layer in range(layers):
            base, prefix = f"{section}/layer{layer}", f"{root}.{layer}."
            attn = prefix + ("attention." if vision else "self_attn.")
            mlp = prefix + ("feed_forward." if vision else "mlp.")
            self.norm(
                base + "/norm1",
                p + "Hidden",
                p + "Norm",
                prefix + ("attention_norm.weight" if vision else "input_layernorm.weight"),
                h,
                vision=vision,
            )
            self.activation_input(base + "/rotate_qkv", p + "Norm", h, vision=vision)
            fused = self.w.combined_projection(
                attn + "qkv_proj.weight", [attn + role + "_proj.weight" for role in ("q", "k", "v")]
            )
            self.projection(base + "/qkv", fused, q + 2 * kv, h, p + "QKV", vision=vision)
            self.emit(
                base + "/rope",
                f"rope-{vision}-grouped",
                lambda: (
                    primitive(e.vision_rope, self.capacity)
                    if vision
                    else primitive(e.text_rope, self.capacity, head_group=8)
                ),
                dict(
                    X=p + "QKV",
                    Q=p + "Q",
                    K=p + "K",
                    V=p + "Value",
                    **(
                        dict(Grid="Grid")
                        if vision
                        else dict(
                            Cos="TextCos", Sin="TextSin", QScale="QueryScale", Ranges="RowRanges"
                        )
                    ),
                ),
                vision=vision,
            )
            self.attention(
                base + "/attention",
                q,
                heads,
                kv_heads,
                dict(
                    Q=p + "Q",
                    K=p + "K",
                    V=p + "Value",
                    Length="VisionLength" if vision else "RowRanges",
                    Y=p + "Attention",
                ),
                vision=vision,
            )
            self.activation_input(
                base + "/quantize_o", p + "Attention", q, compressed=False, vision=vision
            )
            self.projection(base + "/o", attn + "o_proj.weight", h, q, p + "Product", vision=vision)
            self.emit(
                base + "/add1",
                f"add-{h}-tile1024",
                lambda: primitive(e.add, self.capacity, h, block_size=1024),
                dict(X=p + "Product", Residual=p + "Hidden", Y=p + "Hidden"),
                vision=vision,
            )
            self.norm(
                base + "/norm2",
                p + "Hidden",
                p + "Norm",
                prefix + ("ffn_norm.weight" if vision else "post_attention_layernorm.weight"),
                h,
                vision=vision,
            )
            self.activation_input(base + "/rotate_gateup", p + "Norm", h, vision=vision)
            fused = self.w.combined_projection(
                mlp + "gate_up_proj.weight",
                [mlp + role + "_proj.weight" for role in ("gate", "up")],
            )
            self.projection(base + "/gate_up", fused, 2 * f, h, p + "GateUp", vision=vision)
            self.emit(
                base + "/swiglu",
                f"swiglu-{f}-tile1024",
                lambda: primitive(e.activation, self.capacity, f, block_size=1024),
                dict(X=p + "GateUp", Y=p + "FFN"),
                vision=vision,
            )
            self.activation_input(base + "/rotate_down", p + "FFN", f, vision=vision)
            self.projection(
                base + "/down", mlp + "down_proj.weight", h, f, p + "Product", vision=vision
            )
            self.emit(
                base + "/add2",
                f"add-{h}-tile1024",
                lambda: primitive(e.add, self.capacity, h, block_size=1024),
                dict(X=p + "Product", Residual=p + "Hidden", Y=p + "Hidden"),
                vision=vision,
            )

    def text(self):
        from kernels.cuda.model import e8p_q4, embedding as e, encoder_rows

        weight = "language_model.embed_tokens.weight"
        self.emit(
            "text/embedding",
            "embedding",
            lambda: primitive(e8p_q4.embedding, self.capacity, 131072, 2048),
            dict(
                IDs="Input",
                Weight=self.w.immutable(weight + ".packed"),
                Book=self.book,
                Scale=self.w.immutable(weight + ".scale"),
                Signs=self.signs[2048],
                Output="THidden",
            ),
        )
        self.emit(
            "text/overlay",
            "overlay",
            lambda: primitive(e.overlay, self.capacity, 2048, self.capacity),
            dict(Hidden="THidden", Features="Features", Index="FeatureIndex"),
        )
        self.transformer()
        self.norm("text/final_norm", "THidden", "TNorm", "language_model.norm.weight", 2048)
        self.emit(
            "text/pool",
            "pool",
            lambda: primitive(encoder_rows.pool, 2048),
            dict(X="TNorm", Mean="Mean", Segments="Segments", Count="SequenceCount"),
        )
        self.emit(
            "text/normalize",
            "normalize",
            lambda: primitive(encoder_rows.normalize, 2048),
            dict(Mean="Mean", Output="Embedding", Count="SequenceCount"),
            dynamic=False,
        )

    def vision(self):
        from kernels.cuda.model import embedding as e
        from kernels.cuda.vision.encoder import linear

        zero = self.w.constant("PatchBias", np.zeros(1024, np.float16))
        self.emit(
            "vision/patch",
            "patch",
            lambda: primitive(linear, 1024, 588, bm=32),
            dict(
                X="Pixels",
                W=self.w.immutable(
                    "vision_tower.patch_conv.weight", cast="float16", shape=[1024, 588]
                ),
                Bias=zero,
                Y="VHidden",
            ),
            vision=True,
        )
        self.norm(
            "vision/pre_norm", "VHidden", "VHidden", "vision_tower.ln_pre.weight", 1024, vision=True
        )
        self.transformer(vision=True)
        self.norm(
            "vision/projector_norm",
            "VHidden",
            "VNorm",
            "multi_modal_projector.norm.weight",
            1024,
            vision=True,
        )
        self.emit(
            "vision/merge",
            "merge",
            lambda: primitive(e.merge, self.capacity),
            dict(X="VNorm", Grid="Grid", Y="Merged"),
            vision=True,
            divisor=4,
        )
        for role, k, n, source, dest, weight in [
            ("merger", 4096, 1024, "Merged", "ProjectorInput", "patch_merger.merging_layer"),
            ("linear1", 1024, 2048, "ProjectorInput", "ProjectorHidden", "linear_1"),
            ("linear2", 2048, 2048, "ProjectorHidden", "VisionOutput", "linear_2"),
        ]:
            self.activation_input(
                "vision/" + role + "_input",
                source,
                k,
                compressed=role != "linear2",
                vision=True,
                divisor=4,
            )
            self.projection(
                "vision/" + role,
                "multi_modal_projector." + weight + ".weight",
                n,
                k,
                dest,
                vision=True,
                divisor=4,
            )
            if role == "linear1":
                self.emit(
                    "vision/gelu",
                    "gelu",
                    lambda: primitive(e.activation, self.capacity, 2048, kind="gelu"),
                    dict(X=dest, Y=dest),
                    vision=True,
                    divisor=4,
                )

    def publish(self):
        self.publish_encoder(
            architecture="nemotron_embed",
            model_name="Nemotron-3.5-Embed-1B-E8P-Q4A8",
            vocab=131072,
            logits="Embedding",
            weight_scope="Two-stage E8P Q4A8 text, embedding and vision; sensitive outputs W8; patch/norm FP16; one packed bank",
            config_signature=signature(self.config),
            copy_files=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "processor_config.json",
                "chat_template.jinja",
                "config_sentence_transformers.json",
                "sentence_bert_config.json",
                "modules.json",
                "README.md",
                "LICENSE",
            ],
            frontend_files=[
                "tokenizer.json",
                "tokenizer_config.json",
                "processor_config.json",
                "chat_template.jinja",
            ],
            spec_key="embedding",
            spec=dict(
                rows=self.row_spec(),
                dimension=2048,
                output="Embedding",
                program="embed",
                length="Length",
                feature_index="FeatureIndex",
                vision_program="embed_vision",
                pixels="Pixels",
                grid="Grid",
                vision_length="VisionLength",
                vision_output="VisionOutput",
                features="Features",
                image_token_id=10,
                max_patches=self.patches,
                patch_size=14,
                merge_size=2,
            ),
        )


def main():
    p = encoder_parser(__doc__, "nemotron-embed")
    p.add_argument("--max-patches", type=int, default=6400)
    args = p.parse_args()
    if (
        not 512 <= args.context <= 32768
        or args.context % 32
        or not 4 <= args.max_patches <= 12100
        or args.max_patches % 4
    ):
        p.error(
            "Context must be a multiple of 32 in 512..32768 and vision capacity a multiple of four in 4..12100"
        )
    args.engine = args.engine.resolve(strict=True)
    with atomic_model(args.model_output) as root:
        Builder(args, root).publish()


if __name__ == "__main__":
    main()

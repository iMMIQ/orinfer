"""Build a complete-sequence causal Qwen3-Reranker E8P execution package."""

from orinfer_build.compiler.compile import primitive
from orinfer_build.models.encoder import Projections, encoder_parser
from orinfer_build.models.qwen_reranker.config import configuration, signature
from orinfer_build.packaging.publication import atomic_model


class Builder(Projections):
    kernel_prefix = "qwen_reranker"
    norm_epsilon = 1e-6

    def __init__(self, args, root):
        self.init_encoder(
            args,
            root,
            validate_config=configuration,
            capacity=args.context,
            patches=0,
            identity_key="reranker_capacity",
            identity_value=args.context,
            sign_widths=(1024, 2048, 3072),
        )
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
        ids = self.config["quantization_config"]["classifier_token_ids"]
        self.publish_encoder(
            architecture="qwen3_reranker",
            model_name="Qwen3-Reranker-0.6B-E8P-Q4A8",
            vocab=151669,
            logits="ClassLogits",
            weight_scope="Two-stage E8P Q4A8; attention output and protected text blocks W8; norms and original two-class head FP16; one packed embedding bank",
            config_signature=signature(self.config),
            copy_files=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "chat_template.jinja",
                "README.md",
                "LICENSE",
            ],
            frontend_files=[
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "generation_config.json",
            ],
            spec_key="reranking",
            spec=dict(
                rows=self.row_spec(),
                program="rerank",
                length="Length",
                output="ClassLogits",
                false_token_id=ids[0],
                true_token_id=ids[1],
            ),
        )


def main():
    p = encoder_parser(__doc__, "qwen-reranker")
    args = p.parse_args()
    if not 512 <= args.context <= 32768 or args.context % 32:
        p.error("Context must be a multiple of 32 in 512..32768")
    args.engine = args.engine.resolve(strict=True)
    with atomic_model(args.model_output) as root:
        Builder(args, root).publish()


if __name__ == "__main__":
    main()

"""Bind the shared ViT encoder once with runtime patch rows and a merge divisor."""

import math
from orinfer_build.compiler.compile import primitive
from orinfer_build.contracts.vision import validate_vision

PATCHES = 32768


def build(b):
    from kernels.cuda.vision import encoder as e
    from kernels.cuda.vision.features import overlay

    features = b.context
    v = b.config["vision_config"]
    validate_vision(v, b.h)
    w = b.weights
    h, f, out = v["hidden_size"], v["intermediate_size"], v["out_hidden_size"]
    for name, shape, dtype in [
        ("VPixels", [PATCHES, 1536], "f16"),
        ("VGrid", [2], "i32"),
        ("VLength", [1], "i32"),
        ("VHidden", [PATCHES, h], "f16"),
        ("VNorm", [PATCHES, h], "f16"),
        ("VQKV", [PATCHES, 3 * h], "f16"),
        ("VQ", [PATCHES, h], "f16"),
        ("VK", [PATCHES, h], "f16"),
        ("VV", [PATCHES, h], "f16"),
        ("VAttention", [PATCHES, h], "f16"),
        ("VProjection", [PATCHES, h], "f16"),
        ("VFFN", [PATCHES, f], "f16"),
        ("VMerged", [PATCHES // 4, 4 * h], "f16"),
        ("VOutput", [PATCHES // 4, out], "f16"),
    ]:
        w.fixed(name, dtype, shape)
    w.fixed("Features", "f16", [features, out], "sequence")
    w.fixed("FeatureIndex", "i32", [b.context], "sequence")

    if b.config["quantization_config"].get("vision") == "e8p_q4":
        width = max(h, f, 4 * h)
        w.fixed("VActivationA8", "i8", [PATCHES * width])
        w.fixed("VActivationScale", "f16", [PATCHES])
        w.fixed("VProduct", "f32", [PATCHES * max(3 * h, f, 4 * h, out)])

    def weight(name, shape=None):
        return w.immutable("model.visual." + name, cast="float16", shape=shape)

    def emit(section, role, key, fn, args, *, divisor=1):
        b.aot.emit(
            f"qwen_rows/vision/{section}/{role}",
            f"vision-{key}",
            fn,
            args,
            PATCHES // divisor,
            row_divisor=divisor,
        )

    def dense(
        section, role, n, k, inp, dest, prefix, activation="none", divisor=1, separate_bias=False
    ):
        source_weight = "model.visual." + prefix + ".weight"
        if source_weight + ".packed" in w.index["weight_map"]:
            from kernels.cuda.model.rotation_a8 import rotate_activation_a8

            capacity = PATCHES // divisor
            if separate_bias:
                raise ValueError("Quantized patch embedding needs an explicit convolution boundary")
            emit(
                section,
                role + "_rotate",
                f"rotation-a8-{k}",
                lambda: primitive(rotate_activation_a8, PATCHES, k, dynamic_rows=True),
                dict(X=inp, Signs=b.signs[k], Q="VActivationA8", S="VActivationScale"),
                divisor=divisor,
            )
            emit(
                section,
                role + "_project",
                f"q4-a8-{capacity}-{n}-{k}",
                lambda: b.compile_projection(
                    source_weight, "e8p_q4", n, k, capacity, True, "float32"
                ),
                dict(
                    A="VActivationA8",
                    W=w.immutable(source_weight + ".packed"),
                    Book=b.book,
                    S=w.immutable(source_weight + ".scale"),
                    AS="VActivationScale",
                    Out="VProduct",
                ),
                divisor=divisor,
            )
            emit(
                section,
                role,
                f"bias-{n}-{activation}",
                lambda: primitive(e.bias_activation, n, activation),
                dict(X="VProduct", Bias=weight(prefix + ".bias"), Y=dest),
                divisor=divisor,
            )
            return
        emit(
            section,
            role,
            f"linear-{n}-{k}-{activation}-{separate_bias}",
            lambda: primitive(e.linear, n, k, activation, separate_bias=separate_bias),
            dict(
                X=inp, W=weight(prefix + ".weight", [n, k]), Bias=weight(prefix + ".bias"), Y=dest
            ),
            divisor=divisor,
        )

    def norm(section, role, inp, dest, prefix):
        emit(
            section,
            role,
            f"norm-{h}",
            lambda: primitive(e.layer_norm, h),
            dict(X=inp, W=weight(prefix + ".weight"), B=weight(prefix + ".bias"), Y=dest),
        )

    dense("begin", "patch", h, 1536, "VPixels", "VNorm", "patch_embed.proj", separate_bias=True)
    emit(
        "begin",
        "position",
        f"position-{h}",
        lambda: primitive(e.position, h, math.isqrt(v["num_position_embeddings"])),
        dict(X="VNorm", W=weight("pos_embed.weight"), Grid="VGrid", Length="VLength", Y="VHidden"),
    )
    for i in range(v["depth"]):
        s = f"layer{i}"
        p = f"blocks.{i}"
        norm(s, "norm1", "VHidden", "VNorm", p + ".norm1")
        dense(s, "qkv", 3 * h, h, "VNorm", "VQKV", p + ".attn.qkv")
        emit(
            s,
            "rope",
            f"rope-{h}-{v['num_heads']}",
            lambda: primitive(e.qkv_rope, h, v["num_heads"]),
            dict(X="VQKV", Grid="VGrid", Length="VLength", Q="VQ", K="VK", V="VV"),
        )
        emit(
            s,
            "attention",
            f"attention-{h}-{v['num_heads']}",
            lambda: primitive(e.attention, h, v["num_heads"]),
            dict(Q="VQ", K="VK", V="VV", Length="VLength", Y="VAttention"),
        )
        dense(s, "projection", h, h, "VAttention", "VProjection", p + ".attn.proj")
        emit(
            s,
            "residual1",
            f"add-{h}",
            lambda: primitive(e.add, h),
            dict(X="VProjection", Residual="VHidden", Y="VHidden"),
        )
        norm(s, "norm2", "VHidden", "VNorm", p + ".norm2")
        dense(s, "fc1", f, h, "VNorm", "VFFN", p + ".mlp.linear_fc1", "gelu_tanh")
        dense(s, "fc2", h, f, "VFFN", "VProjection", p + ".mlp.linear_fc2")
        emit(
            s,
            "residual2",
            f"add-{h}",
            lambda: primitive(e.add, h),
            dict(X="VProjection", Residual="VHidden", Y="VHidden"),
        )
    norm("end", "norm", "VHidden", "VNorm", "merger.norm")
    dense("end", "merge1", 4 * h, 4 * h, "VNorm", "VMerged", "merger.linear_fc1", "gelu", divisor=4)
    dense("end", "merge2", out, 4 * h, "VMerged", "VOutput", "merger.linear_fc2", divisor=4)
    b.emit(
        "begin/overlay",
        f"vision-overlay-{b.context}",
        lambda: primitive(overlay, 4096, b.h, b.context, features, dynamic_rows=True),
        dict(Embedding="Hidden", Features="Features", Index="FeatureIndex", Position="Step"),
    )
    b.vision = dict(
        dtype="f16",
        hidden=out,
        patch_size=v["patch_size"],
        temporal_patch_size=v["temporal_patch_size"],
        merge_size=v["spatial_merge_size"],
        max_patches=PATCHES,
        max_features=features,
        pixels="VPixels",
        grid="VGrid",
        length="VLength",
        output="VOutput",
        features="Features",
        feature_index="FeatureIndex",
        mrope_positions="MRopePositions",
        plans=[dict(patches=PATCHES, min_patches=4, program="vision")],
        **{
            k: b.config[k]
            for k in ["image_token_id", "vision_start_token_id", "vision_end_token_id"]
        },
    )

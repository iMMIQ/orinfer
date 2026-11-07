"""Bind bounded audio stems and the shared floating encoder components."""

import numpy as np
from orinfer_build.compiler.compile import primitive
from kernels.cuda.audio import encoder as audio
from kernels.cuda.vision import encoder as common
from kernels.cuda.vision.features import overlay
from orinfer_build.models.qwen_asr.tuning import select

CHUNKS = 32
FEATURES = 15600


def build(b):
    w = b.weights
    for name, dtype, shape, scope in [
        ("AudioMel", "f16", [CHUNKS, 128, 100], "workspace"),
        ("AudioConv1", "f16", [CHUNKS, 64, 50, 480], "workspace"),
        ("AudioConv2", "f16", [CHUNKS, 32, 25, 480], "workspace"),
        ("AudioConv3", "f16", [CHUNKS, 16, 13, 480], "workspace"),
        ("AudioFlat", "f16", [CHUNKS * 13, 7680], "workspace"),
        ("AudioStem", "f16", [CHUNKS * 13, 1024], "workspace"),
        ("AudioHidden", "f16", [FEATURES, 1024], "workspace"),
        ("AudioNorm", "f16", [FEATURES, 1024], "workspace"),
        ("AudioQKV", "f16", [FEATURES, 3072], "workspace"),
        ("AudioQ", "f16", [FEATURES, 1024], "workspace"),
        ("AudioK", "f16", [FEATURES, 1024], "workspace"),
        ("AudioV", "f16", [FEATURES, 1024], "workspace"),
        ("AudioAttention", "f16", [FEATURES, 1024], "workspace"),
        ("AudioProjection", "f16", [FEATURES, 1024], "workspace"),
        ("AudioFFN", "f16", [FEATURES, 4096], "workspace"),
        ("AudioOutput", "f16", [FEATURES, 2048], "workspace"),
        ("AudioLength", "i32", [FEATURES // 104], "workspace"),
        ("AudioFeatures", "f16", [FEATURES, 2048], "sequence"),
        ("AudioFeatureIndex", "i32", [b.context], "sequence"),
    ]:
        w.fixed(name, dtype, shape, scope)

    def weight(name):
        return w.immutable(name, cast="float16")

    def emit(name, key, fn, bindings, capacity=FEATURES, multiplier=1):
        b.aot.emit(
            "qwen_rows/audio/" + name,
            "asr-audio-" + key,
            fn,
            bindings,
            capacity * multiplier,
            row_multiplier=multiplier,
        )

    for i, channels, height, width, src in [
        (1, 1, 128, 100, "AudioMel"),
        (2, 480, 64, 50, "AudioConv1"),
        (3, 480, 32, 25, "AudioConv2"),
    ]:
        p = f"model.audio_tower.conv2d{i}"
        emit(
            f"stem/conv{i}",
            f"conv-{channels}-{height}-{width}",
            lambda channels=channels, height=height, width=width: select(
                b.row_tuner, convolution=(channels, height, width)
            ),
            dict(
                X=src,
                W=w.constant(
                    p + ".weight.runtime",
                    w.array(p + ".weight").transpose(0, 2, 3, 1).astype(np.float16),
                ),
                Bias=weight(p + ".bias"),
                Y=f"AudioConv{i}",
            ),
            CHUNKS,
        )
    emit(
        "stem/flatten",
        "flatten",
        lambda: primitive(audio.flatten),
        dict(X="AudioConv3", Y="AudioFlat"),
        CHUNKS,
    )
    zero = w.constant("AudioZeroBias1024", np.zeros(1024, np.float16))
    emit(
        "stem/out",
        "conv-out",
        lambda: select(b.row_tuner, outputs=1024, inputs=7680),
        dict(
            X="AudioFlat", W=weight("model.audio_tower.conv_out.weight"), Bias=zero, Y="AudioStem"
        ),
        CHUNKS,
        13,
    )
    inverse = np.exp(-np.log(10000.0) / 511 * np.arange(512, dtype=np.float32))
    angles = np.arange(13, dtype=np.float32)[:, None] * inverse[None]
    pe = w.constant(
        "AudioSinusoids",
        np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float16),
    )
    emit(
        "stem/position",
        "position",
        lambda: primitive(audio.position),
        dict(X="AudioStem", PE=pe),
        CHUNKS,
        13,
    )

    def norm(name, source, target, prefix):
        emit(
            name,
            "layernorm-1024",
            lambda: primitive(common.layer_norm, 1024, epsilon=1e-5),
            dict(X=source, W=weight(prefix + ".weight"), B=weight(prefix + ".bias"), Y=target),
        )

    def dense(name, n, k, source, target, prefix, activation="none"):
        emit(
            name,
            f"linear-{n}-{k}-{activation}",
            lambda: select(b.row_tuner, outputs=n, inputs=k, activation=activation),
            dict(X=source, W=weight(prefix + ".weight"), Bias=weight(prefix + ".bias"), Y=target),
        )

    def add(name, source):
        emit(
            name,
            "residual-1024",
            lambda: primitive(common.add, 1024),
            dict(X=source, Residual="AudioHidden", Y="AudioHidden"),
        )

    for i in range(24):
        p = f"model.audio_tower.layers.{i}"
        s = f"layer{i}/"
        norm(s + "norm1", "AudioHidden", "AudioNorm", p + ".self_attn_layer_norm")
        qw = w.combined(
            f"AudioL{i}QKV", [p + f".self_attn.{role}_proj.weight" for role in ["q", "k", "v"]]
        )
        qb = w.combined(
            f"AudioL{i}QKVBias", [p + f".self_attn.{role}_proj.bias" for role in ["q", "k", "v"]]
        )
        emit(
            s + "qkv",
            "linear-3072-1024-none",
            lambda: select(b.row_tuner, outputs=3072, inputs=1024),
            dict(X="AudioNorm", W=qw, Bias=qb, Y="AudioQKV"),
        )
        emit(
            s + "split",
            "split-qkv",
            lambda: primitive(audio.split_qkv),
            dict(X="AudioQKV", Q="AudioQ", K="AudioK", V="AudioV"),
        )
        emit(
            s + "attention",
            "window-attention-104-packed",
            lambda: primitive(common.attention, 1024, 16, window=104, windowed_lengths=True),
            dict(Q="AudioQ", K="AudioK", V="AudioV", Length="AudioLength", Y="AudioAttention"),
        )
        dense(s + "out", 1024, 1024, "AudioAttention", "AudioProjection", p + ".self_attn.out_proj")
        add(s + "add1", "AudioProjection")
        norm(s + "norm2", "AudioHidden", "AudioNorm", p + ".final_layer_norm")
        dense(s + "up", 4096, 1024, "AudioNorm", "AudioFFN", p + ".fc1", "gelu")
        dense(s + "down", 1024, 4096, "AudioFFN", "AudioProjection", p + ".fc2")
        add(s + "add2", "AudioProjection")
    norm("end/norm", "AudioHidden", "AudioNorm", "model.audio_tower.ln_post")
    dense(
        "end/project1",
        1024,
        1024,
        "AudioNorm",
        "AudioProjection",
        "model.multi_modal_projector.linear_1",
        "gelu",
    )
    dense(
        "end/project2",
        2048,
        1024,
        "AudioProjection",
        "AudioOutput",
        "model.multi_modal_projector.linear_2",
    )
    b.emit(
        "begin/overlay",
        "asr-feature-overlay",
        lambda: primitive(overlay, 4096, b.h, b.context, FEATURES, dynamic_rows=True),
        dict(
            Embedding="Hidden", Features="AudioFeatures", Index="AudioFeatureIndex", Position="Step"
        ),
    )
    b.audio = dict(
        hidden=b.h,
        max_features=FEATURES,
        stem_chunks=CHUNKS,
        mel="AudioMel",
        stem_output="AudioStem",
        encoder_input="AudioHidden",
        length="AudioLength",
        output="AudioOutput",
        features="AudioFeatures",
        feature_index="AudioFeatureIndex",
        positions="MRopePositions",
        token_id=151676,
        window_size=104,
    )

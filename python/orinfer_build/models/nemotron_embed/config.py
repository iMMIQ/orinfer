"""Validate the pinned 1B checkpoint's arithmetic before conversion or compilation."""

import copy

# Zero-based third text block: measured short-query sensitivity, +24.1 MB.
ROTATED_W8_LAYERS = [2]


TEXT = dict(
    model_type="ministral3",
    hidden_size=2048,
    intermediate_size=6144,
    num_hidden_layers=16,
    num_attention_heads=24,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=131072,
    rms_norm_eps=1e-5,
    hidden_act="silu",
    is_causal=False,
    pooling="avg",
    tie_word_embeddings=True,
    sliding_window=None,
)
VISION = dict(
    model_type="pixtral",
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    head_dim=64,
    patch_size=14,
    num_channels=3,
    hidden_act="silu",
)
ROPE = dict(
    rope_type="yarn",
    rope_theta=1000000.0,
    factor=16.0,
    original_max_position_embeddings=16384,
    beta_fast=32.0,
    beta_slow=1.0,
    apply_yarn_scaling=False,
    llama_4_scaling_beta=0.1,
)


def configuration(config, *, quantized=False):
    if config.get("model_type") != "mistral3" or config.get("architectures") != ["Mistral3Model"]:
        raise ValueError("Expected Nemotron-3.5-Embed-1B Mistral3Model")
    if bool(config.get("quantization_config")) != quantized:
        raise ValueError(
            "Expected completed E8P checkpoint" if quantized else "Expected original BF16"
        )
    if quantized and config["quantization_config"].get("rotated_w8_layers", []) not in (
        [],
        ROTATED_W8_LAYERS,
    ):
        raise ValueError("Unsupported Nemotron rotated-W8 layer policy")
    for section, expected in [("text_config", TEXT), ("vision_config", VISION)]:
        actual = config.get(section, {})
        for key, value in expected.items():
            if actual.get(key) != value or type(actual.get(key)) is not type(value):
                raise ValueError(f"Unsupported Nemotron {section}.{key}")
    for key, value in ROPE.items():
        if config["text_config"].get("rope_parameters", {}).get(key) != value:
            raise ValueError(f"Unsupported Nemotron RoPE {key}")
    if config["text_config"].get("llama_4_scaling") != dict(
        beta=0.1, original_max_position_embeddings=16384
    ):
        raise ValueError("Unsupported Nemotron attention scaling")
    if config["vision_config"].get("rope_parameters") != dict(
        rope_type="default", rope_theta=10000.0
    ):
        raise ValueError("Unsupported Pixtral axial RoPE")
    for key, value in dict(
        is_causal=False,
        pooling="avg",
        spatial_merge_size=2,
        image_token_index=10,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,
        vision_feature_layer=-1,
    ).items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported Nemotron {key}")
    return config["text_config"]


def signature(config):
    """Pin every model arithmetic field, including bidirectionality and pooling."""
    return copy.deepcopy({k: v for k, v in config.items() if k != "quantization_config"})

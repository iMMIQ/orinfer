"""Validate assumptions of the Qwen3.5 image adapter before adopting weights."""

import math


def validate_vision(vision, output_hidden):
    if (
        vision["num_heads"],
        vision["in_channels"],
        vision["patch_size"],
        vision["spatial_merge_size"],
        vision["temporal_patch_size"],
        vision["out_hidden_size"],
        vision["hidden_act"],
    ) != (16, 3, 16, 2, 2, output_hidden, "gelu_pytorch_tanh") or vision["hidden_size"] not in (
        1024,
        1152,
    ):
        raise ValueError("Unsupported vision architecture; need an explicit adapter")
    positions = vision["num_position_embeddings"]
    if (
        vision.get("deepstack_visual_indexes")
        or vision["depth"] <= 0
        or vision["intermediate_size"] <= 0
        or positions <= 0
        or math.isqrt(positions) ** 2 != positions
    ):
        raise ValueError("Unsupported vision positions, depth or DeepStack layout")

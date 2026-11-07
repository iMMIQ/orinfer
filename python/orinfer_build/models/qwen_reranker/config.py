"""Fixed arithmetic and scoring contract for Qwen3-Reranker-0.6B."""

import copy

ROTATED_W8_LAYERS = [0, 1, 2, 3, 8]

EXPECTED = dict(
    model_type="qwen3",
    architectures=["Qwen3ForCausalLM"],
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151669,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    attention_bias=False,
    tie_word_embeddings=True,
    rope_theta=1000000,
    rope_scaling=None,
    sliding_window=None,
    use_sliding_window=False,
)
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def configuration(config, *, quantized=False):
    for key, value in EXPECTED.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError(f"Unsupported Qwen3-Reranker {key}")
    if bool(config.get("quantization_config")) != quantized:
        raise ValueError(
            "Expected completed E8P checkpoint" if quantized else "Expected original BF16"
        )
    if quantized and config["quantization_config"].get("policy") != "qwen3-reranker-e8p-q4a8-v1":
        raise ValueError("Unsupported Qwen3-Reranker precision policy")
    if quantized and config["quantization_config"].get("rotated_w8_layers", []) not in (
        [],
        [0, 1, 2, 3],
        ROTATED_W8_LAYERS,
    ):
        raise ValueError("Unsupported Qwen3-Reranker protected layers")
    return config


def signature(config):
    return copy.deepcopy({k: v for k, v in config.items() if k != "quantization_config"})

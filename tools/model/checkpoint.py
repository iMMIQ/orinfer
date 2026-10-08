"""Bounded, indexed reader for the supported group-128 compressed-tensors W4 checkpoint."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from safetensors import safe_open
from tools.model.publication import file_hash, source_path


class Checkpoint:
    def __init__(self, directory, framework="pt"):
        self.root = Path(directory).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("--checkpoint must be a Hugging Face checkpoint directory")
        self.config = json.loads((self.root / "config.json").read_text())
        text = self.config.get("text_config", self.config)
        expected = dict(
            hidden_size=5120,
            intermediate_size=17408,
            vocab_size=248320,
            num_hidden_layers=64,
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
        if (
            self.config.get("model_type") not in ("qwen3_5", "qwen3_5_text")
            or any(text.get(k) != v for k, v in expected.items())
            or text.get("layer_types")
            != ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]
        ):
            raise ValueError(
                "Builder supports only the registered Qwen3_5 27B dimensions/layer order"
            )
        rope = text.get("rope_parameters", {})
        if (
            rope.get("rope_type") != "default"
            or rope.get("rope_theta") != 10000000
            or rope.get("partial_rotary_factor") != 0.25
            or rope.get("mrope_section") != [11, 11, 10]
            or rope.get("mrope_interleaved") is not True
            or text.get("partial_rotary_factor", 0.25) != 0.25
        ):
            raise ValueError("Checkpoint RoPE differs from the fixed 27B compiler recipe")
        index = self.root / "model.safetensors.index.json"
        self.stack = ExitStack()
        self.readers = {}
        self.mapping = {}
        try:
            if index.exists():
                self.mapping = json.loads(index.read_text())["weight_map"]
                if not self.mapping:
                    raise ValueError("Empty checkpoint index")
                files = set(self.mapping.values())
            else:
                files = {"model.safetensors"}
            for name in sorted(files):
                path = source_path(self.root, name)
                reader = self.stack.enter_context(
                    safe_open(path, framework=framework, device="cpu")
                )
                self.readers[name] = reader
                if not index.exists():
                    self.mapping.update({key: name for key in reader.keys()})
            for key, name in self.mapping.items():
                if key not in self.readers[name].keys():
                    raise ValueError(f"Checkpoint index references missing tensor: {key}")
            key = "model.language_model.layers.0.mlp.down_proj"
            if any(
                key + suffix not in self.mapping
                for suffix in (".weight_packed", ".weight_scale", ".weight_zero_point")
            ):
                raise ValueError(
                    "Expected compressed-tensors asymmetric W4 group128; BF16/AWQ qweight import is not implemented by this builder"
                )
            if (
                self.get_slice(key + ".weight_packed").get_shape() != [5120, 17408 // 8]
                or self.get_slice(key + ".weight_scale").get_shape() != [5120, 17408 // 128]
                or self.get_slice(key + ".weight_zero_point").get_shape()
                != [5120 // 8, 17408 // 128]
            ):
                raise ValueError("Checkpoint packing differs from supported W4 group128")
            for suffix in (".weight_packed", ".weight_zero_point"):
                if self.get_slice(key + suffix).get_dtype() != "I32":
                    raise ValueError(
                        "Packed weights and zero points require signed INT32 containers"
                    )
            identities = {name: file_hash(source_path(self.root, name)) for name in sorted(files)}
            for name in ("config.json", "model.safetensors.index.json"):
                if (self.root / name).exists():
                    identities[name] = file_hash(self.root / name)
            self.identity = dict(
                files=identities,
                sha256=hashlib.sha256(
                    json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            )
        except BaseException:
            self.stack.close()
            raise

    def validate_backbone(self):
        """Reject incomplete or mismatched tensor headers before GPU compilation."""

        def shape(key, expected, dtypes):
            if key not in self.mapping:
                raise ValueError(f"Missing backbone tensor: {key}")
            tensor = self.get_slice(key)
            if tensor.get_shape() not in expected or tensor.get_dtype() not in dtypes:
                raise ValueError(f"Backbone tensor shape/dtype mismatch: {key}")

        def dense(key, dimensions):
            shape(key, [dimensions], ("F16", "BF16", "F32"))

        def packed(key, n, k):
            shape(key + ".weight_packed", [[n, k // 8]], ("I32",))
            shape(key + ".weight_scale", [[n, k // 128]], ("F16", "BF16", "F32"))
            shape(key + ".weight_zero_point", [[n // 8, k // 128]], ("I32",))

        dense("model.language_model.embed_tokens.weight", [248320, 5120])
        dense("lm_head.weight", [248320, 5120])
        dense("model.language_model.norm.weight", [5120])
        for layer in range(64):
            prefix = f"model.language_model.layers.{layer}."
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                dense(prefix + name, [5120])
            for name in ("gate_proj", "up_proj"):
                packed(prefix + "mlp." + name, 17408, 5120)
            packed(prefix + "mlp.down_proj", 5120, 17408)
            if layer % 4 != 3:
                base = prefix + "linear_attn."
                for name, n, k in [
                    ("in_proj_qkv", 10240, 5120),
                    ("in_proj_z", 6144, 5120),
                    ("out_proj", 5120, 6144),
                ]:
                    packed(base + name, n, k)
                for name in ("in_proj_a.weight", "in_proj_b.weight"):
                    dense(base + name, [48, 5120])
                for name in ("A_log", "dt_bias"):
                    dense(base + name, [48])
                dense(base + "norm.weight", [128])
                shape(base + "conv1d.weight", [[10240, 1, 4], [10240, 4]], ("F16", "BF16", "F32"))
            else:
                base = prefix + "self_attn."
                for name, n, k in [
                    ("q_proj", 12288, 5120),
                    ("k_proj", 1024, 5120),
                    ("v_proj", 1024, 5120),
                    ("o_proj", 5120, 6144),
                ]:
                    packed(base + name, n, k)
                dense(base + "q_norm.weight", [256])
                dense(base + "k_norm.weight", [256])

    def get_tensor(self, key):
        return self.readers[self.mapping[key]].get_tensor(key)

    def get_slice(self, key):
        return self.readers[self.mapping[key]].get_slice(key)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

"""Export deterministic Engram addressing metadata; online code uses integers only.

Tokenizer normalization and RNG rules follow DeepSeek's V4.1 reference Engram
implementation (MIT, Copyright (c) 2023 DeepSeek; see LICENSE in this directory).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def validate_hadamard32(rotation):
    """Require the exact uniform signed Walsh bank used by the fast kernel."""
    matrix = np.asarray(rotation, dtype=np.float32)
    if matrix.shape != (32, 32) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0:
        raise ValueError("Expected a finite 32x32 signed Walsh rotation")
    walsh = np.array(
        [[(-1) ** ((i & j).bit_count()) for j in range(32)] for i in range(32)],
        dtype=np.float32,
    )
    if not (
        np.all(np.abs(matrix) == matrix[0, 0]) and np.array_equal(matrix, matrix[:, :1] * walsh)
    ):
        raise ValueError("Fast Engram requires uniform signed Walsh rows in canonical order")


def next_prime(start, seen):
    candidate = start + 1
    while True:
        prime = candidate >= 2 and all(candidate % d for d in range(2, int(candidate**0.5) + 1))
        if prime and candidate not in seen:
            return candidate
        candidate += 1


def compressed_tokens(tokenizer):
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    if set(tokenizer.get_vocab(with_added_tokens=True).values()) != set(range(size)):
        raise ValueError("Engram requires dense tokenizer IDs")
    mapping, keys = [], {}
    for token in range(size):
        text = tokenizer.decode([token], skip_special_tokens=False)
        key = (
            tokenizer.id_to_token(token)
            if "\ufffd" in text
            else normalize.normalize_str(text) or text
        )
        mapping.append(keys.setdefault(key, len(keys)))
    return mapping, len(keys)


def layers(text, vocabulary):
    if text["engram_max_ngram_size"] != 4:
        raise ValueError("This exporter supports the V4.1 four-token Engram contract")
    seen, result = set(), []
    bound = max(1, (np.iinfo(np.int64).max // vocabulary) // 2)
    for layer, rows in zip(text["engram_layer_ids"], text["engram_num_embeddings"], strict=True):
        primes = []
        for _ in range(3):
            current = text["engram_vocab_size"] - 1
            for _ in range(text["engram_n_heads"]):
                current = next_prime(current, seen)
                seen.add(current)
                primes.append(current)
        if sum(primes) != rows:
            raise ValueError("Engram bucket rows differ from checkpoint")
        multipliers = (
            np.random.default_rng(10007 * layer).integers(0, bound, size=4, dtype=np.int64) * 2 + 1
        )
        result.append(dict(layer=layer, rows=rows, multipliers=multipliers.tolist(), primes=primes))
    return result


def export(model, output):
    from tokenizers import Tokenizer

    config_bytes = (model / "config.json").read_bytes()
    tokenizer_bytes = (model / "tokenizer.json").read_bytes()
    config = json.loads(config_bytes)
    text = config["text_config"]
    mapping, vocabulary = compressed_tokens(Tokenizer.from_str(tokenizer_bytes.decode()))
    if vocabulary != text["engram_compressed_vocab_size"]:
        raise ValueError(f"Compressed vocabulary {vocabulary} differs from checkpoint")
    image = config["image_token_id"]
    result = dict(
        schema_version=1,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
        token_map=mapping,
        pad_token=text["engram_pad_token_id"],
        barriers=[image, config.get("image_pad_token_id", image + 1)],
        heads=text["engram_n_heads"],
        layers=layers(text, vocabulary),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, output)  # atomic publication, refuses existing output
        finally:
            temporary.unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.model, args.output)
    print(json.dumps(dict(tokens=len(result["token_map"]), layers=result["layers"])))

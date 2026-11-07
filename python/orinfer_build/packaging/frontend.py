"""Freeze HF added-token overrides into the standalone deployment tokenizer."""

import json
from pathlib import Path


def prepare_generation_config(source, destination):
    """Retain explicit HF defaults, or freeze EOS from this model's own assets."""
    source, destination = Path(source), Path(destination)
    generation = source / "generation_config.json"
    if generation.is_file():
        destination.write_bytes(generation.read_bytes())
        return
    from tokenizers import Tokenizer

    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    eos = text.get("eos_token_id", config.get("eos_token_id", []))
    eos = eos if isinstance(eos, list) else [eos]
    if any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in eos):
        raise ValueError("Invalid configured EOS ID")
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    settings = json.loads((source / "tokenizer_config.json").read_text())
    token = settings.get("eos_token")
    token = token.get("content") if isinstance(token, dict) else token
    if token is not None:
        if not isinstance(token, str) or tokenizer.token_to_id(token) is None:
            raise ValueError("Tokenizer EOS is outside its vocabulary")
        eos.append(tokenizer.token_to_id(token))
    eos = list(dict.fromkeys(eos))
    if not eos or any(tokenizer.id_to_token(token) is None for token in eos):
        raise ValueError("Missing EOS or EOS outside tokenizer vocabulary")
    pad = text.get("pad_token_id", config.get("pad_token_id"))
    if pad is None:
        pad = eos[0]
    if type(pad) is not int or not 0 <= pad <= 0xFFFFFFFF or tokenizer.id_to_token(pad) is None:
        raise ValueError("Invalid configured padding ID")
    destination.write_text(json.dumps(dict(eos_token_id=eos, pad_token_id=pad)) + "\n")


def prepare_chat_template(source, destination):
    """Freeze HF's standalone or embedded text template without rewriting it."""
    source = Path(source)
    template = source / "chat_template.jinja"
    if template.is_file():
        raw = template.read_bytes()
        text = raw.decode("utf-8")
    else:
        config = json.loads((source / "tokenizer_config.json").read_text())
        text = config.get("chat_template")
        if not isinstance(text, str):
            raise ValueError("Expected a standalone or embedded text chat template")
        raw = text.encode("utf-8")
    if not text.strip():
        raise ValueError("Empty chat template")
    Path(destination).write_bytes(raw)


def prepare_tokenizer(source, destination):
    """Match HF token IDs without requiring tokenizer_config parsing online."""
    from tokenizers import AddedToken, Tokenizer

    source, destination = Path(source), Path(destination)
    raw = (source / "tokenizer.json").read_bytes()
    tokenizer = Tokenizer.from_str(raw.decode())
    config = source / "tokenizer_config.json"
    overrides = (
        json.loads(config.read_text()).get("added_tokens_decoder", {}) if config.exists() else {}
    )
    original = tokenizer.to_str()
    for key, values in sorted(overrides.items(), key=lambda item: int(item[0])):
        expected = int(key)
        token = AddedToken(**values)
        existing = tokenizer.token_to_id(token.content)
        if existing is not None and existing != expected:
            raise ValueError(f"Tokenizer ID conflict for {token.content}: {existing} != {expected}")
        if token.special:
            tokenizer.add_special_tokens([token])
        else:
            tokenizer.add_tokens([token])
        if tokenizer.token_to_id(token.content) != expected:
            raise ValueError(f"Tokenizer override would renumber {token.content}")
    # Preserve source bytes when its serialized tokenizer is already complete.
    target = tokenizer.to_str()
    destination.write_bytes(raw if target == original else (target + "\n").encode())

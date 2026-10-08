"""Deterministic code workloads and complete-session regression signatures."""

import hashlib

import torch
from transformers import AutoTokenizer

from tools.model.flash_next.long_context import state_hash


TASKS = {
    "python": (
        "Python",
        "Implement a thread-safe TTL LRU cache in Python using only the standard library. "
        "Include a generic class, constructor validation, get, put, delete, clear, and len. "
        "Use time.monotonic, lazy expiry, OrderedDict and an RLock. An expired get must raise "
        "KeyError. Add unittest tests for eviction, expiry using an injected clock, update, "
        "and deletion. Output the complete source code, without prose.",
    ),
    "rust": (
        "Rust",
        "Write a complete Rust module implementing a streaming binary frame decoder. "
        "The format is a big-endian u32 payload length followed by payload bytes. "
        "Use only std, accept arbitrary chunks, retain incomplete frames, reject frames "
        "above a configurable maximum, and avoid copying the unconsumed buffer on every "
        "push. Include explicit error types, constructor, push, reset and unit tests "
        "covering fragmented headers, fragmented payloads, multiple frames and oversized "
        "frames. Output the complete Rust code without prose.",
    ),
    "typescript": (
        "TypeScript",
        "Implement a fully typed TypeScript async task pool without dependencies. "
        "Provide mapConcurrent<T,R>(items, concurrency, fn, signal?) returning results "
        "in input order. Validate concurrency, cap in-flight tasks, propagate the first "
        "error, honor AbortSignal before starting a task, and settle already started "
        "promises without unhandled rejections. Include usage examples and tests for "
        "ordering, maximum concurrency, errors and cancellation. Output source code "
        "only.",
    ),
}


def requests(checkpoint, lengths, names, thinking, reasoning_effort="xhigh"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = (checkpoint / "chat_template.jinja").read_text()
    # Deterministic repository context, rather than repeated numbered prose.
    context = "".join(
        f"def normalize_record_{i}(record: dict) -> dict:\n"
        f'    key = "field_{i}"\n    value = record.get(key, {i})\n'
        '    return {key: str(value).strip(), "valid": value is not None}\n\n'
        for i in range(max(lengths) // 12 + 32)
    )
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    result = []
    for name in names:
        language, task = TASKS[name]
        for enabled in thinking:
            for requested in lengths:

                def render(count):
                    content = (
                        "Repository reference helpers (not part of the requested implementation):\n"
                        "```python\n" + tokenizer.decode(context_ids[:count]) + "\n```\n\n" + task
                    )
                    messages = [
                        {"role": "system", "content": f"You are a careful {language} programmer."},
                        {"role": "user", "content": content},
                    ]
                    ids = tokenizer.apply_chat_template(
                        messages,
                        chat_template=template,
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=enabled,
                        reasoning_effort=reasoning_effort,
                    )
                    return messages, ids

                messages, ids = render(0)
                count = max(0, requested - len(ids))
                # Retokenization at context boundaries can change a few tokens.
                for _ in range(4):
                    messages, ids = render(count)
                    if len(ids) == requested:
                        break
                    count = max(0, min(len(context_ids), count + requested - len(ids)))
                result.append(
                    {
                        "id": f"{name}-{'thinking' if enabled else 'direct'}-{requested}",
                        "language": language,
                        "thinking": enabled,
                        "requested_length": requested,
                        "reasoning_effort": reasoning_effort if enabled else None,
                        "prompt_ids": ids,
                        "messages": messages,
                    }
                )
    return tokenizer, result


def session_state(session):
    """Hash the live prefix, including CPU cursors and the next draft condition."""
    model, draft = session.target, session.draft
    return (
        state_hash(model),
        state_hash(draft),
        model.position,
        draft.position,
        list(model.history),
        list(draft.history),
        session.pending,
        session.draft_token,
        hashlib.sha256(session.hidden.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
    )

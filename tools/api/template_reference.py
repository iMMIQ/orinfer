#!/usr/bin/env python3
"""Generate offline Transformers token references for Rust chat codec tests."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Lookup a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    requests = [
        {"messages": [{"role": "user", "content": "你好，2+3等于几？"}]},
        {"messages": [{"role": "system", "content": "Answer briefly."},
                      {"role": "user", "content": [{"type": "text", "text": "天气"}]}], "tools": tools},
        {"messages": [{"role": "user", "content": "查一下北京"},
                      {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "lookup", "arguments": '{"city":"北京"}'}}]},
                      {"role": "tool", "tool_call_id": "a", "content": "晴朗"}], "tools": tools},
        {"messages": [{"role": "user", "content": "2+3?"}], "enable_thinking": True, "reasoning_effort": "low"},
    ]
    cases = []
    for request in requests:
        request.update(model="qwen3.8-27b", max_tokens=16)
        # serde_json's object representation uses sorted keys. Canonicalize the
        # Python input before applying the unchanged checkpoint template.
        normalized = json.loads(json.dumps(request, sort_keys=True))
        for message in normalized["messages"]:
            for call in message.get("tool_calls", []):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        ids = tokenizer.apply_chat_template(
            normalized["messages"], tools=normalized.get("tools", []),
            enable_thinking=request.get("enable_thinking", False),
            reasoning_effort=request.get("reasoning_effort", "xhigh"),
            add_generation_prompt=True,
        )
        if hasattr(ids, "get"):
            ids = ids["input_ids"]
        cases.append({"request": request, "ids": ids})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(cases, output, ensure_ascii=False)
    print("Reference lengths:", [len(case["ids"]) for case in cases])
    print("ORIN_CHAT_REFERENCE=" + str(args.output.resolve()))


if __name__ == "__main__":
    main()

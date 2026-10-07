#!/usr/bin/env python3
"""Integration checks against a running real-model Chat Completions server."""
import argparse
import concurrent.futures
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def request(base, path, payload=None):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("ORINFER_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["ORINFER_API_KEY"]
    encoded = None if payload is None else json.dumps(payload).encode()
    return urllib.request.urlopen(
        urllib.request.Request(base + path, data=encoded, headers=headers), timeout=180
    )


def complete(base, payload):
    with request(base, "/chat/completions", payload) as response:
        return json.load(response)


def stream(base, payload):
    chunks = []
    done = False
    with request(base, "/chat/completions", dict(payload, stream=True,
                 stream_options={"include_usage": True})) as response:
        assert response.headers["Content-Type"].startswith("text/event-stream")
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            assert "error" not in chunk, chunk
            chunks.append(chunk)
    assert done, "SSE did not terminate with [DONE]"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"] == [] and "usage" in chunks[-1]
    assert chunks[-2]["choices"][0]["finish_reason"] in ("stop", "length", "tool_calls")
    assert len({chunk["id"] for chunk in chunks}) == 1
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--expect-prefix-cache", action="store_true",
                        help="Require complete prompt reuse on repeated requests")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON result path outside tracked source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = args.base_url.rstrip("/")
    with request(base.removesuffix("/v1"), "/health") as response:
        context = json.load(response)["max_context"]
    with request(base, "/models") as response:
        assert any(model["id"] == args.model for model in json.load(response)["data"])
    common = {"model": args.model, "temperature": 0, "seed": 20261002, "max_tokens": 16}
    text = dict(common, messages=[{"role": "user", "content": "2+3等于几？只回答数字。"}])
    evidence = {}
    first = complete(base, text)
    assert first["choices"][0]["message"]["content"].strip() == "5", first
    assert first["choices"][0]["finish_reason"] == "stop", first
    assert first["usage"]["prompt_tokens"] % 512 != 0, "Expected real arbitrary-length prompt"
    evidence["text"] = first
    print("PASS arbitrary prompt, EOS, usage", flush=True)
    chunks = stream(base, text)
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert content == first["choices"][0]["message"]["content"]
    if args.expect_prefix_cache:
        assert chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == first["usage"]["prompt_tokens"]
    evidence["stream"] = chunks
    print("PASS SSE text matches non-streaming", flush=True)
    stopped = complete(base, dict(text, stop="5"))
    assert not stopped["choices"][0]["message"]["content"] and stopped["choices"][0]["finish_reason"] == "stop"
    evidence["stop"] = stopped
    stochastic = dict(text, temperature=0.7, top_p=0.9)
    a = complete(base, stochastic)
    b = complete(base, stochastic)
    assert a["choices"] == b["choices"], (a, b)
    if args.expect_prefix_cache:
        assert b["usage"]["prompt_tokens_details"]["cached_tokens"] == b["usage"]["prompt_tokens"]
    evidence["seeded"] = [a, b]
    print("PASS stop across tokens and seeded sampling", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        queued = list(pool.map(lambda _: complete(base, text), range(2)))
    assert all(result["choices"] == first["choices"] for result in queued)
    assert queued[0]["id"] != queued[1]["id"]
    evidence["queued_isolation"] = queued
    print("PASS queued requests have private reset state", flush=True)
    with request(base, "/chat/completions", dict(text, stream=True, max_tokens=512)) as response:
        while True:
            line = response.readline()
            assert line, "Stream ended before role delta"
            if line.startswith(b"data: "):
                break
    after_cancel = complete(base, text)
    assert after_cancel["choices"] == first["choices"]
    evidence["after_cancel"] = after_cancel
    print("PASS disconnected request cancels and next request resets", flush=True)
    for invalid in [dict(text, model="unknown"), dict(text, n=2), dict(text, max_tokens=context + 1),
                    dict(text, messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])]:
        try:
            complete(base, invalid)
        except urllib.error.HTTPError as error:
            assert error.code == 400, error.code
            assert "error" in json.load(error)
        else:
            raise AssertionError("Invalid request was accepted")
    print("PASS OpenAI-shaped validation errors", flush=True)
    tools = [{"type": "function", "function": {"name": "read_verification", "description": "Read a local verification code.",
              "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "count": {"type": "integer"}},
                             "required": ["filename", "count"], "additionalProperties": False}}}]
    tool_request = dict(common, max_tokens=128, tools=tools, tool_choice="required", parallel_tool_calls=False,
                        messages=[{"role": "user", "content": "Call read_verification with filename input.txt and count 2. Then report the returned code."}])
    chunks = stream(base, tool_request)
    calls = [call for c in chunks if c["choices"] for call in c["choices"][0]["delta"].get("tool_calls", [])]
    assert calls and all(c["index"] == 0 for c in calls), calls
    call = {"id": calls[0]["id"], "type": "function", "function": {
        "name": calls[0]["function"]["name"],
        "arguments": "".join(c["function"].get("arguments", "") for c in calls),
    }}
    assert len(calls) > 1, "Expected incremental tool arguments"
    assert call["function"]["name"] == "read_verification", call
    assert json.loads(call["function"]["arguments"]) == {"filename": "input.txt", "count": 2}, call
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    followup = dict(common, max_tokens=64, tools=tools, messages=tool_request["messages"] + [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": '{"code":"ORIN_CHECK_5824"}'}])
    final = complete(base, followup)
    assert "ORIN_CHECK_5824" in (final["choices"][0]["message"]["content"] or ""), final
    assert final["choices"][0]["finish_reason"] == "stop", final
    evidence["tool_stream"] = chunks
    evidence["tool_result"] = final
    print("PASS real model tool call, typed arguments and result round-trip", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(evidence, file, ensure_ascii=False, indent=2)
    print(f"All checks passed; evidence: {args.output}")


if __name__ == "__main__":
    main()

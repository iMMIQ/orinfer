#!/usr/bin/env python3
"""Check structured output, tools and probabilities against a real Chat API."""
import argparse
import json
import math
import urllib.error
from pathlib import Path

from smoke import complete, request, stream


def tool_calls(chunks):
    calls = {}
    for chunk in chunks:
        for choice in chunk["choices"]:
            for delta in choice["delta"].get("tool_calls", []):
                call = calls.setdefault(delta["index"], {"function": {"arguments": ""}})
                for key in ("id", "type"):
                    if key in delta:
                        call[key] = delta[key]
                if "name" in delta["function"]:
                    call["function"]["name"] = delta["function"]["name"]
                call["function"]["arguments"] += delta["function"].get("arguments", "")
    return [calls[i] for i in sorted(calls)]


def check_scores(content, scores, count):
    raw = b"".join(bytes(s["bytes"]) for s in scores)
    assert raw.decode() == content, (raw, content)
    for score in scores:
        assert math.isfinite(score["logprob"]) and score["logprob"] <= 0, score
        assert len(score["top_logprobs"]) <= count, score
        assert all(math.isfinite(s["logprob"]) and s["logprob"] <= 0
                   for s in score["top_logprobs"]), score
        probabilities = [s["logprob"] for s in score["top_logprobs"]]
        assert probabilities == sorted(probabilities, reverse=True), score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--tokenizer-dir", type=Path, help="Enable token-ID bias checks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-logprobs", action="store_true",
                        help="Also generate a large 512-token probability response")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = args.base_url.rstrip("/")
    common = {"model": args.model, "temperature": 0, "seed": -20261002,
              "max_completion_tokens": 128,
              "messages": [{"role": "user", "content": "2+3等于几？只回答数字。"}]}
    evidence = {}

    def record(name, result):
        evidence[name] = result
        print(f"PASS {name}", flush=True)

    nullable = complete(base, dict(common, stream=None, tools=None, store=False,
                                  metadata={"test": "compatibility"}, service_tier="auto"))
    assert nullable["choices"][0]["message"]["content"].strip() == "5", nullable
    record("nullable_and_signed_seed", nullable)
    cached = complete(base, dict(common, prompt_cache_key="pi-session-probe", prompt_cache_retention="24h"))
    assert cached["choices"] == nullable["choices"], (cached, nullable)
    record("cache_hints_preserve_output", cached)
    invalid = [({"stream": "yes"}, "stream"), ({"top_logprobs": 3}, "top_logprobs"),
               ({"logprobs": True, "top_logprobs": 21}, "top_logprobs"),
               ({"store": True}, "store"), ({"temperature": -1}, "temperature"),
               ({"reasoning_effort": "invalid"}, "reasoning_effort"),
               ({"thinking_token_budget": -1}, "thinking_token_budget"),
               ({"prompt_cache_key": "x" * 65}, "prompt_cache_key"),
               ({"prompt_cache_retention": "forever"}, "prompt_cache_retention"),
               ({"enable_thinking": True, "thinking_token_budget": 0,
                 "max_completion_tokens": 1}, "max_completion_tokens"),
               ({"logit_bias": {"999999999": 1}}, "logit_bias.999999999"),
               ({"response_format": {"type": "json_schema", "json_schema": {
                   "name": "remote", "schema": {"$ref": "https://example.com/schema"}}}},
                "response_format.json_schema.schema")]
    errors = []
    for fields, param in invalid:
        try:
            complete(base, dict(common, **fields))
        except urllib.error.HTTPError as error:
            body = json.load(error)
            assert error.code == 400, (error.code, body)
            assert error.headers.get("x-request-id"), error.headers
            assert body["error"]["param"] == param, (param, body)
            errors.append(body)
        else:
            raise AssertionError(f"Invalid request accepted: {fields}")
    record("validation_errors", errors)
    with request(base.removesuffix("/v1"), "/health") as response:
        context_limit = json.load(response)["max_context"]
    try:
        complete(base, dict(common, max_completion_tokens=context_limit))
    except urllib.error.HTTPError as error:
        body = json.load(error)
        assert error.code == 400 and body["error"]["code"] == "context_length_exceeded", body
        assert "exceeds the context window" in body["error"]["message"], body
        record("recognizable_context_overflow", body)
    else:
        raise AssertionError("Context overflow accepted")
    for budget in (0, 8, 32):
        payload = dict(common, enable_thinking=True, thinking_token_budget=budget)
        full = complete(base, payload)
        chunks = stream(base, payload)
        assert full["usage"]["completion_tokens_details"]["reasoning_tokens"] <= budget, full
        assert chunks[-1]["usage"]["completion_tokens_details"]["reasoning_tokens"] <= budget, chunks
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        assert text == full["choices"][0]["message"]["content"], (text, full)
        # Forcing an early thinking boundary can change answer style. Check
        # the result and protocol separately from verbatim baseline output.
        assert text.rstrip().endswith("5"), full
        assert full["choices"][0]["finish_reason"] == "stop", full
        record(f"thinking_budget_{budget}", {"full": full, "stream": chunks})
    for path in ("/missing", "/chat/completions"):
        try:
            request(base, path)
        except urllib.error.HTTPError as error:
            assert error.code in (404, 405) and "error" in json.load(error)
        else:
            raise AssertionError(f"Unexpected success: {path}")

    prob_request = dict(common, logprobs=True, top_logprobs=3)
    full = complete(base, prob_request)
    choice = full["choices"][0]
    check_scores(choice["message"]["content"], choice["logprobs"]["content"], 3)
    assert full["choices"][0]["message"] == nullable["choices"][0]["message"]
    chunks = stream(base, prob_request)
    content = "".join(c["choices"][0]["delta"].get("content", "")
                      for c in chunks if c["choices"])
    scores = [s for c in chunks if c["choices"]
              for s in (c["choices"][0]["logprobs"] or {}).get("content", [])]
    check_scores(content, scores, 3)
    assert content == choice["message"]["content"]
    assert scores == choice["logprobs"]["content"], (scores, choice)
    assert all(c["usage"] is None for c in chunks[:-1])
    record("target_logprobs_and_sse", {"full": full, "stream": chunks})
    unicode_request = dict(prob_request, messages=[{"role": "user", "content":
        "请原样输出这串字符，不要解释：中文🙂 café"}])
    unicode_result = complete(base, unicode_request)
    choice = unicode_result["choices"][0]
    check_scores(choice["message"]["content"], choice["logprobs"]["content"], 3)
    record("unicode_token_bytes", unicode_result)
    if args.tokenizer_dir:
        tokenizer = json.loads((args.tokenizer_dir / "tokenizer.json").read_text())
        token_id = tokenizer["model"]["vocab"]["7"]
        biased = complete(base, dict(prob_request, max_completion_tokens=1,
                                     logit_bias={str(token_id): 100}))
        assert biased["choices"][0]["message"]["content"] == "7", biased
        record("logit_bias", biased)

    schema = {"type": "object", "properties": {
        "answer": {"type": "integer", "const": 5},
        "language": {"type": "string", "enum": ["zh"]},
        "items": {"type": "array", "items": {"type": "integer"},
                  "minItems": 2, "maxItems": 2}},
        "required": ["answer", "language", "items"], "additionalProperties": False}
    structured = dict(common, response_format={"type": "json_schema", "json_schema": {
        "name": "answer", "strict": True, "schema": schema}})
    capped_json = complete(base, dict(structured, enable_thinking=True, thinking_token_budget=8))
    assert json.loads(capped_json["choices"][0]["message"]["content"])["answer"] == 5, capped_json
    assert capped_json["usage"]["completion_tokens_details"]["reasoning_tokens"] <= 8, capped_json
    record("thinking_budget_with_schema", capped_json)
    for thinking in (False, True):
        result = complete(base, dict(structured, enable_thinking=thinking,
                                     reasoning_effort="low" if thinking else "none",
                                     logprobs=True, top_logprobs=3,
                                     max_completion_tokens=512 if thinking else 128))
        choice = result["choices"][0]
        assert choice["finish_reason"] == "stop", result
        value = json.loads(choice["message"]["content"])
        assert set(value) == {"answer", "language", "items"} and value["answer"] == 5
        assert value["language"] == "zh" and len(value["items"]) == 2
        assert all(isinstance(i, int) for i in value["items"])
        check_scores(choice["message"]["content"], choice["logprobs"]["content"], 3)
        if thinking:
            assert choice["message"].get("reasoning_content"), result
            assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0
        record(f"schema_thinking_{thinking}", result)
    obj = complete(base, dict(common, response_format={"type": "json_object"},
                              messages=[{"role": "user", "content": 'Return JSON {"ok":true}.'}]))
    assert isinstance(json.loads(obj["choices"][0]["message"]["content"]), dict), obj
    record("json_object", obj)
    chunks = stream(base, structured)
    content = "".join(c["choices"][0]["delta"].get("content", "")
                      for c in chunks if c["choices"])
    assert json.loads(content)["answer"] == 5 and chunks[-2]["choices"][0]["finish_reason"] == "stop"
    record("schema_sse", chunks)
    truncated = complete(base, dict(structured, max_completion_tokens=1))
    assert truncated["choices"][0]["finish_reason"] == "length", truncated
    record("schema_length_limit", truncated)
    tagged = {"type": "object", "properties": {"text": {"type": "string", "const":
        "<think>中文</think><tool_call>literal</tool_call>"}},
        "required": ["text"], "additionalProperties": False}
    literal = complete(base, dict(common, logprobs=True, top_logprobs=3,
                                 response_format={"type": "json_schema", "json_schema": {
        "name": "literal", "strict": True, "schema": tagged}}))
    assert json.loads(literal["choices"][0]["message"]["content"])["text"] == tagged["properties"]["text"]["const"]
    check_scores(literal["choices"][0]["message"]["content"],
                 literal["choices"][0]["logprobs"]["content"], 3)
    record("json_literal_protocol_tags", literal)
    sampled_request = dict(structured, temperature=0.7, top_p=0.9, logprobs=True, top_logprobs=3)
    sampled = complete(base, sampled_request)
    repeated = complete(base, sampled_request)
    assert sampled["choices"] == repeated["choices"], (sampled, repeated)
    choice = sampled["choices"][0]
    check_scores(choice["message"]["content"], choice["logprobs"]["content"], 3)
    assert json.loads(choice["message"]["content"])["answer"] == 5
    record("sampled_constraint_target_probs_and_seed", sampled)

    tools = [{"type": "function", "function": {"name": "read_verification", "strict": True,
              "description": "Read a local verification code.", "parameters": {
                  "type": "object", "properties": {"filename": {"type": "string"},
                                                      "count": {"type": "integer", "const": 2}},
                  "required": ["filename", "count"], "additionalProperties": False}}}]
    tool_request = dict(common, tools=tools, parallel_tool_calls=False,
                        messages=[{"role": "user", "content":
                            "Call read_verification with filename input.txt and count 2."}])
    for label, choice in (("required", "required"), ("named", {"type": "function",
                           "function": {"name": "read_verification"}})):
        chunks = stream(base, dict(tool_request, tool_choice=choice))
        calls = tool_calls(chunks)
        assert len(calls) == 1 and calls[0]["function"]["name"] == "read_verification", calls
        assert json.loads(calls[0]["function"]["arguments"]) == {"filename": "input.txt", "count": 2}, calls
        deltas = [d for c in chunks if c["choices"]
                  for d in c["choices"][0]["delta"].get("tool_calls", [])]
        assert len(deltas) > 1, deltas
        assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls", chunks
        record(f"strict_{label}_tool_sse", chunks)
    reference_tools = json.loads(json.dumps(tools))
    params = reference_tools[0]["function"]["parameters"]
    params["$defs"] = {"filename": {"type": "string", "enum": ["input.txt"]}}
    params["properties"]["filename"] = {"$ref": "#/$defs/filename"}
    referenced = complete(base, dict(tool_request, tools=reference_tools, tool_choice="required"))
    args_value = json.loads(referenced["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args_value == {"filename": "input.txt", "count": 2}, referenced
    record("strict_tool_local_ref", referenced)
    call = calls[0]
    followup = dict(common, tools=tools, tool_choice="none", messages=tool_request["messages"] + [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": '{"code":"ORIN_CHECK_5824"}'}])
    result = complete(base, followup)
    assert "ORIN_CHECK_5824" in result["choices"][0]["message"]["content"], result
    record("strict_tool_history", result)
    capped_tool = complete(base, dict(tool_request, tool_choice="required",
                                     enable_thinking=True, thinking_token_budget=8))
    assert capped_tool["choices"][0]["message"]["tool_calls"], capped_tool
    assert capped_tool["usage"]["completion_tokens_details"]["reasoning_tokens"] <= 8, capped_tool
    record("thinking_budget_with_tool", capped_tool)
    failed_call = json.loads(json.dumps(call))
    failed_call["function"]["arguments"] = '{"filename":'
    recovered = complete(base, dict(tool_request, tool_choice="required", messages=tool_request["messages"] + [
        {"role": "assistant", "content": None, "tool_calls": [failed_call]},
        {"role": "tool", "tool_call_id": failed_call["id"],
         "content": "Invalid truncated arguments. Retry with filename input.txt and count 2."}]))
    assert json.loads(recovered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"filename":"input.txt","count":2}, recovered
    record("failed_tool_history_recovery", recovered)
    auto = complete(base, dict(tool_request, tool_choice="auto", messages=[{
        "role": "user", "content": "Do not call any tool. Reply only OK."}]))
    assert not auto["choices"][0]["message"].get("tool_calls"), auto
    record("strict_auto_plain_text", auto)
    after = complete(base, common)
    assert after["choices"] == nullable["choices"], (after, nullable)
    record("request_state_isolation", after)
    if args.long_logprobs:
        long_request = dict(common, max_completion_tokens=512, logprobs=True, top_logprobs=20,
                            messages=[{"role": "user", "content":
                                "Write a complete Python asyncio HTTP client with retries, rate limiting, "
                                "streaming responses and a comprehensive suite of unit tests. Include all code."}])
        result = complete(base, long_request)
        choice = result["choices"][0]
        check_scores(choice["message"]["content"], choice["logprobs"]["content"], 20)
        assert len(json.dumps(result).encode()) > 512 * 1024, "Expected a large probability response"
        record("large_probability_response", result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(evidence, file, ensure_ascii=False, indent=2)
    print(f"All checks passed; evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()

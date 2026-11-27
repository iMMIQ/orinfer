"""Single-request Ascend comparison with fixed token IDs and cold prefix identity.

TTFT includes localhost HTTP and scheduling. Decode uses the first/last token
events and excludes the final usage message. Raw events make the boundary
auditable.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request
import uuid


def request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def prepare(model, output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    # Raw completion tokens isolate model execution. The deployed V4.1 chat
    # formatter is a custom tokenizer mode, not an HF Jinja chat template.
    prefix_ids = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    cases = []
    for length in (1024, 8192):
        body = (
            "这是一份软件系统运行记录。请分析其中的性能瓶颈和资源依赖。"
            "请求依次执行输入处理、矩阵计算、缓存访问和结果生成；"
            "不同任务之间共享只读权重，各自保存位置与上下文状态。"
        )
        ids = tokenizer.encode(body * length, add_special_tokens=False)
        ids = prefix_ids + ids[: length - len(prefix_ids)]
        assert len(ids) == length
        cases.append(
            {
                "name": f"p{length}",
                "tokens": ids,
                "tokens_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
        )
    output.write_text(json.dumps({"seed": 20261002, "cases": cases}, indent=2) + "\n")


def measure(base, model, case, output_tokens, destination, index):
    payload = {
        "model": model,
        "prompt": case["tokens"],
        "temperature": 0,
        "seed": 20261002,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "cache_salt": uuid.uuid4().hex,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    start = time.perf_counter()
    events, usage, times, token_ids = [], None, [], []
    with request_json(base + "/completions", payload, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                continue
            data = json.loads(raw[6:])
            now = time.perf_counter() - start
            if "error" in data:
                raise RuntimeError(data["error"])
            events.append({"seconds": now, "event": data})
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                generated = choice.get("token_ids") or []
                if generated:
                    times.extend([now] * len(generated))
                    token_ids.extend(generated)
    (destination / f"{case['name']}-r{index}-events.json").write_text(json.dumps(events))
    if (
        usage is None
        or usage["prompt_tokens"] != len(case["tokens"])
        or usage["completion_tokens"] != output_tokens
    ):
        raise ValueError(f"Unexpected usage: {usage}")
    if len(token_ids) != output_tokens:
        raise ValueError(f"Expected token IDs on every event; received {len(token_ids)}")
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cached_tokens", 0) != 0:
        raise ValueError(f"Cold-prefix request unexpectedly hit cache: {usage}")
    return {
        "name": case["name"],
        "input_tokens": len(case["tokens"]),
        "output_tokens": output_tokens,
        "ttft_s": times[0],
        "prefill_tokens_per_s_from_ttft": len(case["tokens"]) / times[0],
        "decode_s": times[-1] - times[0],
        "decode_tokens_per_s": (output_tokens - 1) / (times[-1] - times[0]),
        "e2e_s": times[-1],
        "cached_tokens": details.get("cached_tokens", 0),
        "output_token_ids": token_ids,
        "tokens_sha256": case["tokens_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ready-timeout", type=int, default=2700)
    args = parser.parse_args()
    if args.prepare:
        if args.model_dir is None:
            parser.error("--prepare requires --model-dir")
        prepare(args.model_dir, args.prepare)
        return
    if args.inputs is None or args.output is None or args.model is None:
        parser.error("measurement requires --inputs, --output and --model")
    if args.output_tokens < 2 or args.repeats < 1 or args.ready_timeout < 1:
        parser.error("at least two output tokens, one repeat and a positive timeout are required")
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.ready_timeout
    while True:
        try:
            with request_json(args.base_url + "/models", timeout=5) as response:
                json.load(response)
            break
        except (OSError, urllib.error.URLError) as error:
            if time.monotonic() >= deadline:
                raise TimeoutError("vLLM readiness timeout") from error
            time.sleep(5)
    inputs = json.loads(args.inputs.read_text())["cases"]
    results = []
    for case in inputs:
        measure(args.base_url, args.model, case, 16, args.output, "warm")
        for repeat in range(args.repeats):
            result = measure(
                args.base_url, args.model, case, args.output_tokens, args.output, repeat
            )
            results.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "output_token_ids"}), flush=True
            )
            (args.output / "requests.json").write_text(json.dumps(results, indent=2))
    summary = [
        {
            "name": case["name"],
            **{
                metric: statistics.median(r[metric] for r in results if r["name"] == case["name"])
                for metric in (
                    "ttft_s",
                    "prefill_tokens_per_s_from_ttft",
                    "decode_tokens_per_s",
                    "e2e_s",
                )
            },
        }
        for case in inputs
    ]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

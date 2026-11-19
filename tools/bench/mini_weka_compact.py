"""Replay compact synthetic Weka traces as prefix-preserving Chat requests."""

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import time

from tokenizers import Tokenizer

from orinfer_build.common.files import atomic_json, file_hash
from tools.bench.concurrency import request


# The fork starts both children. The parent continues once both finish.
DEPENDENCIES = ((), (0,), (1,), (1,), (2, 3), (4,), (5,))


def render_trace(item, tokenizer, model):
    events = [
        event["requests"][0] if event["type"] == "subagent" else event for event in item["requests"]
    ]
    parents = (None, 0, 1, 1, 1, 4, 5)

    def code_blocks(hashes):
        for block in hashes:
            yield (
                f"// Review record {block}\n"
                f"fn update_{block}(old: usize) -> usize {{\n"
                f"    let next = old + {block % 103};\n"
                f"    if next > {block % 311} {{ return next / 2; }}\n"
                "    next\n}\n"
                "// Check callers, expected output, and error cases.\n"
                "// Record the observed values, then inspect the next call site.\n"
            )

    def text(hashes, length):
        tokens = tokenizer.encode("".join(code_blocks(hashes)), add_special_tokens=False).ids
        if len(tokens) < length:
            raise ValueError("Generated compact replay text is shorter than its token budget")
        return tokenizer.decode(tokens[:length], skip_special_tokens=False)

    requests = []
    for turn, (event, parent) in enumerate(zip(events, parents, strict=True)):
        hashes = event["hash_ids"]
        if parent is None:
            system_blocks = item["system_tokens"] // item["block_size"]
            system = dict(
                role="system", content=text(hashes[:system_blocks], item["system_tokens"])
            )
            messages = [system]
            start = system_blocks
            suffix = event["in"] - item["system_tokens"]
        else:
            previous = events[parent]
            start = len(previous["hash_ids"])
            if hashes[:start] != previous["hash_ids"]:
                raise ValueError("Compact replay branch lost its declared shared context")
            messages = requests[parent]["messages"] + [
                dict(
                    role="assistant",
                    reasoning_content="I will inspect the relevant code and verify the output.",
                    content=text(previous["hash_ids"][-8:], previous["out"]),
                )
            ]
            suffix = event["in"] - previous["in"]
        messages = messages + [
            dict(
                role="user",
                content=text(hashes[start:], suffix),
            )
        ]
        requests.append(
            dict(
                messages=messages,
                model=model,
                seed=20261002,
                temperature=1.0,
                ignore_eos=True,
                max_completion_tokens=event["out"],
            )
        )
    return requests


def load_requests(manifest, tokenizer, model):
    document = json.loads(manifest.read_text())
    if document["workload"] != "orinfer-synthetic-mini-weka-compact-v1":
        raise ValueError("Expected a compact synthetic Mini-Weka manifest")
    plans = {}
    for window in document["windows"]:
        name = window["trace"]
        path = manifest.parent / "traces" / f"{name}.json"
        if name in plans or file_hash(path) != window["sha256"]:
            raise ValueError(f"Unexpected compact replay trace {name}")
        plans[name] = render_trace(json.loads(path.read_text()), tokenizer, model)
    return plans


def replay(plans, url, concurrency, request_count, wall_limit, records):
    work, completed = Queue(), Queue()
    start = time.monotonic()
    deadline = start + wall_limit
    responses = Counter()
    progressed = {}
    visits = iter(plans)
    ready = deque()
    counts = Counter()

    def worker():
        while task := work.get():
            phase, trace, turn = task
            payload = plans[trace][turn]
            if phase == "warmup":
                payload = dict(payload, max_completion_tokens=1)
            began = time.monotonic()
            try:
                remaining = max(deadline - began, 0.001)
                outcome = request(url, payload, timeout_s=remaining)
            except Exception as error:
                outcome = dict(error=str(error))
            ended = time.monotonic()
            completed.put(
                dict(
                    phase=phase,
                    trace=trace,
                    turn=turn,
                    request=payload,
                    result=outcome,
                    start_s=began - start,
                    end_s=ended - start,
                )
            )

    def add_visit(phase):
        if (trace := next(visits, None)) is not None:
            progressed[trace] = set()
            ready.append((phase, trace, 0))

    def receive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            row = completed.get(timeout=remaining)
        except Empty:
            return None
        if row["end_s"] > wall_limit:
            return None
        records.write(json.dumps(row, ensure_ascii=False) + "\n")
        records.flush()
        if error := row["result"].get("error"):
            raise RuntimeError(f"{row['trace']}/{row['turn']}: {error}")
        counts[row["phase"]] += 1
        progressed[row["trace"]].add(row["turn"])
        usage = row["result"]["usage"]
        responses[(row["phase"], "prompt")] += usage["prompt_tokens"]
        responses[(row["phase"], "generated")] += usage["completion_tokens"]
        responses[(row["phase"], "cached")] += usage.get("prompt_tokens_details", {}).get(
            "cached_tokens", 0
        )
        return row

    for _ in range(concurrency):
        Thread(target=worker, daemon=True).start()
        add_visit("warmup")
    warm_count = len(ready)
    while ready:
        work.put(ready.popleft())
    last_warmup = None
    for _ in range(warm_count):
        row = receive()
        if row is None:
            break
        last_warmup = row["end_s"]
    submitted = 0
    active = 0
    if counts["warmup"] == warm_count:
        ready.extend(("profiling", trace, 1) for trace in progressed)
        while (ready or active) and time.monotonic() < deadline:
            while ready and active < concurrency and submitted < request_count:
                work.put(ready.popleft())
                active += 1
                submitted += 1
            if active == 0 or (row := receive()) is None:
                break
            active -= 1
            trace, turn = row["trace"], row["turn"]
            done = progressed[trace]
            for future, parents in enumerate(DEPENDENCIES):
                if turn in parents and all(i in done for i in parents):
                    ready.append(("profiling", trace, future))
            if len(done) == len(DEPENDENCIES):
                add_visit("profiling")
    for _ in range(concurrency):
        work.put(None)
    return dict(
        wall_limit_s=wall_limit,
        wall_s=time.monotonic() - start,
        warmup_s=last_warmup,
        warmup_completed=counts["warmup"],
        profiling_completed=counts["profiling"],
        profiling_submitted=submitted,
        complete_trajectories=sum(len(done) == len(DEPENDENCIES) for done in progressed.values()),
        **{
            f"{phase}_{metric}_tokens": responses[(phase, metric)]
            for phase in ("warmup", "profiling")
            for metric in ("prompt", "cached", "generated")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "tokenizer", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18828/v1")
    parser.add_argument("--model", default="qwen-flash-next")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-count", type=int, default=64)
    parser.add_argument("--wall-limit", type=int, default=240)
    args = parser.parse_args()
    if min(args.concurrency, args.request_count, args.wall_limit) <= 0:
        parser.error("Concurrency, request count and wall limit must be positive")
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    plans = load_requests(args.manifest, tokenizer, args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()
    with (args.output / "requests.jsonl").open("w") as records:
        result = replay(
            plans, args.url, args.concurrency, args.request_count, args.wall_limit, records
        )
    result.update(
        input_identity=identity, concurrency=args.concurrency, request_count=args.request_count
    )
    atomic_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

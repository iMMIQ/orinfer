# Offline JSONL Batch Inference

`orinfer batch` processes local JSONL files directly through the Rust CLI without starting an HTTP server. It shares the exact same chat templates, tokenizer, sampling, continuous batching, prefix cache, and execution packages as `orinfer serve`.

---

## Quick Start

```bash
orinfer batch /path/to/model \
  --model index-translate-2b \
  --input requests.jsonl \
  --output results.jsonl \
  --max-active-requests 64 \
  --mtp-drafts 0 \
  --prefix-cache-mib 512 \
  --default-request-params '{"temperature":0,"seed":20261002,"enable_thinking":false,"max_completion_tokens":512}'
```

A ready-to-run sample input is provided at [fixtures/translation-batch.jsonl](../fixtures/translation-batch.jsonl).

---

## Scheduling & Prefix-Aware Planning

- **Throughput Policy (Default)**: `batch` defaults to `--scheduler-policy throughput` to minimize total job wall time. It admits new requests continuously as slots free up and chooses joint vs. independent prefill from measured iteration cost, while respecting memory limits, active-request caps, and request isolation. Pass `--scheduler-policy latency` to match online serving behavior.
- **Global Prefix & Length Ordering (`--batch-order auto`)**:
  - Scans text requests upfront using the real chat frontend to extract token prefixes (capped at 4,096 prefix tokens per row and 1/4 of `--preprocess-memory-mib`).
  - **Prefix cache off**: Schedules longer requests earlier to eliminate single-request tail stragglers.
  - **Prefix cache on**: Orders requests along a compressed prefix tree (heavier subtrees first) and emits checkpoint hints for shared branches while avoiding wasted snapshots on unique prompts.
  - Learns output-to-input length ratios from naturally completed rows to pack cohorts efficiently (`max_completion_tokens` remains a hard cap).
  - Pass `--batch-order input` to preserve original file order; multimodal rows stream directly through bounded preprocessing.
- **Tip for unique short translations**: Disable prefix caching (`--prefix-cache-mib 0`) and MTP (`--mtp-drafts 0`) with `--max-active-requests 64` for peak throughput.

---

## Input Format

UTF-8 JSONL, one request per line (blank lines skipped; max **32 MiB** per line). Supports both OpenAI Batch envelopes and direct Chat bodies:

```json
{"custom_id":"paragraph-1","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"user","content":"Translate to Chinese: The service is ready."}]}}
{"custom_id":"paragraph-2","messages":[{"role":"user","content":"Translate to English: 服务已就绪。"}],"max_tokens":128}
```

### Rules & Validation
- **`custom_id`**: Unique 1–256 byte string (defaults to `line-N` if omitted). Duplicate IDs fail before loading the GPU model.
- **`model`**: Defaults to `--model`; explicit values must match.
- **Chat fields**: Supports images (data URI or HTTP(S) URL), tools, JSON Schema, thinking, and sampling defaults just like [Serving Guide](serving.md). Explicit `stream: true` or `stream_options` is rejected as a row error.
- **Error isolation**: Per-line syntax or validation errors write an error record for that row and continue; job-level I/O failures abort the run.

---

## Output Format & Durability

Results are written incrementally in completion order via a bounded background writer thread (flushed and synced roughly every second and on exit; single-process file lock enforced).

**Success record**:
```json
{
  "id": "batch-paragraph-1",
  "custom_id": "paragraph-1",
  "response": {
    "status_code": 200,
    "request_id": "chatcmpl-...",
    "body": {
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "服务已就绪。"}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
    }
  },
  "error": null
}
```

**Error record**: Contains `custom_id`, `"response": null`, and `"error": {"code": "...", "message": "..."}`.

### Standard Output Summary & Exit Codes
On exit, `stdout` emits a JSON summary containing `scheduler_policy`, `batch_order`, `planned_requests`, `planning_prefix_tokens`, `planning_s`, counts for total / succeeded / truncated / failed / incomplete / skipped rows, generated tokens, peak active requests, and end-to-end `output_tps` (including scan, model load, planning, execution, and disk sync).
- **Exit 0**: All rows completed cleanly without errors, truncation, or interruption.
- **Non-zero**: Errors, truncated outputs (`finish_reason: "length"`), or signal interruption occurred.

---

## Resuming Interrupted Jobs (`--resume`)

Each job writes a sidecar metadata file `results.jsonl.meta.json` recording the input SHA256, model identity, request defaults, and MTP setting. Keep it alongside the output file.

To resume after `SIGINT`/`SIGTERM` or transient row failures, re-run the same command with `--resume`:

```bash
orinfer batch /path/to/model \
  --model index-translate-2b \
  --input requests.jsonl \
  --output results.jsonl \
  --max-active-requests 64 \
  --mtp-drafts 0 \
  --prefix-cache-mib 512 \
  --default-request-params '{"temperature":0,"seed":20261002,"enable_thinking":false,"max_completion_tokens":512}' \
  --resume
```

- **What is skipped vs. retried**: Committed normal and truncated responses are skipped; failed and unfinished rows are retried and appended (last record per `custom_id` wins). Incomplete trailing partial lines without a newline are trimmed automatically; corrupted committed lines fail fast.
- **What may change on resume**: Scheduling policy, `--batch-order`, concurrency, and memory/cache budgets can be adjusted. Changing the input file, model, request defaults, or MTP configuration requires a new output file.

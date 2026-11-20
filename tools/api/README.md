# API Validation Tools

Client-side test harnesses for verifying OpenAI Chat Completions compatibility, streaming/isolation behavior, and exact tokenizer/chat-template parity against Hugging Face Transformers.

## Live Server Checks

Run against a live `orinfer serve` instance (set `ORINFER_API_KEY` if authentication is enabled):

```bash
# Core smoke test: variable lengths, EOS, SSE/usage, seed, stop, cancellation, tools
python3 tools/api/smoke.py \
  --base-url http://127.0.0.1:8088/v1 \
  --output artifacts/api-smoke-results.json

# Deep compatibility: nullables, signed seeds, JSON Schema + thinking, strict tools, logprobs, isolation
python3 tools/api/compatibility.py \
  --model qwen-flash-next \
  --tokenizer-dir /path/to/prepared-model \
  --output artifacts/api-compatibility.json
```

Pass `--long-logprobs` to `compatibility.py` to additionally stress-test >512 KiB non-streaming responses (512 tokens × top-20 logprobs).

## Chat Template & Tokenizer Parity (Offline)

Generate reference token IDs using Hugging Face Transformers and verify byte-exact token parity in Rust:

```bash
# Text, tools, tool history, and thinking
python3 tools/api/template_reference.py \
  --tokenizer-dir /path/to/tokenizer-dir \
  --output artifacts/chat-reference.json

ORINFER_TOKENIZER_DIR=/path/to/tokenizer-dir \
ORINFER_CHAT_REFERENCE="$PWD/artifacts/chat-reference.json" \
  cargo test --offline checkpoint_matches_reference_tokens -- --ignored
```

Add `--vision-model /path/to/prepared-model --image /path/to/image.png` to include single-image, multi-image, and image-history placeholder expansion checks.

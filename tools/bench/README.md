# Benchmarking & Stress-Testing Toolkit

Tools in `tools/bench/` measure real-request HTTP/SSE throughput, startup latency, mixed prefill/decode interference, encoder/reranker speedups, and multi-turn agent replay. Always report complete-request metrics alongside exact model/package identities. See [Performance Reference](../../docs/performance.md) for current baselines.

## Tool Overview

| Tool | Target | What It Measures |
| --- | --- | --- |
| `mini_weka_synthetic.py` + `mini_weka_compact.py` | HTTP `/v1` | Reproducible 512-trajectory synthetic multi-turn agent replay (240 s Orin baseline) |
| `agentx_mini.py` | HTTP `/v1` | Fixed 16-window AgentX / Mini-Weka replay for DSv4.1 A2 & vLLM parity |
| `concurrency.py` | HTTP `/v1` | 1–128 concurrent SSE throughput, TTFT quantiles, `/health` scheduler & admission deltas |
| `mixed.py` | HTTP `/v1` | Cold long-prefill injection during active cached decode streams |
| `scheduling.py` | HTTP `/v1` | Fixed-arrival-time mixed trace (arrival error, TTFT, E2E, max chunk gap) |
| `startup.py` | CLI + HTTP | Fresh-process `serve` → `/health` ready → first non-empty SSE chunk → completion |
| `code_generation.py` | HTTP `/v1` | Natural-EOS long code/translation generation with thinking, MTP, and prefix cache |
| `output_stress.py` | HTTP `/v1` | Timed multi-phase concurrency stress (default 1/8/16/32 over 600 s) with answer & loop checks |
| `acceptance.py` | HTTP `/v1` | Heterogeneous delayed-arrival acceptance (text, single/multi-image, multi-turn) |
| `encoders.py` | HTTP `/v1` | Embeddings (`/v1/embeddings`) and reranking (`/v1/rerank`) vs. vLLM BF16 |
| `dspark_costs.py` | Log offline | Extracts A2 DSpark timing priors from `ORINFER_A2_DSPARK_POLICY_TRACE=1` logs |
| `ascend_single.py` / `ascend_regression.py` | Ascend CLI | Single-stream A2 inputs & 5% regression gate |
| `sample_machine.py` / `sample_continuous.py` | Host | Read-only temperature, clock, memory, and power telemetry |
| `validate_plan.py` | CPU CI | Validates benchmark & eval configuration schemas |

## Common Usage Examples

### Synthetic Mini-Weka Replay (Orin)

```bash
uv run --locked python -m tools.bench.mini_weka_synthetic \
  --profile compact --output artifacts/bench/mini-weka-512

uv run --locked python -m tools.bench.mini_weka_compact \
  --manifest artifacts/bench/mini-weka-512/manifest.json \
  --tokenizer /path/to/model --url http://127.0.0.1:18828/v1 \
  --model qwen-flash-next --concurrency 4 --wall-limit 240 \
  --output artifacts/bench/out
```

### HTTP Concurrency, Startup & Mixed-Traffic Benchmarks

```bash
# Concurrency sweep (1..128)
python3 tools/bench/concurrency.py --requests requests.json \
  --concurrency 1 8 32 128 --repetitions 3 --output artifacts/concurrency.json

# Fresh-process startup to first SSE output
python3 -m tools.bench.startup --model /path/to/model --request req.json \
  --output artifacts/startup --server-args --mtp-drafts 0 --cuda-graph decode_only

# Embeddings / reranking vs. vLLM
python3 -m tools.bench.encoders --task rerank --backend orinfer \
  --url http://127.0.0.1:8089 --model bench --requests inputs.json \
  --identity native-identity.json --output native-results.json
python3 -m tools.bench.encoders --compare vllm-results.json native-results.json \
  --output comparison.json
```

For vLLM reranker baselines, launch vLLM with `--chat-template tools/bench/templates/qwen3_reranker.jinja` to match Qwen's official prompt tokens.

## Rust GPU Engine Benchmarks (`--ignored`)

Run under the shared GPU lock with `ORINFER_BATCH_FIXTURE=/abs/path/to/fixture.json`:

```bash
ORINFER_BATCH_FIXTURE=/abs/path/to/fixed-requests.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --locked --offline -p orinfer-engine \
    runtime::requests::tests::benchmark::benchmark_fixed_requests \
    -- --ignored --exact --nocapture --test-threads=1
```

Other engine GPU validation/profiling tests in `orinfer-engine` include `benchmark_prefill_scheduling`, `validate_flash_decode_batches`, `validate_flash_heterogeneous_scheduling`, `validate_flash_long_generation_reuse`, `validate_batched_gpu_sampling_requests`, `validate_lazy_workspace_requests`, and `profile_vision_stages`.

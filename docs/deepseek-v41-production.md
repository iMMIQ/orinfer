# DSv4.1 Ascend 910B Production Deployment

This guide covers deploying **DeepSeek-V4.1 Flash-1017 W4A8** on **8× Ascend 910B (TP8)** with Orinfer **v0.4.4**.

---

## 1. Requirements & Capabilities

- **OS & Runtime**: Linux `aarch64`, `glibc >= 2.34`, **CANN 9.2**, 8 dedicated Ascend 910B NPUs (TP8).
- **Recommended Host**: ≥96 CPU cores and ≥1 TB host RAM when keeping Engram tables resident (~207 GiB) alongside 256 GiB CPU prefix cache offload.
- **Supported Features**: OpenAI HTTP/SSE, continuous batching, DSpark speculative decoding (auto up to 5 drafts by default), device + CPU tiered prefix cache, DSML tool calling, thinking history, and single/multi-image vision (when paired with `vision.safetensors` and a vision-enabled execution package).
- **Package Digest (A2 W4A8)**: `3b3961a745cbabff0523fafe1224571b3582a534d744e1fecb3b9b6371d50ca7` (compact `shared-json-v1` + `rope_f32_v1` A2 package shipped with v0.4.4).

---

## 2. Installation & Upgrade

### Install Engine & Execution Package

```bash
tar -xf orinfer-v0.4.4-linux-aarch64.tar.xz
bin/orinfer --version

bin/orinfer package install \
  orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.4.4.tar.xz \
  --cache /srv/orinfer/execution-cache
```

### Update Model Frontend Metadata (Existing Deployments)

Weights from v0.3.1+ remain unchanged. Create a new model directory (using hardlinks on the same filesystem so the previous directory stays intact for rollback) and pull the matching v0.4.4 metadata:

```bash
uv run --locked python -m orinfer_build.models.deepseek_v41.service --frontend-only \
  --model-dir /data/model/current --output /data/model/dsv41-v0.4.4

hf download immiq/DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 \
  cache/model.json README.md publication.json SHA256SUMS \
  --revision v0.4.4 --local-dir /data/model/dsv41-v0.4.4
```

Run `bin/orinfer validate-model /data/model/dsv41-v0.4.4` to verify frontend, execution package, and weight integrity.

---

## 3. Recommended Service Configuration

Bind `orinfer` to localhost behind your internal API gateway (which handles TLS, user quotas, and SSE pass-through with response buffering disabled). Store `ORINFER_API_KEY` in a protected env file outside the repository.

```bash
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_COREDUMP_SIGNAL=none
export ORINFER_EXECUTION_CACHE=/srv/orinfer/execution-cache
export ORINFER_A2_HOST_TABLE_MIB=212000
export ORINFER_VISION_CACHE_MIB=1024
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_EXEC_TIMEOUT=120

bin/orinfer serve /data/model/dsv41-v0.4.4 \
  --model deepseek-v4.1 \
  --listen 127.0.0.1:8088 \
  --mtp-drafts auto \
  --max-speculative-tokens 0 \
  --graph full \
  --prefix-cache-mib 8192 \
  --kv-offload-mib 262144 \
  --request-body-mib 64 \
  --queue-timeout-ms 60000 \
  --output-timeout-ms 180000 \
  --drain-timeout-ms 120000
```

### Key Sizing Notes
- **Memory**: `--prefix-cache-mib 8192` is **per NPU**; `--kv-offload-mib 262144` is **total across TP ranks** (allocated on demand, does not expand active NPU KV capacity); `ORINFER_A2_HOST_TABLE_MIB=212000` pins ~207 GiB of Engram tables in RAM.
- **CPU / NUMA Placement**: When setting `ORINFER_A2_WORKER_CPUS` to bind the 8 `orinfer-a2-{0..7}` rank threads to NPU-local NUMA cores, leave dedicated unpinned host cores outside that set for `orinfer-gpu`, Tokio HTTP workers, preprocessing, Rayon Engram gathers, and host KV offload so host work does not preempt TP8 rank threads.
- **Startup & Shutdown Probes**: Full host-table loading can take several minutes—configure startup probes to `/health` (HTTP 200) with a ≥600 s grace window. Set your process manager's `SIGKILL` timeout > `--drain-timeout-ms` (120 s) so NPU and host-table teardown completes cleanly.
- **Sampling Defaults**: Unspecified requests default to `temperature=1`, `top_p=1`, `top_k=0` (set `temperature=0` explicitly for greedy output). Output budgets default to 8,192 (non-thinking), 65,536 (thinking), and 131,072 (thinking + `reasoning_effort=max`), clamped to remaining context.

### Optional Offline DSpark Cost Profile

To supply measured timing priors for the current execution package, run representative short traffic with `ORINFER_A2_DSPARK_POLICY_TRACE=1`, extract the profile, and place it in `$ORINFER_EXECUTION_CACHE/profiles/<execution_package>.json` (or set `ORINFER_A2_DSPARK_COST_MODEL`):

```bash
uv run --locked python -m tools.bench.dspark_costs \
  --log service.log \
  --output /srv/orinfer/execution-cache/profiles/<execution_package>.json
```

---

## 4. Concurrency, Timeouts & Client Retry Policy

- **Queue & Concurrency**: Default active cap is 32 requests (start interactive gateways around 16 concurrency); fixed queue depth is 128. Route long-context traffic through a separate gateway lane with higher client timeouts (e.g., 300 s interactive, 600 s+ long-context; 262K is the hard context cap, not a latency SLA).
- **Error Semantics**:
  - **Queue full**: HTTP `429` with `Retry-After: 1`.
  - **Queue deadline (`--queue-timeout-ms`)**: HTTP `408` (non-streaming) or SSE terminal error `request_queue_timeout` (`retryable=true`, `retry_after_ms=1000`). Use exponential backoff with bounded retries.
  - **Output stall (`--output-timeout-ms`)**: `client_output_timeout` (`retryable=false`).
  - **Idempotency**: Stop automatic retries once any text, thinking, or tool-call delta has been received.
- **Device Faults**: Any NPU driver error, rank timeout (`ORINFER_A2_COMMAND_TIMEOUT_MS` default 30 s for fast commands; `ORINFER_A2_HEAVY_COMMAND_TIMEOUT_MS` default 120 s for capture/offload/restore/vision), or HCCL failure (`HCCL_EXEC_TIMEOUT=120`) retires the TP team and exits `orinfer serve` with a non-zero exit code; configure systemd (`Restart=on-failure`) or Kubernetes to restart the service cleanly. Per-request constrained decoding dead-ends fail only that single request without retiring the TP team.

---

## 5. Tools, Vision & Rollback

- **Structured & Tool Output**: Use `strict: true` tools or `response_format: {type: "json_schema", ...}` whenever machine-parseable output is required; always inspect `finish_reason` before executing tools.
- **Vision**: Up to 1,024 visual tokens per image; control total history budgets via `image_processing` (`max_images`, `max_pixels`, `max_tokens_per_image`, `max_tokens`) and `--request-body-mib`.
- **Client Integration**: See [OpenCode](../examples/opencode.json) (`orinfer/deepseek-v4.1`) and [DeepSeek Harness](deepseek-harness.md).
- **Rollback**: Execution packages coexist by digest in the same cache, and model frontends live in versioned directories. Always record `{engine version, model dir, execution package digest, cost profile, CANN env}` before upgrading. To roll back, stop the new service, confirm NPU memory release, start the previous deployment unit, and verify a real request + prefix restore before switching gateway traffic back.

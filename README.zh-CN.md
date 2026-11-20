# Orinfer

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

基于 **Rust** 与 **TileLang** 构建的高性能文本、视觉语言、语音、嵌入与重排推理引擎。

主要面向 **NVIDIA Jetson AGX Orin 64GB（CUDA SM87）**；同时通过后端无关的共享运行时，实验性支持 **华为 Ascend 910B / A2（CANN 9.2）** 上的 DeepSeek-V4.1 Flash-1017 W4A8 TP8 原生服务。

---

## 核心特性

- **OpenAI 与 Ollama API**：支持流式 Chat Completions、Ollama `/api/chat` 与 `/api/generate`、thinking 与 `thinking_token_budget`、函数调用、严格 JSON Schema 约束（`llguidance`）及 `logprobs`。
- **多模态与多任务**：
  - **文本与视觉语言**：支持 Qwen3.8-27B、Qwen3.8-Flash-Next、Qwen3.5-4B、Index-Translate-2B/9B/35B-A3B-preview，以及 Ascend 910B TP8 上的 DSv4.1 Flash-1017。
  - **语音识别**：Qwen3-ASR-1.7B，内置 Rust 音频解码与重采样，支持 `/v1/audio/transcriptions` 与 Chat `input_audio`（见[语音识别指南](docs/audio.md)）。
  - **文档嵌入**：Nemotron-3.5-Embed-1B，支持文本、单图与多图检索，提供 `/v1/embeddings` 与离线 CLI（见[嵌入指南](docs/embeddings.md)）。
  - **文档重排**：Qwen3-Reranker-0.6B，提供 `/v1/rerank` 与离线 CLI（见[重排指南](docs/reranking.md)）。
  - **JSONL 离线批处理**：支持连续批处理、前缀感知规划、逐条落盘与断点续跑（见[离线批处理指南](docs/batch.md)）。
- **量化与显存优化**：Qwen3.5/ASR/Embed/Reranker 采用两级 E8P Q4A8/W8 混合精度；Flash Next 采用 E8P Q2A8 专家 + Q4/W8 投影；支持 INT8 KV 缓存与最高 256K 上下文（见[量化技术说明](docs/quantization.md)）。
- **生产级运行时**：分块 prefill、请求级状态隔离、Radix Prefix Cache 与可选 CPU KV 卸载、CUDA/ACL Graph（`decode_only`、`full`、`off`），以及自适应原生 MTP / DSpark 投机解码（`--mtp-drafts 0..31` 或 `auto`）。

---

## 支持模型一览

| 模型 | 模态 / 任务 | 后端 | 权重精度 | 最大上下文 |
| --- | --- | --- | --- | ---: |
| **Index-Translate-2B / 9B / 35B-A3B** | 文本、单图、多图 | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.5-4B** | 文本、单图、多图 | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.8-27B** | 文本、单图、多图 | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.8-Flash-Next** | 文本、单图、多图 | CUDA SM87 | E8P Q2A8 专家 + Q4/W8 | 256K |
| **Qwen3-ASR-1.7B** | 语音转写 | CUDA SM87 | E8P Q4A8 / W8 + FP16 音频 | 64K |
| **Nemotron-3.5-Embed-1B** | 图文文档嵌入 | CUDA SM87 | E8P Q4A8 / W8 | 32K |
| **Qwen3-Reranker-0.6B** | 查询-文档重排 | CUDA SM87 | E8P Q4A8 / W8 | 32K |
| **DSv4.1 Flash-1017** *(实验性)* | 文本、图片、工具、DSpark | Ascend 910B TP8 | 原生 W4A8 + INT8 KV | 256K |

---

## 快速开始（Jetson AGX Orin）

### 环境要求
- `aarch64` Linux 与 NVIDIA CUDA Driver
- 从 [Releases](https://github.com/iMMIQ/orinfer/releases) 下载预编译 `orinfer` 引擎与匹配的执行包
- 准备好的模型目录（包含配置、tokenizer、chat template 与量化 safetensors，其 `execution_package` digest 须与已安装执行包一致）

### 安装与启动

```bash
# 1. 校验发行包并将 bin/orinfer 加入 PATH
sha256sum --ignore-missing -c SHA256SUMS

# 2. 将匹配的执行包安装到本地缓存
orinfer package install /path/to/matching-execution.tar.xz

# 3. 完整校验模型文件、执行包与权重哈希
orinfer validate-model /path/to/prepared-model

# 4. 启动 OpenAI / Ollama 兼容 HTTP 服务
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

服务默认监听 `[::]:8088`。设置环境变量 `ORINFER_API_KEY` 可为 `/v1` 与 `/api` 启用 Bearer 鉴权。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "用一句话介绍一下你自己。"}],
    "max_tokens": 512,
    "stream": true
  }'
```

多模型管理模式（无路径运行 `orinfer serve`）、Ollama 接口、图片输入，以及 [OpenCode](examples/opencode.json) / [Pi](examples/pi-models.json) / [DeepSeek Harness](docs/deepseek-harness.md) 接入方式，请见[服务指南](docs/serving.md)。

Ascend 910B / A2 部署可运行 `orinfer devices --backend ascend` 与 `orinfer validate-ascend --device 0`，完整说明见 [DSv4.1 A2 生产部署指南](docs/deepseek-v41-production.md)。

---

## 性能速览（Jetson AGX Orin 64GB）

固定英译中翻译基准（2026-10-21，`decode_only`、greedy、seed `20261002`，关闭 thinking 与 prefix cache；单流启用自适应 MTP 最多 3 草稿，多流关闭 MTP）。吞吐单位为 **tok/s**，启动时延为三次全新进程中位数（秒）。

| 模型 | Prefill 512 | Prefill 8K | 单流 Decode (MTP) | 8 并发 Decode | 最高并发 / Decode | 启动→API（秒） | 启动→首输出（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Index-Translate-2B | 8,121.1 | 7,507.7 | 171.5 | 757.4 | 128 / 2,361.1 | 1.12 | 1.26 |
| Qwen3.5-4B | 3,592.6 | 3,417.0 | 90.7 | 360.5 | 128 / 1,071.3 | 1.68 | 1.83 |
| Index-Translate-9B | 2,213.5 | 2,223.3 | 56.5 | 233.9 | 128 / 890.7 | 2.92 | 3.10 |
| Qwen3.8-27B | 658.9 | 598.5 | 33.6 | 67.6 | 64 / 135.9 | 6.68 | 7.05 |
| Index-Translate-35B-A3B | 2,233.2 | 3,222.2 | 83.0 | 338.7 | 128 / 825.0 | 8.01 | 8.18 |
| Qwen3.8-Flash-Next | 925.6 | 858.1 | 56.4 | 157.3 | 32 / 282.9 | 15.79 | 15.96 |

最新的 **Schema 2 + P1–P6 Mini-Weka 端到端基线（2026-11-20）**、vLLM 对照、JSONL 离线吞吐与 Ascend 910B3 TP8 数据，见[性能参考](docs/performance.md)。

---

## 常用启动参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--listen` | `[::]:8088` | HTTP 监听地址 |
| `--default-request-params` | `{}` | JSON 格式的默认 Chat 请求参数 |
| `--cuda-graph` / `--graph` | `decode_only` | Graph 模式（`decode_only`、`full`、`off`） |
| `--mtp-drafts` | `auto` | 自适应 MTP 上限（`auto`、`0` 关闭或 `1..31`） |
| `--resident-experts` | `full` | 专家全量常驻，或设置每层等效 SLRU 容量 `N`（Flash Next） |
| `--prefix-cache-mib` | `4096` | 单卡设备前缀缓存预算，单位 MiB（`0` 关闭） |
| `--kv-offload-mib` | `0` | 全局 CPU 前缀状态缓存预算，单位 MiB（`0` 关闭） |
| `--max-active-requests` | `32` | 活跃请求上限（`1..128`） |
| `--max-batch-tokens` | `128` | 普通 decode 行预算（`1..128`） |
| `--max-speculative-tokens` | `0` | MTP 验证行预算（`0` 使用执行包容量） |
| `--prefill-budget-ms` | `200` | `latency` 策略下的混合 prefill 预测耗时预算 |

执行包默认安装在 `~/.cache/orinfer/packages`（可通过 `ORINFER_EXECUTION_CACHE` 覆盖）。完整参数见 `orinfer serve --help`。

---

## 从源码构建

Rust 负责在线引擎与 CLI；Python 3.12+（`uv`）负责离线量化、TileLang 编译与验证。

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build
make python-env
make check
```

---

## 文档导航

完整索引请访问[文档中心](docs/README.md)（所有详细技术文档均使用英文编写）：

- **服务与客户端**：[Serving Guide](docs/serving.md) · [Offline JSONL Batch](docs/batch.md) · [DeepSeek Harness](docs/deepseek-harness.md) · [DSv4.1 A2 Production](docs/deepseek-v41-production.md)
- **模型与模态**：[Speech Recognition](docs/audio.md) · [Embeddings](docs/embeddings.md) · [Reranking](docs/reranking.md)
- **架构与执行包**：[Model Execution Packages](docs/model-packages.md) · [Ascend A2 Build Pipeline](docs/ascend-a2-build.md) · [E8P Quantization Note](docs/quantization.md)
- **性能与开发**：[Performance Reference](docs/performance.md) · [Development Guide](docs/development.md) · [Build & Release](tools/build/README.md) · [GPU/NPU Kernels](kernels/README.md)

---

## 许可证

采用 **LGPL-3.0-or-later** 许可，详见 [LICENSE](LICENSE)、[COPYING](COPYING) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型权重与第三方组件遵循各自原始许可证。

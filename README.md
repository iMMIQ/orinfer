# Orinfer

LLM and vision-language inference on NVIDIA Jetson, powered by Rust and TileLang.

面向 **Jetson AGX Orin 64GB / CUDA SM87** 的推理引擎：Rust 负责在线运行和服务，TileLang 算子离线编译为独立 GPU 包。支持 Qwen3.8-27B（`Qwen3_5`）的文本、图片和多图，以及 Qwen3.8-Flash-Next（`qwen4_exp`）的文本、图片和多图推理。

## 功能

- OpenAI Chat Completions API，支持流式输出、thinking、采样、严格函数调用、JSON Schema 约束和 token 概率。
- 27B 使用单份 W4；Flash Next 使用自有 E8P Q2 专家和 W8 投影。以 INT8 混合精度计算，关键路径保留 BF16/FP16/FP32。
- 分块 prefill、独立请求状态及 prefix cache，两种模型都支持原生 MTP。两种模型均可通过对应算子包启用连续批处理，内存不足时排队。Flash Next 的 MTP 草稿上限可用 `--mtp-drafts 1..7` 调整，`0` 关闭，默认 `auto` 使用包设置（3 个草稿）。
- 混合架构 prefix cache、INT8 KV 和按需映射内存；模型包可配置最长 256K 上下文。
- CUDA Graph 提供 `decode_only`、`full`、`off` 三种模式，默认 `decode_only`。

## 快速开始

需要 aarch64 Linux、Rust 工具链和 NVIDIA CUDA Driver。开发工具链固定在 `rust-toolchain.toml`。

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build
```

先按[模型构建说明](tools/model/README.md)准备引擎原生模型目录及匹配的模型执行包。目录保留 checkpoint 的配置、tokenizer 和 chat template，权重采用带专用物理布局的分片 safetensors。**当前不能直接加载任意 HF、AWQ 或 GGUF 目录**；仓库及发行包不附带模型权重。Python、PyTorch 和 TileLang 仅用于离线准备，在线服务不需要它们。模型与 kernel 在同一工作区维护，执行逻辑通过 `.so + cubin` 部署包加载，详见[执行包接口与构建](docs/model-packages.md)。

```bash
./target/release/orinfer validate-model /path/to/prepared-model
./target/release/orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

默认监听 `0.0.0.0:8088`。设置 `ORINFER_API_KEY` 可启用 `/v1` 的 Bearer 鉴权。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"max_tokens":512,"stream":true}'
```

提供 `/health`、`/v1/models` 和 `/v1/chat/completions`，支持流式文本、工具、thinking、JSON 约束与图片/多图。[服务使用指南](docs/serving.md)包含参数说明，以及 [OpenCode](examples/opencode.json) 和 [Pi](examples/pi-models.json) 接入配置。

## 常用配置

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `--listen` | `0.0.0.0:8088` | 监听地址 |
| `--default-request-params` | `{}` | JSON 请求默认值，显式请求优先 |
| `--cuda-graph` | `decode_only` | Graph 模式 |
| `--prefix-cache-mib` | `12288` | GPU 前缀缓存预算；0 关闭 |
| `--max-active-requests` | `32` | 活跃请求上限，支持 1–128 |
| `--max-batch-tokens` | `128` | 每轮计算 token 预算 |
| `--prefill-budget-ms` | `200` | 混合 prefill 的预测耗时目标 |

实际并发和上下文容量受执行包及可用内存限制。环境变量统一使用 `ORINFER_*`；共享模型包默认安装到 `~/.cache/orinfer/packages`，可用 `ORINFER_EXECUTION_CACHE` 指定位置。完整配置见[服务使用指南](docs/serving.md)和 `orinfer serve --help`。

## 性能与范围

Jetson AGX Orin 64GB、27B 参考配置开启 MTP 后，Python 归并排序、Rust LRU 缓存和 TypeScript 异步 map 三个代码生成场景的单流 decode 中位数分别为 **26.17、25.23、25.95 tokens/s**。每个场景测量三次，每次生成 128 tokens；decode 按实际输出 tokens 计时，排除首 token。

目前只支持上述硬件和架构，不支持视频、音频、`n>1` 或云端存储接口。JSON Schema 支持范围见[服务使用指南](docs/serving.md)。量化质量使用同源 BF16/FP8 的固定场景和同历史概率对照；完整 LLM benchmark 和独立 BF16 256K 质量评测尚未覆盖，见[质量评测](tools/eval/README.md)。

## 开发与文档

```bash
make python-env  # uv 创建并同步 CPU .venv
make check       # 格式、编译、clippy、Rust/CPU 测试
```

- [构建与发布](tools/build/README.md)：编译镜像、CPU/GPU 检查与发行包。
- [模型与执行包](tools/model/README.md)：离线准备、布局、MTP 和上下文扩展。
- [服务使用](docs/serving.md)：API、图片、缓存、并发和资源配置。
- [图文构建](tools/vision/README.md) · [API 验证](tools/api/README.md) · [TileLang kernels](kernels/README.md)。

在线代码位于 `crates/orinfer-{engine,api,cli}`，模型适配器及计算策略位于 `crates/orinfer-models`，公共 ABI 位于 `crates/orinfer-model-sdk`；GPU 算子与离线工具分别位于 `kernels/`、`tools/`。

## 许可证

**LGPL-3.0-or-later**，见 [LICENSE](LICENSE) 和 [COPYING](COPYING)。第三方依赖及模型权重遵循各自许可证。

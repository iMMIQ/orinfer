# Orinfer

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

以 Rust 和 TileLang 实现的 NVIDIA Jetson 文本、视觉语言与语音推理引擎。

面向 **Jetson AGX Orin 64GB / CUDA SM87**。Rust 负责引擎与 API，TileLang 在部署前将 GPU 算子编译为执行包。支持 Qwen3.8-27B、Qwen3.8-Flash-Next、Qwen3.5-4B，以及 Index-Translate-2B/9B/35B-A3B-preview，以及语音转写模型 Qwen3-ASR-1.7B；各模型包声明输入能力。

共享运行时已将设备执行与调度、模型语义、prefix 状态分离。Ascend 910B / A2 后端提供原生 ACL 资源、异步传输、W4A8 grouped matmul、TileLang AOT 算子和 ACL Graph 验收，实验性 `bench-ascend` 提供 DSv4.1 Flash-1017 W4A8 完整 40 层的单流测量，Ascend 服务执行器仍在接入。在已配置 CANN 环境的主机上运行 `orinfer devices --backend ascend` 可查看设备，`orinfer validate-ascend --device 0` 执行有界原生验收；现有 CUDA 执行包和模型 C ABI 保持兼容。详见[设备后端边界](docs/model-packages.md#设备后端边界)。

## 功能

- OpenAI Chat Completions 和 Ollama 聊天/生成 API：流式输出、thinking、采样、函数调用、JSON Schema 约束及 token 概率。
- [语音识别](docs/audio.md)：Qwen3-ASR-1.7B，Rust 原生文件转写和 Chat 音频输入。
- [JSONL 离线推理](docs/batch.md)：通过 Rust CLI 连续批处理、逐条写入结果和断点续跑。
- Qwen3_5 系列使用 E8P Q4A8/W8 混合权重；Flash Next 使用 E8P Q2 专家和 Q4/W8 投影。敏感路径保留 BF16/FP16/FP32。
- 分块 prefill、独立请求状态、prefix cache、INT8 KV，以及由模型配置的最长 256K 上下文。
- 原生 MTP 根据实测成本和接受率动态选择草稿深度。`--mtp-drafts 1..7` 设置上限，`0` 关闭，`auto` 使用执行包上限。
- CUDA Graph 提供 `decode_only`（默认）、`full` 和 `off` 三种模式。

## 快速开始

部署环境为 **aarch64 Linux、NVIDIA CUDA Driver、已编译的引擎及匹配执行包**。

从 [Releases](https://github.com/iMMIQ/orinfer/releases) 下载引擎归档、匹配执行包和 `SHA256SUMS`。校验文件后解压引擎，将 `bin/orinfer` 加入 `PATH`。准备好的模型单独下载，目录包含配置、tokenizer、chat template 和量化 safetensors；其中的 `execution_package` digest 须与已安装执行包一致。

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/matching-execution.tar.xz
orinfer validate-model /path/to/prepared-model
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

服务默认监听 `[::]:8088`。设置 `ORINFER_API_KEY` 可为 `/v1` 和 `/api` 启用 Bearer 鉴权。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":512,"stream":true}'
```

接口包括 `/health`、`/v1/models` 和 `/v1/chat/completions`。[服务指南](docs/serving.md)提供 Ollama 部署、图片、请求默认值，以及 [OpenCode](examples/opencode.json) / [Pi](examples/pi-models.json) 接入说明。

## 性能

六个文本与视觉语言模型统一在 Jetson AGX Orin 64GB 上测量，使用开发版执行包，日期 **2026-10-21**。输入为固定英译中翻译请求。基准按并发数切换：单流启用自适应 MTP（最多 3 个草稿），多流自动关闭 MTP。

| 模型 | Prefill 512 tokens | Prefill 8K tokens | 单流 decode，开 MTP | 8 并发 decode | 最高并发 / decode | 启动→API 就绪（秒） | 启动→首输出（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Index-Translate-2B | 8121.13 | 7507.67 | 171.47 | 757.39 | 128 / 2361.11 | 1.12 | 1.26 |
| Qwen3.5-4B | 3592.58 | 3416.98 | 90.71 | 360.49 | 128 / 1071.29 | 1.68 | 1.83 |
| Index-Translate-9B | 2213.52 | 2223.34 | 56.49 | 233.87 | 128 / 890.74 | 2.92 | 3.10 |
| Qwen3.8-27B | 658.87 | 598.46 | 33.64 | 67.56 | 64 / 135.89 | 6.68 | 7.05 |
| Index-Translate-35B-A3B-preview | 2233.16 | 3222.22 | 82.98 | 338.65 | 128 / 824.98 | 8.01 | 8.18 |
| Qwen3.8-Flash-Next | 925.57 | 858.11 | 56.35 | 157.25 | 32 / 282.87 | 15.79 | 15.96 |

吞吐单位为 **tok/s**，启动时延单位为秒；多流列为总吞吐。单流输入为 512 或 8192 tokens，输出固定 257 tokens；表中单流 decode 使用 512-token 输入。Prefill 包含请求初始化、输出头和首输出选择，decode 排除首个输出步骤。最高并发列列出 8/16/32/64/128 档中完成测量的最高并发数及对应总 TPS。多流每请求输入 256 tokens、输出 65 tokens，全部首输出就绪后计量后续 64 tokens。

每项预热一次，再测三条完整请求或三组固定并发，表格取中位数。统一使用 `decode_only`、greedy、seed `20261002`，关闭 thinking 和 prefix cache；固定长度诊断忽略 EOS。

启动列取三次全新 Rust `serve` 进程的中位数，分别计量进程启动→`/health` 就绪、进程启动→短英译中首请求的首个非空 SSE 输出（输出预算 32 tokens）。保留系统文件缓存，计时包含模型加载、GPU 初始化及首请求工作；一个 SSE 输出可能包含多个 token。

[性能参考](docs/performance.md)记录执行包身份、8K decode、MTP 轮次、vLLM 对比、启动、离线吞吐及质量范围。复杂图片翻译持续改进中。

## 常用配置

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--listen` | `[::]:8088` | HTTP 监听地址 |
| `--default-request-params` | `{}` | JSON 请求默认值，显式请求字段优先 |
| `--cuda-graph` | `decode_only` | CUDA Graph 模式 |
| `--resident-experts` | `full` | 专家全量常驻；Flash Next 也支持 SSD 按需加载容量 |
| `--prefix-cache-mib` | `4096` | GPU 前缀缓存预算，单位 MiB；0 关闭 |
| `--max-active-requests` | `32` | 活跃请求上限，1–128 |
| `--max-batch-tokens` | `128` | Decode 行预算；混合 prefill 使用执行包容量 |
| `--max-speculative-tokens` | `0` | MTP 验证行预算；0 使用执行包容量 |
| `--prefill-budget-ms` | `200` | 混合 prefill 的预测耗时目标 |

实际并发和上下文容量由执行包及可用内存决定。执行包默认安装到 `~/.cache/orinfer/packages`，可用 `ORINFER_EXECUTION_CACHE` 指定位置。完整参数见 `orinfer serve --help`。

## 开发与文档

Rust 完成在线加载、调度、模型执行与服务；Python 用于离线量化、编译和验证。

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build
make python-env
make check
```

- [服务指南](docs/serving.md) · [离线 JSONL](docs/batch.md) · [性能参考](docs/performance.md)。
- [构建与发布](tools/build/README.md) · [模型执行包](docs/model-packages.md)。
- [模型构建](tools/model/README.md) · [GPU 算子](kernels/README.md) · [质量评测](tools/eval/README.md)。
- [量化技术 Wiki](docs/quantization.md)：E8P Q2 码本、编解码与 Tensor Core 算子，Q2_K 对比及两级 Q4 扩展。

详细指南目前以中文提供。

## 许可证

**LGPL-3.0-or-later**，见 [LICENSE](LICENSE) 和 [COPYING](COPYING)。第三方组件及模型权重遵循各自许可证。

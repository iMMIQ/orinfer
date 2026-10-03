# Orin LLM

用于 Jetson AGX Orin 64GB 的文本推理引擎。在线运行时用 Rust，GPU kernel 用 TileLang，目标固定为 CUDA SM87。

当前支持 Qwen3.8-27B 的文本主干，checkpoint架构为 `Qwen3_5ForConditionalGeneration`：48层 Gated DeltaNet和16层 full attention。支持常驻模型、分块prefill、连续decode、请求状态重置，以及OpenAI Chat Completions API、流式输出和函数工具调用。

## 构建

主机需要aarch64 Linux、Rust、CUDA Driver和本机AOT模型产物。开发环境使用Rust1.98.1；workspace声明的最低版本为1.85。离线kernel编译需要TileLang0.1.13、PyTorch2.9.1、CUDA12.6及NVIDIA Docker。

```bash
cargo fetch --locked  # 首次安装Rust依赖
make check
make build
```

模型权重、cubin和编译缓存不包含在源码库中。[离线构建说明](tools/model/README.md)介绍checkpoint转换、kernel导出和模型组装。

## 运行

```bash
./target/release/orin-llm run-model /path/to/model.json examples/requests.json
```

CLI接收token-ID请求，输出包含生成token、加载时间和请求时延的JSON。`examples/requests.json`提供一个512-token文本请求。需要自行使用模型tokenizer准备其他输入。

```json
{
  "requests": [
    {"id": "example", "input_tokens": [151644, 8948], "max_new_tokens": 32}
  ]
}
```

上面只展示字段格式。当前AOT计划要求实际输入长度为512的整数倍，优先选择512/2048/8192中能整除请求长度的最大计划。具体上下文容量由manifest声明；本机模型为8704 tokens。请求状态不复用。

## Chat API

```bash
./target/release/orin-llm serve /path/to/model.json /path/to/tokenizer-dir \
  --listen 127.0.0.1:8088 --model qwen3.8-27b
```

Tokenizer目录需要与权重匹配的`tokenizer.json`、`chat_template.jinja`和`generation_config.json`。Rust直接渲染checkpoint模板并分词。服务只加载一次模型，通过一个GPU worker执行请求；最多128个等待请求，队列满返回429。GPU worker持有`artifacts/gpu-experiment.lock`；可用`--gpu-lock`指定共享锁路径。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"2+3等于几？"}],"temperature":0,"max_tokens":64,"stream":true}'
```

提供`GET /health`、`GET /v1/models`和`POST /v1/chat/completions`。设置`ORIN_API_KEY`后，`/v1`请求需要对应的Bearer token。支持文本messages、`tools`、`tool_choice`、`parallel_tool_calls`、SSE deltas/`[DONE]`、usage、EOS、stop、temperature、top_p、top_k、presence/frequency penalties和seed。默认关闭thinking；`enable_thinking=true`启用`reasoning_content`输出。默认输出上限512 tokens，可通过`max_tokens`或`max_completion_tokens`调整。

模型的XML工具调用会转换成标准`tool_calls`，arguments为JSON字符串。客户端执行工具，并将带`tool_call_id`的`role=tool`消息连同历史再次发送。函数调用完成后才发送该调用的流式delta；工具参数支持结构校验，暂不提供完整JSON Schema约束解码。`tool_choice=required`或指定函数会加入模板指令并校验结果，模型未遵守时返回生成错误。

API支持任意提示长度：完整块走prefill图，尾部真实tokens走M=1图。尾部最多511 tokens，当前可能显著增加首token等待时间；不会用额外token填充提示。上下文和输出预算超过manifest容量时返回400。`run-model`仍保留固定块性能测试行为。暂不支持多模态、n>1、logprobs、JSON约束输出、prefix cache、MTP或同时驻留多个请求；等待请求串行执行，客户端断开后停止生成。

[OpenCode配置示例](examples/opencode.json)使用`@ai-sdk/openai-compatible`接入`http://127.0.0.1:8088/v1`。复制到独立测试目录后运行：

```bash
opencode run --pure --agent orin --model orin/qwen3.8-27b '读取input.txt并把内容写入output.txt'
python3 tools/api/smoke.py --output artifacts/api-smoke-results.json
```

Smoke工具验证真实模型的文本/SSE、采样、停止词、状态隔离、错误格式和工具结果回传；结果文件不进入源码库。完整OpenCode默认工具集和长系统提示可能超出当前8704-token容量，示例仅开放read/write。

## 实现

- `crates/orin-engine/`：CUDA Driver封装、manifest校验、权重加载、graph执行、KV/GDN/卷积状态和采样。
- `crates/orin-cli/`：命令行、Rust tokenizer/chat template、HTTP/SSE和工具协议。
- `kernels/operators/`：基础TileLang算子。
- `kernels/model/`：模型投影、融合、GDN与attention实现。
- `tools/model/`：离线构建、验证、运行与性能采集。
- `tools/operators/`、`tools/eval/`、`tools/bench/`：kernel、质量及计时检查。

Prefill使用单份W4权重、临时W8/A8和INT8 Tensor Core；512的FFN使用LUT4融合。Decode直接读取同一份W4，GDN持续状态和累积为FP32。权重含量化元数据约14.794GB，平均4.4003bits；显式CUDA allocations约19.713GB，另有driver/module/graph开销。

## 性能与限制

本机单流、无MTP/无prefix、每档三次256输出的中位数：

| 输入tokens | Prefill TPS | Decode TPS |
| --- | ---: | ---: |
| 512 | 664.57 | 10.551 |
| 2048 | 773.25 | 10.481 |
| 8192 | 765.88 | 10.223 |

Prefill计时包含输入复制和同步，排除末位置head；decode排除首token，包含逐token复制和同步。上表是固定块token-ID请求的引擎计时，不包含API分词、排队或M=1输入尾部。当前未提供prefix cache、并行batch或MTP。BF16/FP8量化质量评测尚待补充。

## 许可证

GNU LGPL version 3 or later（`LGPL-3.0-or-later`）。见[LICENSE](LICENSE)及其引用的GPLv3文本[COPYING](COPYING)。外部依赖与模型权重遵循各自许可证。

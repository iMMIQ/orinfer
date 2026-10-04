# Orin LLM

用于 Jetson AGX Orin 64GB 的图文推理引擎。在线运行时用 Rust，GPU kernel 用 TileLang，目标固定为 CUDA SM87。

当前支持 Qwen3.8-27B 的文本主干及图片、多图输入，checkpoint架构为 `Qwen3_5ForConditionalGeneration`：48层 Gated DeltaNet和16层 full attention。支持常驻模型、分块prefill、连续decode、请求状态重置、原生MTP，以及OpenAI Chat Completions API、流式输出和函数工具调用。

## 构建

主机需要aarch64 Linux、Rust、CUDA Driver和本机AOT模型产物。开发环境使用Rust1.98.1；workspace声明的最低版本为1.88。离线kernel编译需要TileLang0.1.13、PyTorch2.9.1、CUDA12.6及NVIDIA Docker。

```bash
cargo fetch --locked  # 首次安装Rust依赖
make check
make build
```

模型目录使用checkpoint的配置、tokenizer和chat template；`cache/weights/`采用标准分片safetensors及HF索引，`cache/model.json`只描述模型数据、状态作用域和算子包身份。packed W4、scale、zero和LUT保留原始字节及`orin.layout.<tensor>`元数据。这是引擎专用物理布局，通用safetensors工具可读取，其他引擎需要适配布局才能执行。

Rust从配置识别注册架构，在代码中生成执行计划；算子包独立保存cubin、ABI、布局和形状契约。加载器先查`ORIN_OPERATOR_CACHE`或`$XDG_CACHE_HOME/orin-llm/operators`（默认`~/.cache/orin-llm/operators`），再查模型内的`cache/operators/`。第一阶段计算策略是INT8为主、质量优先的混合精度，关键路径保留FP16/FP32。`validate-model`校验完整模型；`plan-model`在CPU上输出实际生成的计划。

服务协议位于`orin-api`，CLI只处理命令；`orin-engine`分为加载器、架构注册、算子包、CUDA执行器和生成/视觉/MTP控制模块。权重、请求状态和workspace按作用域分开管理，当前GPU仍逐个处理请求。
模型权重、cubin和编译缓存不包含在源码库中。[离线构建说明](tools/model/README.md)介绍checkpoint转换、kernel导出和模型组装。

## 运行

```bash
./target/release/orin-llm validate-model /path/to/model-dir
./target/release/orin-llm run-model /path/to/model-dir examples/requests.json
```

CLI接收token-ID请求，输出包含生成token、加载时间和请求时延的JSON。`examples/requests.json`提供一个512-token文本请求。需要自行使用模型tokenizer准备其他输入。

```json
{
  "requests": [
    {"id": "example", "input_tokens": [151644, 8948], "max_new_tokens": 32}
  ]
}
```

上面只展示字段格式。`run-model`是固定块、普通decode的诊断基准，输入长度须为某个声明prefill计划的整数倍，选择能整除请求长度的最大计划；基础计划为512/2048/8192。任意长度请求及MTP通过下述Chat API或Rust `Model::generate`执行。具体上下文容量由模型缓存声明；本机部署容量为262144 tokens（256k），提示、历史、图片、thinking与输出合计计入。当前部署的最大prefill块为2048 tokens，长提示分块处理。请求状态不复用。

## Chat API

```bash
./target/release/orin-llm serve /path/to/model-dir \
  --model qwen3.8-27b
```

默认监听`0.0.0.0:8088`，可用`--listen HOST:PORT`覆盖。模型目录内保留与权重匹配的`tokenizer.json`、`chat_template.jinja`和`generation_config.json`。Rust直接渲染checkpoint模板并分词。服务只加载一次模型，通过一个GPU worker执行请求；最多128个等待请求，队列满返回429。GPU worker持有`artifacts/gpu-experiment.lock`；可用`--gpu-lock`指定共享锁路径。

`serve`和`run-model`支持`--cuda-graph decode_only|full|off`，默认`decode_only`。`decode_only`只在生成阶段使用Graph，包含普通decode及MTP草稿、验证、恢复和短步刷新；文本prefill、视觉编码及MTP首次预热直接提交。prefill尾部即使复用decode计划也不使用Graph。`full`捕获并使用全部执行计划；`off`按相同计划逐个提交kernel、copy和memset。Graph模式通过显式加载配置传入引擎。

```bash
./target/release/orin-llm serve /path/to/model-dir --cuda-graph full
./target/release/orin-llm run-model /path/to/model-dir requests.json --cuda-graph off
```

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"2+3等于几？"}],"temperature":0,"max_tokens":64,"stream":true}'
```

提供`GET /health`、`GET /v1/models`和`POST /v1/chat/completions`。设置`ORIN_API_KEY`后，`/v1`请求需要对应的Bearer token。支持文本和图片messages、`tools`、`tool_choice`、`parallel_tool_calls`、SSE deltas/`[DONE]`、usage、EOS、stop、temperature、top_p、top_k、presence/frequency/repetition penalties和seed。temperature、top_p、top_k和repetition_penalty默认采用模型`generation_config.json`，请求可覆盖；presence/frequency penalties默认0。默认关闭thinking；`enable_thinking=true`启用`reasoning_content`输出。默认输出上限512 tokens，可通过`max_tokens`或`max_completion_tokens`调整。

使用包含`mtp`执行计划的模型时，所有支持的采样参数组合及文本、图片、多图请求自动启用MTP，thinking与工具调用沿用相同路径。无惩罚的greedy使用GPU top-1；其他组合对主模型与草稿分别应用相同的历史惩罚、temperature、top-k和top-p，再按`min(1,p/q)`接受草稿，拒绝后从归一化的`(p-q)+`采样修正token，保留主模型的采样分布（[算法来源](https://arxiv.org/abs/2211.17192)）。拒绝时恢复GDN、卷积、位置和有效KV状态。固定seed可复现同模式输出；随机MTP与普通decode不要求同seed输出逐token相同。图片草稿使用对应视觉embedding与MRoPE。输出尾部不足一个验证块时执行普通decode。MTP复用主模型embedding/head，额外草稿权重采用W4；构建方式见[离线构建说明](tools/model/README.md)。API日志记录每个请求的MTP接受数、轮数和分段耗时；是否加速取决于接受率及采样开销。

模型的XML工具调用会转换成标准`tool_calls`，arguments为JSON字符串。客户端执行工具，并将带`tool_call_id`的`role=tool`消息连同历史再次发送。函数调用完成后才发送该调用的流式delta；工具参数支持结构校验，暂不提供完整JSON Schema约束解码。`tool_choice=required`或指定函数会加入模板指令并校验结果，模型未遵守时返回生成错误。

API支持任意提示长度：优先选择能容纳剩余输入的最大prefill块，不足最小块的真实tokens走M=1图，不会用额外token填充提示。基础512-token计划的尾部最多511 tokens，可能显著增加首token等待时间；MTP manifest额外包含2/4/8-token计划，将M=1尾部缩小到最多1 token。上下文和输出预算超过manifest容量时返回400。`run-model`仍保留固定块性能测试行为。暂不支持视频、音频、n>1、logprobs、JSON约束输出、prefix cache或同时驻留多个请求；等待请求串行执行，客户端断开后停止生成。

带视觉编码器的manifest接受用户消息中的`image_url`，支持PNG/JPEG/WebP、HTTP(S) URL和base64 data URI。可按内容顺序交错多张图片与文本，历史消息中的图片也会重新编码。每张图片独立做双向视觉attention，merger输出注入对应image tokens；文本full attention使用交错MRoPE，物理KV位置保持连续。

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "比较两张图片"},
    {"type": "image_url", "image_url": {"url": "https://example.com/first.png"}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}],
  "temperature": 0,
  "max_tokens": 128
}
```

`detail=low`使用约256×256像素预算；`auto/high`使用checkpoint预算与manifest视觉容量中的较小者。保留宽高比并按32像素对齐；每32×32像素占一个提示token。所有图片tokens计入上下文和usage。单图最大patch数、整请求特征数及上下文由manifest限制，超过容量返回400；文本manifest收到图片也返回400。HTTP body上限32 MiB，单张压缩图片上限24 MiB，URL读取超时20秒。不提供图片/prefix缓存。

图片预处理、视觉encoder和真实API验证见[图文构建与测试](tools/vision/README.md)。

[OpenCode配置示例](examples/opencode.json)使用`@ai-sdk/openai-compatible`接入`http://127.0.0.1:8088/v1`。复制到独立测试目录后运行：

```bash
opencode run --pure --agent orin --model orin/qwen3.8-27b '读取input.txt并把内容写入output.txt'
opencode run --pure --agent orin --model orin/qwen3.8-27b '比较两张图片' -f first.png second.png
python3 tools/api/smoke.py --output artifacts/api-smoke-results.json
```

Smoke工具验证真实模型的文本/SSE、采样、停止词、状态隔离、错误格式和工具结果回传；结果文件不进入源码库。示例仅开放read/write，客户端声明的上下文容量应与`/health`中的`max_context`一致。

示例配置声明text/image输入；附图请求需要服务加载视觉manifest。

## 实现

- `crates/orin-engine/`：CUDA Driver封装、manifest校验、权重加载、graph执行、KV/GDN/卷积状态和采样。
- `crates/orin-cli/`：命令行、Rust tokenizer/chat template、HTTP/SSE和工具协议。
- `kernels/operators/`：基础TileLang算子。
- `kernels/vision/`：视觉encoder、特征注入和MRoPE。
- `kernels/model/`：模型投影、融合、GDN与attention实现。
- `tools/model/`：离线构建、验证、运行与性能采集。
- `tools/operators/`、`tools/eval/`、`tools/bench/`：kernel、质量及计时检查。

Prefill使用单份W4权重、临时W8/A8和INT8 Tensor Core；512的FFN使用LUT4融合。Decode直接读取同一份W4，GDN持续状态和累积为FP32。权重含量化元数据约14.794GB，平均4.4003bits；显式CUDA allocations约19.713GB，另有driver/module/graph开销。图文manifest另加约0.921GB视觉权重，来自原始BF16 checkpoint，默认转为FP16存储，合计约4.596bits；默认视觉workspace下显式CUDA allocations约22.021GB。

8704-token容量的分配数字见上。262144-token配置可通过 `tools/model/optimize_kv.py` 使用直接 KV prefill 和 CUDA VMM：加载时固定缓冲区约 18.60 GiB，KV 物理内存按执行位置增长，graph 地址保持稳定；最大 prefill 块为 2048 tokens。FP16 KV 满容量包含主模型和 MTP 共 17 GiB，INT8 group-64 KV 含 FP16 scale 共约 8.77 GiB，另有 driver/module/graph 开销。新请求会回收上一请求的 KV 物理映射。构建与验证方法见[模型构建](tools/model/README.md)。

## 性能与限制

本机单流、无MTP/无prefix、每档三次256输出的中位数：

| 输入tokens | Prefill TPS | Decode TPS |
| --- | ---: | ---: |
| 512 | 664.57 | 10.551 |
| 2048 | 773.25 | 10.481 |
| 8192 | 765.88 | 10.223 |

Prefill计时包含输入复制和同步，排除末位置head；decode排除首token，包含逐token复制和同步。上表是关闭MTP时固定块token-ID请求的引擎计时，不包含API分词、排队或M=1输入尾部。当前未提供prefix cache或并行batch。BF16/FP8量化质量评测尚待补充。

开启MTP后，固定seed20261002、greedy、关闭thinking、单请求128-token代码输出，预热后3次HTTP SSE decode中位数：Python合并排序26.04 TPS、Rust LRU缓存25.11 TPS、TypeScript异步并发映射25.82 TPS。计数通过关闭MTP时的主模型token IDs核对，排除首个输出片段和被拒绝的草稿；完整输出与主模型参考相同。额外MTP草稿权重222,342,144 bytes，含视觉与MTP的常驻权重合计约4.59 bits/parameter。

## 许可证

GNU LGPL version 3 or later（`LGPL-3.0-or-later`）。见[LICENSE](LICENSE)及其引用的GPLv3文本[COPYING](COPYING)。外部依赖与模型权重遵循各自许可证。

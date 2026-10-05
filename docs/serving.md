# 服务使用指南

## 启动与模型加载

```bash
./target/release/orinfer validate-model /path/to/prepared-model
./target/release/orinfer serve /path/to/prepared-model \
  --model qwen3.8-27b --listen 0.0.0.0:8088 --cuda-graph decode_only
```

服务加载一次模型，由单个 GPU worker 执行请求。CPU 使用模型目录中的 `tokenizer.json`、`chat_template.jinja` 和 `generation_config.json` 渲染模板及分词；线上不运行 Python、量化或 kernel JIT。

模型准备时封存上述前端文件的 SHA256。需要有意修改时，先确认它们与 checkpoint 的 token 映射和语义一致，再执行 `python3 tools/model/package.py pin-assets MODEL_DIR` 重新封存；此命令不改权重。模型文件在加载期间必须保持不变。

先查共享算子缓存中的 `<digest>`，缺失时再查模型内的 `cache/operators/<digest>`。`ORINFER_OPERATOR_CACHE` 可覆盖共享缓存位置；默认目录是 `$XDG_CACHE_HOME/orinfer/operators`，未设置 XDG 时为 `~/.cache/orinfer/operators`。存在但损坏的包会报错。专用 safetensors 布局沿用 `orin.layout.<tensor>` 标识，模型和算子包的身份不随项目名称变化。

## Chat API

提供 `GET /health`、`GET /v1/models` 和 `POST /v1/chat/completions`。设置 `ORINFER_API_KEY` 后，`/v1` 请求需携带 `Authorization: Bearer <key>`。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"37×24−19×13等于多少？"}],"temperature":0,"max_tokens":512,"enable_thinking":true,"stream":true,"stream_options":{"include_usage":true}}'
```

支持 `temperature`、`top_p`、`top_k`、presence/frequency/repetition penalties、`seed`、EOS 和 `stop`。temperature、top_p、top_k 和 repetition_penalty 默认采用模型的 generation config；presence/frequency penalties 默认 0。默认关闭 thinking；开启后思考文本位于 `reasoning_content`。

默认输出预算为 512 tokens，可用 `max_tokens` 或 `max_completion_tokens` 调整。提示、历史、图片、thinking 和输出合计计入上下文容量；超出模型包容量时在准入前返回 400。HTTP body 上限为 32 MiB。

流式响应使用 SSE deltas 和 `[DONE]`；请求 `stream_options.include_usage=true` 可获得 usage。客户端断连后取消生成。异常流返回 error，不将失败后的部分输出伪装为成功完成。

### 图片与多图

用户消息可交错文本和多张图片，支持 PNG/JPEG/WebP、HTTP(S) URL 和 base64 data URI；历史消息中的图片也会重新编码。

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "比较两张图片"},
    {"type": "image_url", "image_url": {"url": "https://example.com/first.png"}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}],
  "max_tokens": 512
}
```

`detail=low` 使用约 256×256 像素预算；`auto/high` 使用 checkpoint 预算与模型包视觉容量中的较小者。预处理保持宽高比并按 32 像素对齐，每 32×32 像素占一个提示 token。单张压缩图片上限 24 MiB，URL 读取超时 20 秒；patch、特征及总上下文超出模型包容量时返回 400。纯文本模型收到图片也返回 400。

每张图片独立执行双向视觉 attention，merger 输出注入对应 image tokens；文本使用交错 MRoPE。视觉 encoder 每次仍执行，相同图片的预处理结果可命中文本主干 prefix cache。当前不支持视频或音频。构建与验证见[图文工具](../tools/vision/README.md)。

### 工具调用

支持 `tools`、`tool_choice`、`parallel_tool_calls`。模型的 XML 工具调用转换为标准 `tool_calls`，arguments 为 JSON 字符串。客户端执行工具，再把带 `tool_call_id` 的 `role=tool` 消息连同历史发送回来。

函数调用完成后发送该调用的流式 delta；参数支持结构校验，但暂不提供完整 JSON Schema 约束解码。`tool_choice=required` 或指定函数时加入模板指令并校验结果，模型未遵守时返回生成错误。[OpenCode 示例](../examples/opencode.json)使用 `orinfer` provider 和 agent。

## 执行与缓存

### CUDA Graph

`serve`、`run-model` 和 `score-model` 支持 `--cuda-graph decode_only|full|off`。

- `decode_only`：默认模式。普通 decode、MTP 草稿、验证、恢复和短步刷新使用 Graph；文本 prefill、视觉编码和 MTP 首次预热直接提交。
- `full`：捕获并使用全部执行计划。
- `off`：按相同计划直接提交 kernel、copy 和 memset。

prefill 尾部即使复用 decode 计划，在 `decode_only` 模式下也不使用 Graph。地址及 workspace 变化会使相关 Graph 失效或重新捕获。

### MTP

包含原生 `mtp` 计划的模型自动启用 MTP，适用于文本、图片、多图、thinking、工具调用和支持的采样参数。无惩罚 greedy 使用 GPU top-1；随机采样对 target/draft 应用相同的采样处理，接受概率为 `min(1,p/q)`，拒绝后从归一化的 `(p-q)+` 分布采样，以保持主模型分布。

拒绝时恢复私有 GDN、卷积、位置和有效 KV 状态。固定 seed 可复现同模式结果；随机 MTP 与普通 decode 不要求同 seed 下逐 token 相同。输出尾部不足一个验证块时执行普通 decode。

单请求使用 MTP；2–4 个 decoder 按实测每提交 token 成本选择 MTP 或 target batch，更大并发和混合 prefill 使用 target batch。收益取决于接受率及采样开销。构建方法见[模型工具](../tools/model/README.md)。

### Prefix cache

`--prefix-cache-mib` 默认 12288，单位 MiB，0 关闭复用；按需分配，不在加载时预占。缓存保存完整 KV、FP32 GDN、卷积、位置及 MTP 检查点。Rust radix tree 按 token 和图片身份匹配，并结合实际 prefill/恢复成本选择可用端点。

KV 区间不可变并按引用共享，持续状态单独保存；恢复仍执行 GPU 复制到 Graph 绑定地址。短前缀妨碍大块执行时可能跳过。保存最终提示、周期及按成本准入的分叉检查点；生成结束也可保存已计算的输出前缀。模型重载后缓存清空。

命中免除对应文本主干计算，视觉编码、剩余提示和生成继续执行。`usage.prompt_tokens_details.cached_tokens` 报告实际恢复长度，SSE 需请求 usage。缓存不是 paged attention 或零复制映射，不保证接近满上下文时命中完整提示；内存不足会驱逐或跳过保存。

## 并发、上下文与资源

支持批处理的算子包启用连续 batching；未包含相应算子时串行执行，`/health.continuous_batching` 报告实际模式。每个请求拥有独立 KV、GDN、卷积、位置、视觉和采样状态，权重及执行 workspace 共享。

单请求冷 prefill 使用大块计划；兼容的多个短请求可联合执行。与 decode 混合时，按预测耗时选择小块 prefill。常见 batch 可使用固定形状 kernel，其余 2–128 行可由支持动态行数的 AOT 包执行，线上不编译。请求按迭代加入和结束；不为凑 batch 额外等待。

默认最多 32 个活跃请求、128 个待处理请求，队列满时返回 429。活跃数还受每请求上下文/输出预算、workspace 和可用内存约束。准入检查 CUDA 可用内存和 Linux `MemAvailable`，保留系统余量；不足时先收缩 prefix cache，再排队。冷请求可能延后准入以减少现有 decoder 的停顿，缓存命中请求仍可准入；统计见 `/health`。

模型包决定实际上下文容量，上限可配置为 262144 tokens。长提示分块计算，首次 dense attention 的复杂度不变。INT8 group-64 KV 包含 FP16 scale，CUDA VMM 按已执行位置映射物理页，同时保持 Graph 地址稳定；prefill 的单层 FP16 临时 KV workspace 可共享并按需增长。持续 GDN 状态保持 FP32。上下文扩展、KV 优化和验证见[模型构建](../tools/model/README.md)。

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--listen` | `0.0.0.0:8088` | HTTP 监听地址 |
| `--model` | `qwen3.8-27b` | API 模型 ID |
| `--cuda-graph` | `decode_only` | Graph 模式 |
| `--prefix-cache-mib` | `12288` | GPU 缓存字节预算，0 关闭 |
| `--max-active-requests` | `32` | 活跃请求上限，1–128 |
| `--max-batch-tokens` | `128` | 每轮 target 计算 token 预算，1–128 |
| `--prefill-budget-ms` | `200` | 混合块的预测耗时目标，非严格延迟上限 |
| `--memory-reserve-mib` | `1024` | 系统余量以外的准入保留内存 |
| `--preprocess-workers` | `2` | 同时执行图片/模板/tokenizer CPU 任务数 |
| `--preprocess-memory-mib` | `2048` | 预处理和排队图片张量预算 |
| `--queue-timeout-ms` | `0` | 排队期限，0 不限制等待；不限制已开始计算 |
| `--output-timeout-ms` | `60000` | 输出通道无进展时终止，prefill 不计入 |
| `--drain-timeout-ms` | `30000` | SIGINT/SIGTERM 后 HTTP 排空期限 |
| `--gpu-lock` | `artifacts/gpu-experiment.lock` | GPU worker 使用的共享锁 |

`/health` 显示 starting/ready/draining/failed，并提供调度、准入、缓存恢复和 Graph 统计。GPU worker 故障时返回 503；CUDA 或内部状态故障后需要重启服务。统计和真实 API 性能测量见[性能工具](../tools/bench/README.md)。

## 诊断 CLI

`plan-model` 在 CPU 输出注册架构生成的执行计划，`validate-model` 校验模型、算子包和张量 hash；两者不执行 GPU 推理。

`run-model MODEL_DIR REQUESTS.json` 接收 token-ID 请求，用于固定块、普通 decode 的诊断基准，逐请求重置状态。输入长度须能由声明的 prefill 计划整除；它不代表 API 的任意长度提示、MTP 和缓存路径。任意长度和多请求生成通过 Chat API 或 Rust `Model::generate` 执行。

`score-model` 对冻结 token 历史评分，服务的固定场景检查见[API 验证](../tools/api/README.md)，BF16/FP8 质量对照见[质量评测](../tools/eval/README.md)。

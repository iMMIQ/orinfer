# Orin LLM

用于 Jetson AGX Orin 64GB 的图文推理引擎。在线运行时用 Rust，GPU kernel 用 TileLang，目标固定为 CUDA SM87。

当前支持 Qwen3.8-27B 的文本主干及图片、多图输入，checkpoint架构为 `Qwen3_5ForConditionalGeneration`：48层 Gated DeltaNet和16层 full attention。支持常驻模型、分块prefill、连续decode、请求状态重置、原生MTP，混合架构prefix cache，以及OpenAI Chat Completions API、流式输出和函数工具调用。

## 构建

主机需要aarch64 Linux、Rust、CUDA Driver和本机AOT模型产物。开发环境使用Rust1.98.1；workspace声明的最低版本为1.88。离线kernel编译需要TileLang0.1.13、PyTorch2.9.1、CUDA12.6及NVIDIA Docker。

```bash
cargo fetch --locked  # 首次安装Rust依赖
make check
make build
```

模型目录使用checkpoint的配置、tokenizer和chat template；`cache/weights/`采用标准分片safetensors及HF索引，`cache/model.json`只描述模型数据、状态作用域和算子包身份。packed W4、scale、zero和LUT保留原始字节及`orin.layout.<tensor>`元数据。这是引擎专用物理布局，通用safetensors工具可读取，其他引擎需要适配布局才能执行。

Rust从配置识别注册架构，在代码中生成执行计划；算子包独立保存cubin、ABI、布局和形状契约。加载器先查`ORIN_OPERATOR_CACHE`或`$XDG_CACHE_HOME/orin-llm/operators`（默认`~/.cache/orin-llm/operators`），再查模型内的`cache/operators/`。第一阶段计算策略是INT8为主、质量优先的混合精度，关键路径保留FP16/FP32。可离线加入decode INT8 FFN与GDN输出包：权重维持单份W4，在寄存器解包后执行INT8 MMA；FFN Down及GDN输出采用group-128 activation scale。其余投影与持续状态保持原精度，构建方法见[模型构建](tools/model/README.md)。`validate-model`校验完整模型；`plan-model`在CPU上输出实际生成的计划。

服务协议位于`orin-api`，CLI只处理命令；`orin-engine`分为加载器、架构注册、算子包、CUDA执行器和生成/视觉/MTP控制模块。连续批处理共享权重与workspace，为每个请求保留独立KV、FP32 GDN、卷积、位置、视觉和采样状态。
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

上面只展示字段格式。`run-model`是固定块、普通decode的诊断基准，输入长度须为某个声明prefill计划的整数倍，选择能整除请求长度的最大计划；基础计划为512/2048/8192。任意长度请求及MTP通过下述Chat API或Rust `Model::generate`执行。具体上下文容量由模型缓存声明；本机部署容量为262144 tokens（256k），提示、历史、图片、thinking与输出合计计入。当前部署的最大prefill块为2048 tokens，长提示分块处理。`run-model`逐请求重置状态；Chat API可复用完整前缀检查点。

## Chat API

```bash
./target/release/orin-llm serve /path/to/model-dir \
  --model qwen3.8-27b
```

默认监听`0.0.0.0:8088`，可用`--listen HOST:PORT`覆盖。模型目录内保留与权重匹配的`tokenizer.json`、`chat_template.jinja`和`generation_config.json`。Rust直接渲染checkpoint模板并分词。服务只加载一次模型，通过一个GPU worker执行请求；活跃与待处理请求合计有界，默认容量为160（32+128），满时返回429。GPU worker持有`artifacts/gpu-experiment.lock`；可用`--gpu-lock`指定共享锁路径。

包含批处理算子的模型自动启用continuous batching。默认最多32个活跃请求，混合prefill/decode每轮最多128个target计算tokens；`--max-active-requests 1..128`和`--max-batch-tokens 1..128`调整上限，活跃数还受请求的完整上下文/输出预算、共享prefill workspace及可用显存约束。Orin使用统一内存，准入和缓存预算同时检查CUDA可用内存与Linux `MemAvailable`，为系统保留物理RAM的1/16，再加上`--memory-reserve-mib 1024`配置的余量；不足时先收缩prefix cache，再让新请求排队。权重只常驻一份，私有KV虚拟地址按请求上下文预算预留，空闲槽位扩容时重建相关Graph。新请求加入与结束按迭代处理，断连后释放其槽位，慢客户端输出通过有界非阻塞缓冲传送。

完整提示命中的文本请求支持成批准入：无活跃decoder时最多32个、3秒预算，容纳初次分配CUDA私有槽位的成本；已有decoder时最多8个、100毫秒预算。恢复期间已到达的请求可以加入同一批，不设置等待定时器。预算在每个请求启动完成后检查；一次不可切分的恢复或视觉操作可能超出预算。冷请求、部分命中及图片沿用10毫秒准入预算，显存准入和排队老化策略继续生效。`/health`与[并发测试工具](tools/bench/README.md)提供准入及Graph分项计时。

单请求prefill使用512/2048大块计划；多个短prefill可以拼接执行。声明`prefill_batch_profiles`的算子包还支持512/1024/2048总行数的大块联合prefill：没有decoder等待时，每轮为至多四个请求分配真实分块，共享投影与FFN，卷积、GDN分块扫描、attention和KV保持每请求独立。联合FFN沿用512-token路径的LUT4码本，权重与workspace不重复常驻。与decode混合时按预测耗时选择1/2/4/8/32/64/128-token块，`--prefill-budget-ms 200`是混合块的预测时间目标，首轮估计和不可切分的视觉编码/缓存复制可能超过它。投影与FFN按总行数合批，attention/GDN保持每请求独立。decode算子包覆盖2/4/8/16/32/64/128行，可为常见小batch加入固定行数投影，带动态行数契约的包按实际2..128行复用symbolic-row kernel，由Rust生成执行计划与launch参数，不需要在线Python编译；未升级的包继续补齐到下一档，填充行不进入请求状态。Graph命中时直接重放，不重建执行计划。不含批处理算子的包继续串行执行，`/health.continuous_batching`报告实际模式；离线升级见[模型构建](tools/model/README.md)。

冷请求准入还比较预计首token时间：若按已测decode速率完成现有请求，再用大块prefill，预计比立即混合执行快至少10%，冷请求会暂留队列。完整缓存命中的请求继续准入；已有prefill继续推进，冷请求等待30秒后也恢复通常准入规则。尚无速率观测时直接准入，混合prefill的冷启动速率估计会由实际测量替换。`/health.admission_statistics.cold_deferrals`记录延后准入的候选检查次数，不等于请求数。

包含`batch_gdn`能力的包在M1批处理中使用地址表并行访问私有GDN和卷积状态，卷积直接更新历史；混入短prefill时，decoder子集仍批处理，prefill请求单独执行自己的mixer。GDN持续状态和累积保持FP32，所有路径保持请求状态隔离。

包含`greedy_sampling`能力的算子包在GPU执行带历史惩罚的零温度token选择，使用FP64运算、完整历史词频和确定性的并列排序，不逐步下载完整logits；每请求约2 MiB临时缓冲。正温度及MTP接受/拒绝的分布计算保留CPU路径。

`serve`、`run-model`和`score-model`支持`--cuda-graph decode_only|full|off`，默认`decode_only`。`decode_only`只在生成阶段使用Graph，包含普通decode及MTP草稿、验证、恢复和短步刷新；文本prefill、视觉编码及MTP首次预热直接提交。prefill尾部即使复用decode计划也不使用Graph。`full`捕获并使用全部执行计划；`off`按相同计划逐个提交kernel、copy和memset。Graph模式通过显式加载配置传入引擎。

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

使用包含`mtp`执行计划的模型时，所有支持的采样参数组合及文本、图片、多图请求自动启用MTP，thinking与工具调用沿用相同路径。无惩罚的greedy使用GPU top-1；其他组合对主模型与草稿分别应用相同的历史惩罚、temperature、top-k和top-p，再按`min(1,p/q)`接受草稿，拒绝后从归一化的`(p-q)+`采样修正token，保留主模型的采样分布（[算法来源](https://arxiv.org/abs/2211.17192)）。拒绝时恢复GDN、卷积、位置和有效KV状态。固定seed可复现同模式输出；随机MTP与普通decode不要求同seed输出逐token相同。图片草稿使用对应视觉embedding与MRoPE。输出尾部不足一个验证块时执行普通decode。MTP复用主模型embedding/head，额外草稿权重采用W4；构建方式见[离线构建说明](tools/model/README.md)。连续批处理在独立decode请求时使用MTP；2–4个decoder根据实测的每提交token代价选择交替MTP或target batch，更大并发及混合prefill使用target batch。恢复MTP时从私有hidden环追赶草稿状态，若超出环容量则保持普通decode。API日志记录每个请求的MTP接受数、轮数和分段耗时；是否加速取决于接受率及采样开销。

`serve`默认启用prefix cache，`--prefix-cache-mib 12288`设置实际GPU缓存字节预算，`0`关闭；按需分配，不在加载时预占。Rust压缩radix tree寻找兼容的完整状态检查点，结合在线测量的prefill块耗时、剩余输入分块和恢复复制成本选择命中；短前缀会破坏高效大块执行时跳过。KV区间不可变、按引用共享，GDN/卷积/位置/MTP状态独立保存，恢复复制到原有Graph绑定地址。保存最终提示、每8192 tokens的回退检查点，并按执行成本准入实际分叉点及Chat模板提示的系统/历史边界；生成结束时还保存已经计算的输出前缀，最后一个尚未计算的token由下一轮续接。输出检查点没有有效head时只用于继续输入，不能作为完整提示直接采样。检查点数量由字节预算决定，淘汰结合最近使用、复用次数与到最近有效祖先的重算距离。

图片身份包含预处理后的像素和网格，绑定到该图片的第一个特征token；后面的图片变化不影响前面的有效检查点。模型重载后缓存清空。命中免除该段文本主干计算，视觉编码、剩余提示和生成仍执行。服务保持单GPU执行线程，按剩余prefill与缓存恢复成本选择待准入请求；等待超过两秒后优先处理较早到达的请求，不为凑batch额外等待。

响应的`usage.prompt_tokens_details.cached_tokens`报告实际恢复的tokens；SSE需请求`stream_options.include_usage=true`。日志分别记录token匹配长度、状态恢复长度、查询/恢复/保存耗时、物理/逻辑字节、共享节省与淘汰数量。公共KV共享可以降低持久缓存占用，恢复仍执行GPU复制；缓存不是paged attention或零复制映射。预算不足或无法分配时跳过缓存保存。GDN继续保留FP32，没有使用近似后缀重建或状态量化；长提示首次计算仍包含完整dense attention，复杂度不变。

模型的XML工具调用会转换成标准`tool_calls`，arguments为JSON字符串。客户端执行工具，并将带`tool_call_id`的`role=tool`消息连同历史再次发送。函数调用完成后才发送该调用的流式delta；工具参数支持结构校验，暂不提供完整JSON Schema约束解码。`tool_choice=required`或指定函数会加入模板指令并校验结果，模型未遵守时返回生成错误。

API支持任意提示长度：优先选择能容纳剩余输入的最大prefill块，不足最小块的真实tokens走M=1图，不会用额外token填充提示。基础512-token计划的尾部最多511 tokens，可能显著增加首token等待时间；MTP manifest额外包含2/4/8-token计划，将M=1尾部缩小到最多1 token。上下文和输出预算超过manifest容量时返回400。`run-model`仍保留固定块性能测试行为。暂不支持视频、音频、n>1、logprobs或JSON约束输出；客户端断开后停止其生成。

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

`detail=low`使用约256×256像素预算；`auto/high`使用checkpoint预算与manifest视觉容量中的较小者。保留宽高比并按32像素对齐；每32×32像素占一个提示token。所有图片tokens计入上下文和usage。单图最大patch数、整请求特征数及上下文由manifest限制，超过容量返回400；文本manifest收到图片也返回400。HTTP body上限32 MiB，单张压缩图片上限24 MiB，URL读取超时20秒。视觉encoder每次仍重新执行；相同图片预处理结果可以命中文本主干的prefix cache。

图片预处理、视觉encoder和真实API验证见[图文构建与测试](tools/vision/README.md)。

[OpenCode配置示例](examples/opencode.json)使用`@ai-sdk/openai-compatible`接入`http://127.0.0.1:8088/v1`。复制到独立测试目录后运行：

```bash
opencode run --pure --agent orin --model orin/qwen3.8-27b '读取input.txt并把内容写入output.txt'
opencode run --pure --agent orin --model orin/qwen3.8-27b '比较两张图片' -f first.png second.png
python3 tools/api/smoke.py --output artifacts/api-smoke-results.json
```

Smoke工具验证真实模型的文本/SSE、采样、停止词、状态隔离、错误格式和工具结果回传；`--expect-prefix-cache`额外检查重复提示及SSE的缓存命中usage。结果文件不进入源码库。示例仅开放read/write，客户端声明的上下文容量应与`/health`中的`max_context`一致。

示例配置声明text/image输入；附图请求需要服务加载视觉manifest。

## 实现

- `crates/orin-engine/`：CUDA Driver封装、manifest校验、权重加载、graph执行、KV/GDN/卷积状态和采样。
- `crates/orin-api/`：Rust tokenizer/chat template、HTTP/SSE、调度队列与工具协议。
- `crates/orin-cli/`：命令行入口。
- `kernels/operators/`：基础TileLang算子。
- `kernels/vision/`：视觉encoder、特征注入和MRoPE。
- `kernels/model/`：模型投影、融合、GDN与attention实现。
- `tools/model/`：离线构建、验证、运行与性能采集。
- `tools/operators/`、`tools/eval/`、`tools/bench/`：kernel、质量及计时检查。

Prefill使用单份W4权重、临时W8/A8和INT8 Tensor Core；512的FFN使用LUT4融合。Decode直接读取同一份W4，GDN持续状态和累积为FP32。权重含量化元数据约14.794GB，平均4.4003bits；显式CUDA allocations约19.713GB，另有driver/module/graph开销。图文manifest另加约0.921GB视觉权重，来自原始BF16 checkpoint，默认转为FP16存储，合计约4.596bits；默认视觉workspace下显式CUDA allocations约22.021GB。

8704-token容量的分配数字见上。262144-token配置可通过 `tools/model/optimize_kv.py` 使用直接 KV prefill 和 CUDA VMM：加载时固定缓冲区约 18.60 GiB，KV 物理内存按执行位置增长，graph 地址保持稳定；最大 prefill 块为 2048 tokens。FP16 KV 满容量包含主模型和 MTP 共 17 GiB，INT8 group-64 KV 含 FP16 scale 共约 8.77 GiB，长文本 prefill 可共享按需映射的单层 FP16 临时 KV workspace，每 token 4096 字节、256k 上限 1 GiB；decode 直接读取 INT8 KV。另有 driver/module/graph 开销。请求结束时回收其 KV 物理映射；prefill临时workspace由执行线程共享。构建与验证方法见[模型构建](tools/model/README.md)。

## 性能与限制

本机单流、无MTP/无prefix，含group-128 INT8 GDN输出投影：512行FFN使用BM256/BN64融合tile，2048行FFN使用BM128/BN128、128线程及M分组调度的临时W8 GEMM。每档三次128输出的中位数：

| 输入tokens | Prefill TPS | Decode TPS |
| --- | ---: | ---: |
| 512 | 676.38 | 10.919 |
| 2048 | 818.90 | 10.753 |
| 8192 | 773.50 | 10.150 |

Prefill计时包含输入复制和同步，排除末位置head；decode排除首token，包含逐token复制和同步。上表是关闭MTP时固定块token-ID请求的引擎计时，不包含API分词、排队或M=1输入尾部。上表不代表连续批处理吞吐；并发需通过真实API另测。用户已接受当前测得性能，第一阶段不再以所有长度达到800 TPS为硬性门槛。

文本快速质量对照使用同一uncensored模型的原始BF16 checkpoint，由独立Transformers逐层加载、B1完整序列prefill评分；本引擎执行正常prefill和teacher-forced decode，固定seed20261002与原始token历史。12个场景161个位置的top1一致率为97.52%，候选选择全部位于BF16 top3，平均答案NLL变化为−0.00800；约512/2k/8k的中部资料检索共15个位置，top1全部相同，NLL变化接近0。图片和多图另用同源BF16视觉encoder与文本模型作独立B1参考：8个颜色、场景及顺序任务全部通过，14个标签位置的候选选择有92.86%位于BF16 top3；大小写和前导空白不同需结合任务结果判断。视觉参考使用相同的归一化patch输入，未独立验证图片预处理。第一阶段按约定的快速评测范围验收；完整LLM benchmark留待后续。工具与执行路径见[质量评测](tools/eval/README.md)。

长上下文另有无MTP的容量探针：260096-token输入与2049个固定长度输出使主模型实际计算位置到达262144，中部校验码检索正确；单次prefill约242.21 TPS、后续decode约3.08 TPS，主模型KV峰值8.25 GiB、prefill workspace峰值约0.99 GiB。固定长度探针在EOS后继续执行以验证容量，语义答案只取首个EOS之前；它未覆盖256k的独立BF16质量、Chat API长请求准入、MTP或prefix恢复，不能用来证明这些场景的完整验收。

在线Chat另测260095-token提示、相同请求重放和260126-token多轮续写，开启MTP，三次均正确输出中部校验码并正常结束；超出262144总预算的请求在准入前返回400。默认12 GiB prefix预算下，内存压力会驱逐完整端点，本次重放与续写各命中65536 tokens，剩余约194k tokens重新计算，端到端分别约1107秒和1069秒。256k容量可用，但不能保证近满上下文的完整缓存命中；独立BF16长上下文质量评测仍未覆盖。

开启MTP后，固定seed20261002、greedy、关闭thinking、单请求128-token代码输出，预热后3次HTTP SSE decode中位数：Python合并排序26.04 TPS、Rust LRU缓存25.11 TPS、TypeScript异步并发映射25.82 TPS。计数通过关闭MTP时的主模型token IDs核对，排除首个输出片段和被拒绝的草稿；完整输出与主模型参考相同。额外MTP草稿权重222,342,144 bytes，含视觉与MTP的常驻权重合计约4.59 bits/parameter。

## 许可证

GNU LGPL version 3 or later（`LGPL-3.0-or-later`）。见[LICENSE](LICENSE)及其引用的GPLv3文本[COPYING](COPYING)。外部依赖与模型权重遵循各自许可证。

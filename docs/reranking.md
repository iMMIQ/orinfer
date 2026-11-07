# 查询与文档重排

`Qwen3-Reranker-0.6B` 使用 28 层因果 Qwen3、1024 hidden、16/8 GQA heads、128 head dimension 和逐头 Q/K RMSNorm。输入是指令、查询和文档组成的完整序列；最后位置的 `no`、`yes` logits 给出二分类相关性。API 返回 `sigmoid(yes_logit - no_logit)`，按原始 logit 差降序排序，分数饱和时仍保留排序精度，同分按输入索引稳定排序。

## 构建

来源目录包含官方 `Qwen/Qwen3-Reranker-0.6B` BF16 配置、tokenizer、模板、safetensors/index 及带固定 `repo`、40 位 `revision` 的 `source.json`。

```sh
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/reranker-source" \
  bash tools/environments/cuda/run.sh python/orinfer_build/quantization/qwen_reranker.py \
  artifacts/reranker-converted --source-dir "$PWD/artifacts/reranker-source" \
  --gpu-lock /tmp/qwen-reranker-encoder.lock

make build
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/reranker-converted" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen_reranker/build.py \
  artifacts/reranker-build --source "$PWD/artifacts/reranker-converted" \
  --model-output "$PWD/artifacts/reranker-build/model" --gpu-lock-held
```

operator wrapper 持有公共 GPU 锁，转换器的内部锁使用独立路径。直接调用转换器时沿用默认公共锁。

主矩阵和 embedding 使用两级 E8P Q4、系数 `[4,1]`、beam8 搜索、signed-block128 旋转和 FP16 输出行 scale，激活使用 A8。attention 输出投影保留 W8；前 4 层及第 9 层（索引 0/1/2/3/8）的 Q/K/V 与 FFN 使用相同输入旋转基下的 W8，保护自定义指令和较长文档的相关性概率。norm 与原始 `no`、`yes` 两行输出权重保留 FP16。输出头占用 4096 字节，主 embedding 保持单份 K128 压缩表示。

已有完整 E8P 转换可用 `python -m orinfer_build.quantization.qwen_reranker --source-dir 原始BF16目录 --upgrade-from 已转换目录 --output 新目录` 按行块提升敏感矩阵精度，其余压缩值和二分类头保持原始字节，校验来源后原子发布。

每个查询与文档对保持独立的完整因果序列。同一请求内的多个文档共享投影执行，Q/K/V 与 gate/up 权重按输出行无损拼接。各序列按 32 行对齐，attention、RoPE 和最后位置输出使用各自的真实边界，填充不计入输入；改变边界后可复用同一容量的 Graph。超过单个 packet 行容量时按文档边界分包，结果保持输入索引。默认容量为 32768 token；`--context` 可设置 512–32768 内的 32 倍数。旧执行包仍可逐序列运行。模型配置、原始 yes/no ID 和前端资源固定并校验。

## 使用

```sh
target/release/orinfer serve artifacts/reranker-build/model \
  --model qwen3-reranker-0.6b --listen 127.0.0.1:8089 --cuda-graph full

curl http://127.0.0.1:8089/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-reranker-0.6b","query":"Orin 有多少内存？","documents":["Jetson AGX Orin 64GB 配有 64GB 统一内存。","苹果树在春季开花。"],"top_n":2,"return_documents":true}'
```

`/rerank` 为同一接口的别名，模型管理器也转发这两个入口。`documents` 接受字符串或带 `text` 的对象，最多 128 项；返回的 `index` 对应原输入位置。`return_documents` 默认 false，开启时保留文档对象的其他字段。`top_n` 默认为全部，允许 1–128，超过文档数量时返回全部。`instruction` 可覆盖默认英文检索指令，`ORINFER_API_KEY` 配置 Bearer 认证。

前端按官方前缀、`<Instruct>` / `<Query>` / `<Document>` 正文及关闭 thinking 的后缀分别 tokenize 后拼接。超出上下文返回错误，完整后缀始终保留。查询和指令各不超过 1 MiB，每个文档不超过 4 MiB；空查询和空文档返回错误，HTTP 请求体上限 32 MiB，队列与并发入口有界。

离线 JSON 请求复用同一前端、排序及原生执行流程，另返回各输入顺序下的二分类 logits，便于质量对照：

```sh
target/release/orinfer rerank-model artifacts/reranker-build/model requests.json --cuda-graph full
```

## 验证

```sh
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/reranker-source" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen_reranker/check_kernels.py \
  artifacts/reranker-kernels --original "$PWD/artifacts/reranker-source"

ORINFER_REFERENCE_CHECKPOINT="$PWD/artifacts/reranker-source" \
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/reranker-build/model" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen_reranker/check_model.py \
  artifacts/reranker-quality --original "$PWD/artifacts/reranker-source" \
  --model "$PWD/artifacts/reranker-build/model" --engine "$PWD/target/release/orinfer"
```

共享独立行 kernel 可通过 `bash tools/environments/cuda/run.sh python/orinfer_build/validation/check_encoder_rows.py artifacts/encoder-rows-check` 验证不同 owner、填充与改变边界后的 Graph replay。

GPU 验证覆盖因果 GQA 非对齐尾部、未来 token 隔离、Q/K norm＋RoPE、最后位置二分类头及改变输入后的 Graph replay。完整请求对照原始 BF16 的二分类概率、KL 和排序，覆盖中英文、代码检索、自定义指令、较长文档、重复与反向输入、Graph/direct 一致性。这些用例用于接入验收，目标语料上的完整重排质量需另行评测。

模型权重采用 Apache-2.0，保留官方来源及声明；引擎与 kernel 实现采用 LGPL-3.0-or-later。

构建器用真实权重离线测量短 packet 和最大容量，按实际行数发布所选整数投影；在线直接读取选择表。文本 attention 合并共享 K/V 的查询头，使用有界异步读取，各输入保留独立的真实边界。旋转量化复用寄存器实现，权重保持单份压缩表示。完整请求性能使用[嵌入与重排基准工具](../tools/bench/README.md)对照相同输入的 vLLM BF16；模型包、行容量和各档加速比分别记录。

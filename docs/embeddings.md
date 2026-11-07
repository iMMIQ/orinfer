# 文本和图片文档嵌入

`Nemotron-3.5-Embed-1B` 使用 Ministral3 双向文本编码器和 Pixtral 视觉编码器，输出 2048 维、L2 归一化向量。文本、图片和多图文档可以放入同一向量索引；查询使用文本。

文本采用 `query: ` 或 `passage: ` 前缀，按 checkpoint 的检索模板转义用户特殊标记，不添加生成用 BOS/EOS。平均池化包括前缀和全部实际输入 token。输入上限由执行包声明，最大 32768 token，图片 token 也计入长度。完整序列在每层执行双向注意力。

## 量化和构建

使用官方原始 BF16 checkpoint，输入目录包含 tokenizer、processor、配置、safetensors 分片/index，以及记录 `repo` 和固定 40 位 `revision` 的 `source.json`。

```sh
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/nemotron-source" \
  bash tools/environments/cuda/run.sh python/orinfer_build/quantization/nemotron_embed.py \
  artifacts/nemotron-converted \
  --source-dir "$PWD/artifacts/nemotron-source" \
  --gpu-lock /tmp/nemotron-encoder.lock

make build
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/nemotron-converted" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/nemotron_embed/build.py \
  artifacts/nemotron-build \
  --source "$PWD/artifacts/nemotron-converted" \
  --model-output "$PWD/artifacts/nemotron-build/model" \
  --gpu-lock-held
```

operator wrapper 持有公共 GPU 锁，转换器的内部锁因此使用独立路径。直接调用转换器时沿用默认公共锁。

文本和视觉大矩阵、token embedding 使用 signed-block128 旋转、两级 E8P Q4，系数 `[4,1]`，beam8 搜索和 FP16 输出行 scale。attention 输出投影与最终视觉 projector 保留 W8；第 3 个文本层（索引 2）的 Q/K/V 与 FFN 使用相同输入旋转基下的 W8，以保护短查询质量，增加约 24.1MB。patch convolution 和 norm 使用 FP16。运行时从单份 K128 压缩 bank 执行 A8 整数投影，FP32 累加；生产执行只依赖 Rust 和同包 TileLang cubin。

已有完成的 E8P checkpoint 可从同一 BF16 来源提升该层精度，复用其余压缩值：

```sh
uv run --locked --project tools/environments/cuda python -m orinfer_build.quantization.nemotron_embed \
  --source-dir artifacts/nemotron-source --upgrade-from artifacts/nemotron-converted \
  --output artifacts/nemotron-converted-quality
```

该操作验证原始配置、固定来源及已有分片校验和，按行块读取敏感矩阵，原子发布新的完整转换目录。

构建器验证文本 16 层、2048 hidden、24/8 GQA heads，以及视觉 24 层、1024 hidden 和 14×14 patch。RoPE、attention scaling、空间合并、GELU 和池化语义都固定到配置签名。整数投影使用真实权重离线测量多个行数，发布每种几何选中的动态实现。

默认图片最长边 1120，按 28 对齐，最多 6400 patch。`--max-patches` 可以提高执行包容量到 12100；前端仍使用 checkpoint 的 1120 默认值。图片特征在一次请求内按输入顺序绑定，每张图独立执行视觉注意力；图文 token 随后共同执行文本双向注意力。

## 使用

已量化部署模型提供于 [Hugging Face](https://huggingface.co/immiq/Nemotron-3.5-Embed-1B-Orinfer-E8P-Q4A8) 和 [ModelScope](https://modelscope.cn/models/iammiq/Nemotron-3.5-Embed-1B-Orinfer-E8P-Q4A8)。下载完整仓库可取得权重、前端资源及匹配的 SM87 原生执行包，保留 `cache` 目录结构：

```sh
hf download immiq/Nemotron-3.5-Embed-1B-Orinfer-E8P-Q4A8 \
  --local-dir artifacts/nemotron-build/model
target/release/orinfer validate-model artifacts/nemotron-build/model
```

```sh
target/release/orinfer serve artifacts/nemotron-build/model \
  --model nemotron-3.5-embed-1b --listen 127.0.0.1:8088 --cuda-graph full
```

```sh
curl http://127.0.0.1:8088/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-3.5-embed-1b","input_type":"query","input":["Orin 有多少内存？"]}'
```

文档使用 `input_type: "document"`，这是默认值。图片文档的单项输入为 `{"text":"可选正文","images":["data:image/png;base64,..."]}`；`images` 也接受 HTTP(S) URL，每项最多 8 张图。输入数组最多 128 项，按完整序列依次执行，返回顺序与输入一致。HTTP 队列有界，图片解码、准备数据、上下文和输出等待都有明确限制。

`dimensions` 可以截取向量前缀并重新归一化，`encoding_format` 支持 `float` 和小端 FP32 的 `base64`。超长输入返回错误，不进行静默截断。服务读取 `ORINFER_API_KEY` 时要求 Bearer 认证。

离线 JSON 请求复用同一协议、前端与执行器，图片还可以使用本地路径：

```sh
target/release/orinfer embed-model artifacts/nemotron-build/model requests.json --cuda-graph full
```

执行包沿用 schema 1 与原生 C ABI；`plan-model` 和 `validate-model` 支持检查嵌入模型，部署执行包使用 `package install`。

## 验证

```sh
bash tools/environments/cuda/run.sh python/orinfer_build/models/nemotron_embed/check_kernels.py artifacts/nemotron-kernels

ORINFER_REFERENCE_CHECKPOINT="$PWD/artifacts/nemotron-source" \
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/nemotron-build/model" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/nemotron_embed/check_model.py artifacts/nemotron-quality \
  --original "$PWD/artifacts/nemotron-source" \
  --model "$PWD/artifacts/nemotron-build/model" \
  --engine "$PWD/target/release/orinfer"
```

算子检查覆盖双向 GQA、非对齐尾部、RMSNorm、Pixtral axial RoPE、patch 合并和改变输入后的 Graph replay。完整请求检查与官方 BF16 比较中文、英文、图片、图文和多图向量及检索排序，并验证反向输入顺序、变长工作区复用和 Graph/direct 一致性。这些用例用于接入验收，完整检索质量需要在目标语料上评测。

模型权重遵循 [OpenMDW-1.1](https://openmdw.ai/)，Ministral 基座遵循 Apache-2.0；保留 checkpoint 的 README 和来源 revision。引擎及 kernel 实现遵循本仓库 LGPL-3.0-or-later。

同一请求内的多条输入可共享文本主干投影。各输入按 32 行对齐，保持独立的双向 attention、RoPE 位置和平均池化边界，填充不计入输入或池化；图片特征按 owner 写入各自的文本位置。Q/K/V 与 gate/up 使用无损拼接后的单份权重 bank。超过 packet 容量时按完整输入分包，输出顺序保持不变。新构建器的 `--context` 使用 512–32768 内的 32 倍数，旧执行包继续使用逐输入执行。

构建器用真实权重离线测量短 packet 和最大容量，按实际行数发布所选整数投影；在线直接读取选择表。文本 attention 合并共享 K/V 的查询头，使用有界异步读取，各输入保留独立的真实边界。旋转量化复用寄存器实现，权重保持单份压缩表示。完整请求性能使用[嵌入与重排基准工具](../tools/bench/README.md)对照相同输入的 vLLM BF16；模型包、行容量和各档加速比分别记录。

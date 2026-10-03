# Orin LLM

用于 Jetson AGX Orin 64GB 的文本推理引擎。在线运行时用 Rust，GPU kernel 用 TileLang，目标固定为 CUDA SM87。

当前支持 Qwen3.8-27B 的文本主干，checkpoint架构为 `Qwen3_5ForConditionalGeneration`：48层 Gated DeltaNet和16层 full attention。支持模型加载、分块prefill、连续decode、请求状态重置和greedy生成。

## 构建

主机需要aarch64 Linux、Rust、CUDA Driver和本机AOT模型产物。开发环境使用Rust1.98.1；workspace声明的最低版本为1.85。离线kernel编译需要TileLang0.1.13、PyTorch2.9.1、CUDA12.6及NVIDIA Docker。

```bash
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

## 实现

- `crates/orin-engine/`：CUDA Driver封装、manifest校验、权重加载、graph执行、KV/GDN/卷积状态和采样。
- `crates/orin-cli/`：命令行入口。
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

Prefill计时包含输入复制和同步，排除末位置head；decode排除首token，包含逐token复制和同步。当前未提供HTTP接口、任意输入尾部、prefix cache、并发调度或MTP。BF16/FP8量化质量评测尚待补充。

## 许可证

GNU LGPL version 3 or later（`LGPL-3.0-or-later`）。见[LICENSE](LICENSE)及其引用的GPLv3文本[COPYING](COPYING)。外部依赖与模型权重遵循各自许可证。

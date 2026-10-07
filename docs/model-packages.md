# 模型执行包

在线引擎通过版本化 C ABI 加载模型执行逻辑。每个架构族、计算策略独立实现并发布一个包，包含原生 `.so`、匹配的 cubin 和配置/布局契约；权重仍在模型目录中。A8/A4 是计算策略，包应声明实际混合精度，不代表所有算子精度相同。

核心负责 API、调度、CUDA context/stream、内存、graph 和 prefix 索引；包负责配置校验、模型执行顺序、融合、动态 batch、状态绑定和视觉位置。增加模型族不需要在核心添加枚举或分派分支。当前在线包为 `orinfer-qwen3_5-a8`；Flash Next 保持现有离线验证路径。

```text
MODEL_DIR/
  config.json, tokenizer.json, chat_template.jinja, ...
  cache/model.json
  cache/weights/*.safetensors
  cache/packages/<sha256>/
    package.json
    lib/model.so
    kernels/*.cubin
    LICENSE
```

`model.json` schema 1 的 `execution_package` 固定整个包。共享缓存位于 `~/.cache/orinfer/packages`，可通过 `ORINFER_EXECUTION_CACHE` 覆盖。包 schema 1 / runtime ABI 1 描述目标 SM87、架构与计算策略、支持的配置、buffer 契约、kernel ABI 和 `.so` 身份。包自身通过 `describe` 返回身份及能力，运行时与 manifest 交叉检查。包版本可独立于核心版本；破坏协议兼容性的变化必须更新 ABI。

`.so` 使用 `orinfer-model-sdk`，不依赖 `orinfer-engine`。入口 `orinfer_model_v1` 返回 C 函数表，包含 describe/create/batch/visual 和对应资源释放函数。完整 C 声明位于 [`orinfer_model.h`](../crates/orinfer-model-sdk/include/orinfer_model.h)。Rust 数据结构及 trait 不跨动态库边界。

创建时用 JSON 传递配置及数据契约；热路径的 batch 程序使用 C 数组，一次返回 kernel/copy/zero 指令、buffer views、动态 launch 参数和请求状态绑定。原生内存由分配方释放，库保持加载直到模型及 CUDA 资源释放完成。模型实例与 GPU 执行线程绑定。

包不创建独立 CUDA context 或隐藏 stream，不直接管理引擎的显存。核心执行明确的程序；CUDA graph 命中时不再跨包重建程序。状态绑定缓存最多 16 个槽位/段长组合，仅缓存逻辑名称，GPU 地址仍从当前请求 arena 解析。prefix 身份包含包 digest；不同执行策略和状态布局不会复用旧快照。

## 构建与更新

模型实现保存在独立仓库 [orinfer-qwen3_5-a8](https://github.com/iMMIQ/orinfer-qwen3_5-a8)。使用同版本 SDK 源码离线构建：

```bash
cd /path/to/orinfer-qwen3_5-a8
bash build.sh /path/to/orinfer/crates/orinfer-model-sdk
```

更新已准备模型的执行库时，发布到一个新目录：

```bash
cd /path/to/orinfer
PYTHONPATH=. .venv/bin/python -m tools.model.attach_execution \
  --model /path/to/prepared-model --output /path/to/new-model \
  --library /path/to/liborinfer_qwen3_5_a8.so \
  --package orinfer-qwen3_5-a8 --version 0.1.1
```

更新工具校验原执行库与 kernel 资产，复用权重和 cubin，替换 `.so` 后通过加载器校验计划，再原子发布。源目录不会修改。模型数据与包仅支持当前 schema 1 结构，不读取旧算子包或 schema 2。

新准备的 Qwen 模型通过 `ORINFER_MODEL_LIBRARY=/path/to/library.so` 指定原生执行库；默认查找本项目 `target/release/liborinfer_qwen3_5_a8.so`。发布和安装工具：

```bash
PYTHONPATH=. .venv/bin/python tools/model/package.py archive /path/to/package /path/to/package.tar.gz
PYTHONPATH=. .venv/bin/python tools/model/package.py install /path/to/package.tar.gz ~/.cache/orinfer/packages
orinfer plan-model /path/to/new-model
orinfer validate-model /path/to/new-model
```

安装校验 `.so` 和 kernel 资产后原子写入缓存。`plan-model` 会在 CPU 加载原生执行库并生成计划；`validate-model` 还校验权重和全部 kernel 文件，均不初始化 GPU。

## 验证

主引擎执行 `make check`，包含 C ABI 跨语言加载、错误拒绝和更新原子性测试。模型仓库独立执行格式、clippy 和单元测试。更新还应比较完整程序、launch 参数、buffer 与状态契约；GPU 验证需要匹配模型权重，覆盖请求状态、graph 输入变化、prefix 恢复及 MTP 分支。

通用加载链的 GPU 夹具可独立运行：

```bash
bash tools/operators/run.sh tools/model/validate_model_package_gpu.py artifacts/model-package/check01
```

夹具覆盖三种 graph 模式、输入变化和完整 logits，不代替真实模型质量或性能验收。

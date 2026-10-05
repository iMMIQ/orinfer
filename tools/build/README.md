# 构建、检查与发布

在线服务只依赖 Rust 二进制、CUDA 驱动、准备好的模型目录和算子包。Python、PyTorch、TileLang 均属于离线环境。

## 编译环境

```bash
make compiler-image
make check-offline
```

`compiler.Dockerfile`使用公开 NVIDIA Jetson CUDA 12.6/PyTorch 2.9.1 镜像的固定 digest，安装指定版本的 TileLang 0.1.13 编译栈，不复制模型或编译缓存。`compiler-requirements.txt`固定覆盖包版本，其余依赖由基底镜像 digest 固定。默认镜像名为`orin-llm-compiler:0.1.0`，可用`ORIN_OPERATOR_IMAGE`覆盖。

GPU编译入口：

```bash
ORIN_CHECKPOINT_DIR=/path/to/checkpoint \
  bash tools/operators/run.sh tools/model/build.py artifacts/build/text512 \
  --checkpoint /path/to/checkpoint --dense-u4 --prefill-w4a8
```

GPU入口使用调用者UID/GID及补充设备组，避免生成root所有的模型/cache。仓库与输出目录按当前绝对路径挂载；外部checkpoint通过`ORIN_CHECKPOINT_DIR`只读挂载。完整计划与后续组装参数见[模型构建说明](../model/README.md)。此构建器只接受已支持的Qwen3_5 27B asymmetric compressed-tensors W4/group128，包括标准safetensors分片索引；不接受BF16、AWQ qweight或其他架构。检查checkpoint、计算来源hash后才编译，复用权重必须具有相同来源身份。

新环境的最小GPU编译验收无需checkpoint：

```bash
bash tools/operators/run.sh tools/build/smoke.py artifacts/build/compiler-smoke
```

它从空缓存编译实际生产norm，检查非对齐尾部、改变输入的graph replay并导出AOT。

## 自动检查

`make check`包含格式、编译、clippy、Rust/CPU Python测试和验收配置校验；GitHub Actions执行同一入口。依赖安装：`python3 -m pip install -r tools/build/cpu-requirements.txt`；Rust版本由`rust-toolchain.toml`固定。

`make check-offline`在编译镜像中执行需要Torch的CPU FP8/BF16 MTP导入检查，不使用GPU。

```bash
make check-gpu MODEL=/path/to/prepared-model OUTPUT=artifacts/check-gpu/new-run
```

该入口生成原生Chat模板的文本、图片、多图夹具，固定seed20261002，依次验证off/decode_only/full模式的动态batch、取消/重排/复用隔离、请求丢弃清理和同历史概率。原串行计划只在测试构建中作为独立参考存在，生产整请求生成和服务使用同一状态机。此检查不替代独立BF16质量验收。

GPU入口获取`artifacts/gpu-experiment.lock`；锁被占用时退出。它不会停止服务、调整功耗/频率或清理全局缓存；先释放自己部署的空闲服务，保留其他服务。

## 发布

提交源码、更新workspace版本并建立指向该提交的annotated tag后，运行：

```bash
python3 tools/release/package.py --version v0.1.1 \
  --operators /path/to/operators.tar.gz --output artifacts/releases/v0.1.1
```

默认仅生成和校验产物。工具要求工作区干净，使用`Cargo.lock`构建，验证aarch64/版本，打包完整源码、许可证、依赖声明、模型打包工具和架构契约，并验证独立算子包。权重不进入发行包。

实际发布时添加`--publish --notes /path/to/release-notes.md`。工具核对tag与源码revision，推送tag，创建GitHub Release，再重新下载并核对所有上传资产的SHA256。已有release不会覆盖。

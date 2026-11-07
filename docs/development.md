# 开发目录与产物管理

Rust workspace 负责在线推理。`crates/orinfer-engine/src/models` 保存模型专用 provider，
`backend` 保存设备资源、Graph、通信和传输。prefix、调度及 MTP 策略继续复用共享组件。
DSv4.1 位于 `models/deepseek_v41`，原 `backend::ascend::model` 诊断路径保留兼容导出。
CANN 通信适配器由引擎的 `native/ascend` 构建，保留嵌入库的 glibc 2.34 上限检查。

## 离线源码

| 目录 | 职责 |
|---|---|
| `kernels/cuda` | CUDA 生产 TileLang 算子 |
| `kernels/ascend` | Ascend 生产 TileLang 算子及第三方许可证 |
| `kernels/shared` | 两个前端共用的动态行契约 |
| `python/orinfer_build/contracts` | 几何、布局、host ABI 与绑定契约 |
| `python/orinfer_build/compiler` | AOT、原生绑定生成、后端编译与缓存 |
| `python/orinfer_build/models` | 模型组装与专用离线流程 |
| `python/orinfer_build/tuning` | 候选选择和性能测量 |
| `python/orinfer_build/packaging` | 前端资源、执行包和原子发布 |
| `python/orinfer_build/common` | 路径、文件身份、锁和测试发现 |
| `tests/cuda`、`tests/ascend` | 设备验证程序 |
| `tests/support` | 验证夹具 |
| `tools` | 发布、压测、开发命令与环境脚本 |

`make python-env` 安装可编辑的 `orinfer_build` 包及 CPU 依赖。CUDA 独立环境继续由
`tools/environments/cuda` 管理，并引用同一源码包；CANN 9.2 编译前端由
`tools/environments/ascend/run.sh` 管理。环境检查按后端选择模块。

模型构建、量化和算子验证使用 `python -m orinfer_build...` 或 `python -m tests...`。
旧 `tools/model`、`tools/operators` 和 `tools/quantization` 委托入口已移除。
历史执行契约中的 Python factory 模块名仍由离线导入器映射，映射不进入 Rust 在线推理。

内部组装直接调用 transform/build 函数，投影调优接收显式的 `Options`；CLI 只解析参数。
Ascend Graph 测量位于 `tuning/ascend.py`。本地权重读取复用 `weights/local.py` 的持久
文件句柄、`pread` 和公共 safetensors 元数据契约，模型适配器负责物理布局与数值解码。
执行包定位复用 `packaging.publication.resolve_package`，与 Rust 一样优先使用已安装缓存，
缺失时读取模型自带包；存在但损坏的缓存报错。

## 工作区和数据

将开发 worktree 放在仓库外。权重、执行缓存、实验记录和发布包按生命周期分开。
例如使用 `/data/worktrees/orinfer` 保存 worktree，以 `/data/orinfer` 保存工作数据：

```bash
git worktree add /data/worktrees/orinfer/my-task -b codex/my-task
cd /data/worktrees/orinfer/my-task
make python-env
uv run --locked python -m tools.dev.workspace init \
  --work-root /data/orinfer --name my-task
uv run --locked python -m tools.dev.workspace inventory
```

初始化创建忽略的 `artifacts/layout.json` 和目录链接：`builds` 按工作区隔离，`cache`
保存可重建缓存，`runs` 保存实验记录，`releases` 保存冻结发布产物，`locks` 保存设备锁。
`target` 指向该工作区自己的 Cargo 目录。已有数据或不匹配的链接使初始化报错，
工具不自动覆盖或删除。

`ORINFER_WORK_ROOT` 可覆盖工作数据根目录，`ORINFER_GPU_LOCK` 可显式指定锁。
已有 `artifacts/gpu-experiment.lock` 链接继续解析到原锁文件，保留同一个 inode。
设备工作仍须串行，原有服务以及频率、功耗和缓存设置保持原约束。

历史路径可保留到新目录的链接。清理前确认发布清单、冻结基线与 worktree 引用，
工作区通过 `git worktree remove` 管理。`workspace` 命令只初始化和盘点，不执行清理。
实验记录引用独立的不可变权重和执行包。

## 检查与缓存

Rust 工具链和 workspace 最低版本统一为稳定版 1.99.1。依赖更新要求发布满 7 天，
覆盖 Cargo 直接与传递依赖、两个 uv 环境的实际锁定文件、构建依赖、CI actions/uv、
Jetson PyTorch wheel 和 Ascend TileLang wheel。uv 通过 `exclude-newer = "7 days"`
在解析时过滤新文件；Rust 精确版本及完整锁文件的发布时间记录在
`configs/dependency-releases.json`，由官方 registry、Rust 发布元数据和 GitHub Release 生成。

更新 Cargo 精确版本与锁文件后，运行以下流程：

```bash
uv lock --upgrade
uv lock --upgrade --project tools/environments/cuda
python3 -m tools.dev.dependencies refresh
make python-env
cargo fetch --locked
make check
```

`refresh` 只读取发布元数据，拒绝未满 7 天、缺少日期、被撤回或来源不匹配的版本。
Cargo 的版本解析不会执行依赖代码；锁定版本通过发布时间审计后再下载和构建。
被拒绝的传递版本可用 `cargo update -p NAME --precise VERSION` 选择已满等待期的版本，再重新审计。
`make python-env`、`make cuda-env`、`make build`、`make check` 和发行构建先离线核对
锁文件及依赖声明与发布时间记录。输入变动后须重新审计；常规检查不联网。
设备绑定的 wheel、CANN 9.2 和已有算子数值/ABI 契约保持显式固定，升级后按实际契约验证。

`make check` 执行统一 CPU 检查。Python 单测按 `test_*.py` 自动发现，默认进入 CPU
组；需要离线 CUDA 环境的文件声明 `TEST_ENVIRONMENT = "cuda"`。`make check-offline`
执行 CUDA 离线检查，`make check-ascend-offline` 在无设备容器中执行 Ascend 导入检查。
Rust 单测保留在各 crate 内，设备测试继续显式选择夹具执行。

候选改动先运行相关检查，提交前统一运行一次 `make check`。纯路径迁移验证导入、
构建、资源与打包，沿用未改变契约的数值和状态验收；执行契约变化再补对应设备短测。

离线缓存身份包含生产组件及其导入依赖、编译器和契约。内联组件进入依赖身份，
无关测试和其他模型的 Python 源码不会仅因相邻目录而使缓存失效。模型库保留 Rust、
SDK 和锁文件身份。正常在线加载不增加内容 SHA256 扫描；完整校验继续由安装器、
`validate-model` 和显式 `--verify-weights` 负责。

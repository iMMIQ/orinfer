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
| `tools` | 发布、压测、开发命令与原构建命令的薄入口 |

`make python-env` 安装可编辑的 `orinfer_build` 包及 CPU 依赖。CUDA 独立环境继续由
`tools/environments/cuda` 管理，并引用同一源码包；CANN 9.2 编译前端由
`tools/environments/ascend/run.sh` 管理。环境检查按后端选择模块。

原模型构建和量化命令保留委托入口，内部调用使用 `orinfer_build`。历史执行契约中的
Python factory 名由离线导入器映射到新模块，映射不进入 Rust 在线推理。

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

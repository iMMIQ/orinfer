# 工程约定

本文件适用于整个仓库；用户明确指令优先。

## 范围

- 硬件固定本机Jetson AGX Orin 64GB，aarch64/CUDA SM87。
- 支持Qwen3.8-27B文本、图片和多图输入，实际checkpoint为Qwen3_5架构；不包含视频。
- 在线运行时用Rust，生产GPU kernel以TileLang为源；Python只用于离线转换、编译、参考和验证。
- 公共运行时按manifest执行，不硬编码模型层数、维度或层序；后续架构由adapter和执行计划扩展。
- 当前性能优化已结束。新增优化、prefix、并发、MTP或其他模型须以新的用户任务为依据。
- 项目许可证为LGPL-3.0-or-later，保留第三方版权和许可证声明。

## 实现

- 按真实checkpoint配置核对GDN、GQA、RoPE、zero-centered RMSNorm、embedding和head语义。
- GDN持续状态为FP32；KV、卷积、GDN和位置状态均属于请求私有状态。
- 改变prefill/decode、恢复或分支行为时校验全部状态，不只校验KV。
- 单份常驻W4；scale、zero、LUT、padding和持久重复表示计入权重平均bits。临时W8不作永久权重缓存。
- 在线模型采用HF配置/tokenizer目录和`cache/manifest.json`（schema 2）；初始化tensor只从标准分片safetensors读取，按HF index定位并校验payload hash、dtype、shape与物理layout。旧裸权重模型仅作为离线中间产物，经`tools/model/prepare.py`打包；在线不提供旧模型加载分支。算子夹具格式独立。
- Rust按实际导出host ABI绑定参数、grid/block/shared；检查CUDA返回值、unsafe边界、资源生存期和stream顺序。
- CUDA graph需要稳定地址和预分配workspace；改输入后replay验证实际执行。
- Cooperative launch需显式声明并校验occupancy，不假定普通launch可跨block同步。
- 性能以完整链及真实模型为准。合成探针、微测外推和四输出profile不能替代完整请求TPS。
- 质量对照官方BF16或同模型FP8，固定seed20261002及teacher-forced历史；token不完全一致不是自动失败。任务结果、target NLL和长上下文另行判断。

## 开发

- 先用rg检查调用者和依赖；保持修改小而完整。不要默认启动subagent。
- `make check`执行Rust格式/编译/clippy/测试及CPU评测检查；GPU验证用`bash tools/operators/run.sh`。
- GPU工作通过`artifacts/gpu-experiment.lock`串行，保留其他服务；不自行改全局频率、功耗或清缓存。
- 新kernel验证数值、非对齐尾部和实际graph replay；状态改动另测分块、恢复和请求隔离。
- 源码库保留实现、必要构建工具、测试、示例与使用文档。不提交研究过程、agent交付记录、历史实验报告或本机环境证据。
- 权重、二进制、cubin、cache、日志及trace保留在忽略的artifacts/target目录；不提交密钥或token。

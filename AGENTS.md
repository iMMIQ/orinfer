# 工程约定

本文件适用于整个仓库；用户明确指令优先。

## 范围

- 硬件固定本机Jetson AGX Orin 64GB，aarch64/CUDA SM87。
- 支持Qwen3.8-27B文本、图片和多图输入，实际checkpoint为Qwen3_5架构；不包含视频。
- 在线运行时用Rust，生产GPU kernel以TileLang为源；Python只用于离线转换、编译、参考和验证。
- 架构通过Rust模块注册，按checkpoint配置实例化执行顺序；算子包提供实现/ABI/布局契约，不携带执行程序。公共CUDA执行器不硬编码模型层数、维度或层序。
- 第一阶段包含prefix cache与多请求batching；缓存恢复和batch调度必须保持完整KV/GDN/卷积/MTP状态及请求隔离。
- 项目许可证为LGPL-3.0-or-later，保留第三方版权和许可证声明。

## 实现

- 按真实checkpoint配置核对GDN、GQA、RoPE、zero-centered RMSNorm、embedding和head语义。
- GDN持续状态为FP32；KV、卷积、GDN和位置状态均属于请求私有状态。
- 改变prefill/decode、恢复或分支行为时校验全部状态，不只校验KV。
- 单份常驻W4；scale、zero、LUT、padding和持久重复表示计入权重平均bits。临时W8不作永久权重缓存。
- Flash Next扩展采用Q2A8和质量优先的混合精度；其QSA长上下文KV使用group-64对称INT8及FP16 scale，压缩索引与未完成块也纳入请求私有状态；Q2专家直接索引常驻权重库，不按步复制权重或缓存整份展开W8。HC保留BF16权重；PLE表使用受预算限制的CPU缓存，其n-gram历史与卷积历史必须纳入请求状态及prefix恢复。
- Flash Next使用我们自己的整数E8P＋旋转量化表示，运行时反量化到INT8计算，不沿用社区Q2量化编码。当前优先完成算子和真实端到端推理，允许不校准或小样本校准；原始BF16按矩阵或专家分块读取，不需要保存完整BF16模型。完整校准需单独验证专家覆盖和层间误差，不能把反量化后的社区Q2称为原始BF16基线。
- 在线模型采用HF配置/tokenizer目录和`cache/model.json`数据描述；标准分片safetensors保留当前W4表示，校验payload hash、dtype、shape与物理layout。架构自动匹配独立算子包，第一阶段采用INT8为主、质量优先的混合精度，关键部分可FP16/FP32；不实现force_int8或其他格式导入。离线AOT manifest只作构建中间产物，不提供旧在线格式兼容分支。算子夹具格式独立。
- 服务协议位于orinfer-api；CLI只处理命令。CUDA执行器、加载器、架构计划、生成/视觉/MTP控制分离；权重、序列状态和workspace分作用域管理。
- Rust按实际导出host ABI绑定参数、grid/block/shared；检查CUDA返回值、unsafe边界、资源生存期和stream顺序。
- CUDA graph需要稳定地址和预分配workspace；改输入后replay验证实际执行。
- Cooperative launch需显式声明并校验occupancy，不假定普通launch可跨block同步。
- 性能以完整链及真实模型为准。合成探针、微测外推和四输出profile不能替代完整请求TPS。
- 质量对照官方BF16或同模型FP8，固定seed20261002及teacher-forced历史；token不完全一致不是自动失败。任务结果、target NLL和长上下文另行判断。

## 开发

- 先用rg检查调用者和依赖；保持修改小而完整。不要默认启动subagent。
- CPU Python依赖用uv管理项目`.venv`，通过`make python-env`同步固定版本，不直接更新主机Python环境。`make check`执行Rust格式/编译/clippy/测试及CPU评测检查；GPU验证用`bash tools/operators/run.sh`的固定编译镜像。
- GPU工作通过`artifacts/gpu-experiment.lock`串行，保留其他服务；不自行改全局频率、功耗或清缓存。
- 新kernel验证数值、非对齐尾部和实际graph replay；状态改动另测分块、恢复和请求隔离。
- 源码库保留实现、必要构建工具、测试、示例与使用文档。不提交研究过程、agent交付记录、历史实验报告或本机环境证据。
- 权重、二进制、cubin、cache、日志及trace保留在忽略的artifacts/target目录；不提交密钥或token。

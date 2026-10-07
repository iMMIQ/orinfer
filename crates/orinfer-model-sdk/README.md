# orinfer-model-sdk

独立模型执行包的共享数据契约和 C ABI v1。包只依赖此 SDK，不依赖主引擎；Rust 实现可以导出 `cdylib`，其他语言实现可使用 [C 头文件](include/orinfer_model.h)。

配置与计划描述在创建阶段序列化；batch 使用 C 数组传递。内存由分配方释放，原生库保持加载到对象销毁。SDK 中的 Rust trait 只用于包内实现，不跨库传递。

协议和资源所有权见[模型执行包](../../docs/model-packages.md)。ABI 版本与 crate 版本分别管理；修改跨库结构或函数表必须更新 ABI，不能仅增加字段。

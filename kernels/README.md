# TileLang kernels

目标为CUDA SM87。`operators/`包含基础算子，`model/`包含模型使用的投影、融合、GDN和attention实现，`projections/`包含基础投影的共用实现。

kernel经离线编译生成cubin和真实host ABI，在线由Rust CUDA Driver执行。导出格式与模型组装入口见[模型工具](../tools/model/README.md)。新增实现应检查完整数值、非对齐尾部、状态语义和改变输入后的graph replay。

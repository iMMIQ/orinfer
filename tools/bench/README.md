# 性能工具

`reference_bench.py`测量兼容OpenAI SSE接口的prefill/TTFT与decode时间；`test_reference_bench.py`检查token chunk和计时间隔。

`run_aot_artifact.sh MANIFEST NEW_OUTPUT`构建Rust CLI并执行AOT fixture，保留实际运行身份和机器状态。模型完整推理使用`tools/model/run.sh`。

`sample_machine.py`和`sample_continuous.py`只读记录机器温度、频率、内存和功耗；不会改动全局设置。GPU运行入口共享`artifacts/gpu-experiment.lock`。

`validate_plan.py`校验benchmark与快速评测配置，属于CPU检查，不产生性能测量。

`concurrency.py`以真实SSE请求测量1..128并发，输入为`{ "cases": [{ "id": "case", "request": { ...Chat API body... } }] }`：

```bash
python3 tools/bench/concurrency.py --requests requests.json \
  --concurrency 1 2 4 8 16 32 128 --repetitions 3 --output artifacts/concurrency.json
```

报告包括真实usage输出token数、整轮吞吐、TTFT分位数、缓存tokens及每请求结果。吞吐包含排队、prefill、HTTP和输出，不能直接当作GPU纯decode TPS；流式chunk间隔也不能当作token ITL。重复请求会命中prefix cache；测冷请求时需构造不同输入或关闭cache。`/health.scheduler_statistics`提供prefill实际计算tokens、引擎提交的decode tokens、batch直方图、混合迭代及峰值活跃数。

健康接口还提供准入、request startup、prefix恢复、prefill完成、batch输入准备/计划/提交及Graph统计。`batch_execution`记录batch Graph命中、未命中、捕获的kernel/copy/memset操作数、淘汰、地址变化失效和分项耗时；单请求/MTP程序运行期间的新捕获另记`sequence_captures/sequence_capture_s`。这些是累计主机计时，replay包含同步等待，不能视为CUDA event的GPU时间。`request_start_s`包含`prefix_restore_s`，batch Graph分项包含在`compute_s`中；单程序捕获可能发生在startup或计算阶段，不能把这些重叠计时重复相加。

工具在发送请求前、整轮完成后取得健康快照，并输出`scheduler_delta`和`admission_delta`；累计峰值不作差值。测量期间应独占服务，重启导致计数器回退时会报错。`admission_statistics`记录各轮处理的请求数、完整缓存命中的文本请求数、准入直方图及墙钟时间；它包含策略、显存准入检查和request startup。

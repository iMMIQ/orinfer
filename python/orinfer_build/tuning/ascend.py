"""Shared Ascend Graph measurement for offline tuning and validation."""


def npu_benchmark(run, repetitions, warmup, *, calls_per_replay=1):
    """Time each call, amortizing host submission for independent projections."""
    import statistics
    import torch

    if type(calls_per_replay) is not int or not 1 <= calls_per_replay <= 64:
        raise ValueError("NPU replay call count must be in 1..64")
    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls_per_replay):
            run()
    for _ in range(2):
        graph.replay()
    torch.npu.synchronize()
    values = []
    for _ in range(repetitions):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / calls_per_replay)
    return dict(median_ms=statistics.median(values)), graph

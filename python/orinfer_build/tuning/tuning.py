"""Common offline candidate construction, cache records and Graph measurement."""

import hashlib
import json
import math


def record_path(root, identity):
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return root / (key + ".json")


def read_record(path, identity, validate=lambda record: True):
    try:
        record = json.loads(path.read_text())
        if record["identity"] == identity and validate(record):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def compile_candidates(configs, build, *, tolerate_failures=False):
    kernels, legal, rejected = [], [], []
    for config in configs:
        try:
            kernel = build(config)
        except Exception as error:
            if not tolerate_failures:
                raise
            rejected.append(dict(config=config, stage="compile", reason=str(error)))
            continue
        kernels.append(kernel)
        legal.append(config)
    if not kernels:
        raise ValueError(f"All tile candidates failed compilation: {rejected}")
    return kernels, legal, rejected


def measure(run, *, budget_ms, max_repetitions, change=None, check=None, benchmark=None):
    if benchmark is None:
        from orinfer_build.compiler.cuda.common import benchmark

    repetitions = 1
    if budget_ms:
        pilot, _ = benchmark(run, repetitions=1, warmup=1)
        repetitions = max(1, min(max_repetitions, math.ceil(budget_ms / pilot["median_ms"])))
    timing, graph = benchmark(run, repetitions=repetitions, warmup=0)
    if change is not None:
        change()
        try:
            graph.replay()
            check()
        finally:
            change()
    return timing["median_ms"]

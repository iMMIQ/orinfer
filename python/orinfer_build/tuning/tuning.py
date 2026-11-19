"""Common offline candidate construction, cache records and Graph measurement."""

import hashlib
import json
import math

from orinfer_build.common.files import atomic_json


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


def write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, record)


def geometric_mean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def lowest_total_latency(records, candidates):
    return min(candidates, key=lambda candidate: sum(row["times"][candidate] for row in records))


def retain_valid_timings(records, invalid):
    """A failed validation excludes a candidate at every measured row."""
    for record in records:
        record["times"] = {
            key: ms for key, ms in record["times"].items() if int(key) not in invalid
        }


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


def measure(
    run, *, budget_ms, max_repetitions, change=None, check=None, benchmark=None, check_initial=False
):
    if check_initial:
        run()
        check()

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

"""Replay fixed Chat API arrival times to measure scheduling under mixed load.

Input: {requests:[{id,at_s,request:Chat API request}]}. Arrivals are independent
of completions. Chunk gaps describe observable streaming pauses, not token ITL.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import threading
import time
import urllib.request

from tools.bench.concurrency import counter_delta, percentile, request


def validate_trace(trace):
    if not isinstance(trace, dict):
        raise ValueError('Trace must be an object')
    rows = trace.get('requests')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError('Require 1..128 trace requests')
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Trace request must be an object')
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('Require unique nonempty string request IDs')
        ids.add(identifier)
        arrival = row.get('at_s')
        if (isinstance(arrival, bool) or not isinstance(arrival, (int, float))
                or not math.isfinite(arrival) or arrival < 0):
            raise ValueError('Arrival time must be finite and nonnegative')
        if not isinstance(row.get('request'), dict):
            raise ValueError('Require a Chat request object')
    return rows


def summarize(rows, wall_s):
    passed = [r for r in rows if 'error' not in r]
    tokens = sum(r['usage']['completion_tokens'] for r in passed)
    metrics = dict(complete=len(passed) == len(rows), completed=len(passed),
                   failed=len(rows)-len(passed), wall_s=wall_s,
                   committed_output_tokens=tokens, committed_output_tps=tokens/wall_s)
    prompts = sum(r['usage'].get('prompt_tokens', 0) for r in passed)
    cached = [(r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0)
              for r in passed]
    metrics['prefix_cache'] = dict(prompt_tokens=prompts, cached_tokens=sum(cached),
        request_hit_rate=sum(n > 0 for n in cached)/len(passed) if passed else None,
        token_hit_rate=sum(cached)/prompts if prompts else None)
    for name in ('ttft_s', 'wall_s', 'chunk_gap_max_s', 'arrival_lateness_s'):
        values = [r[name] for r in passed if r.get(name) is not None]
        metrics[name.removesuffix('_s')] = {
            'p50_s': percentile(values, .5) if values else None,
            'p95_s': percentile(values, .95) if values else None,
            'p99_s': percentile(values, .99) if values else None,
            'max_s': max(values) if values else None,
        }
    return metrics


def replay(base_url, trace):
    rows = validate_trace(trace)
    health_url = base_url.rstrip('/').removesuffix('/v1') + '/health'
    def health():
        with urllib.request.urlopen(health_url, timeout=5) as response:
            return json.load(response)
    before = health()
    gate = threading.Event()
    epoch = [0.]
    def invoke(row):
        gate.wait()
        due = epoch[0] + row['at_s']
        time.sleep(max(0., due-time.perf_counter()))
        actual = time.perf_counter()
        result = dict(id=row['id'], at_s=row['at_s'],
                      arrival_lateness_s=max(0., actual-due))
        try:
            result.update(request(base_url, row['request'], collect_arrivals=True))
        except Exception as error:
            result.update(error=str(error), wall_s=time.perf_counter()-actual)
        return result
    with ThreadPoolExecutor(max_workers=len(rows)) as pool:
        futures = [pool.submit(invoke, row) for row in rows]
        epoch[0] = time.perf_counter() + .05
        gate.set()
        results = [future.result() for future in futures]
    wall_s = time.perf_counter()-epoch[0]
    after = health()
    return dict(scope='Fixed arrivals; HTTP wall time includes queueing and prefill. '
                     'Streaming chunk gaps are not token ITL.',
                metrics=summarize(results, wall_s), rows=results,
                scheduler_delta=counter_delta(before.get('scheduler_statistics', {}),
                                              after.get('scheduler_statistics', {})),
                admission_delta=counter_delta(before.get('admission_statistics', {}),
                                              after.get('admission_statistics', {})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8088/v1')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Require a fresh output file')
    report = replay(args.base_url, json.loads(args.requests.read_text()))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['metrics'], ensure_ascii=False))
    if not report['metrics']['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

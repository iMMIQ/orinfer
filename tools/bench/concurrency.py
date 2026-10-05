"""Measure real concurrent Chat API requests, including queueing and streaming.

Token counts come from usage. Content chunks are not counted as tokens: MTP and
UTF-8 decoding can put multiple tokens into a single event. Aggregate throughput
uses the whole round's wall time and must not be reported as GPU-only decode TPS.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request


def events(response):
    pending = []
    for raw in response:
        line = raw.decode().rstrip('\r\n')
        if not line and pending:
            yield '\n'.join(pending)
            pending = []
        elif line.startswith('data:'):
            pending.append(line[5:].lstrip())
    if pending:
        raise ValueError('Unterminated SSE event')


def request(base_url, body, barrier=None, *, on_content=None, collect_arrivals=False):
    headers = {'Content-Type': 'application/json'}
    if os.getenv('ORINFER_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['ORINFER_API_KEY']
    payload = dict(body, stream=True, stream_options={'include_usage': True})
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(), headers=headers)
    if barrier:
        barrier.wait(timeout=30)
    start = time.perf_counter()
    usage, finish, done = None, None, False
    arrivals, content, reasoning = [], [], []
    with urllib.request.urlopen(req, timeout=1800) as response:
        for event in events(response):
            at = time.perf_counter() - start
            if event == '[DONE]':
                done = True
                break
            value = json.loads(event)
            if value.get('error'):
                raise ValueError(value['error'])
            usage = value.get('usage') or usage
            for choice in value.get('choices', []):
                delta = choice.get('delta', {})
                if delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'):
                    arrivals.append(at)
                    if on_content is not None:
                        on_content(at)
                content.append(delta.get('content', ''))
                reasoning.append(delta.get('reasoning_content', ''))
                finish = choice.get('finish_reason') or finish
    if not done or not usage or not finish:
        raise ValueError('Incomplete stream or missing usage')
    gaps = [b-a for a, b in zip(arrivals, arrivals[1:])]
    result = dict(usage=usage, finish_reason=finish, content=''.join(content),
        reasoning=''.join(reasoning), ttft_s=arrivals[0] if arrivals else None,
        wall_s=time.perf_counter()-start,
        chunk_gap_p50_s=statistics.median(gaps) if gaps else None,
        chunk_gap_max_s=max(gaps) if gaps else None)
    if collect_arrivals:
        result.update(started_monotonic_s=start, arrivals_s=arrivals)
    return result


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values)-1, max(0, math.ceil(len(values)*fraction)-1))]


def counter_delta(before, after):
    """Differences of cumulative counters, including nested graph/histogram data."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    out = {}
    for key in before.keys() | after.keys():
        a, b = before.get(key, 0), after.get(key, 0)
        if isinstance(b, dict):
            out[key] = counter_delta(a if isinstance(a, dict) else {}, b)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if key.startswith('peak_') or key.startswith('max_'):
                continue  # Cumulative maxima cannot be subtracted per round.
            if b < a:
                raise ValueError('Service counters reset during the benchmark')
            out[key] = b-a
    return out


def run_round(base_url, cases, count):
    barrier = threading.Barrier(count)
    stop = threading.Event()
    samples = []
    url = base_url.rstrip('/').removesuffix('/v1') + '/health'
    def health():
        with urllib.request.urlopen(url, timeout=2) as response:
            return json.load(response)
    before = health()
    def monitor():
        while not stop.is_set():
            try:
                samples.append(health())
            except (OSError, ValueError):
                pass
            stop.wait(.1)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(request, base_url, cases[i % len(cases)]['request'], barrier)
                       for i in range(count)]
            rows = [future.result() for future in futures]
        wall = time.perf_counter()-start
    finally:
        stop.set()
        thread.join(timeout=2)
    after = health()
    output = sum(row['usage']['completion_tokens'] for row in rows)
    prompt = sum(row['usage']['prompt_tokens'] for row in rows)
    cached = sum(row['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
                 for row in rows)
    ttft = [row['ttft_s'] for row in rows if row['ttft_s'] is not None]
    return dict(concurrency=count, wall_s=wall, committed_output_tokens=output,
        committed_output_tps=output/wall, prompt_tokens=prompt, cached_tokens=cached,
        observed_peak_active=max((s.get('active_requests',0) for s in samples),default=None),
        observed_peak_queued=max((s.get('queued_requests',0) for s in samples),default=None),
        scheduler_first=before.get('scheduler_statistics'),
        scheduler_last=after.get('scheduler_statistics'),
        scheduler_delta=counter_delta(before.get('scheduler_statistics'), after.get('scheduler_statistics')),
        admission_first=before.get('admission_statistics'),
        admission_last=after.get('admission_statistics'),
        admission_delta=counter_delta(before.get('admission_statistics'), after.get('admission_statistics')),
        ttft_p50_s=percentile(ttft, .5) if ttft else None,
        ttft_p95_s=percentile(ttft, .95) if ttft else None, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True,
                        help='JSON {cases:[{id,request:OpenAI Chat request}]}')
    parser.add_argument('--base-url', default='http://127.0.0.1:8088/v1')
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1,2,4,8,16,32,128])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1 or any(n < 1 or n > 128 for n in args.concurrency):
        parser.error('Require a fresh output, positive repetitions and concurrency 1..128')
    cases = json.loads(args.requests.read_text())['cases']
    if not cases:
        parser.error('At least one request case is required')
    rows = []
    report = dict(scope='Real committed output tokens / round wall time, including queueing, prefill, HTTP and streaming. Chunk gaps are not token ITL.', rows=rows)
    for count in args.concurrency:
        for repetition in range(args.repetitions):
            row = run_round(args.base_url, cases, count)
            row['repetition'] = repetition
            rows.append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(count, repetition, round(row['committed_output_tps'], 3), 'output TPS', flush=True)


if __name__ == '__main__':
    main()

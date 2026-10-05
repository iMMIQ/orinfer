"""Inject a cold Chat request into warm, actively streaming decoders.

Usage supplies committed token counts. Arrival gaps measure SSE fragments,
not token ITL, because UTF-8 decoding and MTP can group several tokens.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request

from tools.bench.concurrency import counter_delta, percentile, request


def overlapping_gaps(row, begin, end):
    times = [row['started_monotonic_s'] + t for t in row['arrivals_s']]
    return [b-a for a, b in zip(times, times[1:]) if a < end and b > begin]


def run_round(base_url, anchor, cold, count, trigger_chunks):
    url = base_url.rstrip('/').removesuffix('/v1') + '/health'

    def health():
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.load(response)

    before = health()
    if before.get('active_requests') or before.get('queued_requests'):
        raise ValueError('Require an idle service before each round')
    ready = threading.Event()
    guard = threading.Lock()
    chunks = [0] * count
    samples = []
    stop = threading.Event()

    def observe(index):
        def callback(_elapsed):
            with guard:
                chunks[index] += 1
                if all(n >= trigger_chunks for n in chunks):
                    ready.set()
        return callback

    def monitor():
        while not stop.is_set():
            try:
                samples.append(health())
            except (OSError, ValueError):
                pass
            stop.wait(.1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    origin = time.perf_counter()
    injection = None
    injected_health = None
    cold_row = None
    try:
        with ThreadPoolExecutor(max_workers=count+1) as pool:
            futures = [pool.submit(request, base_url, anchor,
                                   on_content=observe(i), collect_arrivals=True)
                       for i in range(count)]
            if cold is not None:
                deadline = time.monotonic()+60
                while not ready.wait(.1):
                    if any(f.done() for f in futures):
                        raise ValueError('An anchor finished before the injection trigger')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Anchors did not begin sustained streaming')
                if any(f.done() for f in futures):
                    raise ValueError('Require live decoders when injecting the cold request')
                injected_health = health()
                injection = time.perf_counter()
                cold_future = pool.submit(request, base_url, cold['request'],
                                          collect_arrivals=True)
            rows = [f.result() for f in futures]
            if cold is not None:
                cold_row = cold_future.result()
                if cold_row['ttft_s'] is None:
                    raise ValueError('Cold request produced no visible content')
                expected = cold.get('expected_prompt_tokens')
                if expected is not None and cold_row['usage']['prompt_tokens'] != expected:
                    raise ValueError('Online prompt length differs from the prepared fixture')
                target = cold.get('target_text')
                if target is not None:
                    cold_row['task_passed'] = cold_row['content'].strip('`\"\' \n') == target
                cached = cold_row['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
                if cached > cold_row['usage']['prompt_tokens'] * .05:
                    raise ValueError('Cold input reused more than five percent of its prefix')
        wall = time.perf_counter()-origin
        deadline = time.monotonic()+10
        after = health()
        while after.get('active_requests') or after.get('queued_requests'):
            if time.monotonic() >= deadline:
                raise ValueError('Service did not release completed requests')
            time.sleep(.05)
            after = health()
    finally:
        stop.set()
        thread.join(timeout=6)
    all_rows = rows + ([cold_row] if cold_row is not None else [])
    admission = counter_delta(before.get('admission_statistics'), after.get('admission_statistics'))
    if admission and admission.get('requests') != len(all_rows):
        raise ValueError('Unexpected service traffic during the benchmark')
    for row in rows:
        cached = row['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if cached != row['usage']['prompt_tokens']:
            raise ValueError('Warm anchor did not fully reuse its prompt')
    begin = injection if injection is not None else origin
    end = (cold_row['started_monotonic_s']+cold_row['ttft_s']
           if cold_row is not None else origin+wall)
    gaps = [gap for row in rows for gap in overlapping_gaps(row, begin, end)]
    anchor_end = max(row['started_monotonic_s']+row['wall_s'] for row in rows)
    return dict(anchor_count=count, cold_case=cold['id'] if cold is not None else None,
                wall_s=wall, origin_monotonic_s=origin,
                injection_elapsed_s=injection-origin if injection is not None else None,
                anchor_output_tokens=sum(r['usage']['completion_tokens'] for r in rows),
                anchor_output_tps=sum(r['usage']['completion_tokens'] for r in rows)/(anchor_end-origin),
                committed_output_tps=sum(r['usage']['completion_tokens'] for r in all_rows)/wall,
                anchor_overlap_chunk_gaps=len(gaps),
                anchor_overlap_chunk_gap_p95_s=percentile(gaps, .95) if gaps else None,
                anchor_overlap_chunk_gap_max_s=max(gaps) if gaps else None,
                scheduler_delta=counter_delta(before.get('scheduler_statistics'), after.get('scheduler_statistics')),
                admission_delta=admission, injected_health=injected_health,
                observed_peak_active=max((s['active_requests'] for s in samples), default=None),
                anchors=rows, cold=cold_row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8088/v1')
    parser.add_argument('--anchor-counts', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--trigger-chunks', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.trigger_chunks < 1 or any(n < 1 or n > 127 for n in args.anchor_counts):
        parser.error('Require a fresh output, positive trigger and anchor counts in 1..127')
    spec = json.loads(args.requests.read_text())
    anchor = spec['anchor']
    if any(body.get('temperature') != 0 or body.get('seed') != 20261002
           for body in [anchor]+[c['request'] for c in spec['cold_cases']]):
        parser.error('Require greedy requests with seed20261002')
    warm = request(args.base_url, dict(anchor, max_tokens=1))
    report = dict(status='running', seed=20261002, warmup=warm, rows=[],
                  scope='Real API with configured adaptive MTP; usage output counts / wall time. SSE fragment gaps are not token ITL. All anchors must be fully prefix-cached, cold prefix reuse <=5%.')
    for count in args.anchor_counts:
        cold_cases = [case for case in spec['cold_cases']
                      if case.get('anchor_count', count) == count]
        if not cold_cases:
            parser.error(f'Provide a fresh cold case for anchor count {count}')
        for cold in [None]+cold_cases:
            row = run_round(args.base_url, anchor, cold, count, args.trigger_chunks)
            report['rows'].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(count, row['cold_case'], round(row['anchor_output_tps'], 3),
                  'anchor output TPS', row['anchor_overlap_chunk_gap_max_s'],
                  'max SSE gap', flush=True)
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

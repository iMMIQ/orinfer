"""Run bounded heterogeneous Chat workloads with arrivals and task checks.

Input contains cases {id, request, expected_words} and rounds {id, jobs}.
Each job names a case and optional delay_s. This is functional acceptance;
usage throughput includes HTTP, queueing, prefill and committed generation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

from tools.bench.concurrency import counter_delta, request


def validate(spec):
    cases = {case['id']: case for case in spec['cases']}
    if not cases or len(cases) != len(spec['cases']) or not spec['rounds']:
        raise ValueError('Require unique cases and nonempty rounds')
    for case in cases.values():
        if case['request'].get('seed') != 20261002 or case['request'].get('temperature') != 0:
            raise ValueError('Require greedy requests and fixed seed20261002')
        if not case.get('expected_words'):
            raise ValueError('Each task needs an explicit expected answer')
    for group in spec['rounds']:
        if not 1 <= len(group['jobs']) <= 128:
            raise ValueError('Require 1..128 submitted jobs')
        for job in group['jobs']:
            if job['case'] not in cases or not 0 <= job.get('delay_s', 0) <= 60:
                raise ValueError('Unknown case or invalid arrival delay')
    return cases


def check(row, case):
    # Only explicit label/short-answer tasks use case/whitespace normalization.
    actual = re.findall(r'[\w]+', row['content'].lower())
    expected = re.findall(r'[\w]+', ' '.join(case['expected_words']).lower())
    if actual != expected:
        raise ValueError(f"{case['id']}: expected {expected}, got {row['content']!r}")
    row['task_passed'] = True
    return row


def run(base, spec, output):
    cases = validate(spec)
    health_url = base.rstrip('/').removesuffix('/v1') + '/health'
    def health():
        with urllib.request.urlopen(health_url, timeout=10) as response:
            return json.load(response)
    report = dict(status='running', seed=20261002, warmups=[], rounds=[],
                  scope='Task acceptance; committed usage counts / client wall time; SSE chunks are not token ITL.')
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    before = health()
    if before['active_requests'] or before['queued_requests']:
        raise ValueError('Require an idle service')
    for name in spec.get('warmup', []):
        report['warmups'].append(dict(id=name, **check(request(base, cases[name]['request']), cases[name])))
        save()
    for group in spec['rounds']:
        before = health()
        barrier = threading.Barrier(len(group['jobs']))
        stop = threading.Event()
        samples = []
        def monitor():
            while not stop.is_set():
                samples.append(health())
                stop.wait(.2)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        origin = time.perf_counter()
        def job(item):
            barrier.wait(timeout=30)
            time.sleep(item.get('delay_s', 0))
            case = cases[item['case']]
            row = check(request(base, case['request']), case)
            return dict(id=item['case'], delay_s=item.get('delay_s', 0), **row)
        try:
            with ThreadPoolExecutor(max_workers=len(group['jobs'])) as pool:
                futures = [pool.submit(job, item) for item in group['jobs']]
                rows = [future.result() for future in futures]
            after = health()
            if after['active_requests'] or after['queued_requests']:
                raise ValueError('Completed requests retained active slots')
        finally:
            stop.set()
            monitor_thread.join(timeout=12)
        wall = time.perf_counter()-origin
        admission = counter_delta(before['admission_statistics'], after['admission_statistics'])
        if admission['requests'] != len(rows):
            raise ValueError('Unexpected external traffic')
        tokens = sum(r['usage']['completion_tokens'] for r in rows)
        report['rounds'].append(dict(id=group['id'], submitted=len(rows), rows=rows,
            wall_s=wall, committed_output_tokens=tokens, output_tps=tokens/wall,
            peak_active=max((s['active_requests'] for s in samples), default=0),
            peak_queued=max((s['queued_requests'] for s in samples), default=0),
            scheduler_delta=counter_delta(before['scheduler_statistics'], after['scheduler_statistics']),
            admission_delta=admission))
        save()
        print(group['id'], len(rows), 'tasks passed', flush=True)
    report.update(status='complete', after=health())
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8088/v1')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a fresh path')
    run(args.base_url, json.loads(args.requests.read_text()), args.output)


if __name__ == '__main__':
    main()

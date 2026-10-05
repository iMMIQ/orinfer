"""Measure committed Chat API output using validated native MTP token IDs.

The expected report is produced by validate_mtp_generation. Exact decoded
prefix matching determines the first chunk's real token count; SSE events,
retokenized fragments and speculative proposals are never counted as tokens.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

from tokenizers import Tokenizer


def events(response):
    pending = []
    for line in response:
        line = line.decode().rstrip('\r\n')
        if not line:
            if pending:
                yield '\n'.join(pending)
                pending = []
        elif line.startswith('data:'):
            pending.append(line[5:].lstrip())
    if pending:
        raise ValueError('Unterminated SSE event')


def request(base_url, body, ids, tokenizer):
    payload = dict(body, stream=True, stream_options={'include_usage': True})
    headers = {'Content-Type': 'application/json'}
    if os.getenv('ORINFER_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['ORINFER_API_KEY']
    req = urllib.request.Request(base_url + '/chat/completions',
        data=json.dumps(payload).encode(), headers=headers)
    start = time.perf_counter()
    pieces, arrivals, usage, finish, done = [], [], None, None, False
    with urllib.request.urlopen(req, timeout=600) as response:
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
                if delta.get('reasoning_content') or delta.get('tool_calls'):
                    raise ValueError('This benchmark expects plain code output')
                if delta.get('content'):
                    pieces.append(delta['content'])
                    arrivals.append({'elapsed_s': at, 'content': delta['content']})
                finish = choice.get('finish_reason') or finish
    if not done or not usage or finish != 'length' or not arrivals:
        raise ValueError('Expected complete fixed-length streaming output')
    if usage['completion_tokens'] != len(ids):
        raise ValueError('Actual completion-token count differs from native reference')
    content = ''.join(pieces)
    expected = tokenizer.decode(ids, skip_special_tokens=False)
    if content != expected:
        raise ValueError('API output differs from the validated native MTP reference')
    first_text = arrivals[0]['content']
    matches = [n for n in range(1, len(ids) + 1)
        if tokenizer.decode(ids[:n], skip_special_tokens=False) == first_text]
    if len(matches) != 1:
        raise ValueError('Cannot uniquely determine real first-chunk token count')
    first_count = matches[0]
    interval = arrivals[-1]['elapsed_s'] - arrivals[0]['elapsed_s']
    return dict(usage=usage, content=content, validated_output_equal=True,
        first_chunk_tokens=first_count, ttft_s=arrivals[0]['elapsed_s'],
        decode_s=interval, decode_tps=(len(ids) - first_count) / interval,
        wall_s=time.perf_counter() - start, arrivals=arrivals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8088/v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1:
        parser.error('Require a fresh output and positive repetition count')
    cases = json.loads(args.requests.read_text())['cases']
    reference = json.loads(args.expected.read_text())
    if reference['status'] != 'passed':
        raise ValueError('Target/MTP verification must pass before API comparison')
    expected = {r['case']: r['output_tokens'] for r in reference['rows']
        if r['mtp'] and not r['warmup']}
    tokenizer = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
    rows = []
    for repetition in range(args.repetitions + 1):
        for case in cases:
            row = request(args.base_url, case['request'], expected[case['id']], tokenizer)
            row.update(case=case['id'], repetition=repetition, warmup=repetition == 0)
            rows.append(row)
            print(case['id'], repetition, round(row['decode_tps'], 3), 'TPS', flush=True)
            args.output.write_text(json.dumps(dict(status='running', rows=rows), indent=2) + '\n')
    medians = {c['id']: statistics.median(r['decode_tps'] for r in rows
        if r['case'] == c['id'] and not r['warmup']) for c in cases}
    args.output.write_text(json.dumps(dict(status='passed', rows=rows,
        median_decode_tps=medians, seed=20261002,
        scope='Verified real output tokens after first received chunk divided by first-to-last content arrival interval. HTTP and decoding included; rejected draft tokens excluded.'), indent=2) + '\n')


if __name__ == '__main__':
    main()

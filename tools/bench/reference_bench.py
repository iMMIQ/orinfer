"""Measured reference HTTP throughput plus separately labeled CUDA diagnostics."""
import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from scoring_common import execution_identity

SEED = 20261002


def sse_events(response):
    data = []
    for raw in response:
        line = raw.decode().rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip())
    if data:
        raise ValueError('SSE stream ended in an unterminated event')


def stream_request(config, ids, count, timing_id=None):
    payload = {'model': config['served_model'], 'prompt': ids, 'max_tokens': count,
               'temperature': 0, 'top_p': 1, 'seed': SEED, 'ignore_eos': True,
               'add_special_tokens': False, 'return_token_ids': True, 'stream': True,
               'stream_options': {'include_usage': True}}
    if timing_id:
        payload['vllm_xargs'] = {'orin_timing': timing_id}
    request = urllib.request.Request(config['url'] + '/v1/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    arrivals, tokens, pieces = [], [], []
    usage, finish, done = None, None, False
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            for event in sse_events(response):
                arrived = time.perf_counter() - started
                if event == '[DONE]':
                    done = True
                    break
                value = json.loads(event)
                if value.get('error'):
                    raise ValueError(value['error'])
                usage = value.get('usage') or usage
                for choice in value.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise ValueError('Unexpected batched stream')
                    new = choice.get('token_ids') or []
                    if new:
                        arrivals.append({'elapsed_s': arrived, 'token_count': len(new)})
                        tokens.extend(new)
                    pieces.append(choice.get('text', ''))
                    finish = choice.get('finish_reason') or finish
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc
    wall = time.perf_counter() - started
    if not done or finish != 'length' or not usage or not arrivals:
        raise ValueError('Incomplete fixed-work response')
    if usage['prompt_tokens'] != len(ids) or usage['completion_tokens'] != count or len(tokens) != count:
        raise ValueError('Incorrect executed workload or missing actual token IDs')
    if (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0):
        raise ValueError('Unexpected prefix hit in uncached reference')
    first, last = arrivals[0]['elapsed_s'], arrivals[-1]['elapsed_s']
    subsequent = count - arrivals[0]['token_count']
    return {'input_tokens': len(ids), 'output_tokens': count, 'client_ttft_s': first,
            'input_tokens_per_client_ttft': len(ids) / first,
            'client_decode_interval_s': last - first,
            'client_decode_tps': subsequent / (last - first) if subsequent and last > first else None,
            'first_chunk_tokens': arrivals[0]['token_count'], 'wall_s': wall, 'usage': usage,
            'token_ids': tokens, 'content': ''.join(pieces), 'arrivals': arrivals,
            'diagnostic_timing_id': timing_id,
            'decode_scope': 'Tokens after the first received token chunk / first-to-last token arrival interval.'}


def diagnostics(output, tag, count, length):
    path = output / 'timings' / (tag + '.jsonl')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) == count:
                break
        time.sleep(0.1)
    else:
        raise ValueError('Missing diagnostic CUDA events')
    prefill = [r for r in rows if r['mode'] == 'prefill']
    if sum(r['executed_input_tokens'] for r in prefill) != length:
        raise ValueError('CUDA diagnostic did not cover all actual prefill tokens')
    execute_s = sum(r['execute_gpu_ms'] for r in prefill) / 1000
    trunk_s = sum(r['trunk_gpu_ms'] for r in prefill) / 1000
    return {'steps': rows, 'prefill_trunk_gpu_s': trunk_s,
            'prefill_execute_gpu_s': execute_s, 'input_tokens_per_execute_gpu_s': length / execute_s,
            'scope': 'execute: GPU timeline spanning runner preparation, trunk and last-position head; excludes sampling/server/queue. trunk: model forward including graph replay; excludes head.',
            'limitations': 'CUDA event diagnostics are instrumented and reported separately from normal HTTP TPS; GPU timeline may include host submission gaps.'}


def run(config, output, seed_ids, lengths, count, repetitions):
    report = {'schema': 1, 'seed': SEED, 'outputs': count, 'repetitions': repetitions,
              'execution_identity': execution_identity(output),
              'seed_token_ids': seed_ids, 'completed': [],
              'reference_identity_sha256': json.loads((output / 'reference-lock.json').read_text())['identity_sha256'],
              'host_meminfo_before': Path('/proc/meminfo').read_text(),
              'metric_policy': 'Uncached/no MTP; normal client TPS and instrumented runner CUDA diagnostics remain separate.'}
    path = output / 'performance.json'
    started = time.monotonic()
    for length in lengths:
        ids = (seed_ids * ((length + len(seed_ids) - 1) // len(seed_ids)))[:length]
        warmup = stream_request(config, ids, min(32, count))
        if not report['completed']:
            report['first_request'] = warmup
        runs = []
        for index in range(repetitions):
            measured = stream_request(config, ids, count)
            runs.append(measured)
            report['in_progress'] = {'input_tokens': length, 'warmup': warmup, 'runs': runs}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print('reference', length, index + 1, 'input/TTFT', round(measured['input_tokens_per_client_ttft'], 2),
                  'decode', round(measured['client_decode_tps'], 2), flush=True)
        tag = uuid.uuid4().hex
        diagnostic = stream_request(config, ids, count, tag)
        diagnostic['gpu'] = diagnostics(output, tag, count, length)
        report['completed'].append({'input_tokens': length, 'warmup': warmup, 'runs': runs,
                                    'median_input_tokens_per_client_ttft': statistics.median(r['input_tokens_per_client_ttft'] for r in runs),
                                    'median_client_decode_tps': statistics.median(r['client_decode_tps'] for r in runs),
                                    'median_client_ttft_s': statistics.median(r['client_ttft_s'] for r in runs),
                                    'diagnostic': diagnostic})
        report.pop('in_progress', None)
        report['collection_wall_s'] = time.monotonic() - started
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    report.update(complete=True, host_meminfo_after=Path('/proc/meminfo').read_text())
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/reference.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/reference'))
    parser.add_argument('--seed-tokens', type=Path, required=True, help='JSON array of input token IDs')
    parser.add_argument('--lengths', default='512,2048,8192')
    parser.add_argument('--outputs', type=int, default=256)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.outputs < 2 or args.runs < 1:
        parser.error('Need >=2 output tokens and >=1 repetition')
    config = json.loads(args.config.read_text())
    if config != json.loads((args.output / 'reference-lock.json').read_text())['config']:
        raise ValueError('Reference settings changed after identity locking')
    with (args.output / 'tegrastats-performance.log').open('w') as hardware_log:
        telemetry = subprocess.Popen(['/usr/bin/tegrastats', '--interval', '1000'],
                                     stdout=hardware_log, stderr=subprocess.STDOUT)
        try:
            run(config, args.output, json.loads(args.seed_tokens.read_text()),
                list(map(int, args.lengths.split(','))), args.outputs, args.runs)
        finally:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()

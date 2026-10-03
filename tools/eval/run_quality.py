"""Freeze and compare real model probes through the isolated vLLM reference."""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from quick_quality import analyze
from scoring_common import SEED, execution_identity, pair_probes, safe_run_id, validate_probes, validate_task


def client_identity():
    directory = Path(__file__).resolve().parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ['run_quality.py', 'scoring_common.py', 'quick_quality.py']}


def post(url, endpoint, payload):
    request = urllib.request.Request(url + endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def complete(config, prompt, count, spec=None, ignore_eos=False):
    payload = {'model': config['served_model'], 'prompt': prompt, 'max_tokens': count,
               'temperature': 0, 'top_p': 1, 'repetition_penalty': 1, 'presence_penalty': 0,
               'frequency_penalty': 0, 'seed': SEED, 'return_token_ids': True,
               'add_special_tokens': False, 'ignore_eos': ignore_eos}
    if spec is not None:
        payload['vllm_xargs'] = {'orin_probe': json.dumps(spec, separators=(',', ':'))}
    started = time.monotonic()
    result = post(config['url'], '/v1/completions', payload)
    if result['usage']['prompt_tokens'] != len(prompt):
        raise ValueError('Server changed actual input token IDs')
    return result['choices'][0], time.monotonic() - started


def collect(output, spec):
    path = output / 'probes' / (spec['run_id'] + '.jsonl')
    if not path.exists():
        raise ValueError('Worker emitted no real scoring records')
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_spec(case_id, mode, **kwargs):
    return {'run_id': uuid.uuid4().hex, 'case_id': case_id, 'mode': mode, **kwargs}


def score_case(config, output, case, mode, baseline=None):
    targets = case['output_ids']
    query_ids = [[x['token_id'] for x in row['top3']] for row in baseline] if baseline else []
    spec = run_spec(case['id'], mode, targets=targets, query_ids=query_ids)
    prompt = case['prompt_ids']
    if mode == 'prefill':
        spec['prompt_length'] = len(prompt)
        response, wall = complete(config, prompt + targets, 1, spec, ignore_eos=True)
    else:
        response, wall = complete(config, prompt, len(targets), spec, ignore_eos=True)
        if response['token_ids'] != targets:
            raise ValueError('Teacher-forced decode did not commit the frozen tokens')
    probes = validate_probes(collect(output, spec), prompt, targets, case['id'], mode)
    return probes, wall


def freeze(config, fixtures, output, identity):
    started = time.monotonic()
    frozen = {'schema': 1, 'reference_identity_sha256': identity, 'seed': SEED,
              'execution_identity': execution_identity(output),
              'client_source_sha256': client_identity(),
              'fixtures_sha256': hashlib.sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest(),
              'reference_kind': 'Community W4A16 native reference, not official BF16.',
              'cases': []}
    for scenario in fixtures['cases']:
        tokenized = post(config['url'], '/tokenize', {
            'model': config['served_model'], 'messages': scenario['messages'],
            'add_generation_prompt': True, 'add_special_tokens': False,
            'chat_template_kwargs': {'enable_thinking': False}})
        prompt = tokenized['tokens']
        if len(prompt) != tokenized['count']:
            raise ValueError('Tokenization length mismatch')
        spec = run_spec(scenario['id'], 'decode')
        choice, wall = complete(config, prompt, fixtures['generation']['max_new_tokens'], spec)
        probes = sorted(collect(output, spec), key=lambda x: x['position'])
        targets = [row['reference_token_id'] for row in probes]
        returned = choice['token_ids']
        if not targets or len(targets) > fixtures['generation']['max_new_tokens']:
            raise ValueError('Invalid baseline generation length')
        # Some APIs omit the terminating EOS from displayed output IDs.
        if returned != targets and not (returned == targets[:-1] and choice['finish_reason'] == 'stop'):
            raise ValueError('Captured greedy history differs from committed baseline output')
        case = {'id': scenario['id'], 'messages': scenario['messages'], 'prompt_ids': prompt,
                'output_ids': targets, 'response_token_ids': returned, 'text': choice['text'],
                'finish_reason': choice['finish_reason'], 'stop_reason': choice.get('stop_reason'),
                'validation': scenario['validation'], 'task_check': validate_task(scenario['validation'], choice['text']),
                'decode': validate_probes(probes, prompt, targets, scenario['id'], 'decode'),
                'free_generation_wall_s': wall}
        case['prefill'], case['prefill_scoring_wall_s'] = score_case(config, output, case, 'prefill')
        frozen['cases'].append(case)
        frozen['collection_wall_s'] = time.monotonic() - started
        (output / 'baseline.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
        print('froze', case['id'], len(prompt), len(targets), case['task_check']['status'], flush=True)
    frozen['complete'] = True
    (output / 'baseline.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    return frozen


def compare(config, output, frozen, identity, label):
    safe_run_id(label)
    if not frozen.get('complete') or frozen['seed'] != SEED:
        raise ValueError('Baseline is incomplete or uses a different seed')
    started = time.monotonic()
    current_execution = execution_identity(output)
    paired, cases = [], []
    result_path = output / (label + '.json')
    for case in frozen['cases']:
        timings = {}
        for mode in ['prefill', 'decode']:
            probes, timings[mode] = score_case(config, output, case, mode, case[mode])
            paired.extend(pair_probes(case[mode], probes))
        choice, timings['free_generation'] = complete(config, case['prompt_ids'], 48)
        cases.append({'case_id': case['id'], 'text': choice['text'], 'token_ids': choice['token_ids'],
                      'finish_reason': choice['finish_reason'], 'task_check': validate_task(case['validation'], choice['text']),
                      'baseline_task_check': case['task_check'], 'wall_s': timings})
        report = dict(analyze(paired), cases=cases, complete=False,
                      baseline_identity_sha256=frozen['reference_identity_sha256'], candidate_identity_sha256=identity,
                      baseline_execution_identity_sha256=frozen['execution_identity']['sha256'],
                      candidate_execution_identity_sha256=current_execution['sha256'],
                      client_source_sha256=client_identity(),
                      candidate_evaluation_wall_s=time.monotonic() - started,
                      comparison_label=label)
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (output / (label + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in paired))
        print('compared', case['id'], cases[-1]['task_check']['status'], flush=True)
    report['complete'] = True
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report['aggregate'], indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'compare'])
    parser.add_argument('--config', type=Path, default=Path('configs/reference.json'))
    parser.add_argument('--fixtures', type=Path, default=Path('fixtures/quick-quality-scenarios.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/reference'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--label', default='repeatability')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    lock = json.loads((args.output / 'reference-lock.json').read_text())
    if config != lock['config']:
        raise ValueError('Reference settings changed after identity locking')
    if args.action == 'freeze':
        freeze(config, json.loads(args.fixtures.read_text()), args.output, lock['identity_sha256'])
    else:
        baseline = args.baseline or args.output / 'baseline.json'
        compare(config, args.output, json.loads(baseline.read_text()), lock['identity_sha256'], args.label)

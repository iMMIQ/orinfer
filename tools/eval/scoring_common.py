"""Backend-independent identities and pairing for real next-token probes."""
import hashlib
import json
import math
import re
from pathlib import Path

SEED = 20261002


def execution_identity(output):
    server = json.loads((output / 'server.json').read_text())
    environment = json.loads((output / 'environment.json').read_text())
    if server['status'] != 'ready':
        raise ValueError('Reference server is not Ready')
    for name, expected in server['hook_sha256'].items():
        path = Path(server['source_snapshot']) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen reference source changed')
    identity = {'config': server['config'], 'hook_sha256': server['hook_sha256'], 'environment': environment}
    identity['sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return identity


def context_hash(token_ids):
    if not token_ids or any(type(x) is not int or x < 0 for x in token_ids):
        raise ValueError('A nonempty actual token-ID history is required')
    data = {'token_ids': token_ids, 'positions': list(range(len(token_ids)))}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', value):
        raise ValueError('Invalid probe run ID')
    return value


def validate_task(rule, text):
    """Run deterministic data checks; never execute generated code."""
    kind, expected = rule['kind'], rule['expected']
    if kind == 'manual':
        return {'status': 'manual_review', 'criteria': expected}
    value = text.strip()
    try:
        if kind == 'exact':
            passed = value == expected
        elif kind == 'integer':
            passed = bool(re.fullmatch(r'[+-]?\d+', value)) and int(value) == expected
        elif kind == 'number':
            passed = math.isfinite(float(value)) and math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-9)
        elif kind == 'json_equal':
            actual = json.loads(value)
            # Canonical JSON preserves true vs 1, unlike Python dict equality.
            passed = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        else:
            raise ValueError(f'Unknown validation kind: {kind}')
    except (json.JSONDecodeError, ValueError, OverflowError):
        if kind not in {'exact', 'integer', 'number', 'json_equal'}:
            raise
        passed = False
    return {'status': 'pass' if passed else 'fail', 'kind': kind, 'expected': expected}


def validate_probes(probes, prompt_ids, target_ids, case_id, mode):
    ordered = sorted(probes, key=lambda x: x['position'])
    if len(ordered) != len(target_ids):
        raise ValueError(f'{case_id}/{mode}: missing or extra probe positions')
    for position, row in enumerate(ordered):
        if (row['case_id'], row['execution_mode'], row['position'], row['seed']) != (case_id, mode, position, SEED):
            raise ValueError('Probe identity, seed or position mismatch')
        if row['context_sha256'] != context_hash(prompt_ids + target_ids[:position]):
            raise ValueError('Scorer used a different teacher-forced history')
        if row['reference_token_id'] != target_ids[position]:
            raise ValueError('Scorer used a different reference token')
    return ordered


def pair_probes(baseline, candidate):
    if len(baseline) != len(candidate):
        raise ValueError('Probe count mismatch')
    rows = []
    for a, b in zip(baseline, candidate):
        keys = ['case_id', 'execution_mode', 'position', 'seed', 'context_sha256', 'reference_token_id']
        if any(a[k] != b[k] for k in keys):
            raise ValueError('Probe identities or histories differ')
        required = {str(x['token_id']) for x in a['top3']} | {str(a['reference_token_id'])}
        if not required <= b['queried_logprobs'].keys():
            raise ValueError('Candidate failed to explicitly score reference IDs')
        rows.append({
            'case_id': a['case_id'], 'execution_mode': a['execution_mode'], 'position': a['position'],
            'seed': SEED, 'baseline_context_sha256': a['context_sha256'],
            'candidate_context_sha256': b['context_sha256'], 'baseline_top3': a['top3'],
            'candidate_top3': b['top3'], 'candidate_logprobs_on_baseline_top3': {
                str(x['token_id']): b['queried_logprobs'][str(x['token_id'])] for x in a['top3']},
            'reference_token_id': a['reference_token_id'],
            'baseline_reference_logprob': a['reference_logprob'],
            'candidate_reference_logprob': b['reference_logprob'],
            # This is the unforced greedy choice. The teacher-forced token is distinct.
            'candidate_selected_token_id': b['top3'][0]['token_id'],
        })
    return rows

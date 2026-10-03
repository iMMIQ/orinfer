"""Analyze paired next-token probes. Inputs are JSONL from engine scoring adapters.

Both engines must score identical token histories. Candidate adapters must return
probabilities for all baseline top-3 IDs, even if absent from candidate top-3.
This measures a local probability diagnostic, not a full benchmark task score.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SEED = 20261002

def top3(items):
    if len(items) != 3 or len({x['token_id'] for x in items}) != 3:
        raise ValueError('Exactly three distinct top token IDs are required')
    for x in items:
        if not isinstance(x['token_id'], int) or not math.isfinite(x['logprob']) or x['logprob'] > 0:
            raise ValueError('Invalid token ID or full-vocabulary log probability')
    return sorted(items, key=lambda x: (-x['logprob'], x['token_id']))

def analyze(rows):
    groups = defaultdict(list)
    identities = set()
    for r in rows:
        if r['seed'] != SEED or r['baseline_context_sha256'] != r['candidate_context_sha256']:
            raise ValueError('Seed/history mismatch; paired probabilities cannot be compared')
        for key in ['baseline_context_sha256', 'candidate_context_sha256']:
            digest = r[key]
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Context hash must be an actual SHA256 digest')
        identity = (r['case_id'], r['execution_mode'], r['position'])
        if identity in identities:
            raise ValueError('Duplicate scoring position')
        identities.add(identity)
        a, b = top3(r['baseline_top3']), top3(r['candidate_top3'])
        lp = r['candidate_logprobs_on_baseline_top3']
        base_ids = {x['token_id'] for x in a}
        if not all(str(x) in lp for x in base_ids):
            raise ValueError('Missing baseline top-3 probability; query it explicitly, never fill with zero')
        vals = [lp[str(x['token_id'])] for x in a]
        if any(not math.isfinite(v) or v > 0 for v in vals):
            raise ValueError('Invalid candidate log probabilities')
        if any(total > 1.00001 for total in [sum(math.exp(x['logprob']) for x in a),
                sum(math.exp(x['logprob']) for x in b), sum(math.exp(v) for v in vals)]):
            raise ValueError('Probabilities must use the full-vocabulary normalization')
        # Require consistent scores when a token appears in both candidate views.
        for token in b:
            if token['token_id'] in base_ids and abs(token['logprob'] - lp[str(token['token_id'])]) > 1e-5:
                raise ValueError('Candidate top-3 and queried probabilities disagree')
        selected = r.get('candidate_selected_token_id', b[0]['token_id'])
        if selected != b[0]['token_id']:
            raise ValueError('Fast paired suite assumes greedy selection with stable token-ID tie breaking')
        ref = r['reference_token_id']
        base_lp = r['baseline_reference_logprob']
        candidate_lp = r['candidate_reference_logprob']
        if any(not math.isfinite(v) or v > 0 for v in [base_lp, candidate_lp]):
            raise ValueError('Invalid teacher-forced target score')
        if ref in base_ids:
            known = next(x['logprob'] for x in a if x['token_id'] == ref)
            if abs(known - base_lp) > 1e-5 or abs(lp[str(ref)] - candidate_lp) > 1e-5:
                raise ValueError('Reference-token score does not match top-3 query')
        stat = dict(top1_agreement=selected == a[0]['token_id'], selected_in_baseline_top3=selected in base_ids,
                    top3_overlap=len(base_ids & {x['token_id'] for x in b})/3,
                    mean_abs_probability_error=statistics.mean(abs(math.exp(x['logprob'])-math.exp(v)) for x,v in zip(a, vals)),
                    mean_abs_logprob_error=statistics.mean(abs(x['logprob']-v) for x,v in zip(a, vals)),
                    delta_target_nll=base_lp-candidate_lp,
                    high_confidence=math.exp(a[0]['logprob']) >= 0.8,
                    low_margin=math.exp(a[0]['logprob'])-math.exp(a[1]['logprob']) <= 0.01,
                    baseline_top1_probability=math.exp(a[0]['logprob']),
                    baseline_top1_top2_probability_gap=math.exp(a[0]['logprob'])-math.exp(a[1]['logprob']))
        groups[(r['case_id'], r['execution_mode'])].append(stat)
    def summarize(values):
        confident = [x for x in values if x['high_confidence']]
        low_margin = [x for x in values if x['low_margin']]
        return dict(positions=len(values), **{key: statistics.mean(x[key] for x in values) for key in [
            'top1_agreement','selected_in_baseline_top3','top3_overlap','mean_abs_probability_error','mean_abs_logprob_error','delta_target_nll']},
            high_confidence_positions=len(confident),
            high_confidence_top1_disagreement=statistics.mean(not x['top1_agreement'] for x in confident) if confident else None,
            low_margin_positions=len(low_margin),
            low_margin_top1_disagreement=statistics.mean(not x['top1_agreement'] for x in low_margin) if low_margin else None)
    if not groups:
        raise ValueError('No scoring records')
    return dict(seed=SEED, aggregate=summarize([x for values in groups.values() for x in values]),
                by_case_and_mode=[dict(case_id=k[0],execution_mode=k[1],**summarize(v)) for k,v in sorted(groups.items())],
                limitations=['Probability and token-selection diagnostics are not task accuracy.',
                             'No full-vocabulary KL is inferred from top-3.',
                             'Confidence >=0.8 and probability gap <=0.01 are reporting buckets, not quality acceptance thresholds.',
                             'Backend/revision/template/quantization metadata must accompany scoring artifacts.'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    result = analyze(rows)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result['aggregate'], indent=2))

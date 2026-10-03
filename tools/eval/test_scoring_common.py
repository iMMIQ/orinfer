"""Protocol failures must not turn into plausible-looking quality results."""
import copy
import unittest

from scoring_common import SEED, context_hash, pair_probes, safe_run_id, validate_probes, validate_task


def probe():
    return {'case_id': 'case', 'execution_mode': 'decode', 'position': 0, 'seed': SEED,
            'context_sha256': context_hash([10, 11]), 'reference_token_id': 3,
            'reference_logprob': -0.1,
            'top3': [{'token_id': 3, 'logprob': -0.1}, {'token_id': 4, 'logprob': -3}, {'token_id': 5, 'logprob': -4}],
            'queried_logprobs': {'3': -0.1, '4': -3, '5': -4}}


class ScoringProtocolTests(unittest.TestCase):
    def test_real_histories_and_position_order(self):
        first = probe()
        second = dict(first, position=1, context_sha256=context_hash([10, 11, 3]))
        self.assertEqual(validate_probes([second, first], [10, 11], [3, 3], 'case', 'decode'), [first, second])
        second['context_sha256'] = context_hash([10, 11, 9])
        with self.assertRaises(ValueError):
            validate_probes([first, second], [10, 11], [3, 3], 'case', 'decode')

    def test_missing_explicit_queries_rejected_even_with_top3(self):
        row = copy.deepcopy(probe())
        del row['queried_logprobs']['5']
        with self.assertRaises(ValueError):
            pair_probes([probe()], [row])

    def test_pair_cannot_mix_modes_or_changed_contexts(self):
        for key, value in [('execution_mode', 'prefill'), ('context_sha256', context_hash([10, 12]))]:
            row = dict(probe(), **{key: value})
            with self.assertRaises(ValueError):
                pair_probes([probe()], [row])

    def test_unforced_greedy_is_not_teacher_forced_token(self):
        row = copy.deepcopy(probe())
        row['top3'] = [{'token_id': 8, 'logprob': -0.1}, {'token_id': 3, 'logprob': -3}, {'token_id': 4, 'logprob': -4}]
        paired = pair_probes([probe()], [row])[0]
        self.assertEqual(paired['reference_token_id'], 3)
        self.assertEqual(paired['candidate_selected_token_id'], 8)

    def test_task_check_preserves_json_boolean_type_and_marks_manual(self):
        rule = {'kind': 'json_equal', 'expected': {'ok': True, 'count': 3}}
        self.assertEqual(validate_task(rule, '{"count":3,"ok":true}')['status'], 'pass')
        self.assertEqual(validate_task(rule, '{"count":3,"ok":1}')['status'], 'fail')
        self.assertEqual(validate_task({'kind': 'number', 'expected': 12.5}, 'NaN')['status'], 'fail')
        self.assertEqual(validate_task({'kind': 'manual', 'expected': 'review code'}, 'anything')['status'], 'manual_review')

    def test_run_ids_cannot_escape_artifact_directory(self):
        with self.assertRaises(ValueError):
            safe_run_id('../../outside')
        self.assertEqual(safe_run_id('case-123'), 'case-123')


if __name__ == '__main__':
    unittest.main()

"""Synthetic correctness/rejection tests, not actual model quality measurements."""
import copy
import math
import unittest

from quick_quality import SEED, analyze


def record():
    scores = [{"token_id": i, "logprob": math.log(p)} for i, p in [(1, 0.8), (2, 0.1), (3, 0.05)]]
    return {
        "case_id": "example", "execution_mode": "decode", "position": 0,
        "seed": SEED, "baseline_context_sha256": "a" * 64, "candidate_context_sha256": "a" * 64,
        "baseline_top3": scores, "candidate_top3": copy.deepcopy(scores),
        "candidate_logprobs_on_baseline_top3": {str(x["token_id"]): x["logprob"] for x in scores},
        "reference_token_id": 1, "baseline_reference_logprob": math.log(0.8),
        "candidate_reference_logprob": math.log(0.8),
    }


class QualityTests(unittest.TestCase):
    def test_identical_scores(self):
        summary = analyze([record()])["aggregate"]
        self.assertEqual(summary["top1_agreement"], 1)
        self.assertEqual(summary["mean_abs_probability_error"], 0)
        self.assertEqual(summary["delta_target_nll"], 0)

    def test_candidate_selection_outside_baseline_top3(self):
        row = record()
        row["candidate_top3"] = [{"token_id": i, "logprob": math.log(p)} for i, p in [(4, 0.4), (1, 0.3), (2, 0.1)]]
        row["candidate_logprobs_on_baseline_top3"]["1"] = math.log(0.3)
        row["candidate_reference_logprob"] = math.log(0.3)
        summary = analyze([row])["aggregate"]
        self.assertEqual(summary["top1_agreement"], 0)
        self.assertEqual(summary["selected_in_baseline_top3"], 0)
        self.assertAlmostEqual(summary["top3_overlap"], 2 / 3)
        self.assertAlmostEqual(summary["delta_target_nll"], math.log(0.8 / 0.3))

    def test_history_seed_and_missing_probabilities_rejected(self):
        mutations = [
            ("candidate_context_sha256", "b" * 64),
            ("seed", 42),
            ("candidate_logprobs_on_baseline_top3", {"1": math.log(0.8), "2": math.log(0.1)}),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                row = record()
                row[key] = value
                with self.assertRaises(ValueError):
                    analyze([row])

    def test_duplicate_position_rejected(self):
        with self.assertRaises(ValueError):
            analyze([record(), record()])

    def test_conflicting_probability_views_rejected(self):
        row = record()
        row["candidate_logprobs_on_baseline_top3"]["2"] = math.log(0.09)
        with self.assertRaises(ValueError):
            analyze([row])

    def test_top3_renormalization_would_hide_loss(self):
        row = record()
        row["candidate_top3"] = [{"token_id": x["token_id"], "logprob": x["logprob"] - math.log(0.95)} for x in row["baseline_top3"]]
        row["candidate_logprobs_on_baseline_top3"] = {str(x["token_id"]): x["logprob"] for x in row["candidate_top3"]}
        row["candidate_reference_logprob"] = row["candidate_top3"][0]["logprob"]
        # Valid-looking subset probabilities cannot prove full-vocabulary normalization.
        # The scorer must supply it; the analyzer preserves, rather than normalizes, inputs.
        self.assertGreater(analyze([row])["aggregate"]["mean_abs_probability_error"], 0)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            analyze([])


if __name__ == "__main__":
    unittest.main()

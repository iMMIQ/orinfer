from copy import deepcopy
import unittest

from tools.bench.ascend_regression import compare


class RegressionTests(unittest.TestCase):
    def report(self):
        return dict(
            requests=[
                dict(
                    name=name,
                    tokens_sha256=name,
                    input_tokens=length,
                    output_tokens=2,
                    prefill_chunks=[length],
                    repeat=0,
                    ttft_s=time,
                    decode_s=0.02,
                    e2e_s=time + 0.02,
                    output_token_ids=[1, 2],
                )
                for name, length, time in [("short", 17, 0.05), ("long", 2048, 0.4)]
            ]
        )

    def test_single_slow_shape_rejects_a_faster_aggregate(self):
        baseline = self.report()
        candidate = deepcopy(baseline)
        candidate["requests"][0]["ttft_s"] *= 1.3
        candidate["requests"][1]["ttft_s"] *= 0.5
        for request in candidate["requests"]:
            request["e2e_s"] = request["ttft_s"] + request["decode_s"]
        result = compare(baseline, candidate)
        self.assertFalse(result["passed"])
        self.assertFalse(result["workloads"][0]["passed"])
        self.assertTrue(result["workloads"][1]["passed"])

    def test_missing_shapes_and_changed_chunk_contracts_are_rejected(self):
        baseline = self.report()
        for candidate in [dict(requests=baseline["requests"][:1]), deepcopy(baseline)]:
            if len(candidate["requests"]) == 2:
                candidate["requests"][0]["prefill_chunks"] = [16, 1]
            with self.assertRaises(ValueError):
                compare(baseline, candidate)

    def test_output_changes_are_reported_without_claiming_quality_acceptance(self):
        baseline = self.report()
        candidate = deepcopy(baseline)
        candidate["requests"][0]["output_token_ids"] = [3, 4]
        result = compare(baseline, candidate)
        self.assertTrue(result["passed"])
        self.assertEqual(result["workloads"][0]["output_differences"], 1)

    def test_request_preparation_cost_and_timing_scope_are_checked(self):
        baseline = self.report()
        for request in baseline["requests"]:
            request["request_ttft_s"] = request["ttft_s"] + 0.01
            request["request_e2e_s"] = request["e2e_s"] + 0.01
        candidate = deepcopy(baseline)
        candidate["requests"][0]["request_ttft_s"] += 0.05
        self.assertFalse(compare(baseline, candidate)["passed"])
        del candidate["requests"][0]["request_ttft_s"]
        with self.assertRaises(ValueError):
            compare(baseline, candidate)

    def test_logit_collection_must_match_between_performance_runs(self):
        baseline = dict(self.report(), logits_dump=False)
        candidate = dict(self.report(), logits_dump=True)
        with self.assertRaises(ValueError):
            compare(baseline, candidate)


if __name__ == "__main__":
    unittest.main()

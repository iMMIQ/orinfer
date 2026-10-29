"""Boundary acceptance must expose slow tails rather than average them away."""

from copy import deepcopy
import unittest

from tools.bench.ascend_boundaries import summarize


class BoundaryTests(unittest.TestCase):
    def fixture(self, times):
        inputs, report = dict(cases=[]), dict(context_capacity=4096, requests=[], preparations=[])
        for n, elapsed in times:
            name = f"length{n}"
            inputs["cases"].append(
                dict(name=name, requests=[dict(owner=91, tokens=[7] * n, output=True)], repeats=5)
            )
            report["requests"].extend(
                dict(
                    name=name,
                    repeat=i,
                    batch=1,
                    input_tokens=n,
                    prefill_s=elapsed,
                    request_s=elapsed + 0.001,
                )
                for i in range(5)
            )
            report["preparations"].append(
                dict(
                    name=name,
                    owner_admission_s=0.1,
                    reset_s=0.001,
                    graph_preparation_s=1,
                    first_prefill_s=elapsed,
                )
            )
        return inputs, report

    def test_one_slow_overflow_fails_even_with_fast_other_lengths(self):
        result = summarize(*self.fixture([(1023, 0.2), (1024, 0.2), (2048, 0.4), (2049, 0.45)]))
        self.assertFalse(result["passed"])
        self.assertEqual([p["after"] for p in result["boundaries"] if not p["passed"]], [2049])

    def test_smooth_growth_passes_and_cold_capture_is_reported(self):
        result = summarize(*self.fixture([(2048, 0.4), (2049, 0.4002)]))
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["points"][0]["first_shape_request_s"], 1.501)
        self.assertEqual(result["points"][0]["samples"], 5)

    def test_request_preparation_regression_is_not_hidden_by_forward_time(self):
        inputs, report = self.fixture([(2048, 0.4), (2049, 0.4002)])
        for q in report["requests"][5:]:
            q["request_s"] += 0.1
        self.assertFalse(summarize(inputs, report)["passed"])

    def test_ten_percent_limit_applies_to_throughput(self):
        result = summarize(*self.fixture([(2048, 0.4), (2049, 0.444)]))
        self.assertTrue(result["passed"])
        self.assertFalse(summarize(*self.fixture([(2048, 0.4), (2049, 0.45)]))["passed"])

    def test_changed_prefix_missing_repeats_and_invalid_timings_are_rejected(self):
        original = self.fixture([(2048, 0.4), (2049, 0.4002)])
        for mutate in (
            lambda i, r: i["cases"][1]["requests"][0]["tokens"].__setitem__(0, 8),
            lambda i, r: r["requests"].pop(),
            lambda i, r: r["requests"][0].update(prefill_s=float("nan")),
            lambda i, r: r["requests"][0].update(input_tokens=2047),
            lambda i, r: i["cases"][1].update(decode_steps=1),
        ):
            inputs, report = deepcopy(original)
            mutate(inputs, report)
            with self.assertRaises(ValueError):
                summarize(inputs, report)


if __name__ == "__main__":
    unittest.main()

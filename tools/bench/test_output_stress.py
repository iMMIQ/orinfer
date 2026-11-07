import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers

from tools.bench.output_stress import repetition, run, summarize


class OutputStressTests(unittest.TestCase):
    def test_cycles_in_middle_and_short_periods_are_detected(self):
        for unit in [[9], [9, 8], list(range(137))]:
            result = repetition(list(range(500, 540)) + unit * 400 + [9999, 9998])
            self.assertGreaterEqual(result["repeated_tokens"], 400 * len(unit))
            self.assertGreaterEqual(result["repeats"], 3)

    def test_unique_output_and_two_legitimate_quotes_are_not_cycles(self):
        for ids in [[], list(range(1024)), list(range(500)) * 2]:
            self.assertEqual(repetition(ids)["repeated_tokens"], 0)

    def test_errors_and_truncation_remain_visible(self):
        rows = [
            dict(case="one", error="failed"),
            dict(
                case="one",
                usage={"completion_tokens": 20},
                wall_s=2,
                ttft_s=1,
                decode_tps=19,
                finish_reason="length",
                issues=["empty_answer"],
            ),
        ]
        result = summarize(rows, 4)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["output_issues"], 1)
        self.assertEqual(result["truncated"], 1)
        self.assertEqual(result["completed_output_tps"], 5)

    def test_runner_ignores_saved_truncation_and_retains_detected_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "one": 1}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.enable_truncation(1)
            tokenizer.save(str(root / "tokenizer.json"))
            (root / "cases.json").write_text(json.dumps(dict(cases=[dict(id="cycle", request={})])))
            args = argparse.Namespace(
                cases=root / "cases.json",
                tokenizer=root / "tokenizer.json",
                output=root / "out",
                base_url="http://localhost/v1",
                phases=["1:0.005"],
                timeout=1,
            )

            def response(*args, **kwargs):
                return dict(
                    content="one " * 400,
                    reasoning="",
                    wall_s=1,
                    ttft_s=0.1,
                    usage=dict(completion_tokens=400),
                    finish_reason="stop",
                )

            with (
                patch("tools.bench.output_stress.request", side_effect=response),
                patch(
                    "urllib.request.urlopen",
                    side_effect=lambda *a, **k: io.BytesIO(b'{"status":"ready"}'),
                ),
                patch("builtins.print"),
            ):
                self.assertEqual(run(args), 1)
            rows = [
                json.loads(line)
                for line in (args.output / "responses.jsonl").read_text().splitlines()
            ]
            self.assertTrue(rows)
            self.assertIn("content:repeated_cycle", rows[0]["issues"])
            self.assertEqual(rows[0]["content_repetition"]["repeated_tokens"], 400)


if __name__ == "__main__":
    unittest.main()

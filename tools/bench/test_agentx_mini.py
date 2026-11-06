"""Keep short replay timing and actual cache-bust inputs reproducible."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.bench import agentx_entry, agentx_mini


class MiniIdentityTests(unittest.TestCase):
    def test_fixed_identity_reaches_the_installed_cli_without_changing_arguments(self):
        package = types.ModuleType("aiperf")
        runner = types.ModuleType("aiperf.cli_runner")
        package.cli_runner = runner
        runner._make_benchmark_run = lambda config, **kwargs: kwargs
        argv = [
            "entry",
            "--harness",
            "/tmp/agentx",
            "--benchmark-id",
            "fixed-source",
            "profile",
            "--random-seed",
            "42",
        ]

        def invoke(path, run_name):
            self.assertEqual((path, run_name), ("/tmp/agentx", "__main__"))
            self.assertEqual(sys.argv, ["/tmp/agentx", "profile", "--random-seed", "42"])
            run = runner._make_benchmark_run(None, benchmark_id="random-run", random_seed=42)
            self.assertEqual(run, dict(benchmark_id="fixed-source", random_seed=42))

        with (
            patch.dict(sys.modules, {"aiperf": package, "aiperf.cli_runner": runner}),
            patch.object(sys, "argv", argv),
            patch.object(agentx_entry.runpy, "run_path", side_effect=invoke),
        ):
            agentx_entry.main()

    def test_clock_waits_for_a_request_record_and_uses_issue_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(agentx_mini.first_credit_ns(root))
            path = root / "profile_export.jsonl"
            path.write_text('{"metadata":')
            self.assertIsNone(agentx_mini.first_credit_ns(root))
            # A slow CLI starts at t=0, issues at t=16, and completes at t=20.
            record = dict(
                metadata=dict(credit_issued_ns=16_000_000_000, request_end_ns=20_000_000_000)
            )
            path.write_text(json.dumps(record) + '\n{"metadata":')
            self.assertEqual(agentx_mini.first_credit_ns(root), 16_000_000_000)


if __name__ == "__main__":
    unittest.main()

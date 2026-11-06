"""Keep short replay timing and actual cache-bust inputs reproducible."""

import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.bench import agentx_entry, agentx_mini


class MiniIdentityTests(unittest.TestCase):
    def test_duration_pressure_keeps_the_same_shared_request_contract(self):
        args = SimpleNamespace(
            harness=Path("/env/bin/agentx"),
            benchmark_id="fixed-source",
            url="http://localhost:18300",
            model="deepseek-v41-w4a8",
            tokenizer=Path("/tokenizer"),
            dataset=Path("/dataset"),
            output=Path("/output"),
            request_count=100000,
            concurrency=16,
            benchmark_duration=600,
            enable_thinking=True,
        )
        command = agentx_mini.command(args)
        for flag, value in [
            ("--request-count", "100000"),
            ("--concurrency", "16"),
            ("--benchmark-duration", "600"),
        ]:
            self.assertEqual(command[command.index(flag) + 1], value)
        extra = json.loads(command[command.index("--extra-inputs") + 1])
        self.assertEqual(extra["reasoning_effort"], "high")
        self.assertTrue(extra["chat_template_kwargs"]["enable_thinking"])

    def test_reference_snapshot_sums_engine_gauges_and_preserves_context(self):
        metrics = (
            'vllm:num_requests_running{engine="0"} 7\n'
            'vllm:num_requests_running{engine="1"} 2\n'
            'vllm:num_requests_waiting{engine="0"} 3\n'
        )
        args = SimpleNamespace(backend="vllm", url="http://localhost", model="test")
        with patch.object(agentx_mini, "fetch", side_effect=[metrics, ""]):
            health, raw = agentx_mini.snapshot(args, {"max_model_len": 262144})
        self.assertEqual((health["active_requests"], health["queued_requests"]), (9, 3))
        self.assertEqual(health["max_context"], 262144)
        self.assertEqual(raw, metrics)

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

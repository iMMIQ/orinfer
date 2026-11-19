from collections import Counter
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from orinfer_build.common.files import file_hash
from tools.bench.mini_weka_compact import DEPENDENCIES, TracePlans, parse_args, render_trace, replay
from tools.bench.mini_weka_synthetic import TOOL_INPUT_TOKENS, generate, main, trace


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return SimpleNamespace(ids=list(map(ord, text)))

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(map(chr, ids))


def generate_manifest(output, **options):
    generate(output, **options)
    return output / "manifest.json"


class SyntheticWorkloadTests(unittest.TestCase):
    def test_compact_default_is_balanced_and_has_disjoint_lanes(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "compact"
            manifest = generate_manifest(output, profile="compact")
            data = json.loads(manifest.read_bytes())
            self.assertEqual(data["workload"], "orinfer-synthetic-mini-weka-compact-v1")
            self.assertEqual(data["trajectories"], 512)
            self.assertEqual(len(data["windows"]), 512)
            self.assertEqual(len(list((output / "traces").glob("*.json"))), 512)
            sizes, used_blocks = Counter(), set()
            for lane, window in enumerate(data["windows"]):
                name = f"synthetic-mini-{lane:03d}"
                self.assertEqual(window["trace"], name)
                path = output / "traces" / f"{name}.json"
                self.assertEqual(window["sha256"], file_hash(path))
                item = json.loads(path.read_bytes())
                self.assertEqual(item["id"], name)
                cold_kib = (2, 3, 4, 6)[lane % 4]
                self.assertEqual(item["requests"][0]["in"], cold_kib * 1024)
                sizes[cold_kib] += 1
                events = [
                    event["requests"][0] if event["type"] == "subagent" else event
                    for event in item["requests"]
                ]
                blocks = {block for event in events for block in event["hash_ids"]}
                self.assertEqual(min(blocks), lane * 4096 + 1)
                self.assertTrue(used_blocks.isdisjoint(blocks))
                used_blocks.update(blocks)
                if lane < 16:
                    original = trace(
                        f"synthetic-mini-{lane:02d}",
                        cold_kib * 1024,
                        lane * 4096 + 1,
                        TOOL_INPUT_TOKENS["compact"],
                    )
                    self.assertEqual(dict(item, id=original["id"]), original)
            self.assertEqual(sizes, dict.fromkeys((2, 3, 4, 6), 128))

    def test_long_default_keeps_sixteen_trajectories(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "long"
            manifest = generate_manifest(output, profile="long")
            data = json.loads(manifest.read_bytes())
            self.assertEqual(data["trajectories"], 16)
            self.assertEqual(len(data["windows"]), 16)
            self.assertEqual(data["windows"][-1]["trace"], "synthetic-mini-15")
            cold_inputs = Counter(
                json.loads((output / "traces" / (window["trace"] + ".json")).read_bytes())[
                    "requests"
                ][0]["in"]
                // 1024
                for window in data["windows"]
            )
            self.assertEqual(cold_inputs, dict.fromkeys((8, 24, 48, 80), 4))

    def test_cli_rejects_invalid_trajectory_counts_before_creating_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            for count in (0, -4, 5):
                with self.subTest(count=count):
                    stderr = io.StringIO()
                    argv = ["generator", "--output", str(output), "--trajectories", str(count)]
                    with patch("sys.argv", argv), contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as error:
                            main()
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("positive multiple of 4", stderr.getvalue())
                    self.assertFalse(output.exists())


class CompactReplayTests(unittest.TestCase):
    def test_fork_inherits_the_parent_chat_messages(self):
        item = trace("example", 2048, 1, TOOL_INPUT_TOKENS["compact"])
        turns = render_trace(item, CharacterTokenizer(), "test")
        parent = turns[1]["messages"]
        self.assertEqual(turns[2]["messages"][:-2], parent)
        self.assertEqual(turns[3]["messages"][:-2], parent)
        self.assertEqual(turns[4]["messages"][:-2], parent)
        self.assertEqual(turns[2]["messages"][-2], turns[3]["messages"][-2])
        self.assertNotEqual(turns[2]["messages"][-1], turns[3]["messages"][-1])

    def test_join_waits_for_both_children(self):
        plans = {
            f"trace-{i}": [dict(max_completion_tokens=8) for _ in DEPENDENCIES] for i in range(2)
        }
        records = io.StringIO()

        def result(_url, payload, *, timeout_s):
            return {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": payload["max_completion_tokens"],
                }
            }

        with patch("tools.bench.mini_weka_compact.request", side_effect=result):
            stats = replay(plans, "unused", 2, 12, 5, records)
        completed = {trace: set() for trace in plans}
        for line in records.getvalue().splitlines():
            row = json.loads(line)
            done = completed[row["trace"]]
            self.assertTrue(set(DEPENDENCIES[row["turn"]]) <= done)
            done.add(row["turn"])
        self.assertEqual(stats["warmup_generated_tokens"], 2)
        self.assertEqual(stats["profiling_generated_tokens"], 96)
        self.assertEqual(stats["complete_trajectories"], 2)

    def test_lazy_replay_matches_eager_replay_and_releases_finished_plans(self):
        class Rendered(list):
            pass

        with TemporaryDirectory() as directory:
            manifest = generate_manifest(Path(directory) / "compact", trajectories=4)
            plans = TracePlans(manifest, CharacterTokenizer(), "test")
            eager = {name: plans[name] for name in plans}
            rendered = []

            def checked_render(item, tokenizer, model):
                self.assertTrue(all(reference() is None for reference in rendered))
                turns = Rendered(render_trace(item, tokenizer, model))
                rendered.append(weakref.ref(turns))
                return turns

            def result(_url, payload, *, timeout_s):
                return {
                    "usage": {
                        "prompt_tokens": sum(len(msg["content"]) for msg in payload["messages"]),
                        "completion_tokens": payload["max_completion_tokens"],
                        "prompt_tokens_details": {"cached_tokens": 50},
                    }
                }

            def collect(source):
                records = io.StringIO()
                stats = replay(source, "unused", 1, 100000, 5, records)
                for key in ("wall_s", "warmup_s", "load_trajectories_s"):
                    stats.pop(key)
                rows = [json.loads(line) for line in records.getvalue().splitlines()]
                for row in rows:
                    row.pop("start_s")
                    row.pop("end_s")
                return stats, rows

            with patch("tools.bench.mini_weka_compact.request", side_effect=result):
                expected = collect(eager)
                with patch(
                    "tools.bench.mini_weka_compact.render_trace", side_effect=checked_render
                ):
                    actual = collect(plans)
            self.assertEqual(actual, expected)
            self.assertEqual(actual[0]["profiling_completed"], 27)
            self.assertEqual(actual[0]["complete_trajectories"], len(plans))
            self.assertEqual(len(rendered), len(plans))
            self.assertTrue(all(reference() is None for reference in rendered))

    def test_lazy_loader_verifies_even_the_last_unrendered_trace(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "compact"
            manifest = generate_manifest(output, trajectories=4)
            (output / "traces" / "synthetic-mini-03.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unexpected compact replay trace"):
                TracePlans(manifest, CharacterTokenizer(), "test")

    def test_cli_default_request_count(self):
        args = parse_args(
            ["--manifest", "manifest.json", "--tokenizer", "model", "--output", "results"]
        )
        self.assertEqual(args.request_count, 100000)

import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.bench.mini_weka_compact import DEPENDENCIES, render_trace, replay
from tools.bench.mini_weka_synthetic import TOOL_INPUT_TOKENS, trace


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return SimpleNamespace(ids=list(map(ord, text)))

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(map(chr, ids))


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

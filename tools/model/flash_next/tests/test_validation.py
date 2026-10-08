import copy
from types import SimpleNamespace
import unittest

import numpy as np

from tools.eval.scoring_common import context_hash, pair_probes
from tools.model.flash_next.roles import role
from tools.model.flash_next.validation import probe, validate_snapshot, baseline_probes


class ValidationTests(unittest.TestCase):
    def test_reject_snapshot_before_live_mutation(self):
        value = SimpleNamespace(shape=(2, 3), dtype="float32", device="cuda:0")
        states = {"gdn": value, "ple": value}
        good = {"states": copy.deepcopy(states), "history": [2, 4], "position": 7}
        validate_snapshot(good, states, 16, 8, 2)
        for key, bad in [
            ("position", True),
            ("position", 17),
            ("history", [1, 2, 3]),
            ("history", [-1]),
            ("history", [8]),
            ("history", [True]),
            ("states", {"gdn": value}),
        ]:
            candidate = copy.deepcopy(good)
            candidate[key] = bad
            with self.assertRaises(ValueError):
                validate_snapshot(candidate, states, 16, 8, 2)
        candidate = copy.deepcopy(good)
        candidate["states"]["ple"].device = "cpu"
        with self.assertRaises(ValueError):
            validate_snapshot(candidate, states, 16, 8, 2)
        self.assertEqual(states["ple"].device, "cuda:0")

    def test_compact_prefix_shapes_and_cpu_restore(self):
        states = {
            "3:key": SimpleNamespace(shape=(262144, 2, 256), dtype="float16", device="cuda:0"),
            "3:index": SimpleNamespace(shape=(65536, 128), dtype="float16", device="cuda:0"),
            "3:pending": SimpleNamespace(shape=(4, 128), dtype="float16", device="cuda:0"),
        }
        saved = copy.deepcopy(states)
        saved["3:key"].shape = (2051, 2, 256)
        saved["3:index"].shape = (512, 128)
        for value in saved.values():
            value.device = "cpu"
        snap = {"position": 2051, "history": [1, 2], "states": saved}
        kwargs = {"prefix_divisors": {"3:key": 1, "3:index": 4}, "allow_cpu": True}
        validate_snapshot(snap, states, 262144, 8, 2, **kwargs)
        wrong = copy.deepcopy(snap)
        wrong["states"]["3:index"].shape = (513, 128)
        with self.assertRaises(ValueError):
            validate_snapshot(wrong, states, 262144, 8, 2, **kwargs)
        with self.assertRaises(ValueError):
            validate_snapshot(snap, states, 262144, 8, 2, **dict(kwargs, allow_cpu=False))
        self.assertEqual(
            role("model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"),
            "blk.3.index_qk.weight",
        )

    def test_scores_query_reference_ids_without_argmax_requirement(self):
        prompt, target = [4, 3], [2, 1]
        candidate = probe(
            np.array([1000.0, 1002.0, 1001.0, 999.0, 998.0]), prompt, target, 1, "x", [0, 2]
        )
        baseline = probe(np.array([1002.0, 1001.0, 1000.0, 999.0, 998.0]), prompt, target, 1, "x")
        paired = pair_probes([baseline], [candidate])[0]
        self.assertEqual(paired["candidate_selected_token_id"], 1)
        self.assertEqual(candidate["context_sha256"], context_hash([4, 3, 2]))
        self.assertAlmostEqual(
            sum(np.exp(x["logprob"]) for x in candidate["top3"]),
            sum(np.exp(np.array([0.0, 2.0, 1.0])))
            / sum(np.exp(np.array([0.0, 2.0, 1.0, -1.0, -2.0]))),
        )
        with self.assertRaises(ValueError):
            probe([0.0, float("nan")], prompt, target, 0, "x")

    def test_unknown_or_ambiguous_parameter_is_rejected(self):
        name = "model.language_model.layers.1.linear_attn.A_log"
        self.assertEqual(role(name), "blk.1.ssm_a_log")
        with self.assertRaises(ValueError):
            role(name.replace("model.language_model", "modelXlanguage_model"))
        with self.assertRaises(ValueError):
            role("model.language_model.layers.1.unknown.weight")

    def test_baseline_requires_original_source_and_exact_forced_history(self):
        case = {"id": "x", "prompt_ids": [3, 4], "target_ids": [1, 2]}
        rows = [
            probe([0.0, 1.0, 2.0, 3.0, 4.0], case["prompt_ids"], case["target_ids"], i, "x")
            for i in range(2)
        ]
        quantization = {"source": "Qwen/original", "source_revision": "0" * 40}
        report = {
            "complete": True,
            "baseline_precision": "original-bf16",
            "contract": {"source": "Qwen/original", "revision": "0" * 40, "seed": 20261002},
            "probes": rows,
        }
        self.assertEqual(baseline_probes(report, quantization, [case]), rows)
        for key, value in [("complete", False), ("baseline_precision", "community-q2")]:
            bad = copy.deepcopy(report)
            bad[key] = value
            with self.assertRaises(ValueError):
                baseline_probes(bad, quantization, [case])
        bad = copy.deepcopy(report)
        bad["contract"]["revision"] = "1" * 40
        with self.assertRaises(ValueError):
            baseline_probes(bad, quantization, [case])
        bad = copy.deepcopy(report)
        bad["probes"][1]["context_sha256"] = rows[0]["context_sha256"]
        with self.assertRaises(ValueError):
            baseline_probes(bad, quantization, [case])


if __name__ == "__main__":
    unittest.main()

"""Owner KV bindings must survive prefill implementation selection."""

from orinfer_build.common.paths import ROOT


import unittest
import re

from orinfer_build.models.qwen35.parallel import attention_pointers


class AttentionPointersTests(unittest.TestCase):
    def test_packet_column_order_matches_native_pointer_table(self):
        from orinfer_build.models.qwen35.packet import COLUMNS

        source = (ROOT / "crates/orinfer-models/src/qwen3_5/rows.rs").read_text()
        declaration = re.search(r"const PACKET_COLUMNS: &\[&str\] = &\[(.*?)\];", source, re.S)
        self.assertIsNotNone(declaration)
        self.assertEqual(tuple(re.findall(r'"([^"]+)"', declaration[1])), COLUMNS)
        self.assertEqual(len(COLUMNS), len(set(COLUMNS)))

    def test_staged_prefill_cannot_replace_owner_int8_kv(self):
        staged = dict(K="decoded-key", V="decoded-value", live_k="decoded-key", Q="query")
        for aliases in (False, True):
            with self.subTest(aliases=aliases):
                prepared = {
                    "live_" + name.lower() if aliases else name: {"name": "owner-" + name}
                    for name in ("K", "V", "KS", "VS")
                }
                result = attention_pointers(staged, prepared)
                self.assertEqual(result["Q"], "query")
                for name in ("K", "V", "KS", "VS"):
                    self.assertEqual(result[name], {"name": "owner-" + name})
                self.assertEqual(result["live_k"], result["K"])
                self.assertEqual(staged["K"], "decoded-key")

    def test_missing_owner_scale_does_not_fall_back_to_prefill_storage(self):
        with self.assertRaisesRegex(ValueError, "Missing prepared KV pointer KS"):
            attention_pointers(dict(KS={"name": "prefill-scale"}), dict(K="k", V="v"))

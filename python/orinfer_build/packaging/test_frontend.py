"""Standalone runtime encoding must match HF's configured added tokens."""

import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers

from orinfer_build.packaging.frontend import (
    prepare_chat_template,
    prepare_generation_config,
    prepare_tokenizer,
)


class FrontendTests(unittest.TestCase):
    def test_generation_defaults_preserve_explicit_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b'{"eos_token_id": [2], "temperature": 0.7}\r\n'
            (root / "generation_config.json").write_bytes(original)
            prepare_generation_config(root, root / "prepared.json")
            self.assertEqual((root / "prepared.json").read_bytes(), original)

    def test_missing_generation_defaults_keep_model_and_chat_terminators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(models.WordLevel({"?": 0, "end": 1, "im_end": 2}, "?"))
            tokenizer.save(str(root / "tokenizer.json"))
            (root / "tokenizer_config.json").write_text(json.dumps(dict(eos_token="im_end")))
            config = dict(text_config=dict(eos_token_id=1))
            (root / "config.json").write_text(json.dumps(config))
            prepare_generation_config(root, root / "prepared.json")
            self.assertEqual(
                json.loads((root / "prepared.json").read_text()),
                dict(eos_token_id=[1, 2], pad_token_id=1),
            )
            for invalid in (True, -1, 3, None):
                config["text_config"]["eos_token_id"] = invalid
                (root / "config.json").write_text(json.dumps(config))
                with self.subTest(eos=invalid), self.assertRaises(ValueError):
                    prepare_generation_config(root, root / "invalid.json")

    def test_embedded_chat_template_is_preserved_and_standalone_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedded = "{{ messages[0].content }}\n{% if enable_thinking %}<think>{% endif %}"
            (root / "tokenizer_config.json").write_text(json.dumps(dict(chat_template=embedded)))
            target = root / "prepared.jinja"
            prepare_chat_template(root, target)
            self.assertEqual(target.read_text(), embedded)
            standalone = b"{{ messages[-1].content }}\r\n"
            (root / "chat_template.jinja").write_bytes(standalone)
            prepare_chat_template(root, target)
            self.assertEqual(target.read_bytes(), standalone)

    def test_missing_empty_or_named_only_template_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for template in (None, " ", {"tool_use": "{{ messages }}"}):
                (root / "tokenizer_config.json").write_text(
                    json.dumps(dict(chat_template=template))
                )
                with self.subTest(template=template), self.assertRaises(ValueError):
                    prepare_chat_template(root, root / "prepared.jinja")

    def test_configured_tokens_are_recognized_without_changing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(models.WordLevel({"?": 0, "<think>": 1, "</think>": 2}, "?"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(root / "tokenizer.json"))
            original = (root / "tokenizer.json").read_bytes()
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    dict(
                        added_tokens_decoder={
                            str(i): dict(content=s, normalized=False, special=False)
                            for i, s in ((1, "<think>"), (2, "</think>"))
                        }
                    )
                )
            )
            self.assertNotEqual(tokenizer.encode("<think></think>").ids, [1, 2])
            prepare_tokenizer(root, root / "prepared.json")
            prepared = Tokenizer.from_file(str(root / "prepared.json"))
            self.assertEqual(prepared.encode("<think></think>").ids, [1, 2])
            self.assertEqual(prepared.decode([1, 2], skip_special_tokens=True), "<think> </think>")
            self.assertEqual(prepared.get_vocab(), tokenizer.get_vocab())
            self.assertEqual((root / "tokenizer.json").read_bytes(), original)
            prepare_tokenizer(root, root / "prepared-again.json")
            self.assertEqual(
                (root / "prepared.json").read_bytes(), (root / "prepared-again.json").read_bytes()
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps(dict(added_tokens_decoder={"5": dict(content="new-token")}))
            )
            with self.assertRaises(ValueError):
                prepare_tokenizer(root, root / "bad.json")

    def test_complete_tokenizer_is_copied_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(models.WordLevel({"?": 0}, "?"))
            tokenizer.save(str(root / "tokenizer.json"))
            prepare_tokenizer(root, root / "prepared.json")
            self.assertEqual(
                (root / "tokenizer.json").read_bytes(), (root / "prepared.json").read_bytes()
            )

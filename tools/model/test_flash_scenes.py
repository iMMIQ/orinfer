"""Compiler-container CPU checks; no model weights or network required."""
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools.model.flash_scenes import scenes


class SceneTests(unittest.TestCase):
    def test_fixed_histories_and_thinking_use_actual_template_tokens(self):
        tags = ['<|im_start|>','<|im_end|>','<think>','</think>']
        backend = Tokenizer(models.WordLevel({'[UNK]':0,**{s:i+1 for i,s in enumerate(tags)}},unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',additional_special_tokens=tags)
        template = ("{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}"
                    "{{ '<|im_start|>assistant\\n<think>\\n' if enable_thinking else "
                    "'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}")
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            (Path(directory)/'chat_template.jinja').write_text(template)
            loaded,cases = scenes(directory)
        self.assertEqual(len(cases),8)
        for case in cases:
            self.assertIsInstance(case['prompt_ids'],list)
            self.assertTrue(all(type(i) is int for i in case['prompt_ids']))
            rendered = loaded.apply_chat_template(case['messages'],chat_template=template,tokenize=False,
                                                 add_generation_prompt=True,enable_thinking=case['thinking'])
            self.assertEqual(case['prompt_ids'],loaded.encode(rendered,add_special_tokens=False))
            if case['thinking']:self.assertTrue(rendered.endswith('<think>\n'))
            else:self.assertTrue(rendered.endswith('<think>\n\n</think>\n\n'))
        multi = cases[-1]
        self.assertEqual(sum(i == loaded.convert_tokens_to_ids('<|im_end|>') for i in multi['prompt_ids']),3)


if __name__ == '__main__':unittest.main()

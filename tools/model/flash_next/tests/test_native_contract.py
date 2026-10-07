"""Host validation must reject invalid draft input before planning GPU work."""
from types import SimpleNamespace
import unittest

import torch

from tools.model.flash_next.native import Model
from tools.model.flash_next.tests.test_config import fixture


class NativeContractTests(unittest.TestCase):
    def test_configuration_is_checked_before_cuda_allocation(self):
        config = fixture()
        config['text_config']['hidden_size'] = 1280
        with self.assertRaisesRegex(ValueError, 'hidden_size'):
            Model(SimpleNamespace(config=config), 512, None)

    def test_missing_and_mismatched_hidden_are_rejected_before_plan(self):
        model = Model.__new__(Model)
        model.transaction = None
        model.draft_vocab = None
        model.position = 0
        model.capacity = 512
        model.is_mtp = True
        model.position_gpu = torch.zeros(1, dtype=torch.int32)
        def forbidden(*args, **kwargs):
            self.fail('Invalid MTP condition reached planning')
        model.plan = forbidden
        for hidden in (None, torch.zeros(1, model.C, model.H),
                       torch.zeros(2, model.C, model.H, dtype=torch.float16),
                       torch.zeros(1, model.C, model.H - 1, dtype=torch.float16)):
            with self.subTest(hidden=None if hidden is None else (hidden.shape, hidden.dtype)):
                with self.assertRaisesRegex(ValueError, 'HC conditions'):
                    model.execute([37], hidden=hidden)


if __name__ == '__main__':
    unittest.main()

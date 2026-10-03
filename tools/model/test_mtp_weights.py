"""CPU checks for native FP8 draft import; run in the build container."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tools.model.mtp_weights import tensor_rows


class NativeMtpWeights(unittest.TestCase):
    def test_fp8_scale_blocks_and_partial_row_ranges(self):
        # Deliberately cross both a row and column scale boundary. Values and
        # scales are exactly representable, so this catches swapped axes and
        # treating weight_scale_inv as its reciprocal without loose tolerance.
        weight = torch.ones(256, 384).to(torch.float8_e4m3fn)
        scales = torch.tensor([[.5, 1., 2.], [4., 8., 16.]], dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'weights.safetensors'
            save_file({'mtp.fc.weight': weight, 'mtp.fc.weight_scale_inv': scales}, file)
            with safe_open(file, framework='pt', device='cpu') as source:
                actual = tensor_rows(source, 'mtp.fc.weight', 127, 130)
            expected = torch.stack((torch.cat((torch.full((128,), .5), torch.ones(128), torch.full((128,), 2.))),
                                    torch.cat((torch.full((128,), 4.), torch.full((128,), 8.), torch.full((128,), 16.))),
                                    torch.cat((torch.full((128,), 4.), torch.full((128,), 8.), torch.full((128,), 16.)))))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bf16_weights_preserve_values_without_scale_or_norm_offset(self):
        weight = torch.tensor([[.25, -.5], [1., 2.]], dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'weights.safetensors'
            save_file({'mtp.fc.weight': weight}, file)
            with safe_open(file, framework='pt', device='cpu') as source:
                actual = tensor_rows(source, 'mtp.fc.weight', 0, 2)
            torch.testing.assert_close(actual, weight.float(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

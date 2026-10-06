from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.quantization.flash_next import bf16, digest, verify


class FlashConversionTests(unittest.TestCase):
    def test_original_bf16_bit_conversion(self):
        bits = np.array([0,0x3f80,0xbf80,0x4000],np.uint16)
        np.testing.assert_array_equal(bf16(bits.tobytes(),(2,2)),[[0,1],[-1,2]])

    def test_resume_requires_matching_hash_contract_and_layout(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'quant.safetensors'
            task = {'shape':[2,3,128]}
            data = {'indices':np.zeros((2,1,3,16),np.uint16),'table':np.zeros((256,8),np.int8),
                    'scales':np.ones((2,3),np.float16),'signs':np.ones(128,np.int8)}
            save_file(data,str(path),metadata={'contract':'test'})
            record = {'sha256':digest(path)}
            verify(path,task,record,'test')
            with self.assertRaises(ValueError):verify(path,task,record,'changed')
            with self.assertRaises(ValueError):verify(path,{'shape':[1,3,128]},record,'test')
            with self.assertRaises(ValueError):verify(path,task,{'sha256':'0'*64},'test')
            data['scales'][0,0] = np.nan
            path.unlink();save_file(data,str(path),metadata={'contract':'test'})
            with self.assertRaises(ValueError):verify(path,task,{'sha256':digest(path)},'test')


if __name__ == '__main__':unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.model.safetensors_source import Source, validate_header


class OriginalSourceTests(unittest.TestCase):
    def test_local_reads_exact_expert_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = np.arange(3*5*8,dtype=np.uint16).reshape(3,5,8)
            save_file({'bank':raw},str(root/'shard.safetensors'))
            (root/'index.json').write_text(json.dumps({'weight_map':{'bank':'shard.safetensors'}}))
            s = Source(root/'index.json',directory=root,cache=root/'headers')
            self.assertEqual(s.rows('bank',1,1),raw[1].tobytes())
            self.assertEqual(s.rows('bank',0,3),raw.tobytes())
            for start,count in ((-1,1),(3,1),(0,4),(0,0)):
                with self.assertRaises(ValueError):s.rows('bank',start,count)
            with self.assertRaises(EOFError):s.read('shard.safetensors',10000,1)
            with self.assertRaises(ValueError):s.read('../escape',0,1)

    def test_reject_invalid_header_extents(self):
        for t in ({'dtype':'BF16','shape':[3,8],'data_offsets':[0,47]},
                  {'dtype':'BF16','shape':[3,-8],'data_offsets':[0,48]},
                  {'dtype':'BF16','shape':[3,8],'data_offsets':[4,52]},
                  {'dtype':'BAD','shape':[3,8],'data_offsets':[0,48]}):
            with self.assertRaises(ValueError):validate_header({'bad':t})

    def test_remote_requires_immutable_revision_and_safe_index_paths(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'index.json'
            path.write_text(json.dumps({'weight_map':{'x':'../escape'}}))
            with self.assertRaises(ValueError):Source(path,repo='Qwen/model',revision='0'*40)
            path.write_text(json.dumps({'weight_map':{'x':'weights.safetensors'}}))
            with self.assertRaises(ValueError):Source(path,repo='Qwen/model',revision='main')


if __name__ == '__main__':unittest.main()

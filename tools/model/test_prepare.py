"""Container roundtrip and failed-publication checks; no GPU or checkpoint needed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from safetensors import safe_open
from tools.model.prepare import _write_containers
from tools.model.publication import staged_directory


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.checkpoint = self.root / 'checkpoint'
        self.output = self.root / 'prepared'
        self.source.mkdir()
        self.checkpoint.mkdir()
        for name in ('generation_config.json', 'tokenizer.json'):
            (self.checkpoint / name).write_text('{}')
        (self.checkpoint / 'config.json').write_text('{"text_config":{"vocab_size":8}}')
        (self.checkpoint / 'chat_template.jinja').write_text('{{ messages }}')
        self.data = {'Packed': bytes.fromhex('10325476fffeffff'),
                     'Norm': bytes.fromhex('0080c17f')}
        self.manifest = {'schema_version': 1, 'target': 'sm_87', 'vocab': 8,
                         'weight_bytes': 12, 'weight_parameters': 16,
                         'toolchain': {'tilelang': 'test'}, 'kernels': [], 'buffers': []}
        for name, dtype in [('Packed', 'i32'), ('Norm', 'bf16')]:
            data = self.data[name]
            (self.source / f'{name}.bin').write_bytes(data)
            self.manifest['buffers'].append({
                'name': name, 'dtype': dtype, 'shape': [2], 'layout': f'test_{dtype}',
                'alignment': 256, 'access': 'read',
                'data': {'file': f'{name}.bin', 'sha256': hashlib.sha256(data).hexdigest()}})
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.source / 'model.json').write_text(json.dumps(self.manifest))

    def run_prepare(self, size=8):
        with staged_directory(self.output) as staging:
            result = _write_containers(self.source / 'model.json', self.checkpoint, staging, size)
        return result

    def test_roundtrip_all_bits_shards_and_checkpoint_files(self):
        report = self.run_prepare()
        self.assertEqual(report['shard_count'], 2)
        self.assertEqual(report['tensor_bytes'], 12)
        index = json.loads((self.output / 'cache/weights/model.safetensors.index.json').read_text())
        manifest = json.loads((self.output / 'cache/manifest.json').read_text())
        self.assertEqual(manifest['schema_version'], 2)
        for buffer in manifest['buffers']:
            name = buffer['name']
            path = self.output / 'cache/weights' / index['weight_map'][name]
            with safe_open(path, framework='np') as reader:
                self.assertEqual(reader.get_slice(name).get_shape(), [2])
                self.assertEqual(reader.metadata()[f'orin.layout.{name}'], buffer['layout'])
            raw = path.read_bytes()
            header_size = int.from_bytes(raw[:8], 'little')
            header = json.loads(raw[8:8 + header_size])
            begin, end = header[name]['data_offsets']
            self.assertEqual(raw[8 + header_size + begin:8 + header_size + end], self.data[name])
            self.assertEqual(buffer['data'], {'tensor': name, 'sha256': hashlib.sha256(self.data[name]).hexdigest()})
        self.assertEqual((self.output / 'config.json').read_bytes(), (self.checkpoint / 'config.json').read_bytes())

    def test_multiple_tensor_owners_survive_single_shard_serialization(self):
        report = self.run_prepare(size=1024)
        self.assertEqual(report['shard_count'], 1)
        path = next((self.output / 'cache/weights').glob('*.safetensors'))
        raw = path.read_bytes()
        header_size = int.from_bytes(raw[:8], 'little')
        header = json.loads(raw[8:8 + header_size])
        for name, expected in self.data.items():
            begin, end = header[name]['data_offsets']
            self.assertEqual(raw[8 + header_size + begin:8 + header_size + end], expected)

    def test_corruption_never_publishes_partial_directory(self):
        (self.source / 'Norm.bin').write_bytes(b'bad!')
        with self.assertRaisesRegex(ValueError, 'sha256'):
            self.run_prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.prepared-*')))

    def test_refuses_overwrite_vocabulary_mismatch_and_escaping_source(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_prepare()
        self.output.rmdir()
        (self.checkpoint / 'config.json').write_text('{"vocab_size":9}')
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            self.run_prepare()
        (self.checkpoint / 'config.json').write_text('{"vocab_size":8}')
        self.manifest['buffers'][0]['data']['file'] = '../source/Packed.bin'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.run_prepare()


if __name__ == '__main__':
    unittest.main()

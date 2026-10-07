"""Publication failures must not expose partial models or modify their source."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools.model.publication import atomic_model, clone_model, commit_package, load_model



class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        weights = self.source / 'cache/weights'
        weights.mkdir(parents=True)
        (weights / 'model.safetensors').write_bytes(b'immutable fixture weights')
        (weights / 'model.safetensors.index.json').write_text('{}')
        (self.source / 'config.json').write_text('{}')
        self.package = dict(kernels=[], revision=1)
        raw = json.dumps(self.package).encode()
        digest = hashlib.sha256(raw).hexdigest()
        self.operator = self.source / 'cache/packages' / digest
        self.operator.mkdir(parents=True)
        (self.operator / 'package.json').write_bytes(raw)
        self.data = dict(execution_package=digest, metadata={})
        (self.source / 'cache/model.json').write_text(json.dumps(self.data))
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, ORINFER_EXECUTION_CACHE=str(self.root / 'cache')).start()

    @patch('tools.model.publication.subprocess.run')
    def test_validation_precedes_publication_and_metadata_does_not_alias(self, validate):
        destination = self.root / 'output'
        original = (self.operator / 'package.json').read_bytes()
        with atomic_model(destination, Path('/usr/bin/true')) as staging:
            data, origin, package = load_model(self.source)
            operator = clone_model(self.source, staging, origin)
            index = staging / 'cache/weights/model.safetensors.index.json'
            index.write_text('{"changed":true}')
            package['revision'] = 2
            digest = commit_package(staging, operator, data, package)
            self.assertFalse(destination.exists())
            self.assertEqual((self.operator / 'package.json').read_bytes(), original)
        validate.assert_called_once()
        self.assertTrue(destination.exists())
        self.assertEqual((self.source / 'cache/weights/model.safetensors.index.json').read_text(), '{}')
        published = destination / 'cache/packages' / digest / 'package.json'
        self.assertEqual(hashlib.sha256(published.read_bytes()).hexdigest(), digest)
        self.assertEqual((destination / 'cache/weights/model.safetensors').stat().st_ino,
                         (self.source / 'cache/weights/model.safetensors').stat().st_ino)

    @patch('tools.model.publication.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'validator'))
    def test_failed_validation_cleans_staging_and_leaves_output_absent(self, validate):
        destination = self.root / 'output'
        with self.assertRaises(subprocess.CalledProcessError):
            with atomic_model(destination, Path('/usr/bin/true')) as staging:
                clone_model(self.source, staging, self.operator)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob('.output-*')), [])

    def test_corrupted_source_and_destination_inside_source_are_rejected(self):
        with self.assertRaises(ValueError):
            clone_model(self.source, self.source / 'nested', self.operator)
        (self.operator / 'package.json').write_text('{}')
        with self.assertRaises(ValueError):
            load_model(self.source)

    def test_model_clone_retains_cpu_tables_and_independent_metadata(self):
        cpu = self.source / 'cache/cpu'
        cpu.mkdir()
        (cpu / 'inputs.json').write_text('{"version":1}')
        (cpu / 'rows.safetensors').write_bytes(b'immutable CPU rows')
        destination = self.root / 'with-cpu'
        clone_model(self.source, destination, self.operator)
        copied = destination / 'cache/cpu'
        self.assertEqual((copied / 'rows.safetensors').stat().st_ino, (cpu / 'rows.safetensors').stat().st_ino)
        (copied / 'inputs.json').write_text('{"version":2}')
        self.assertEqual((cpu / 'inputs.json').read_text(), '{"version":1}')

    def test_compiler_failure_and_existing_output_preserve_ownership(self):
        destination = self.root / 'output'
        with self.assertRaises(RuntimeError):
            with atomic_model(destination, Path('/usr/bin/true')) as staging:
                clone_model(self.source, staging, self.operator)
                raise RuntimeError('compilation failed')
        self.assertFalse(destination.exists())
        destination.mkdir()
        with self.assertRaises(ValueError):
            with atomic_model(destination):
                self.fail('Existing output must not be entered')

    @patch('tools.model.publication.subprocess.run')
    @patch('tools.model.upgrade_mtp.bind_rows', side_effect=lambda kernel, directory, names, rows, slot: dict(kernel, name=slot))
    def test_mtp_publication_retains_descriptor_changes(self, bind, validate):
        from tools.model.upgrade_mtp import upgrade
        spec = dict(input='MtpInput', position='MtpPosition',
                    warm_plans=[dict(tokens=2, program='mtp_warm_m2')])
        self.data.update(buffer_scopes={}, metadata=dict(
            mtp=spec, vision=dict(mrope_positions='MRope', feature_index='FeatureIndex'),
            input='Input', position='Position', max_context=128, buffers=[]))
        package = dict(buffer_contracts=[], kernels=[
            dict(name='decode/begin/k1', args=[]),
            dict(name='prefill_m2/layer3/k1', args=[dict(name='MRope'), dict(name='L3_KPages')]),
            dict(name='mtp_warm_m2/body/k2', args=[]),
            dict(name='mtp_warm_m2/body/k8', args=[]),
        ])
        raw = json.dumps(package).encode()
        digest = hashlib.sha256(raw).hexdigest()
        self.operator = self.operator.with_name(digest)
        self.operator.mkdir()
        (self.operator / 'package.json').write_bytes(raw)
        self.data['execution_package'] = digest
        (self.source / 'cache/model.json').write_text(json.dumps(self.data))
        output = self.root / 'mtp'
        upgrade(self.source, output, Path('/usr/bin/true'))
        descriptor = json.loads((output / 'cache/model.json').read_text())
        self.assertEqual(descriptor['metadata']['mtp']['feature_index'], 'MtpFeatureIndex')
        self.assertEqual(descriptor['metadata']['mtp']['verification_logits'], 'SequenceLogits')
        self.assertEqual(descriptor['metadata']['buffers'][0]['shape'], [128])
        self.assertEqual(descriptor['buffer_scopes']['MtpFeatureIndex'], 'sequence')


if __name__ == '__main__':
    unittest.main()

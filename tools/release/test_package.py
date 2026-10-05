import hashlib
from pathlib import Path
import tempfile
import unittest
from tools.release.package import archive, verify


class ReleaseTests(unittest.TestCase):
    def test_reproducible_archive_and_download_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root/'bundle'
            bundle.mkdir()
            (bundle/'source.txt').write_text('fixed source')
            archive(bundle, root/'one.tar.gz', 100)
            archive(bundle, root/'two.tar.gz', 100)
            self.assertEqual((root/'one.tar.gz').read_bytes(), (root/'two.tar.gz').read_bytes())
            digest = hashlib.sha256((root/'one.tar.gz').read_bytes()).hexdigest()
            (root/'SHA256SUMS').write_text(f'{digest}  one.tar.gz\n')
            verify(root)
            (root/'one.tar.gz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify(root)

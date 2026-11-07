"""New unit tests enter an environment without editing a central test list."""

from pathlib import Path
import tempfile
import unittest

from orinfer_build.common.testing import test_modules


class DiscoveryTests(unittest.TestCase):
    def test_new_cpu_tests_are_found_without_importing_other_environments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "python/orinfer_build/models/example"
            folder.mkdir(parents=True)
            (folder / "test_new.py").write_text(
                "raise AssertionError('do not execute discovery')\n"
            )
            (folder / "test_reference.py").write_text(
                'TEST_ENVIRONMENT = "cuda"\nimport missing_torch\n'
            )
            self.assertEqual(test_modules(root), ["orinfer_build.models.example.test_new"])
            self.assertEqual(
                test_modules(root, "cuda"), ["orinfer_build.models.example.test_reference"]
            )

    def test_previously_unregistered_suites_are_selected(self):
        selected = test_modules()
        self.assertIn("orinfer_build.models.deepseek_v41.test_fuse_ffn_collectives", selected)
        self.assertIn("orinfer_build.models.flash_next.tests.test_hc_bf16", selected)

"""Cache dependencies follow production imports rather than neighboring tests."""

from pathlib import Path
import ast
import tempfile
import unittest

from orinfer_build.common.sources import component_identity
from orinfer_build.common.paths import ROOT


class SourceClosureTests(unittest.TestCase):
    def test_inlined_dependency_invalidates_cache_unrelated_sources_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "kernels/cuda"
            package.mkdir(parents=True)
            (package / "projection.py").write_text("from kernels.cuda import arithmetic\n")
            (package / "arithmetic.py").write_text("VALUE = 1\n")
            (package / "test_projection.py").write_text("assert False\n")

            def key():
                return component_identity(["kernels.cuda.projection"], root=root)

            first = key()
            (package / "test_projection.py").write_text("assert True\n")
            (root / "unrelated.py").write_text("VALUE = 9\n")
            self.assertEqual(key(), first)
            (package / "arithmetic.py").write_text("VALUE = 2\n")
            self.assertNotEqual(key(), first)

    def test_build_library_does_not_depend_on_command_implementations(self):
        violations = []
        for path in (ROOT / "python/orinfer_build").rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module]
                    if isinstance(node, ast.ImportFrom) and node.module
                    else []
                )
                if any(name.startswith("tools.") for name in modules):
                    violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

"""Discover CPU/offline unittest suites by source, with explicit environments."""

from orinfer_build.common.paths import ROOT


import argparse
import ast
from pathlib import Path
import unittest


def test_modules(root=ROOT, environment="cpu"):
    result = []
    for directory in ["python/orinfer_build", "tools", "tests"]:
        base = Path(root) / directory
        if not base.exists():
            continue
        for current, directories, files in base.walk():
            directories[:] = [
                d
                for d in directories
                if not d.startswith(".") and d not in {"environments", "__pycache__"}
            ]
            for name in files:
                if not name.startswith("test_") or not name.endswith(".py"):
                    continue
                path = current / name
                selected = (
                    "cuda"
                    if "/tests/cuda/" in str(path) or "/validation/cuda/" in str(path)
                    else "cpu"
                )
                for node in ast.parse(path.read_text()).body:
                    if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "TEST_ENVIRONMENT" for t in node.targets
                    ):
                        selected = ast.literal_eval(node.value)
                if selected not in {"cpu", "cuda", "ascend"}:
                    raise ValueError(f"Unknown test environment: {path}")
                if selected == environment:
                    relative = path.relative_to(root)
                    if relative.parts[0] == "python":
                        relative = relative.relative_to("python")
                    result.append(".".join(relative.with_suffix("").parts))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["cpu", "cuda", "ascend"], default="cpu")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    modules = test_modules(environment=args.environment)
    if not modules:
        parser.error("No test modules found for this environment")
    print(f"Selected {len(modules)} {args.environment} test modules", flush=True)
    if args.list:
        print("\n".join(modules))
        return
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in modules
    )
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())


if __name__ == "__main__":
    main()

"""Check offline module references inside the compiler image without using a GPU."""

import importlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    modules = sorted(
        path
        for directory in ("kernels", "tools")
        for path in (root / directory).rglob("*.py")
        if not path.name.startswith("test_") and path.name != "__init__.py"
    )
    failures = []
    for path in modules:
        name = ".".join(path.relative_to(root).with_suffix("").parts)
        try:
            importlib.import_module(name)
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Imported {len(modules)} offline modules")


if __name__ == "__main__":
    main()

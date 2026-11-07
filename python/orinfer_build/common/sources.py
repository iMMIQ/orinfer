"""Static offline source dependencies; imports never execute during discovery."""

from orinfer_build.common.paths import ROOT


import ast
from pathlib import Path

from orinfer_build.common.files import source_hash


def module_path(name, root=ROOT):
    base = Path(root) / ("python" if name.startswith("orinfer_build.") else "")
    path = base.joinpath(*name.split("."))
    for candidate in [path.with_suffix(".py"), path / "__init__.py"]:
        if candidate.is_file():
            return candidate
    return None


def source_closure(modules, root=ROOT):
    pending, seen, paths = list(modules), set(), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = module_path(name, root)
        if path is None:
            continue
        paths.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = (
                        name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
                    )
                    parent = package[: len(package) - node.level + 1]
                    base = (
                        ".".join([*parent, *node.module.split(".")])
                        if node.module
                        else ".".join(parent)
                    )
                else:
                    base = node.module or ""
                pending.append(base)
                pending.extend(base + "." + a.name for a in node.names)
    return sorted(paths)


def component_identity(modules, *, extra=(), root=ROOT):
    resources = Path(root) / "python/orinfer_build/common/legacy_modules.json"
    return source_hash(
        root,
        [*source_closure(modules, root), *extra, *([resources] if resources.is_file() else [])],
    )

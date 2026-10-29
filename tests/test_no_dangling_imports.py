"""Every `raster_parser` import in the package, scripts and tests names a module that exists.

Static (reads the source with `ast`, imports nothing), so heavy dependencies don't matter.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNED = ("raster_parser", "scripts", "tests")


def _imported_modules(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return [n for n in names if n == "raster_parser" or n.startswith("raster_parser.")]


def _exists(module: str) -> bool:
    path = REPO.joinpath(*module.split("."))
    return path.with_suffix(".py").is_file() or path.is_dir()  # a folder is a namespace package


def test_every_raster_parser_import_resolves() -> None:
    dangling = []
    for folder in SCANNED:
        for path in sorted((REPO / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            dangling += [
                f"{path.relative_to(REPO).as_posix()} -> {module}"
                for module in _imported_modules(tree)
                if not _exists(module)
            ]
    assert dangling == []

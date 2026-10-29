"""Package layout: shared code in `commons/`, data types in `models/`, phases named `phase_<what it does>`."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import raster_parser

PKG = Path(raster_parser.__file__).parent

# Packages that follow the layout rules; upstream's older modules are moved into them over time.
LAYOUT_PACKAGES = (
    "commons",
    "models",
    "phase_native_read",
    "phase_vector_classify",
    "phase_raster_vectorize",
)


@pytest.mark.parametrize(
    "module",
    [
        "raster_parser.phase_native_read.native_read",
        "raster_parser.phase_vector_classify.v1_vector_class.parse",
        "raster_parser.phase_raster_vectorize.v1_raster_to_vec.adapter",
    ],
)
def test_phase_modules_import(module):
    importlib.import_module(module)


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_commons_never_imports_a_phase():
    bad = [
        f"{path.relative_to(PKG)} -> {name}"
        for path in (PKG / "commons").rglob("*.py")
        for name in _imports(path)
        if name.startswith("raster_parser.phase_")
    ]
    assert bad == []


def test_data_types_are_defined_only_in_models():
    """Dataclasses and enums live in `raster_parser/models/`, never next to the code using them."""
    bad = []
    for package in LAYOUT_PACKAGES:
        if package == "models":
            continue
        for path in (PKG / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ClassDef):
                    continue
                decorated = any("dataclass" in ast.unparse(d) for d in node.decorator_list)
                enum = any(ast.unparse(b) in ("Enum", "IntEnum") for b in node.bases)
                if decorated or enum:
                    bad.append(f"{path.relative_to(PKG)}:{node.name}")
    assert bad == []


def test_models_never_import_a_phase_or_core():
    bad = [
        f"{path.relative_to(PKG)} -> {name}"
        for path in (PKG / "models").rglob("*.py")
        for name in _imports(path)
        if name.startswith(("raster_parser.phase_", "raster_parser.core"))
    ]
    assert bad == []

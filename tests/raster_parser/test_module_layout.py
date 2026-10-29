"""Package layout: snake_case modules, phases named `phase_<what it does>`, data types in `models/`,
shared code in `commons/`."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import raster_parser

PKG = Path(raster_parser.__file__).parent


@pytest.mark.parametrize(
    "module",
    [
        "raster_parser.phase_native_read.native_read",
        "raster_parser.phase_raster_vectorize.stub.stub",
        "raster_parser.phase_raster_vectorize.v1_raster_to_vec.adapter",
        "raster_parser.phase_vector_classify.stub.stub",
        "raster_parser.phase_vector_classify.v1_vector_class.parse",
        "raster_parser.phase_output_assemble.organize",
    ],
)
def test_phase_modules_import(module):
    importlib.import_module(module)


def test_every_module_and_package_name_is_snake_case():
    bad = [
        str(p.relative_to(PKG))
        for p in PKG.rglob("*")
        if p.name != "__pycache__"
        and (p.is_dir() or p.suffix in (".py", ".xba"))
        and p.stem != p.stem.lower()
    ]
    assert bad == []


def test_libreoffice_macro_module_matches_its_file():
    from raster_parser.phase_raster_vectorize.v1_raster_to_vec import (
        s1_clear_page,
        s1_libreoffice_macro,
    )

    assert s1_libreoffice_macro.MACRO_SCRIPT_FP.name == "delete_drawings.xba"
    assert s1_libreoffice_macro.MACRO_SCRIPT_FP.is_file()
    # LibreOffice names the Basic module after the file stem.
    assert "Standard.delete_drawings." in s1_clear_page.MACRO_URL


def _imports(path: Path) -> set[str]:
    import ast

    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def _backend_of(module: str) -> tuple[str, str] | None:
    parts = module.split(".")
    if len(parts) >= 3 and parts[0] == "raster_parser" and parts[1].startswith("phase_"):
        return parts[1], parts[2]
    return None


def test_backends_never_import_a_sibling_backend():
    bad = []
    for path in PKG.glob("phase_*/*/**/*.py"):
        own = _backend_of(
            ".".join(("raster_parser",) + path.relative_to(PKG).with_suffix("").parts)
        )
        for name in _imports(path):
            other = _backend_of(name)
            if other and other[0] == own[0] and other[1] != own[1] and not other[1].endswith(".py"):
                bad.append(f"{path.relative_to(PKG)} -> {name}")
    assert bad == []


def test_commons_never_imports_a_phase():
    bad = [
        f"{path.relative_to(PKG)} -> {name}"
        for path in (PKG / "commons").rglob("*.py")
        for name in _imports(path)
        if name.startswith("raster_parser.phase_")
    ]
    assert bad == []


def test_pipeline_code_logs_instead_of_printing():
    """The pipeline (core, commons, phases, extract_vectors_from_png) logs; `evaluation` report helpers may
    print their tables."""
    import ast

    pipeline = [p for p in PKG.rglob("*.py") if p.relative_to(PKG).parts[0] != "evaluation"]
    bad = [
        f"{path.relative_to(PKG)}:{node.lineno}"
        for path in pipeline
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
    ]
    assert bad == []


def test_import_leaves_sys_path_and_heavy_backends_alone():
    import subprocess
    import sys

    code = (
        "import sys; before = list(sys.path); import raster_parser; "
        "assert sys.path == before, 'sys.path changed'; "
        "heavy = [m for m in sys.modules if m.endswith(('v1_raster_to_vec.adapter', "
        "'v1_vector_class.parse', 'v2_vector_class.parse'))]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=PKG.parent)


def test_data_types_are_defined_only_in_models():
    """Dataclasses and enums live in `raster_parser/models/`; the evaluation label schema (pydantic) is
    the one other place that declares types."""
    import ast

    bad = []
    for path in PKG.rglob("*.py"):
        parts = path.relative_to(PKG).parts
        if parts[0] in ("models", "evaluation"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            decorated = any("dataclass" in ast.unparse(d) for d in node.decorator_list)
            enum = any(ast.unparse(b) in ("Enum", "IntEnum") for b in node.bases)
            if decorated or enum:
                bad.append(f"{'/'.join(parts)}:{node.name}")
    assert bad == []


def test_models_never_import_a_phase_or_core():
    bad = [
        f"{path.relative_to(PKG)} -> {name}"
        for path in (PKG / "models").rglob("*.py")
        for name in _imports(path)
        if name.startswith(("raster_parser.phase_", "raster_parser.core"))
    ]
    assert bad == []

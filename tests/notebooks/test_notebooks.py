"""Every shipped notebook is well formed, committed without outputs, and imports only what exists.
Running them is opt-in (`RUN_NOTEBOOKS=1`, needs nbconvert + ipykernel)."""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
EXPECTED = {
    "dashed_line_collinear_lab",
    "filter_showcase",
    "liu_dori_showcase",
    "pdf_dataset_timing",
    "parallel_groups_lab",
    "pattern_texture_lab",
    "vector_intersection_lab",
    "gt_fidelity_showcase",
    "gt_problem_scan",
}


def _import_or_skip(name: str):
    """Import `name`; skip when it is a third-party package that is not installed (the notebook
    group's IPython, say) -- a missing `raster_parser` / `notebooks` module still fails."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in ("raster_parser", "notebooks"):
            raise
        pytest.skip(f"{exc.name} is not installed (uv sync --group notebook)")


def _code(path: Path) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    # drop IPython magics (`%matplotlib inline`) so the cell parses as Python
    return [
        "".join(line for line in c["source"] if not line.lstrip().startswith(("%", "!")))
        for c in cells
        if c["cell_type"] == "code"
    ]


def test_expected_notebooks_are_present():
    assert {p.stem for p in NOTEBOOKS} == EXPECTED


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_committed_without_outputs(path):
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), f"{path.name} carries saved outputs"
            assert cell.get("execution_count") is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_imports_resolve(path):
    for source in _code(path):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = _import_or_skip(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name) or importlib.util.find_spec(
                        f"{node.module}.{alias.name}"
                    ), f"{path.name}: {node.module}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "rastervec" not in alias.name


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_source_package_names_left(path):
    text = path.read_text(encoding="utf-8")
    assert "rastervec" not in text and "P3_Vector_Parsing" not in text


@pytest.mark.skipif(not os.environ.get("RUN_NOTEBOOKS"), reason="opt-in: RUN_NOTEBOOKS=1")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes(path, tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=900",
            "--output-dir",
            str(tmp_path),
            path.name,
        ],
        cwd=path.parent,
        check=True,
    )

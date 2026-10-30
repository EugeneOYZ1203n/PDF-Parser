"""pipeline_report_viewer: resolve_doc_dir and the page mapping (no Tk)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("tkinter")  # the viewer is a Tk app; Ubuntu ships Tk as python3-tk

_MOD = Path(__file__).resolve().parents[2] / "scripts" / "pipeline_report_viewer.py"
_spec = importlib.util.spec_from_file_location("pipeline_report_viewer", _MOD)
prv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prv)


def test_resolve_doc_dir_ok(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"source_pdf": "x", "pages": [0]}))
    label, manifest = prv.resolve_doc_dir(tmp_path)
    assert label == tmp_path.name
    assert manifest == tmp_path / "manifest.json"


def test_resolve_doc_dir_missing(tmp_path):
    with pytest.raises(SystemExit):
        prv.resolve_doc_dir(tmp_path)


def test_cli_rejects_three_dirs(tmp_path):
    with pytest.raises(SystemExit):
        prv.main([str(tmp_path), str(tmp_path), str(tmp_path)])


def test_pages_are_the_union_of_every_folder():
    assert prv.union_pages([[1, 2], [1, 2, 3]]) == [1, 2, 3]
    assert prv.union_pages([[], [2, 0]]) == [0, 2]
    assert prv.union_pages([[]]) == []


def test_layer_pos_follows_each_folders_own_pages():
    assert prv.layer_pos([1, 2, 3], 3) == 2
    assert prv.layer_pos([2, 3], 3) == 1  # page 1 was out of range for this folder's PDF
    assert prv.layer_pos([2, 3], 1) is None


def test_a_folder_with_no_pages_exits_with_a_message(tmp_path):
    """Every configured page was out of range for the input's PDF: the report wrote `pages: []`."""
    import pymupdf

    pdf = tmp_path / "one_page.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(str(pdf))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"source_pdf": str(pdf), "pages": [], "layers": []})
    )
    with pytest.raises(SystemExit, match="no pages to show"):
        prv.ViewerApp([tmp_path])

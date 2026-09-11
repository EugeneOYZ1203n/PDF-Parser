"""pipeline_report_viewer: resolve_doc_dir (no Tk)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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

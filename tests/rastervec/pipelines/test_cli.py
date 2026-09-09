from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.pipelines import _cli
from rastervec.pipelines.sub_pipelines import ocr as ocr_mod


def test_build_arg_parser_current_has_no_fast():
    p = _cli.build_arg_parser("current")
    args = p.parse_args(["--pdf", "x.pdf", "--page", "2", "--no-fast", "-v"])
    assert args.pdf == "x.pdf" and args.page == 2 and args.no_fast and args.verbose


def test_build_arg_parser_legacy_rejects_no_fast():
    p = _cli.build_arg_parser("legacy")
    with pytest.raises(SystemExit):
        p.parse_args(["--pdf", "x.pdf", "--no-fast"])


def test_main_current_returns_zero(tmp_pdf_path, monkeypatch, text):
    def fake_recognize_unique_segments(uniques, *, recognize_fn=None):
        return [text(text="TXT", bbox=(0.0, 0.0, 1.0, 1.0), source="ocr") for _ in uniques]

    monkeypatch.setattr(ocr_mod, "_recognize_unique_segments", fake_recognize_unique_segments)
    doc = fitz.open()
    doc.new_page(width=200, height=100).insert_text((10, 20), "Hi", fontsize=10)
    path = tmp_pdf_path(doc)
    assert _cli.main("current", ["--pdf", str(path), "--page", "0", "--no-fast"]) == 0

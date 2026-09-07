from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import TextVectorResult
from rastervec.pipelines import _cli
from rastervec.pipelines.sub_pipelines import ocr as ocr_mod


class _StubRenderOCR:
    def __init__(self, backend=None):
        pass

    def recognize_segmented(self, seg, cluster, page):
        return TextVectorResult(
            paths=cluster, text="TXT", confidence=0.9, bbox=(0, 0, 1, 1),
            ocr_bbox=None, rotation_used=0, page_index=page.meta.index, words=None,
        )


def test_build_arg_parser_current_has_no_fast():
    p = _cli.build_arg_parser("current")
    args = p.parse_args(["--pdf", "x.pdf", "--page", "2", "--no-fast", "-v"])
    assert args.pdf == "x.pdf" and args.page == 2 and args.no_fast and args.verbose


def test_build_arg_parser_legacy_rejects_no_fast():
    p = _cli.build_arg_parser("legacy")
    with pytest.raises(SystemExit):
        p.parse_args(["--pdf", "x.pdf", "--no-fast"])


def test_main_current_returns_zero(tmp_pdf_path, monkeypatch):
    monkeypatch.setattr(ocr_mod, "RenderOCR", _StubRenderOCR)
    doc = fitz.open()
    doc.new_page(width=200, height=100).insert_text((10, 20), "Hi", fontsize=10)
    path = tmp_pdf_path(doc)
    assert _cli.main("current", ["--pdf", str(path), "--page", "0", "--no-fast"]) == 0

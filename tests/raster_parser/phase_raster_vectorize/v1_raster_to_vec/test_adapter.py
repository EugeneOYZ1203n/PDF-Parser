"""`v1_raster_to_vec.adapter.extract`: the run settings reach the raster pass (LibreOffice and OCR faked)."""

from __future__ import annotations

from pathlib import Path

from raster_parser.models.page import Page, PageMeta
from raster_parser.models.settings import RunSettings
from raster_parser.phase_raster_vectorize.v1_raster_to_vec import (
    adapter,
    raster_page,
    s1_clear_page,
)


def test_run_settings_are_forwarded_to_the_raster_pass(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}
    monkeypatch.setattr(
        s1_clear_page, "clear_page", lambda *a, **k: (tmp_path, tmp_path / "page_cleared.pdf")
    )
    monkeypatch.setattr(s1_clear_page, "page_is_nonempty", lambda *a, **k: True)

    def fake_raster_page(*args, **kwargs):
        seen.update(kwargs)
        return [], []

    monkeypatch.setattr(raster_page, "extract_raster_page", fake_raster_page)
    page = Page(doc_path="x.pdf", meta=PageMeta(0, 0, (0, 0, 10, 10), 0, 10, 10), pymupdf_page=None)
    adapter.extract(
        [],
        page,
        RunSettings(ocr_dpi=150, use_angle_cls=False, workers=3, dump_dir=tmp_path, debug=True),
    )
    assert seen["use_angle_cls"] is False
    assert seen["keep_intermediates"] is True  # RunSettings.debug
    assert (seen["dpi"], seen["workers"]) == (150, 3)

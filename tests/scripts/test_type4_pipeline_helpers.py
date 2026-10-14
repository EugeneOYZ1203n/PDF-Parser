"""Unit tests for Type-4 CLI helpers (collect / chunk / stems) without OCR."""

from __future__ import annotations

import pytest

pytest.skip("depends on raster_parser.models / serialisation / parsing, which were removed; this tool is kept unchanged until it is ported or removed", allow_module_level=True)


import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "type4_pipeline",
    REPO_ROOT / "scripts" / "type4_pipeline.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

_chunked = _mod._chunked
collect_pdfs = _mod.collect_pdfs
unique_stems = _mod.unique_stems
pdf_page_count = _mod.pdf_page_count
native_texts = _mod.native_texts


def test_chunked_splits_and_clamps() -> None:
    assert _chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert _chunked([1, 2, 3], 0) == [[1], [2], [3]]
    assert _chunked([], 5) == []


def test_collect_pdfs_nested_and_dedupe(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.txt").write_bytes(b"x")

    found = collect_pdfs([tmp_path, tmp_path / "a.pdf"])
    rel = sorted(p.relative_to(tmp_path).as_posix() for p in found)
    assert rel == ["a.pdf", "sub/b.pdf"]


def test_collect_pdfs_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        collect_pdfs([tmp_path / "missing"])


def test_unique_stems_disambiguates_collisions(tmp_path: Path) -> None:
    d1 = tmp_path / "alpha"
    d2 = tmp_path / "beta"
    d1.mkdir()
    d2.mkdir()
    p1 = d1 / "page.pdf"
    p2 = d2 / "page.pdf"
    p1.write_bytes(b"%PDF")
    p2.write_bytes(b"%PDF")
    stems = unique_stems([p1, p2])
    assert stems[p1] != stems[p2]
    assert "page" in stems[p1]
    assert "page" in stems[p2]


def test_pdf_page_count_and_native_texts(synthetic_pdf: Path) -> None:
    assert pdf_page_count(str(synthetic_pdf)) == 1
    import pymupdf as pymupdf

    doc = pymupdf.open(synthetic_pdf)
    try:
        words = native_texts(doc[0])
    finally:
        doc.close()
    assert {t.word for t in words} >= {"hello", "world"}

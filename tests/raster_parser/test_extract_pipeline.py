"""Unit tests for extract pipeline steps (no Paddle)."""

from __future__ import annotations

from pathlib import Path


from raster_parser.models import Vector
from raster_parser.parsing.parser import PdfRasterParser
from raster_parser.rendering.pdf_render.reconstruct import filter_text_vectors
from raster_parser.ocr.wordgrouping import cluster_by_seqno, convert_vectors_to_glyphs


def _vector(seqno: int, type_: str, *, fill: tuple | None = None) -> Vector:
    return Vector(
        items=[],
        type=type_,
        rect=(float(seqno), 0.0, float(seqno + 10), 10.0),
        fill=fill,
        color=(0.0, 0.0, 0.0),
        width=1.0,
        even_odd=False,
        close_path=False,
        dashes="",
        fill_opacity=1.0,
        stroke_opacity=1.0,
        seqno=seqno,
        layer=None,
    )


def _split_strokes(all_vectors: list[Vector]) -> tuple[list[Vector], list[Vector]]:
    fills = filter_text_vectors(all_vectors)
    fill_ids = {v.seqno for v in fills}
    strokes = [v for v in all_vectors if v.seqno not in fill_ids]
    return strokes, fills


def test_parser_context_manager_closes_doc(synthetic_pdf: Path) -> None:
    with PdfRasterParser(str(synthetic_pdf), page=0) as parser:
        parser.extract_text()
    assert parser.doc.is_closed


def test_parser_close_is_idempotent(synthetic_pdf: Path) -> None:
    parser = PdfRasterParser(str(synthetic_pdf), page=0)
    parser.close()
    parser.close()
    assert parser.doc.is_closed


def test_parser_extracts_native_text(synthetic_pdf: Path) -> None:
    parser = PdfRasterParser(str(synthetic_pdf), page=0)
    words = parser.extract_text()
    assert len(words) == 2
    assert {w.word for w in words} == {"hello", "world"}


def test_parser_extracts_drawings(synthetic_pdf: Path) -> None:
    parser = PdfRasterParser(str(synthetic_pdf), page=0)
    vectors = parser.extract_vectors()
    assert len(vectors) == 1
    assert vectors[0].type == "s"


def test_filter_text_vectors_keeps_black_fills() -> None:
    stroke = _vector(1, "s")
    fill = _vector(2, "f", fill=(0.0, 0.0, 0.0))
    _, fills = _split_strokes([stroke, fill])
    assert fills == [fill]


def test_split_strokes_excludes_fill_glyphs() -> None:
    stroke = _vector(1, "s")
    fill = _vector(2, "f", fill=(0.0, 0.0, 0.0))
    strokes, fills = _split_strokes([stroke, fill])
    assert strokes == [stroke]
    assert fills == [fill]


def test_white_fill_stays_in_strokes_not_ocr_fills() -> None:
    stroke = _vector(1, "s")
    white = _vector(2, "f", fill=(1.0, 1.0, 1.0))
    strokes, fills = _split_strokes([stroke, white])
    assert fills == []
    assert strokes == [stroke, white]


def test_cluster_by_seqno_groups_adjacent_fills() -> None:
    fills = [_vector(1, "f", fill=(0.0, 0.0, 0.0)), _vector(2, "f", fill=(0.0, 0.0, 0.0))]
    groups = cluster_by_seqno(convert_vectors_to_glyphs(fills), page_rotation=0)
    assert len(groups) == 1
    assert len(groups[0].words) == 2


def test_cluster_by_seqno_splits_spaced_fills(synthetic_pdf_with_fills: Path) -> None:
    parser = PdfRasterParser(str(synthetic_pdf_with_fills), page=0)
    fills = filter_text_vectors(parser.extract_vectors())
    groups = cluster_by_seqno(convert_vectors_to_glyphs(fills), page_rotation=0)
    assert len(fills) == 3
    assert len(groups) == 1


def test_parser_fill_pdf_vector_counts(synthetic_pdf_with_fills: Path) -> None:
    parser = PdfRasterParser(str(synthetic_pdf_with_fills), page=0)
    all_vectors = parser.extract_vectors()
    strokes, fills = _split_strokes(all_vectors)
    assert len(all_vectors) == 4
    assert len(strokes) == 1
    assert strokes[0].type == "s"
    assert len(fills) == 3
    assert all(v.type == "f" for v in fills)

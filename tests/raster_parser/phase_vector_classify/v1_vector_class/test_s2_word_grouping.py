"""v1_vector_class step 2: word grouping (copied from tests/raster_parser/test_extract_pipeline.py)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from raster_parser.models.vector import Vector
from raster_parser.models.vector_classify import Orientation
from raster_parser.phase_vector_classify.v1_vector_class.s1_fill_filter import filter_text_vectors
from raster_parser.phase_vector_classify.v1_vector_class.s2_word_grouping import (
    cluster_by_seqno,
    orientation_of_word_group,
)


def _vector(seqno: int, type_: str, *, fill: tuple | None = None) -> Vector:
    return Vector(
        items=[],
        type=type_,
        rect=(float(seqno), 0.0, float(seqno + 10), 10.0),
        fill=fill,
        color=(0.0, 0.0, 0.0),
        width=1.0,
        even_odd=False,
        closePath=False,
        dashes="",
        fill_opacity=1.0,
        stroke_opacity=1.0,
        seqno=seqno,
        layer=None,
        lineCap=0,
        lineJoin=0.0,
        scissor=None,
        blendmode=None,
        isolated=False,
        knockout=False,
        opacity=None,
        page_index=0,
    )


def _page_vectors(pdf: Path) -> list[Vector]:
    """The page's drawings as shared `Vector`s."""
    with pymupdf.open(pdf) as doc:
        return [
            Vector.from_pymupdf(d, page_index=0, seq=i)
            for i, d in enumerate(doc[0].get_drawings(extended=True))
        ]


def test_cluster_by_seqno_groups_adjacent_fills() -> None:
    fills = [_vector(1, "f", fill=(0.0, 0.0, 0.0)), _vector(2, "f", fill=(0.0, 0.0, 0.0))]
    groups = cluster_by_seqno(fills, page_rotation=0)
    assert len(groups) == 1
    assert groups[0].vectors == fills
    assert groups[0].bbox == (1.0, 0.0, 12.0, 10.0)


def test_cluster_by_seqno_splits_spaced_fills(synthetic_pdf_with_fills: Path) -> None:
    fills = filter_text_vectors(_page_vectors(synthetic_pdf_with_fills))
    groups = cluster_by_seqno(fills, page_rotation=0)
    assert len(fills) == 3
    assert len(groups) == 1


def test_fill_pdf_vector_counts(synthetic_pdf_with_fills: Path) -> None:
    all_vectors = _page_vectors(synthetic_pdf_with_fills)
    fills = filter_text_vectors(all_vectors)
    fill_ids = {v.seqno for v in fills}
    strokes = [v for v in all_vectors if v.seqno not in fill_ids]
    assert len(all_vectors) == 4
    assert len(strokes) == 1
    assert strokes[0].type == "s"
    assert len(fills) == 3
    assert all(v.type == "f" for v in fills)


@pytest.mark.parametrize(
    ("page_rotation", "horizontal_run", "expected"),
    [
        (0, True, Orientation.HORIZONTAL),
        (0, False, Orientation.VERTICAL),
        (90, True, Orientation.VERTICAL),
        (90, False, Orientation.HORIZONTAL),
        (180, True, Orientation.HORIZONTAL),
        (180, False, Orientation.VERTICAL),
        (270, True, Orientation.VERTICAL),
        (270, False, Orientation.HORIZONTAL),
    ],
)
def test_orientation_of_word_group(page_rotation: int, horizontal_run: bool, expected) -> None:
    fills = [_vector(i, "f", fill=(0.0, 0.0, 0.0)) for i in (1, 2)]
    if not horizontal_run:
        fills[1].rect = (1.0, 20.0, 11.0, 30.0)
    assert orientation_of_word_group(fills, page_rotation) == expected


def test_unknown_page_rotation_falls_back_to_page_default() -> None:
    fills = [_vector(i, "f", fill=(0.0, 0.0, 0.0)) for i in (1, 2)]
    assert orientation_of_word_group(fills, 45) == Orientation.HORIZONTAL


def test_single_glyph_falls_back_to_page_default() -> None:
    glyphs = [_vector(1, "f", fill=(0.0, 0.0, 0.0))]
    assert orientation_of_word_group(glyphs, 0) == Orientation.HORIZONTAL
    assert orientation_of_word_group(glyphs, 270) == Orientation.VERTICAL

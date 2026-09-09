"""Tests for `Vector`/`Text`'s fitz round-trip (`from_pymupdf`/`to_pymupdf`)
and `Text`'s computed `angle()`/`quad()`.
"""
from __future__ import annotations

from math import cos, hypot, radians, sin
from pathlib import Path

import pymupdf as fitz
import pytest

from rastervec.helpers.geometry import round_color
from rastervec.models import Text, Vector

REFERENCES_DIR = Path(__file__).resolve().parents[1] / "references"
REFERENCE_PDFS = sorted(REFERENCES_DIR.glob("test_pdfs_*.pdf"))

# Fields `Vector.from_pymupdf` deliberately normalizes away from PyMuPDF's own
# raw (version-dependent) shape -- a round-trip test must compare against the
# *normalized* expected value for these, not the raw original, since
# `to_pymupdf()` faithfully reflects the normalized `Vector`, not the raw input.
_NORMALIZED_FIELDS = {"lineCap", "lineJoin", "even_odd", "isolated", "knockout", "layer", "color", "fill"}


def _expected_normalized(key: str, raw_value):
    if key == "lineCap":
        if isinstance(raw_value, (list, tuple)):
            return int(raw_value[0]) if raw_value else 0
        return int(raw_value or 0)
    if key == "lineJoin":
        return int(raw_value or 0)
    if key in ("even_odd", "isolated", "knockout"):
        return bool(raw_value)
    if key == "layer":
        return raw_value or None
    if key in ("color", "fill"):
        return round_color(raw_value)
    raise AssertionError(f"no normalization rule for {key!r}")


def _assert_vector_round_trips(drawing: dict, *, page_index: int = 0, seq: int = 0) -> None:
    vector = Vector.from_pymupdf(drawing, page_index=page_index, seq=seq)
    back = vector.to_pymupdf()
    for key, value in back.items():
        if key not in drawing:
            continue  # this PyMuPDF version doesn't carry this key at all
        expected = _expected_normalized(key, drawing[key]) if key in _NORMALIZED_FIELDS else drawing[key]
        assert value == expected, f"{key}: expected {expected!r}, got {value!r}"


# --------------------------------------------------------------------------
# Vector round-trip -- hand-built mock drawing dicts, one per item kind.
# --------------------------------------------------------------------------
def _mock_drawing(items: list[tuple], *, kind_type: str = "s", **overrides) -> dict:
    base = {
        "type": kind_type,
        "items": items,
        "color": (0.0, 0.0, 0.0),
        "fill": None,
        "width": 1.0,
        "dashes": "[] 0",
        "closePath": False,
        "lineCap": (0, 0, 0),
        "lineJoin": 0.0,
        "even_odd": False,
        "stroke_opacity": 1.0,
        "fill_opacity": None,
        "layer": "",
        "rect": fitz.Rect(0, 0, 10, 10),
        "scissor": None,
        "seqno": 0,
    }
    base.update(overrides)
    return base


def test_vector_round_trips_line_item():
    d = _mock_drawing([("l", fitz.Point(0, 0), fitz.Point(10, 10))])
    _assert_vector_round_trips(d)


def test_vector_round_trips_rect_item():
    d = _mock_drawing([("re", fitz.Rect(0, 0, 10, 10), 1)])
    _assert_vector_round_trips(d)


def test_vector_round_trips_quad_item():
    quad = fitz.Quad((0, 0), (10, 0), (0, 10), (10, 10))
    d = _mock_drawing([("qu", quad)])
    _assert_vector_round_trips(d)


def test_vector_round_trips_curve_item():
    d = _mock_drawing([
        ("c", fitz.Point(0, 0), fitz.Point(3, 0), fitz.Point(7, 10), fitz.Point(10, 10)),
    ])
    _assert_vector_round_trips(d)


def test_vector_round_trips_filled_item():
    d = _mock_drawing(
        [("re", fitz.Rect(0, 0, 20, 20), 1)],
        kind_type="f", color=None, fill=(1.0, 0.0, 0.0), fill_opacity=0.8,
    )
    vector = Vector.from_pymupdf(d, page_index=0, seq=0)
    assert vector.fill == (1.0, 0.0, 0.0)
    _assert_vector_round_trips(d)


# --------------------------------------------------------------------------
# Vector round-trip -- against the real test_pdfs_* reference fixtures.
# --------------------------------------------------------------------------
@pytest.mark.skipif(not REFERENCE_PDFS, reason="tests/references/test_pdfs_*.pdf not generated")
@pytest.mark.parametrize("pdf_path", REFERENCE_PDFS, ids=lambda p: p.stem)
def test_vector_round_trips_every_reference_pdf_drawing(pdf_path):
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        drawings = page.get_drawings()
        assert drawings, "reference PDF has no drawings"
        for seq, drawing in enumerate(drawings):
            _assert_vector_round_trips(drawing, seq=seq)
    finally:
        doc.close()


@pytest.mark.skipif(not REFERENCE_PDFS, reason="tests/references/test_pdfs_*.pdf not generated")
def test_reference_pdfs_cover_every_item_kind_and_a_filled_vector():
    for pdf_path in REFERENCE_PDFS:
        doc = fitz.open(str(pdf_path))
        try:
            drawings = doc[0].get_drawings()
            kinds = {item[0] for d in drawings for item in d["items"]}
            assert {"l", "c", "qu", "re"} <= kinds, f"{pdf_path.name} missing item kind(s): {kinds}"
            assert any(d.get("fill") is not None for d in drawings), f"{pdf_path.name} has no filled vector"
        finally:
            doc.close()


# --------------------------------------------------------------------------
# Text round-trip -- mock (word tuple, span dict) pairs, plus real PDFs.
# --------------------------------------------------------------------------
def _mock_span(*, direction=(1.0, 0.0), wmode=0, font="helv", size=10.0, color=0) -> dict:
    return {
        "size": size, "flags": 0, "font": font, "color": color,
        "ascender": 0.8, "descender": -0.2, "text": "Hi", "origin": (0.0, 10.0),
        "bbox": (0.0, 0.0, 10.0, 10.0), "dir": direction, "wmode": wmode,
    }


def test_text_round_trips_word_and_span():
    raw_word = (0.0, 0.0, 10.0, 10.0, "Hi", 1, 2, 3)
    span = _mock_span()

    text = Text.from_pymupdf(raw_word, span, page_index=5, seqno=7)
    back_word, back_span = text.to_pymupdf()

    assert back_word == raw_word
    assert back_span == span
    assert text.font == "helv"
    assert text.font_size == 10.0
    assert text.page_index == 5 and text.seqno == 7


def test_text_round_trips_with_no_matched_span():
    raw_word = (0.0, 0.0, 10.0, 10.0, "Hi", 0, 0, 0)

    text = Text.from_pymupdf(raw_word, None)
    back_word, back_span = text.to_pymupdf()

    assert back_word == raw_word
    assert back_span is None
    assert text.direction == (1.0, 0.0)
    assert text.font == "" and text.font_size == 0.0


@pytest.mark.skipif(not REFERENCE_PDFS, reason="tests/references/test_pdfs_*.pdf not generated")
@pytest.mark.parametrize("pdf_path", REFERENCE_PDFS, ids=lambda p: p.stem)
def test_text_round_trips_every_reference_pdf_word(pdf_path):
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        words = page.get_text("words")
        assert words, "reference PDF has no text"
        text_dict = page.get_text("dict")
        spans = []
        for block in text_dict["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    spans.append({**span, "dir": line["dir"], "wmode": line["wmode"]})

        for seq, raw_word in enumerate(words):
            word_bbox = fitz.Rect(raw_word[:4])
            best = max(
                spans, key=lambda s: (word_bbox & fitz.Rect(s["bbox"])).get_area(), default=None,
            )
            text = Text.from_pymupdf(raw_word, best, page_index=0, seqno=seq)
            back_word, back_span = text.to_pymupdf()
            assert back_word == raw_word
            assert back_span == best
    finally:
        doc.close()


# --------------------------------------------------------------------------
# angle() / quad() accuracy
# --------------------------------------------------------------------------
def _independent_oriented_quad(bbox, direction) -> tuple:
    """Recomputes an oriented quad independently of `helpers.geometry.
    make_oriented_quad` (projecting the bbox corners onto the direction
    axis by hand) so `Text.quad()` isn't just checked against its own
    implementation."""
    x0, y0, x1, y1 = bbox
    dx, dy = direction
    length = hypot(dx, dy)
    dx, dy = dx / length, dy / length
    nx, ny = -dy, dx
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    along = [cx * dx + cy * dy for cx, cy in corners]
    normal = [cx * nx + cy * ny for cx, cy in corners]
    a0, a1 = min(along), max(along)
    n0, n1 = min(normal), max(normal)
    ac, nc = (a0 + a1) / 2, (n0 + n1) / 2
    ha, hn = (a1 - a0) / 2, (n1 - n0) / 2

    def pt(a, n):
        return (ac * dx + nc * nx + a * dx + n * nx, ac * dy + nc * ny + a * dy + n * ny)

    return (pt(-ha, -hn), pt(ha, -hn), pt(ha, hn), pt(-ha, hn))


def _make_text(*, bbox=(0.0, 0.0, 10.0, 4.0), direction=(1.0, 0.0)) -> Text:
    return Text(
        text="x", bbox=bbox, direction=direction, origin=(bbox[0], bbox[3]),
        font="helv", font_size=10.0, color=None, flags=0,
        ascender=None, descender=None, wmode=0,
        block_no=0, line_no=0, word_no=0, page_index=0, seqno=0,
    )


@pytest.mark.parametrize("angle_deg", [0.0, 45.0, 90.0, 135.0, 180.0, -30.0])
def test_angle_matches_direction_vector(angle_deg):
    direction = (cos(radians(angle_deg)), sin(radians(angle_deg)))
    text = _make_text(direction=direction)

    got = text.angle()

    # Normalize both to the same [-180, 180) range before comparing.
    def norm(a):
        return ((a + 180) % 360) - 180

    assert norm(got) == pytest.approx(norm(angle_deg), abs=1e-6)


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 90.0, 200.0])
def test_quad_matches_independent_computation(angle_deg):
    direction = (cos(radians(angle_deg)), sin(radians(angle_deg)))
    bbox = (5.0, 5.0, 25.0, 15.0)
    text = _make_text(bbox=bbox, direction=direction)

    got = text.quad()
    expected = _independent_oriented_quad(bbox, direction)

    for (gx, gy), (ex, ey) in zip(got, expected):
        assert gx == pytest.approx(ex, abs=1e-6)
        assert gy == pytest.approx(ey, abs=1e-6)


def test_quad_axis_aligned_matches_bbox_corners():
    bbox = (0.0, 0.0, 10.0, 4.0)
    text = _make_text(bbox=bbox, direction=(1.0, 0.0))

    ul, ur, lr, ll = text.quad()

    assert ul == pytest.approx((0.0, 0.0))
    assert ur == pytest.approx((10.0, 0.0))
    assert lr == pytest.approx((10.0, 4.0))
    assert ll == pytest.approx((0.0, 4.0))

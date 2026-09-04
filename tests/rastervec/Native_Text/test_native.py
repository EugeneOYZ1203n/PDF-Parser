from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.Native_Text.native import Native
from rastervec.Reader.reader import Reader


def test_extract_text_basic_horizontal(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 20), "text": "Hello"}]}]
    )
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)
        words = Native().extract_text(page)

    assert len(words) == 1
    word = words[0]
    assert word.text == "Hello"
    assert word.angle == pytest.approx(0.0, abs=1e-6)
    assert word.orientation_source == "text-span"
    assert word.bbox[0] == pytest.approx(10, abs=1)


def test_extract_text_multi_word_span_gives_each_word_its_own_origin(
    synthetic_pdf_factory, tmp_pdf_path,
):
    # Two words on the same line/span used to both get the *span's* origin
    # (the whole span's baseline-start point), so every word beyond the
    # first rendered at the same insertion point -- see Renderer.
    # render_reconstructed_page, which draws at word.origin. Each word must
    # get its own origin, on the same baseline but at its own leading edge.
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 20), "text": "Hello World"}]}]
    )
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)
        words = Native().extract_text(page)

    assert [w.text for w in words] == ["Hello", "World"]
    hello, world = words
    assert hello.origin is not None and world.origin is not None
    assert hello.origin != world.origin
    assert hello.origin[1] == pytest.approx(world.origin[1], abs=1e-6)  # same baseline
    assert hello.origin[0] < world.origin[0]  # World starts further right
    assert world.origin[0] == pytest.approx(world.bbox[0], abs=1.0)


def test_extract_text_rotated(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [
            {
                "width": 200,
                "height": 200,
                "texts": [
                    {"point": (50, 150), "text": "VertText", "rotate": 90}
                ],
            }
        ]
    )
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)
        words = Native().extract_text(page)

    assert len(words) == 1
    word = words[0]
    assert word.text == "VertText"
    assert word.orientation_source == "text-span"
    # rotate=90 in pymupdf's insert_text produces dir=(0, -1)
    assert word.direction[0] == pytest.approx(0.0, abs=1e-6)
    assert word.direction[1] == pytest.approx(-1.0, abs=1e-6)

    # The critical regression check: the quad's long axis must follow the
    # text direction, not the axis-aligned bbox's width/height. The
    # axis-aligned bbox for this vertical text is narrow (~width) and
    # tall (~height) -- along the text's reading direction (vertical),
    # the quad's "along" extent (projected onto direction) must roughly
    # equal the axis-aligned bbox HEIGHT, and its "normal" extent must
    # roughly equal the bbox WIDTH. If _build_oriented_quad regressed to
    # using bbox.width/height directly as along/normal, this still passes
    # for exactly-vertical text (since it only swaps which axis is which,
    # not more subtly) -- so we additionally assert the quad is NOT
    # axis-aligned (its corners' x-coordinates must differ from a simple
    # bbox rectangle in the expected rotated pattern).
    quad_xs = [p[0] for p in word.quad]
    quad_ys = [p[1] for p in word.quad]
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = word.bbox

    # direction (0, -1): "along" axis is vertical, "normal" axis is
    # horizontal -- so the quad's projected along-extent (its y spread)
    # should match the bbox height, and its normal-extent (x spread)
    # should match the bbox width, i.e. the quad's bounding box coincides
    # with the original bbox for a purely-vertical direction.
    assert max(quad_xs) - min(quad_xs) == pytest.approx(bbox_x1 - bbox_x0, abs=1.0)
    assert max(quad_ys) - min(quad_ys) == pytest.approx(bbox_y1 - bbox_y0, abs=1.0)


def test_match_word_to_span_prefers_max_overlap():
    native = Native()
    bbox = fitz.Rect(0, 0, 10, 10)
    low_overlap_span = {"bbox": fitz.Rect(8, 8, 20, 20)}
    high_overlap_span = {"bbox": fitz.Rect(0, 0, 10, 10)}

    result = native._match_word_to_span(bbox, [low_overlap_span, high_overlap_span])

    assert result is high_overlap_span


def test_match_word_to_span_no_spans_returns_none():
    native = Native()
    assert native._match_word_to_span(fitz.Rect(0, 0, 10, 10), []) is None


def test_build_oriented_quad_horizontal_matches_bbox_corners():
    native = Native()
    bbox = fitz.Rect(0, 0, 10, 4)

    ul, ur, lr, ll = native._build_oriented_quad(bbox, 1.0, 0.0)

    assert ul == pytest.approx((0.0, 0.0))
    assert ur == pytest.approx((10.0, 0.0))
    assert lr == pytest.approx((10.0, 4.0))


def test_build_oriented_quad_vertical_swaps_extents():
    """Regression test for the along/normal-extent bug: for a bbox whose
    axis-aligned width/height do NOT match the text's along/normal
    extents (i.e. vertical direction on a wide-short bbox), the quad must
    still come out oriented along (dx, dy), not simply matching bbox
    corners in the naive horizontal order.
    """
    native = Native()
    # A bbox that is wide (20) and short (5) -- but the text direction is
    # vertical, so a correct implementation reorients around that
    # direction rather than reusing bbox.width as the along-extent.
    bbox = fitz.Rect(0, 0, 20, 5)

    ul, ur, lr, ll = native._build_oriented_quad(bbox, 0.0, 1.0)

    # Naive (buggy) horizontal-order corners would be:
    naive_ul = (bbox.x0, bbox.y0)
    naive_ur = (bbox.x1, bbox.y0)

    quad_ul = (round(ul[0], 3), round(ul[1], 3))
    quad_ur = (round(ur[0], 3), round(ur[1], 3))

    assert (quad_ul, quad_ur) != (naive_ul, naive_ur)


def test_seq_assigns_reading_order(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [
            {
                "width": 200,
                "height": 100,
                "texts": [
                    {"point": (10, 20), "text": "First"},
                    {"point": (10, 60), "text": "Second"},
                ],
            }
        ]
    )
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)
        words = Native().extract_text(page)

    words_by_seq = sorted(words, key=lambda w: w.seq)
    assert [w.text for w in words_by_seq] == ["First", "Second"]


def test_extract_records_carries_word_and_line_metadata(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 20), "text": "Hello World"}]}]
    )
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)
        records = Native().extract_records(page)

    assert [r.text for r in records] == ["Hello", "World"]
    hello, world = records
    assert hello.wmode == 0
    assert hello.line_no == world.line_no
    assert hello.word_no != world.word_no
    assert hello.font_size == pytest.approx(world.font_size)


def test_no_matching_span_falls_back():
    native = Native()
    bbox = fitz.Rect(0, 0, 10, 10)

    word = native._to_text_word((bbox, "orphan"), None, page_index=0, seq=0)

    assert word.angle == 0.0
    assert word.orientation_source == "fallback"
    assert word.text == "orphan"

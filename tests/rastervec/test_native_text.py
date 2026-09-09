from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec import native_text as native
from rastervec.native_text import _Span
from rastervec.Reader.reader import Reader


def _span(bbox: fitz.Rect, **overrides) -> _Span:
    defaults = dict(
        bbox=bbox, text="x", font="helv", font_size=10.0, flags=0, color=None,
        origin=None, direction=(1.0, 0.0), ascender=None, descender=None, wmode=0,
        raw={"dir": (1.0, 0.0), "wmode": 0, "font": "helv", "size": 10.0},
    )
    defaults.update(overrides)
    return _Span(**defaults)


def test_extract_text_basic_horizontal(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 20), "text": "Hello"}]}]
    )
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)
        words = native.extract_native_text(page)

    assert len(words) == 1
    word = words[0]
    assert word.text == "Hello"
    assert word.angle() == pytest.approx(0.0, abs=1e-6)
    assert word.source == "native"
    assert word.raw_span is not None
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
        words = native.extract_native_text(page)

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
        words = native.extract_native_text(page)

    assert len(words) == 1
    word = words[0]
    assert word.text == "VertText"
    assert word.raw_span is not None
    # rotate=90 in pymupdf's insert_text produces dir=(0, -1)
    assert word.direction[0] == pytest.approx(0.0, abs=1e-6)
    assert word.direction[1] == pytest.approx(-1.0, abs=1e-6)

    # The critical regression check: the quad's long axis must follow the
    # text direction, not the axis-aligned bbox's width/height. The
    # axis-aligned bbox for this vertical text is narrow (~width) and
    # tall (~height) -- along the text's reading direction (vertical),
    # the quad's "along" extent (projected onto direction) must roughly
    # equal the axis-aligned bbox HEIGHT, and its "normal" extent must
    # roughly equal the bbox WIDTH.
    quad = word.quad()
    quad_xs = [p[0] for p in quad]
    quad_ys = [p[1] for p in quad]
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = word.bbox

    # direction (0, -1): "along" axis is vertical, "normal" axis is
    # horizontal -- so the quad's projected along-extent (its y spread)
    # should match the bbox height, and its normal-extent (x spread)
    # should match the bbox width, i.e. the quad's bounding box coincides
    # with the original bbox for a purely-vertical direction.
    assert max(quad_xs) - min(quad_xs) == pytest.approx(bbox_x1 - bbox_x0, abs=1.0)
    assert max(quad_ys) - min(quad_ys) == pytest.approx(bbox_y1 - bbox_y0, abs=1.0)


def test_match_word_to_span_prefers_max_overlap():
    bbox = fitz.Rect(0, 0, 10, 10)
    low_overlap_span = _span(fitz.Rect(8, 8, 20, 20))
    high_overlap_span = _span(fitz.Rect(0, 0, 10, 10))

    result = native._match_word_to_span(bbox, [low_overlap_span, high_overlap_span])

    assert result is high_overlap_span


def test_match_word_to_span_rejects_tiny_overlap():
    bbox = fitz.Rect(0, 0, 10, 10)  # area 100
    # overlaps by only a 1x1 corner = 1% of the word's area
    barely = _span(fitz.Rect(9, 9, 30, 30))
    assert native._match_word_to_span(bbox, [barely]) is None


def test_match_word_to_span_no_spans_returns_none():
    assert native._match_word_to_span(fitz.Rect(0, 0, 10, 10), []) is None


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
        words = native.extract_native_text(page)

    words_by_seq = sorted(words, key=lambda w: w.seqno)
    assert [w.text for w in words_by_seq] == ["First", "Second"]


def test_extract_records_carries_word_and_line_metadata(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 20), "text": "Hello World"}]}]
    )
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)
        records = native.extract_native_text(page)

    assert [r.text for r in records] == ["Hello", "World"]
    hello, world = records
    assert hello.wmode == 0
    assert hello.line_no == world.line_no
    assert hello.word_no != world.word_no
    assert hello.font_size == pytest.approx(world.font_size)


def test_no_matching_span_falls_back():
    raw_word = (0.0, 0.0, 10.0, 10.0, "orphan", 0, 0, 0)

    word = native._to_word(0, 0, raw_word, None)

    assert word.angle() == 0.0
    assert word.raw_span is None
    assert word.text == "orphan"

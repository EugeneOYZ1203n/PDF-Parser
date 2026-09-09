"""Renderer tests -- combines the old `test_pdf.py`/`test_png.py`/
`test_svg.py` into one file targeting the new `Vector`/`Text` models.

Even/odd fill-rule tests are dropped: a `Vector`'s items are never
decomposed, and `_shapes.replay_drawing_paths` already replays a whole
`Vector` as one composite path carrying its own real `even_odd` flag --
see PyMuPDF's own "Drawing and Graphics" documentation page for how
`Shape.finish(even_odd=...)` handles a multi-contour fill; nothing here
re-derives that behavior, so there is nothing project-specific left to
test once `Vector` stopped being split into standalone items.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image

from rastervec.models import PageMeta, Text, Vector
from rastervec.native_text import extract_native_text
from rastervec.Reader.reader import Reader
from rastervec.renderer import (
    cluster_frame_size,
    page_points_to_pixel,
    pixel_to_page_bbox,
    render_boxes_pdf,
    render_page_svg,
    render_reconstructed_page,
    render_reconstructed_pdf,
    render_vector_cluster,
)
from rastervec.renderer.png import _cluster_frame
from rastervec.Vector.vector import extract_vectors

REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"
REFERENCE_PDFS = sorted(REFERENCES_DIR.glob("test_pdfs_*.pdf"))


def _meta(width=200.0, height=100.0) -> PageMeta:
    return PageMeta(index=0, number=1, mediabox=(0, 0, width, height), rotation=0, width=width, height=height)


# --------------------------------------------------------------------------
# render_reconstructed_page / render_reconstructed_pdf -- basic shape checks
# --------------------------------------------------------------------------
def test_render_reconstructed_page_size_matches_zoomed_page_meta():
    image = render_reconstructed_page(_meta(), zoom=2.0)
    assert image.size == (400, 200)


def test_render_reconstructed_page_draws_native_words(text):
    word = text(text="Hi", bbox=(10, 10, 30, 25))

    blank = render_reconstructed_page(_meta(), zoom=2.0)
    with_text = render_reconstructed_page(_meta(), native_words=[word], zoom=2.0)

    assert blank.convert("L").getextrema() == (255, 255)
    darkest, _lightest = with_text.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_draws_drawing_vectors(vector):
    v = vector(kind="l", bbox=(10, 10, 60, 60), color=(0, 0, 0), width=2)

    image = render_reconstructed_page(_meta(), drawing_vectors=[v], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_draws_ocr_results(text):
    result = text(text="Hello", bbox=(10, 10, 60, 30), source="ocr", confidence=0.9)

    image = render_reconstructed_page(_meta(), ocr_results=[result], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_shrinks_ocr_text_to_fit_narrow_bbox(text):
    # A long string in a narrow bbox must not raise or overflow the page --
    # width-fit should shrink the height-derived fontsize further.
    result = text(text="A very long piece of OCR'd text", bbox=(10, 10, 30, 20), source="ocr")

    image = render_reconstructed_page(_meta(), ocr_results=[result], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def _ink_pixels(image: "Image.Image") -> int:
    return int(np.count_nonzero(np.asarray(image.convert("L")) < 250))


def test_render_reconstructed_page_ocr_text_sized_from_bbox_height_not_font_size(text):
    # OCR `Text` carries font_size=0.0 (it's never measured). The old
    # `_place_word` path rendered it at ~1 pt; it must now be sized from
    # the bbox height, the same as an equivalent ground-truth text box.
    bbox = (20.0, 20.0, 150.0, 55.0)
    ocr_word = text(text="SCHEDULE", bbox=bbox, source="ocr", font_size=0.0)

    from_ocr = render_reconstructed_page(_meta(), ocr_results=[ocr_word], zoom=3.0)
    from_box = render_reconstructed_page(_meta(), text_boxes=[("SCHEDULE", bbox, 0.0)], zoom=3.0)

    ocr_ink = _ink_pixels(from_ocr)
    assert ocr_ink > 400  # nowhere near a 1 pt rendering
    # both paths go through the same height-sizing helper -> same ink
    assert ocr_ink == pytest.approx(_ink_pixels(from_box), rel=0.02)


def test_render_reconstructed_page_fills_width_by_widening_word_gaps(text):
    # A short multi-word string in a wide bbox: the gaps between words are
    # widened so ink reaches both the left and right edges of the bbox.
    bbox = (10.0, 40.0, 190.0, 60.0)
    word = text(text="PANEL SCHEDULE", bbox=bbox, source="ocr", font_size=0.0)

    image = render_reconstructed_page(_meta(), ocr_results=[word], zoom=4.0)
    W, H = image.size
    band = round(W * 0.08)
    assert _has_ink(image.crop((0, 0, band, H))), "text does not reach the left edge"
    assert _has_ink(image.crop((W - band, 0, W, H))), "text does not reach the right edge"


def test_render_reconstructed_pdf_multiword_text_stays_word_searchable():
    # Widening word gaps must keep each word a single drawn token, so the
    # selectable-text output of render_reconstructed_pdf stays greppable.
    pdf_bytes = render_reconstructed_pdf(
        _meta(width=400.0), text_boxes=[("PANEL SCHEDULE NOTES", (10, 10, 390, 34), 0.0)],
    )
    doc = fitz.open("pdf", pdf_bytes)
    try:
        page_text = doc[0].get_text()
        assert "PANEL" in page_text
        assert "SCHEDULE" in page_text
        assert "NOTES" in page_text
    finally:
        doc.close()


def test_render_reconstructed_page_arbitrary_angle_text_does_not_raise(text):
    word = text(text="Hi", bbox=(10, 10, 30, 25), direction=(0.6, 0.8))  # ~53 degrees

    image = render_reconstructed_page(_meta(), native_words=[word], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def _reconstructed_line_dirs(pdf_bytes: bytes) -> list[tuple[float, float]]:
    doc = fitz.open("pdf", pdf_bytes)
    try:
        return [
            tuple(round(c, 3) for c in ln["dir"])
            for b in doc[0].get_text("dict")["blocks"]
            for ln in b.get("lines", [])
        ]
    finally:
        doc.close()


@pytest.mark.parametrize("direction", [(0.0, -1.0), (0.0, 1.0), (0.6, 0.8), (-0.6, 0.8)])
def test_render_reconstructed_pdf_preserves_native_text_direction(text, direction):
    # A word whose direction has a non-zero y component must reconstruct
    # with that same direction -- not mirrored about the x-axis (the morph
    # rotation turns opposite to the get_text `dir` angle convention, so the
    # renderer negates the angle).
    import math

    n = math.hypot(*direction)
    expected = (round(direction[0] / n, 3), round(direction[1] / n, 3))
    word = text(text="Xy", bbox=(90, 40, 110, 160), direction=direction, font_size=12)

    pdf = render_reconstructed_pdf(_meta(width=200, height=200), native_words=[word])

    assert _reconstructed_line_dirs(pdf) == [expected]


def test_render_reconstructed_pdf_preserves_ocr_and_label_text_rotation(text):
    ocr_word = text(text="Up", bbox=(90, 40, 110, 160), direction=(0.0, 1.0), source="ocr")
    pdf_ocr = render_reconstructed_pdf(_meta(width=200, height=200), ocr_results=[ocr_word])
    assert _reconstructed_line_dirs(pdf_ocr) == [(0.0, 1.0)]

    pdf_box = render_reconstructed_pdf(
        _meta(width=200, height=200), text_boxes=[("Up", (90, 40, 110, 160), 90.0)],
    )
    assert _reconstructed_line_dirs(pdf_box) == [(0.0, 1.0)]


def test_render_reconstructed_pdf_reproduces_blend_mode(vector):
    # A Multiply-blended stroke over a solid fill must composite the way the
    # source did (here red x cyan -> ~black), not paint fully opaque red.
    cyan = vector(kind="re", bbox=(0, 70, 200, 130), fill=(0, 1, 1))
    red = vector(kind="l", bbox=(0, 100, 200, 100), color=(1, 0, 0), width=30, blendmode="Multiply")

    pdf = render_reconstructed_pdf(_meta(width=200, height=200), drawing_vectors=[cyan, red])

    doc = fitz.open("pdf", pdf)
    try:
        assert len(doc[0].get_drawings()) == 2
        pm = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        px = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)
        crossing = px[200, 200]  # centre of the page, on the red line over cyan
        assert int(crossing.max()) < 60, f"blend not applied, got {crossing.tolist()}"
    finally:
        doc.close()


def test_render_reconstructed_page_skips_blank_text(text):
    blank_word = text(text="   ")

    image = render_reconstructed_page(_meta(), native_words=[blank_word], zoom=2.0)

    assert image.convert("L").getextrema() == (255, 255)


def test_render_reconstructed_page_draws_text_boxes():
    blank = render_reconstructed_page(_meta(), zoom=2.0)
    with_text = render_reconstructed_page(
        _meta(), text_boxes=[("Ground truth", (10, 10, 120, 30), 0.0)], zoom=2.0,
    )

    assert blank.convert("L").getextrema() == (255, 255)
    darkest, _lightest = with_text.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_pdf_returns_openable_pdf_with_text_and_drawings(vector):
    v = vector(kind="l", bbox=(10, 40, 90, 40), color=(0, 0, 0), width=2)

    pdf_bytes = render_reconstructed_pdf(
        _meta(), text_boxes=[("HELLO", (10, 10, 90, 30), 0.0)], drawing_vectors=[v],
    )

    doc = fitz.open("pdf", pdf_bytes)
    try:
        assert doc.page_count == 1
        page = doc[0]
        assert "HELLO" in page.get_text()
        assert len(page.get_drawings()) >= 1
    finally:
        doc.close()


def test_render_reconstructed_pdf_page_size_matches_page_meta():
    doc = fitz.open("pdf", render_reconstructed_pdf(_meta()))
    try:
        assert (round(doc[0].rect.width), round(doc[0].rect.height)) == (200, 100)
    finally:
        doc.close()


def test_render_boxes_pdf_accepts_2_and_3_tuples():
    boxes = [
        ((10, 10, 50, 40), (0.0, 0.7, 0.0), "[4 3] 0"),  # dashed
        ((60, 10, 100, 40), (0.85, 0.0, 0.0), None),      # solid via explicit None
        ((110, 10, 150, 40), (0.95, 0.75, 0.0)),          # solid, legacy 2-tuple
    ]

    doc = fitz.open("pdf", render_boxes_pdf(_meta(), boxes))
    try:
        assert (round(doc[0].rect.width), round(doc[0].rect.height)) == (200, 100)
        assert len(doc[0].get_drawings()) >= 3
    finally:
        doc.close()


# --------------------------------------------------------------------------
# New: text-scaling / perimeter-coverage -- reconstructed text must
# actually reach the edges of its own bbox, not render shrunken into a
# corner or the center.
# --------------------------------------------------------------------------
def _has_ink(band: "Image.Image") -> bool:
    darkest, _lightest = band.convert("L").getextrema()
    return darkest < 250


def test_reconstructed_text_reaches_all_four_perimeter_bands():
    # Left/right is a genuine width-fit check: a bbox sized to the text's
    # own natural width (plus a small margin) leaves render_reconstructed_
    # page's width fill (here a single word -> horizontal stretch) almost
    # nothing to do, so ink must span essentially the whole width and
    # reach both the left and right 10% bands.
    #
    # Top/bottom is deliberately a *positioning* check instead of the same
    # kind of scaling check: a font's nominal ascender/descender metrics
    # (what height-derived fontsize is computed from) reach noticeably
    # higher/lower than any real glyph's actual ink for a short, arbitrary
    # string (e.g. an all-caps string has no descender ink at all) -- no
    # choice of text reliably touches the literal top/bottom edge of a
    # tightly-fit bbox for reasons that have nothing to do with the
    # renderer being tested. Placing separate text right at the top edge
    # and right at the bottom edge of a taller page instead checks the
    # thing that's actually meaningful here: text placed near an edge
    # visibly renders there, not compressed away from it.
    text_str = "MW"
    fontsize = 24.0
    margin = 2.0
    base_font = fitz.Font("helv")
    text_width = base_font.text_length(text_str, fontsize=fontsize)
    line_height = fontsize * (base_font.ascender - base_font.descender)

    width = text_width + 2 * margin
    height = 300.0
    meta = _meta(width=width, height=height)
    zoom = 4.0

    boxes = [
        (text_str, (margin, height / 2 - line_height / 2, width - margin, height / 2 + line_height / 2), 0.0),
        ("Top", (margin, 2, width - margin, 2 + line_height), 0.0),
        ("Bot", (margin, height - 2 - line_height, width - margin, height - 2), 0.0),
    ]
    image = render_reconstructed_page(meta, text_boxes=boxes, zoom=zoom)
    W, H = image.size
    band_w, band_h = round(W * 0.10), round(H * 0.10)

    left_band = image.crop((0, 0, band_w, H))
    right_band = image.crop((W - band_w, 0, W, H))
    top_band = image.crop((0, 0, W, band_h))
    bottom_band = image.crop((0, H - band_h, W, H))

    assert _has_ink(left_band), "reconstructed text does not reach the left 10% band"
    assert _has_ink(right_band), "reconstructed text does not reach the right 10% band"
    assert _has_ink(top_band), "reconstructed text does not reach the top 10% band"
    assert _has_ink(bottom_band), "reconstructed text does not reach the bottom 10% band"


# --------------------------------------------------------------------------
# New: extract -> render -> re-extract round trip against a reference PDF.
# --------------------------------------------------------------------------
def _page_similarity(img_a: "Image.Image", img_b: "Image.Image") -> float:
    """1.0 = identical, 0.0 = maximally different (mean absolute grayscale
    pixel difference, normalized to [0, 1])."""
    a = np.asarray(img_a.convert("L").resize((200, 200)), dtype=np.float64)
    b = np.asarray(img_b.convert("L").resize((200, 200)), dtype=np.float64)
    return 1.0 - float(np.mean(np.abs(a - b))) / 255.0


@pytest.mark.skipif(not REFERENCE_PDFS, reason="tests/references/test_pdfs_*.pdf not generated")
@pytest.mark.parametrize("pdf_path", REFERENCE_PDFS, ids=lambda p: p.stem)
def test_extract_render_reextract_round_trip(pdf_path):
    import io

    with Reader(str(pdf_path)) as reader:
        page = reader.get_page(0)
        native = extract_native_text(page)
        vectors = extract_vectors(page)
        original_pixmap = page.fitz_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        original_image = Image.open(io.BytesIO(original_pixmap.tobytes("png")))
        meta = page.meta

    pdf_bytes = render_reconstructed_pdf(meta, native_words=native, drawing_vectors=vectors)
    doc = fitz.open("pdf", pdf_bytes)
    try:
        generated_page = doc[0]
        generated_pixmap = generated_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        generated_image = Image.open(io.BytesIO(generated_pixmap.tobytes("png")))

        # (a) rasterized pages should be roughly similar -- text
        # reconstruction is approximate (no font-family match, base14
        # only), so this is a loose "roughly matches" tolerance, not
        # pixel-exact.
        similarity = _page_similarity(original_image, generated_image)
        assert similarity > 0.75, f"reconstructed page too different from original (similarity={similarity:.3f})"

        # (b) re-extracted native text / vectors roughly agree with the
        # originals in count -- exact text content is approximate (font
        # substitution can shift bbox-derived rewrap), so this checks
        # presence/count, not byte-identical text.
        reextracted_words = generated_page.get_text("words")
        assert len(reextracted_words) == len(native)

        reextracted_drawings = generated_page.get_drawings()
        assert len(reextracted_drawings) == len(vectors)
    finally:
        doc.close()


# --------------------------------------------------------------------------
# render_vector_cluster / pixel <-> page transform (from the old test_png.py)
# --------------------------------------------------------------------------
def test_render_vector_cluster_requires_at_least_one_vector():
    with pytest.raises(ValueError):
        render_vector_cluster([], 150)


def test_render_vector_cluster_draws_something(vector):
    v = vector(kind="re", bbox=(0, 0, 20, 10), fill=(0, 0, 0))
    image = render_vector_cluster([v], dpi=150)

    assert image.width > 0 and image.height > 0
    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255  # something was actually drawn, not a blank white page


def test_render_vector_cluster_blank_stroke_and_fill_stays_blank(vector):
    # No stroke color and no fill -> nothing should render.
    v = vector(kind="re", bbox=(0, 0, 20, 10), color=None, fill=None)
    image = render_vector_cluster([v], dpi=150)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest == 255


def test_render_vector_cluster_line_kind(vector):
    v = vector(kind="l", bbox=(0, 0, 20, 20), color=(0, 0, 0), width=2)
    image = render_vector_cluster([v], dpi=150)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_vector_cluster_reuses_doc_without_bleeding_between_calls(vector):
    """The shared per-process render document (rastervec.renderer.png's
    _get_render_doc) must never hold more than one page at a time, and
    successive calls with different content must not bleed into each
    other -- a regression test for the fitz.Document-reuse performance
    fix."""
    from rastervec.renderer import png as png_module

    filled = vector(kind="re", bbox=(0, 0, 20, 10), fill=(0, 0, 0))
    blank = vector(kind="re", bbox=(0, 0, 20, 10), color=None, fill=None)

    image_filled = render_vector_cluster([filled], dpi=150)
    assert png_module._render_doc.page_count == 0
    image_blank = render_vector_cluster([blank], dpi=150)
    assert png_module._render_doc.page_count == 0

    assert image_filled.convert("L").getextrema()[0] < 255
    assert image_blank.convert("L").getextrema()[0] == 255


def test_pixel_to_page_bbox_round_trips_cluster_frame(vector):
    v = vector(kind="re", bbox=(0, 0, 20, 10), fill=(0, 0, 0))
    dpi = 150
    zoom = dpi / 72.0
    x0, y0, pad_x, pad_y = _cluster_frame([v])

    # A pixel-space point at the padded top-left corner should map back to
    # the cluster's own bbox origin in page space.
    page_bbox = pixel_to_page_bbox(
        [v], dpi, [(pad_x * zoom, pad_y * zoom), ((pad_x + 20) * zoom, (pad_y + 10) * zoom)],
    )
    assert page_bbox == pytest.approx((x0, y0, x0 + 20, y0 + 10))


def test_page_points_to_pixel_inverts_pixel_to_page_bbox(vector):
    v = vector(kind="re", bbox=(3, 7, 23, 17), fill=(0, 0, 0))
    dpi = 200
    corners = [(5.0, 9.0), (21.0, 9.0), (21.0, 15.0), (5.0, 15.0)]

    pixel = page_points_to_pixel([v], dpi, corners)
    back = pixel_to_page_bbox([v], dpi, pixel)

    assert back == pytest.approx((5.0, 9.0, 21.0, 15.0))


def test_cluster_frame_horizontal_padding_more_generous_than_vertical(vector):
    # A tall bbox (height 200) pushes both fraction-based margins well past
    # the stroke-safety floor, so the asymmetry actually engages: pad_x
    # (30% of height) should end up well past pad_y (5% of height).
    v = vector(kind="re", bbox=(0, 0, 20, 200), fill=(0, 0, 0))
    _x0, _y0, pad_x, pad_y = _cluster_frame([v])
    assert pad_x == pytest.approx(60.0)  # 200 * 0.30
    assert pad_y == pytest.approx(10.0)  # 200 * 0.05
    assert pad_x > pad_y


def test_cluster_frame_size_matches_render_vector_cluster_bbox_plus_padding(vector):
    v = vector(kind="re", bbox=(0, 0, 20, 10), fill=(0, 0, 0))
    width, height = cluster_frame_size([v])
    x0, y0, pad_x, pad_y = _cluster_frame([v])
    assert width == pytest.approx(20 + 2 * pad_x)
    assert height == pytest.approx(10 + 2 * pad_y)


# --------------------------------------------------------------------------
# render_page_svg (from the old test_svg.py)
# --------------------------------------------------------------------------
def test_render_page_svg_returns_svg_string(synthetic_pdf_factory):
    from rastervec.models import Page

    doc = synthetic_pdf_factory([
        {"texts": [{"point": (20, 40), "text": "hello"}]},
    ])
    try:
        fitz_page = doc[0]
        page = Page(
            doc_path="<mem>",
            meta=PageMeta(
                index=0, number=1, mediabox=tuple(fitz_page.mediabox),
                rotation=fitz_page.rotation, width=fitz_page.rect.width,
                height=fitz_page.rect.height,
            ),
            fitz_page=fitz_page,
        )
        svg = render_page_svg(page)
    finally:
        doc.close()

    assert isinstance(svg, str)
    assert "<svg" in svg

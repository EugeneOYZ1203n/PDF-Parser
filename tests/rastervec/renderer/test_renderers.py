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

from rastervec.commons.models import PageMeta, Text, Vector
from rastervec.P1_Reading_Native.native_text import extract_native_text
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.commons.renderer import (
    dpi_for_cluster,
    page_points_to_pixel,
    pad_image_uniform,
    pixel_to_page_bbox,
    rasterize_pdf,
    render_boxes_pdf,
    render_cluster_with_dynamic_dpi,
    render_page_svg,
    render_quads_pdf,
    render_text_pdf,
    render_vector_cluster,
)
from rastervec.P1_Reading_Native.vector_extract import extract_vectors
from rastervec.P4_Output_Organization import render_output_page, render_output_pdf

REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"
REFERENCE_PDFS = sorted(REFERENCES_DIR.glob("test_pdfs_*.pdf"))


def _meta(width=200.0, height=100.0, rotation=0) -> PageMeta:
    return PageMeta(index=0, number=1, mediabox=(0, 0, width, height), rotation=rotation, width=width, height=height)


# --------------------------------------------------------------------------
# P4 render_output_page / render_output_pdf -- the final reconstructed page
# --------------------------------------------------------------------------
def test_render_output_page_size_matches_zoomed_page_meta():
    image = render_output_page(_meta(), [], [], zoom=2.0)
    assert image.size == (400, 200)


def test_render_output_page_draws_native_words(text):
    word = text(text="Hi", bbox=(10, 10, 30, 25))

    blank = render_output_page(_meta(), [], [], zoom=2.0)
    with_text = render_output_page(_meta(), [word], [], zoom=2.0)

    assert blank.convert("L").getextrema() == (255, 255)
    darkest, _lightest = with_text.convert("L").getextrema()
    assert darkest < 255


def test_render_output_page_draws_drawing_vectors(vector):
    v = vector(kind="l", bbox=(10, 10, 60, 60), color=(0, 0, 0), width=2)

    image = render_output_page(_meta(), [], [v], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_output_page_draws_ocr_results(text):
    result = text(text="Hello", bbox=(10, 10, 60, 30), source="ocr", confidence=0.9)

    image = render_output_page(_meta(), [result], [], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def _ink_pixels(image: "Image.Image") -> int:
    return int(np.count_nonzero(np.asarray(image.convert("L")) < 250))


def _ink_bbox(image: "Image.Image", zoom: float) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(np.asarray(image.convert("L")) < 128)
    return (xs.min() / zoom, ys.min() / zoom, (xs.max() + 1) / zoom, (ys.max() + 1) / zoom)


def test_long_text_in_narrow_box_is_squeezed_into_the_box(text):
    # Rotate-then-scale-to-fit: a long string in a narrow box is scaled
    # down along its own direction, so its ink stays inside the box.
    bbox = (10.0, 10.0, 60.0, 30.0)
    result = text(text="A very long piece of OCR text", bbox=bbox, source="ocr")

    image = render_output_page(_meta(), [result], [], zoom=4.0)

    x0, y0, x1, y1 = _ink_bbox(image, 4.0)
    assert x0 >= bbox[0] - 0.5 and x1 <= bbox[2] + 0.5
    assert y0 >= bbox[1] - 0.5 and y1 <= bbox[3] + 0.5


def test_ocr_text_sized_from_box_not_font_size(text):
    # OCR `Text` carries font_size=0.0 -- the box decides the size, the same
    # as for an equivalent ground-truth text box.
    bbox = (20.0, 20.0, 150.0, 55.0)
    ocr_word = text(text="SCHEDULE", bbox=bbox, source="ocr", font_size=0.0)

    from_ocr = render_output_page(_meta(), [ocr_word], [], zoom=3.0)
    from_box = rasterize_pdf(render_text_pdf(_meta(), [("SCHEDULE", bbox, 0.0)]), zoom=3.0)

    ocr_ink = _ink_pixels(from_ocr)
    assert ocr_ink > 400  # nowhere near a 1 pt rendering
    assert ocr_ink == pytest.approx(_ink_pixels(from_box), rel=0.02)


def test_native_text_ignores_extracted_font_size(text):
    # Native words follow the same box rule -- font_size plays no part.
    bbox = (20.0, 20.0, 150.0, 55.0)
    small = render_output_page(_meta(), [text(text="SCHEDULE", bbox=bbox, font_size=4.0)], [], zoom=3.0)
    large = render_output_page(_meta(), [text(text="SCHEDULE", bbox=bbox, font_size=40.0)], [], zoom=3.0)
    assert _ink_pixels(small) == _ink_pixels(large)


def test_multiword_text_fills_box_width_as_one_scaled_string(text):
    # A short multi-word string in a wide box is scaled along its own
    # direction (not word-gap-justified) so its ink spans the box width.
    bbox = (10.0, 40.0, 190.0, 60.0)
    word = text(text="PANEL SCHEDULE", bbox=bbox, source="ocr", font_size=0.0)

    image = render_output_page(_meta(), [word], [], zoom=4.0)

    x0, _y0, x1, _y1 = _ink_bbox(image, 4.0)
    assert x0 == pytest.approx(bbox[0], abs=2.0)
    assert x1 == pytest.approx(bbox[2], abs=2.0)


def test_render_output_pdf_multiword_text_stays_word_searchable(text):
    t = text(text="PANEL SCHEDULE NOTES", bbox=(10, 10, 390, 34), source="ocr")
    doc = fitz.open("pdf", render_output_pdf(_meta(width=400.0), [t], []))
    try:
        page_text = doc[0].get_text()
        assert "PANEL" in page_text
        assert "SCHEDULE" in page_text
        assert "NOTES" in page_text
    finally:
        doc.close()


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
@pytest.mark.parametrize("source", ["native", "ocr"])
def test_render_output_pdf_preserves_text_direction(text, direction, source):
    # A word whose direction has a non-zero y component must reconstruct
    # with that same direction -- not mirrored about the x-axis.
    import math

    n = math.hypot(*direction)
    expected = (round(direction[0] / n, 3), round(direction[1] / n, 3))
    word = text(text="Xy", bbox=(90, 40, 110, 160), direction=direction, source=source)

    pdf = render_output_pdf(_meta(width=200, height=200), [word], [])

    assert _reconstructed_line_dirs(pdf) == [expected]


def test_label_text_box_rotation_preserved():
    pdf_box = render_text_pdf(_meta(width=200, height=200), [("Up", (90, 40, 110, 160), 90.0)])
    assert _reconstructed_line_dirs(pdf_box) == [(0.0, 1.0)]


def test_rotated_text_ink_stays_inside_its_box(text):
    # A 90-degree word in a tall box: font size from the box *width* (the
    # text's own height), ink inside the box -- the old renderer sized it
    # from the axis-aligned height and spilled past the box.
    bbox = (90.0, 40.0, 110.0, 160.0)
    word = text(text="UPWARD", bbox=bbox, direction=(0.0, -1.0), source="ocr")

    image = render_output_page(_meta(width=200, height=200), [word], [], zoom=4.0)

    x0, y0, x1, y1 = _ink_bbox(image, 4.0)
    assert x0 >= bbox[0] - 0.5 and x1 <= bbox[2] + 0.5
    assert y0 == pytest.approx(bbox[1], abs=3.0) and y1 == pytest.approx(bbox[3], abs=3.0)


def test_render_output_pdf_reproduces_blend_mode(vector):
    # A Multiply-blended stroke over a solid fill must composite the way the
    # source did (here red x cyan -> ~black), not paint fully opaque red.
    cyan = vector(kind="re", bbox=(0, 70, 200, 130), fill=(0, 1, 1))
    red = vector(kind="l", bbox=(0, 100, 200, 100), color=(1, 0, 0), width=30, blendmode="Multiply")

    pdf = render_output_pdf(_meta(width=200, height=200), [], [cyan, red])

    doc = fitz.open("pdf", pdf)
    try:
        assert len(doc[0].get_drawings()) == 2
        pm = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        px = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)
        crossing = px[200, 200]  # centre of the page, on the red line over cyan
        assert int(crossing.max()) < 60, f"blend not applied, got {crossing.tolist()}"
    finally:
        doc.close()


def test_render_output_page_skips_blank_text(text):
    blank_word = text(text="   ")

    image = render_output_page(_meta(), [blank_word], [], zoom=2.0)

    assert image.convert("L").getextrema() == (255, 255)


def test_render_text_pdf_draws_text_boxes():
    blank = rasterize_pdf(render_text_pdf(_meta(), []), zoom=2.0)
    with_text = rasterize_pdf(
        render_text_pdf(_meta(), [("Ground truth", (10, 10, 120, 30), 0.0)]), zoom=2.0,
    )

    assert blank.convert("L").getextrema() == (255, 255)
    darkest, _lightest = with_text.convert("L").getextrema()
    assert darkest < 255


def test_render_output_pdf_returns_openable_pdf_with_text_and_drawings(text, vector):
    v = vector(kind="l", bbox=(10, 40, 90, 40), color=(0, 0, 0), width=2)
    t = text(text="HELLO", bbox=(10, 10, 90, 30), source="ocr")

    doc = fitz.open("pdf", render_output_pdf(_meta(), [t], [v]))
    try:
        assert doc.page_count == 1
        page = doc[0]
        assert "HELLO" in page.get_text()
        assert len(page.get_drawings()) >= 1
    finally:
        doc.close()


def test_render_output_pdf_page_size_matches_page_meta():
    doc = fitz.open("pdf", render_output_pdf(_meta(), [], []))
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


def test_render_quads_pdf_draws_rotated_quads_exactly():
    quad = ((50.0, 20.0), (90.0, 40.0), (80.0, 60.0), (40.0, 40.0))
    doc = fitz.open("pdf", render_quads_pdf(_meta(), [(quad, (0.0, 0.0, 1.0))]))
    try:
        [drawing] = doc[0].get_drawings()
        pts = set()
        for item in drawing["items"]:  # MuPDF may fold a closed 4-gon into one "qu"
            corners = [item[1].ul, item[1].ur, item[1].lr, item[1].ll] if item[0] == "qu" else item[1:]
            pts |= {(round(p.x, 2), round(p.y, 2)) for p in corners}
        assert pts == set(quad)
    finally:
        doc.close()


def _content_streams(pdf_bytes: bytes) -> int:
    doc = fitz.open("pdf", pdf_bytes)
    try:
        return sum(1 for x in doc[0].get_contents() if (doc.xref_stream(x) or b"").strip())
    finally:
        doc.close()


def test_render_boxes_pdf_batches_into_one_content_stream():
    # One `page.draw_rect` per box made one content stream each (quadratic
    # to build); every box shares one commit.
    boxes = [((i, i, i + 5, i + 5), (1.0, 0.0, 0.0) if i % 2 else (0.0, 0.0, 1.0)) for i in range(50)]
    boxes.append(((0, 0, 5, 5), (0.0, 0.0, 0.0), "[2 2] 0"))
    assert _content_streams(render_boxes_pdf(_meta(), boxes)) == 1
    assert _content_streams(render_boxes_pdf(_meta(), [])) == 0  # blank layer stays blank


def test_render_text_pdf_batches_text_into_one_content_stream():
    boxes = [(f"W{i}", (10, 5 + i * 8, 60, 12 + i * 8), 0.0) for i in range(10)]
    assert _content_streams(render_text_pdf(_meta(), boxes)) == 1
    assert _content_streams(render_text_pdf(_meta(), [])) == 0


# --------------------------------------------------------------------------
# Perimeter coverage -- reconstructed text must reach the edges of its own
# bbox, not render shrunken into a corner or the centre.
# --------------------------------------------------------------------------
def _has_ink(band: "Image.Image") -> bool:
    darkest, _lightest = band.convert("L").getextrema()
    return darkest < 250


def test_reconstructed_text_reaches_all_four_perimeter_bands():
    # Left/right is a width-fit check (the string is scaled along its own
    # direction to the box width). Top/bottom is a positioning check: a
    # font's ascender/descender span (what the box height maps to) reaches
    # past any real glyph's ink, so separate text is placed right at the
    # top and bottom edges of a tall page instead.
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
    image = rasterize_pdf(render_text_pdf(meta, boxes), zoom=zoom)
    W, H = image.size
    band_w, band_h = round(W * 0.10), round(H * 0.10)

    assert _has_ink(image.crop((0, 0, band_w, H))), "text does not reach the left 10% band"
    assert _has_ink(image.crop((W - band_w, 0, W, H))), "text does not reach the right 10% band"
    assert _has_ink(image.crop((0, 0, W, band_h))), "text does not reach the top 10% band"
    assert _has_ink(image.crop((0, H - band_h, W, H))), "text does not reach the bottom 10% band"


# --------------------------------------------------------------------------
# extract -> render -> re-extract round trip against a reference PDF.
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

    doc = fitz.open("pdf", render_output_pdf(meta, native, vectors))
    try:
        generated_page = doc[0]
        generated_pixmap = generated_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        generated_image = Image.open(io.BytesIO(generated_pixmap.tobytes("png")))

        # Text is approximate (base14 Helvetica, box-fit), so a loose
        # "roughly matches" tolerance, not pixel-exact.
        similarity = _page_similarity(original_image, generated_image)
        assert similarity > 0.75, f"reconstructed page too different from original (similarity={similarity:.3f})"

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
    """The shared per-process render document (rastervec.commons.renderer.png's
    _get_render_doc) must never hold more than one page at a time, and
    successive calls with different content must not bleed into each
    other -- a regression test for the fitz.Document-reuse performance
    fix."""
    from rastervec.commons.renderer import png as png_module

    filled = vector(kind="re", bbox=(0, 0, 20, 10), fill=(0, 0, 0))
    blank = vector(kind="re", bbox=(0, 0, 20, 10), color=None, fill=None)

    image_filled = render_vector_cluster([filled], dpi=150)
    assert png_module._render_doc.page_count == 0
    image_blank = render_vector_cluster([blank], dpi=150)
    assert png_module._render_doc.page_count == 0

    assert image_filled.convert("L").getextrema()[0] < 255
    assert image_blank.convert("L").getextrema()[0] == 255


def test_pixel_to_page_bbox_maps_origin_to_bbox_origin(vector):
    """The render carries no border, so pixel (0, 0) *is* the cluster's own
    bbox origin -- nothing to subtract."""
    v = vector(kind="re", bbox=(3, 7, 23, 17), fill=(0, 0, 0))
    dpi = 150
    zoom = dpi / 72.0

    page_bbox = pixel_to_page_bbox([v], dpi, [(0.0, 0.0), (20 * zoom, 10 * zoom)])
    assert page_bbox == pytest.approx((3, 7, 23, 17))


def test_page_points_to_pixel_inverts_pixel_to_page_bbox(vector):
    v = vector(kind="re", bbox=(3, 7, 23, 17), fill=(0, 0, 0))
    dpi = 200
    corners = [(5.0, 9.0), (21.0, 9.0), (21.0, 15.0), (5.0, 15.0)]

    pixel = page_points_to_pixel([v], dpi, corners)
    back = pixel_to_page_bbox([v], dpi, pixel)

    assert back == pytest.approx((5.0, 9.0, 21.0, 15.0))


def test_render_vector_cluster_canvas_is_exactly_the_union_bbox(vector):
    """No border of any kind: all padding lives in
    commons.renderer.ocr_prep.pad_image_uniform, and pixel_to_page_bbox
    depends on the bbox *being* the frame."""
    v = vector(kind="re", bbox=(5, 5, 25, 15), fill=(0, 0, 0))
    image = render_vector_cluster([v], dpi=144)  # zoom 2.0

    assert (image.width, image.height) == (40, 20)


def test_render_vector_cluster_degenerate_flat_cluster_still_renders(vector):
    """A zero-height cluster (a single horizontal rule) must not build a
    0-px canvas -- the canvas-side floor is a degeneracy guard, not
    padding."""
    v = vector(kind="l", bbox=(0, 10, 40, 10), color=(0, 0, 0), width=1)
    image = render_vector_cluster([v], dpi=144)

    assert image.width > 0 and image.height > 0


# --------------------------------------------------------------------------
# ocr_prep -- padding + dynamic-dpi rendering shared by the P3 backends'
# own paddle_engine.py/radon.py (moved out of per-backend duplicates)
# --------------------------------------------------------------------------
def test_pad_image_uniform_adds_border_sized_from_larger_dimension():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    padded, (pad_x, pad_y) = pad_image_uniform(img, fraction=0.1)
    assert pad_x == pad_y == 2
    assert padded.shape == (14, 24, 3)
    assert (padded[0, 0] == 255).all()


def test_pad_image_uniform_empty_input_returns_unchanged():
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    padded, offset = pad_image_uniform(img)
    assert padded is img
    assert offset == (0, 0)


def test_pad_image_uniform_handles_grayscale_and_color_the_same_way():
    gray = np.zeros((10, 20), dtype=np.uint8)
    color = np.zeros((10, 20, 3), dtype=np.uint8)

    gray_padded, gray_offset = pad_image_uniform(gray, fraction=0.1)
    color_padded, color_offset = pad_image_uniform(color, fraction=0.1)

    assert gray_offset == color_offset == (2, 2)
    assert gray_padded.shape == (14, 24)
    assert color_padded.shape == (14, 24, 3)


def test_dpi_for_cluster_bumps_small_cluster_up(vector):
    v = vector(bbox=(0.0, 0.0, 1.0, 1.0))
    assert dpi_for_cluster([v], dpi=72, padding=0.0, min_render_side_px=200, max_render_dpi=4800) > 72


def test_dpi_for_cluster_never_reduces_dpi(vector):
    v = vector(bbox=(0.0, 0.0, 1000.0, 1000.0))
    assert dpi_for_cluster(
        [v], dpi=300, padding=0.0, min_render_side_px=200, max_render_dpi=4800,
    ) == 300


def test_dpi_for_cluster_accounts_for_padding(vector):
    v = vector(bbox=(0.0, 0.0, 50.0, 50.0))
    kwargs = dict(dpi=72, min_render_side_px=200, max_render_dpi=4800)
    assert dpi_for_cluster([v], padding=0.0, **kwargs) >= dpi_for_cluster([v], padding=50.0, **kwargs)


def test_dpi_for_cluster_caps_at_max_render_dpi(vector):
    v = vector(bbox=(0.0, 0.0, 0.1, 0.1))
    assert dpi_for_cluster(
        [v], dpi=72, padding=0.0, min_render_side_px=200, max_render_dpi=600,
    ) == 600


def test_render_cluster_with_dynamic_dpi_matches_manual_composition(vector):
    v = vector(kind="re", bbox=(0.0, 0.0, 1.0, 1.0), fill=(0, 0, 0))

    image, dpi_used = render_cluster_with_dynamic_dpi(
        [v], base_dpi=72, min_render_side_px=200, max_render_dpi=4800,
    )

    expected_dpi = dpi_for_cluster(
        [v], dpi=72, padding=0.0, min_render_side_px=200, max_render_dpi=4800,
    )
    assert dpi_used == expected_dpi
    assert dpi_used > 72
    expected_image = render_vector_cluster([v], dpi_used)
    assert image.size == expected_image.size


# --------------------------------------------------------------------------
# render_page_svg (from the old test_svg.py)
# --------------------------------------------------------------------------
def test_render_page_svg_returns_svg_string(synthetic_pdf_factory):
    from rastervec.commons.models import Page

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

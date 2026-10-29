"""`commons/renderer/pdf.py` one-layer wrappers and output_assemble's `render_output_pdf` /
`render_output_page` (the final reconstructed page)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image

from raster_parser.models.source import Source
from raster_parser.models.page import PageMeta
from raster_parser.models.vector import Vector
from raster_parser.phase_native_read.native_text import extract_native_text
from raster_parser.phase_native_read.reader import Reader
from raster_parser.commons.renderer.pdf import (
    rasterize_pdf,
    render_boxes_pdf,
    render_quads_pdf,
    render_text_pdf,
)
from raster_parser.phase_native_read.vector_extract import extract_vectors
from raster_parser.phase_output_assemble import render_output_page, render_output_pdf

REFERENCES_DIR = Path(__file__).resolve().parents[3] / "references"
REFERENCE_PDFS = sorted(REFERENCES_DIR.glob("test_pdfs_*.pdf"))


def _meta(width=200.0, height=100.0, rotation=0) -> PageMeta:
    return PageMeta(
        index=0,
        number=1,
        mediabox=(0, 0, width, height),
        rotation=rotation,
        width=width,
        height=height,
    )


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
    result = text(text="Hello", bbox=(10, 10, 60, 30), source=Source.NATIVE_VECTOR, confidence=0.9)

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
    result = text(text="A very long piece of OCR text", bbox=bbox, source=Source.NATIVE_VECTOR)

    image = render_output_page(_meta(), [result], [], zoom=4.0)

    x0, y0, x1, y1 = _ink_bbox(image, 4.0)
    assert x0 >= bbox[0] - 0.5 and x1 <= bbox[2] + 0.5
    assert y0 >= bbox[1] - 0.5 and y1 <= bbox[3] + 0.5


def test_ocr_text_sized_from_box_not_font_size(text):
    # OCR `Text` carries font_size=0.0 -- the box decides the size, the same
    # as for an equivalent ground-truth text box.
    bbox = (20.0, 20.0, 150.0, 55.0)
    ocr_word = text(text="SCHEDULE", bbox=bbox, source=Source.NATIVE_VECTOR, font_size=0.0)

    from_ocr = render_output_page(_meta(), [ocr_word], [], zoom=3.0)
    from_box = rasterize_pdf(render_text_pdf(_meta(), [("SCHEDULE", bbox, 0.0)]), zoom=3.0)

    ocr_ink = _ink_pixels(from_ocr)
    assert ocr_ink > 400  # nowhere near a 1 pt rendering
    assert ocr_ink == pytest.approx(_ink_pixels(from_box), rel=0.02)


def test_native_text_ignores_extracted_font_size(text):
    # Native words follow the same box rule -- font_size plays no part.
    bbox = (20.0, 20.0, 150.0, 55.0)
    small = render_output_page(
        _meta(), [text(text="SCHEDULE", bbox=bbox, font_size=4.0)], [], zoom=3.0
    )
    large = render_output_page(
        _meta(), [text(text="SCHEDULE", bbox=bbox, font_size=40.0)], [], zoom=3.0
    )
    assert _ink_pixels(small) == _ink_pixels(large)


def test_multiword_text_fills_box_width_as_one_scaled_string(text):
    # A short multi-word string in a wide box is scaled along its own
    # direction (not word-gap-justified) so its ink spans the box width.
    bbox = (10.0, 40.0, 190.0, 60.0)
    word = text(text="PANEL SCHEDULE", bbox=bbox, source=Source.NATIVE_VECTOR, font_size=0.0)

    image = render_output_page(_meta(), [word], [], zoom=4.0)

    x0, _y0, x1, _y1 = _ink_bbox(image, 4.0)
    assert x0 == pytest.approx(bbox[0], abs=2.0)
    assert x1 == pytest.approx(bbox[2], abs=2.0)


def test_render_output_pdf_multiword_text_stays_word_searchable(text):
    t = text(text="PANEL SCHEDULE NOTES", bbox=(10, 10, 390, 34), source=Source.NATIVE_VECTOR)
    doc = pymupdf.open("pdf", render_output_pdf(_meta(width=400.0), [t], []))
    try:
        page_text = doc[0].get_text()
        assert "PANEL" in page_text
        assert "SCHEDULE" in page_text
        assert "NOTES" in page_text
    finally:
        doc.close()


def _reconstructed_line_dirs(pdf_bytes: bytes) -> list[tuple[float, float]]:
    doc = pymupdf.open("pdf", pdf_bytes)
    try:
        return [
            tuple(round(c, 3) for c in ln["dir"])
            for b in doc[0].get_text("dict")["blocks"]
            for ln in b.get("lines", [])
        ]
    finally:
        doc.close()


@pytest.mark.parametrize("direction", [(0.0, -1.0), (0.0, 1.0), (0.6, 0.8), (-0.6, 0.8)])
@pytest.mark.parametrize("source", [Source.NATIVE_TEXT, Source.NATIVE_VECTOR])
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
    word = text(text="UPWARD", bbox=bbox, direction=(0.0, -1.0), source=Source.NATIVE_VECTOR)

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

    doc = pymupdf.open("pdf", pdf)
    try:
        assert len(doc[0].get_drawings()) == 2
        pm = doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        px = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)
        crossing = px[200, 200]  # centre of the page, on the red line over cyan
        assert int(crossing.max()) < 60, f"blend not applied, got {crossing.tolist()}"
    finally:
        doc.close()


def _page_content(pdf: bytes) -> str:
    doc = pymupdf.open("pdf", pdf)
    try:
        return b"".join(doc.xref_stream(x) for x in doc[0].get_contents()).decode()
    finally:
        doc.close()


def _reextract(pdf: bytes) -> list[dict]:
    doc = pymupdf.open("pdf", pdf)
    try:
        return doc[0].get_drawings()
    finally:
        doc.close()


def test_hairline_stroke_keeps_its_colour_and_zero_width(vector):
    # A `0 w` hairline (CAD exports draw nearly every line this way) must not
    # fall through `Shape.finish(width=0)`, which drops the colour and paints
    # the stroke in the default black at 1pt.
    magenta = vector(kind="l", bbox=(10, 50, 190, 50), color=(1, 0, 1), width=0.0)

    pdf = render_output_pdf(_meta(), [], [magenta])

    content = _page_content(pdf)
    assert "0 w" in content
    assert "1 0 1 RG" in content
    [d] = _reextract(pdf)
    assert d["type"] == "s"
    assert d["width"] == 0
    assert tuple(d["color"]) == (1.0, 0.0, 1.0)


def test_hairline_fill_and_stroke_keeps_its_stroke(vector):
    tri = vector(
        type="fs",
        color=(0, 0, 0),
        fill=(0, 1, 1),
        width=0.0,
        closePath=True,
        items=[("l", (10, 10), (40, 10)), ("l", (40, 10), (10, 40)), ("l", (10, 40), (10, 10))],
        bbox=(10, 10, 40, 40),
    )

    pdf = render_output_pdf(_meta(), [], [tri])

    ops = _page_content(pdf).split()
    assert "B" in ops and "f" not in ops
    [d] = _reextract(pdf)
    assert d["type"] == "fs"
    assert d["width"] == 0
    assert [tuple(p) for it in d["items"] for p in it[1:]] == [
        (10, 10),
        (40, 10),
        (40, 10),
        (10, 40),
        (10, 40),
        (10, 10),
    ]


def test_nonzero_width_and_fill_only_unchanged(vector):
    thin = vector(kind="l", bbox=(10, 50, 190, 50), color=(0, 0, 1), width=0.5)
    fill_only = vector(kind="re", type="f", bbox=(10, 60, 50, 90), color=None, fill=(1, 0, 0))

    pdf = render_output_pdf(_meta(), [], [thin, fill_only])

    ops = _page_content(pdf).split()
    assert ".5" in ops and ops[ops.index(".5") + 1] == "w"
    assert "0" not in [ops[i - 1] for i, op in enumerate(ops) if op == "w"]
    assert [d["type"] for d in _reextract(pdf)] == ["s", "f"]


def _coloured_vectors(vector) -> list[Vector]:
    magenta_line = vector(kind="l", bbox=(0, 5, 40, 5), color=(1, 0, 1), width=0.0)
    cyan_tri = vector(
        type="f",
        color=None,
        fill=(0, 1, 1),
        items=[("l", (0, 10), (30, 10)), ("l", (30, 10), (0, 40)), ("l", (0, 40), (0, 10))],
        bbox=(0, 10, 30, 40),
    )
    return [magenta_line, cyan_tri]


def test_final_page_keeps_real_colours(vector):
    pdf = render_output_pdf(_meta(), [], _coloured_vectors(vector))
    paints = [(d["type"], d.get("color"), d.get("fill")) for d in _reextract(pdf)]
    assert paints == [("s", (1.0, 0.0, 1.0), None), ("f", None, (0.0, 1.0, 1.0))]


def test_render_output_page_skips_blank_text(text):
    blank_word = text(text="   ")

    image = render_output_page(_meta(), [blank_word], [], zoom=2.0)

    assert image.convert("L").getextrema() == (255, 255)


def test_render_text_pdf_draws_text_boxes():
    blank = rasterize_pdf(render_text_pdf(_meta(), []), zoom=2.0)
    with_text = rasterize_pdf(
        render_text_pdf(_meta(), [("Ground truth", (10, 10, 120, 30), 0.0)]),
        zoom=2.0,
    )

    assert blank.convert("L").getextrema() == (255, 255)
    darkest, _lightest = with_text.convert("L").getextrema()
    assert darkest < 255


def test_render_output_pdf_returns_openable_pdf_with_text_and_drawings(text, vector):
    v = vector(kind="l", bbox=(10, 40, 90, 40), color=(0, 0, 0), width=2)
    t = text(text="HELLO", bbox=(10, 10, 90, 30), source=Source.NATIVE_VECTOR)

    doc = pymupdf.open("pdf", render_output_pdf(_meta(), [t], [v]))
    try:
        assert doc.page_count == 1
        page = doc[0]
        assert "HELLO" in page.get_text()
        assert len(page.get_drawings()) >= 1
    finally:
        doc.close()


def test_render_output_pdf_page_size_matches_page_meta():
    doc = pymupdf.open("pdf", render_output_pdf(_meta(), [], []))
    try:
        assert (round(doc[0].rect.width), round(doc[0].rect.height)) == (200, 100)
    finally:
        doc.close()


def test_render_boxes_pdf_accepts_2_and_3_tuples():
    boxes = [
        ((10, 10, 50, 40), (0.0, 0.7, 0.0), "[4 3] 0"),  # dashed
        ((60, 10, 100, 40), (0.85, 0.0, 0.0), None),  # solid via explicit None
        ((110, 10, 150, 40), (0.95, 0.75, 0.0)),  # solid, legacy 2-tuple
    ]

    doc = pymupdf.open("pdf", render_boxes_pdf(_meta(), boxes))
    try:
        assert (round(doc[0].rect.width), round(doc[0].rect.height)) == (200, 100)
        assert len(doc[0].get_drawings()) >= 3
    finally:
        doc.close()


def test_render_quads_pdf_draws_rotated_quads_exactly():
    quad = ((50.0, 20.0), (90.0, 40.0), (80.0, 60.0), (40.0, 40.0))
    doc = pymupdf.open("pdf", render_quads_pdf(_meta(), [(quad, (0.0, 0.0, 1.0))]))
    try:
        [drawing] = doc[0].get_drawings()
        pts = set()
        for item in drawing["items"]:  # MuPDF may fold a closed 4-gon into one "qu"
            corners = (
                [item[1].ul, item[1].ur, item[1].lr, item[1].ll] if item[0] == "qu" else item[1:]
            )
            pts |= {(round(p.x, 2), round(p.y, 2)) for p in corners}
        assert pts == set(quad)
    finally:
        doc.close()


def _content_streams(pdf_bytes: bytes) -> int:
    doc = pymupdf.open("pdf", pdf_bytes)
    try:
        return sum(1 for x in doc[0].get_contents() if (doc.xref_stream(x) or b"").strip())
    finally:
        doc.close()


def test_render_boxes_pdf_batches_into_one_content_stream():
    # One `page.draw_rect` per box made one content stream each (quadratic
    # to build); every box shares one commit.
    boxes = [
        ((i, i, i + 5, i + 5), (1.0, 0.0, 0.0) if i % 2 else (0.0, 0.0, 1.0)) for i in range(50)
    ]
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
    base_font = pymupdf.Font("helv")
    text_width = base_font.text_length(text_str, fontsize=fontsize)
    line_height = fontsize * (base_font.ascender - base_font.descender)

    width = text_width + 2 * margin
    height = 300.0
    meta = _meta(width=width, height=height)
    zoom = 4.0

    boxes = [
        (
            text_str,
            (margin, height / 2 - line_height / 2, width - margin, height / 2 + line_height / 2),
            0.0,
        ),
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
        original_pixmap = page.pymupdf_page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        original_image = Image.open(io.BytesIO(original_pixmap.tobytes("png")))
        meta = page.meta

    doc = pymupdf.open("pdf", render_output_pdf(meta, native, vectors))
    try:
        generated_page = doc[0]
        generated_pixmap = generated_page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        generated_image = Image.open(io.BytesIO(generated_pixmap.tobytes("png")))

        # Text is approximate (base14 Helvetica, box-fit), so a loose
        # "roughly matches" tolerance, not pixel-exact.
        similarity = _page_similarity(original_image, generated_image)
        assert similarity > 0.75, (
            f"reconstructed page too different from original (similarity={similarity:.3f})"
        )

        reextracted_words = generated_page.get_text("words")
        assert len(reextracted_words) == len(native)

        reextracted_drawings = generated_page.get_drawings()
        assert len(reextracted_drawings) == len(vectors)
    finally:
        doc.close()


def test_layer_slug_makes_file_safe_names():
    from raster_parser.commons.renderer.pdf import layer_slug

    assert layer_slug("kept (off-grid)") == "kept_off_grid"
    assert layer_slug("!!!") == "layer"
    assert layer_slug("!!!", fallback="step") == "step"

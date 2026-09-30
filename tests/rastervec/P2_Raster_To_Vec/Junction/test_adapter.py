from __future__ import annotations

import numpy as np
import pytest

from rastervec.commons.models import Image, Page
from rastervec.P2_Raster_To_Vec.Junction import adapter
from rastervec.P2_Raster_To_Vec.Junction.paddle_engine import OcrBox

# No PaddleOCR in unit tests: detection finds nothing unless a test says so.
_NO_OCR = {
    "detect_many": lambda images: [[] for _ in images],
    "recognize": lambda crops: [OcrBox("", 0.0) for _ in crops],
    "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
}


def _two_color_image(transform=None) -> Image:
    arr = np.full((150, 200, 3), 255, dtype=np.uint8)
    arr[39:42, 10:190] = (0, 0, 0)          # black horizontal line
    arr[60:140, 99:102] = (220, 20, 20)     # red vertical line
    return Image(
        array=arr, bbox=(0.0, 0.0, 200.0, 150.0), dpi=72.0, source="embedded",
        transform=transform,
    )


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=150.0), fitz_page=None)


def test_extract_streams_layers_and_matches_batch_render_debug(page_meta):
    page = _page(page_meta)
    images = [_two_color_image()]

    seen: list[tuple] = []
    vectors, texts = adapter.extract(
        images, page, on_debug_layer=lambda *layer: seen.append(layer), ocr_fns=_NO_OCR,
    )
    assert texts == []
    streamed = [(layer[0], layer[1]) for layer in seen]
    assert len(set(streamed)) == len(streamed)  # each layer once per page

    debug_out: dict = {}
    vectors2, _ = adapter.extract(images, page, debug_out=debug_out, ocr_fns=_NO_OCR)
    batch = [(layer[0], layer[1]) for layer in adapter.render_debug(debug_out, page.meta)]

    assert streamed == batch
    assert len(vectors) == len(vectors2)
    for stage in ("color_separation", "text_removal", "enhance", "vector_render", "vector_diff", "ocr"):
        assert stage in {s for s, _ in streamed}
    for _stage, _label, hexcolor, pdf_bytes in seen:
        assert pdf_bytes[:4] == b"%PDF"
        assert hexcolor.startswith("#")
    labels = {code_label for code_label, _img, _arr in debug_out["diff_codes"]}
    assert "total" in labels and len(labels) == 3  # total + black + red layers


def test_vectors_carry_their_layer_color(page_meta):
    vectors, _ = adapter.extract([_two_color_image()], _page(page_meta), ocr_fns=_NO_OCR)
    colors = {tuple(round(c, 1) for c in v.color) for v in vectors}
    assert any(c[0] < 0.2 and c[1] < 0.2 for c in colors)   # black
    assert any(c[0] > 0.7 and c[1] < 0.3 for c in colors)   # red
    horizontal = [v for v in vectors if v.color[0] < 0.2]
    x0 = min(v.rect[0] for v in horizontal)
    x1 = max(v.rect[2] for v in horizontal)
    assert x0 < 20 and x1 > 180


def test_ocr_hits_become_texts_and_are_erased(page_meta):
    arr = np.full((150, 200, 3), 255, dtype=np.uint8)
    arr[20:32, 20:120] = 0                  # "text" block
    arr[100:103, 10:190] = 0                # a real line elsewhere
    image = Image(array=arr, bbox=(0.0, 0.0, 200.0, 150.0), dpi=72.0, source="embedded")

    def detect_text_only(images):
        """Fake detector: the top-most ink connected component only."""
        import cv2

        out = []
        for im in images:
            ink = (np.asarray(im).min(axis=2) < 128).astype(np.uint8)
            n, _lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            if n <= 1:
                out.append([])
                continue
            top = min(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_TOP])
            x0, y0, w, h = (int(v) for v in stats[top, :4])
            x1, y1 = x0 + w, y0 + h
            out.append([np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)])
        return out

    fns = {
        "detect_many": detect_text_only,
        "recognize": lambda crops: [OcrBox("HELLO", 0.95) for _ in crops],
        "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
    }
    vectors, texts = adapter.extract([image], _page(page_meta), ocr_fns=fns)
    assert [t.text for t in texts] == ["HELLO"]
    t = texts[0]
    assert t.source == "ocr"
    r, g, b = (t.color >> 16) & 255, (t.color >> 8) & 255, t.color & 255
    assert max(r, g, b) < 20  # ink centroid is (quantized) black
    assert t.bbox == pytest.approx((20, 20, 120, 32), abs=1.5)
    assert t.direction == pytest.approx((1.0, 0.0), abs=1e-6)
    # the text block was erased before tracing; only the real line is traced
    assert vectors and all(v.rect[1] > 90 for v in vectors)


def test_transform_maps_quarter_turn_placement():
    # image x -> page +y, image y -> page -x (a 90-degree placement)
    img = Image(
        array=np.zeros((10, 20), np.uint8), bbox=(50, 0, 60, 20), dpi=72.0,
        source="embedded", transform=(0.0, 20.0, -10.0, 0.0, 60.0, 0.0),
    )
    to_page = adapter._make_to_page(img, 20, 10)
    assert to_page((0, 0)) == pytest.approx((60, 0))
    assert to_page((20, 0)) == pytest.approx((60, 20))
    assert to_page((0, 10)) == pytest.approx((50, 0))
    assert adapter._px_per_pt(img, 20, 10) == pytest.approx(1.0)
    oriented = adapter._oriented_for_page(np.arange(200).reshape(10, 20), img)
    assert oriented.shape == (20, 10)
    # page top-right (row 0, last col) must be image pixel (0, 0)
    assert oriented[0, -1] == 0


def test_tracing_debug_layers_draw_points_and_lines_not_boxes(page_meta):
    import pymupdf as fitz

    arr = np.full((150, 200, 3), 255, dtype=np.uint8)
    arr[39:42, 20:180] = 0                  # T: horizontal bar
    arr[40:130, 99:102] = 0                 #    + vertical stem
    image = Image(array=arr, bbox=(0.0, 0.0, 200.0, 150.0), dpi=72.0, source="embedded")
    seen: dict = {}
    adapter.extract([image], _page(page_meta), ocr_fns=_NO_OCR,
                    on_debug_layer=lambda s, l, _h, pdf: seen.__setitem__((s, l), pdf))

    def drawings(key):
        with fitz.open(stream=seen[key], filetype="pdf") as doc:
            return doc[0].get_drawings()

    for key in [("graph_build", "chains"), ("graph_build", "junctions"), ("graph_build", "endpoints"),
                ("polyline_fit", "segments"), ("polyline_fit", "segment endpoints")]:
        ds = drawings(key)
        assert ds, key
        assert all(item[0] != "re" for d in ds for item in d["items"]), key
    assert any(item[0] == "l" for d in drawings(("polyline_fit", "segments")) for item in d["items"])
    # dots are filled circles (bezier items); a T has 3 free ends and >= 1 junction
    endpoint_dots = [d for d in drawings(("graph_build", "endpoints")) if d.get("fill")]
    assert sum(1 for d in endpoint_dots for it in d["items"] if it[0] == "c") >= 3 * 4
    assert ("graph_build", "chain bbox") not in seen
    assert ("polyline_fit", "segment bbox") not in seen

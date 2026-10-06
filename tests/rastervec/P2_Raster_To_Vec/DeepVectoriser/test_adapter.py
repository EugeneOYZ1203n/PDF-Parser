from __future__ import annotations

import numpy as np
import pytest

from rastervec.commons.models import Image, Page
from rastervec.P2_Raster_To_Vec.DeepVectoriser import adapter, geometry as geo
from rastervec.P2_Raster_To_Vec.DeepVectoriser.inference import TilePrediction
from rastervec.P2_Raster_To_Vec.DeepVectoriser.paddle_engine import OcrBox

_NO_OCR = {
    "detect_many": lambda images: [[] for _ in images],
    "recognize": lambda crops: [OcrBox("", 0.0) for _ in crops],
    "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
}


def _fake_model(tiles):
    """One horizontal stroke through each tile's ink (row mean, full x span)."""
    out = []
    for t in tiles:
        ys, xs = np.nonzero(t < 128)
        p = TilePrediction(n_queries=16)
        if len(xs):
            y = ys.mean() + 0.5
            p.strokes.append(geo.line_to_cubic((xs.min(), y), (xs.max() + 1.0, y))[None])
            p.confidences.append(0.9)
            p.n_confident = 1
        out.append(p)
    return out


def _image(transform=None) -> Image:
    # 72 dpi placement -> resampled x300/72 before tiling
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    arr[39:42, 10:190] = (0, 0, 0)
    return Image(array=arr, bbox=(0.0, 0.0, 200.0, 100.0), dpi=72.0, source="embedded", transform=transform)


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def _span(vectors):
    pts = [p for v in vectors for it in v.items for p in it[1:]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), max(xs), float(np.mean(ys))


def test_line_is_vectorized_merged_and_mapped_to_page(page_meta):
    vectors, texts = adapter.extract([_image()], _page(page_meta), ocr_fns=_NO_OCR, model_fn=_fake_model)
    assert texts == []
    assert len(vectors) == 1  # tile seams joined into one stroke
    v = vectors[0]
    assert v.type == "s" and all(it[0] == "l" for it in v.items)
    x0, x1, y = _span(vectors)
    assert x0 == pytest.approx(10, abs=1.0) and x1 == pytest.approx(190, abs=1.0)
    assert y == pytest.approx(40.5, abs=1.0)
    assert v.color == pytest.approx((0.0, 0.0, 0.0), abs=0.1)
    assert v.width == pytest.approx(3.0, abs=1.0)  # 3 px at 72 dpi = 3 pt


def test_quarter_turn_placement(page_meta):
    # unit square -> page: x' = 100 * fy, y' = 200 * (1 - fx) (a 90-degree placement)
    transform = (0.0, -200.0, 100.0, 0.0, 0.0, 200.0)
    vectors, _ = adapter.extract([_image(transform)], _page(page_meta), ocr_fns=_NO_OCR, model_fn=_fake_model)
    pts = [p for v in vectors for it in v.items for p in it[1:]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert np.mean(xs) == pytest.approx(40.5, abs=1.0)            # the line is now vertical
    assert min(ys) == pytest.approx(10, abs=1.0) and max(ys) == pytest.approx(190, abs=1.0)


def test_debug_layers_stream_and_batch_match(page_meta):
    page = _page(page_meta)
    seen: list[tuple] = []
    adapter.extract([_image()], page, on_debug_layer=lambda *layer: seen.append(layer),
                    ocr_fns=_NO_OCR, model_fn=_fake_model)
    streamed = [(s, label) for s, label, _h, _b in seen]
    assert len(set(streamed)) == len(streamed)
    debug_out: dict = {}
    adapter.extract([_image()], page, debug_out=debug_out, ocr_fns=_NO_OCR, model_fn=_fake_model)
    assert [(s, label) for s, label, _h, _b in adapter.render_debug(debug_out, page.meta)] == streamed
    stages = {s for s, _ in streamed}
    for stage in ("color_separation", "ocr", "text_removal", "tiles", "strokes", "merge",
                  "vector_diff"):
        assert stage in stages
    assert all(b[:4] == b"%PDF" for *_x, b in seen)


def test_pale_ink_layer_is_still_vectorized(page_meta):
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    arr[39:42, 10:190] = (242, 236, 109)  # pale yellow: gray ~225
    image = Image(array=arr, bbox=(0.0, 0.0, 200.0, 100.0), dpi=72.0, source="embedded")
    vectors, _ = adapter.extract([image], _page(page_meta), ocr_fns=_NO_OCR, model_fn=_fake_model)
    assert len(vectors) == 1
    assert vectors[0].color == pytest.approx((242 / 255, 236 / 255, 109 / 255), abs=0.05)


def test_saturated_tile_is_resplit(page_meta):
    calls: list[int] = []

    def saturating(tiles):
        calls.append(tiles[0].shape[0])
        preds = _fake_model(tiles)
        if tiles[0].shape[0] == 128:
            for p in preds:
                p.n_confident = p.n_queries  # pretend every query fired
        return preds

    vectors, _ = adapter.extract([_image()], _page(page_meta), ocr_fns=_NO_OCR, model_fn=saturating)
    assert 128 in calls and 64 in calls
    x0, x1, _y = _span(vectors)
    assert x0 == pytest.approx(10, abs=1.5) and x1 == pytest.approx(190, abs=1.5)

from __future__ import annotations

import numpy as np
import pytest

from rastervec.commons.models import Image, Page
from rastervec.P2_Raster_To_Vec.DeepTechVec import adapter, geometry as geo
from rastervec.P2_Raster_To_Vec.DeepTechVec.paddle_engine import OcrBox

_NO_OCR = {
    "detect_many": lambda images: [[] for _ in images],
    "recognize": lambda crops: [OcrBox("", 0.0) for _ in crops],
    "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
}


def _fake_model(patches):
    """One horizontal line through each patch's ink (row mean, full x span,
    width = ink rows)."""
    out = []
    for t in patches:
        ys, xs = np.nonzero(t < 128)
        if not len(xs):
            out.append([])
            continue
        y = ys.mean() + 0.5
        out.append([geo.Prim(np.array([[xs.min(), y], [xs.max() + 1.0, y]], float), float(len(np.unique(ys))))])
    return out


def _image(transform=None) -> Image:
    # 72 dpi placement -> resampled x300/72 before patching
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    arr[39:42, 10:190] = (0, 0, 0)
    return Image(array=arr, bbox=(0.0, 0.0, 200.0, 100.0), dpi=72.0, source="embedded", transform=transform)


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def _points(vectors):
    return np.array([p for v in vectors for it in v.items for p in it[1:]], float)


@pytest.mark.parametrize("refine_iters", [0, 10])
def test_line_is_vectorized_merged_and_mapped_to_page(page_meta, refine_iters):
    vectors, texts = adapter.extract([_image()], _page(page_meta), ocr_fns=_NO_OCR, model_fn=_fake_model,
                                     refine_iters=refine_iters)
    assert texts == []
    assert len(vectors) == 1  # patches clipped to their cores, then merged into one line
    v = vectors[0]
    assert v.type == "s" and [it[0] for it in v.items] == ["l"]
    pts = _points(vectors)
    assert pts[:, 0].min() == pytest.approx(10, abs=1.0) and pts[:, 0].max() == pytest.approx(190, abs=1.0)
    assert pts[:, 1].mean() == pytest.approx(40.5, abs=0.75)
    assert v.color == pytest.approx((0.0, 0.0, 0.0), abs=0.1)
    assert v.width == pytest.approx(3.0, abs=1.0)  # predicted width: 3 px at 72 dpi = 3 pt


def test_quarter_turn_placement(page_meta):
    # unit square -> page: x' = 100 * fy, y' = 200 * (1 - fx) (a 90-degree placement)
    transform = (0.0, -200.0, 100.0, 0.0, 0.0, 200.0)
    vectors, _ = adapter.extract([_image(transform)], _page(page_meta), ocr_fns=_NO_OCR, model_fn=_fake_model,
                                 refine_iters=0)
    pts = _points(vectors)
    assert pts[:, 0].mean() == pytest.approx(40.5, abs=1.0)            # the line is now vertical
    assert pts[:, 1].min() == pytest.approx(10, abs=1.0) and pts[:, 1].max() == pytest.approx(190, abs=1.0)


def test_curve_primitives_become_cubic_items(page_meta):
    def curvy(patches):
        return [[geo.Prim(np.array([[0.0, 60.0], [32.0, 0.0], [64.0, 60.0]]), 2.0)] if (t < 128).any() else []
                for t in patches]

    vectors, _ = adapter.extract([_image()], _page(page_meta), ocr_fns=_NO_OCR, model_fn=curvy,
                                 refine_iters=0)
    assert vectors and any(it[0] == "c" for v in vectors for it in v.items)


def test_debug_layers_stream_and_batch_match(page_meta):
    page = _page(page_meta)
    seen: list[tuple] = []
    adapter.extract([_image()], page, on_debug_layer=lambda *layer: seen.append(layer),
                    ocr_fns=_NO_OCR, model_fn=_fake_model, refine_iters=0)
    streamed = [(s, label) for s, label, _h, _b in seen]
    assert len(set(streamed)) == len(streamed)
    debug_out: dict = {}
    adapter.extract([_image()], page, debug_out=debug_out, ocr_fns=_NO_OCR, model_fn=_fake_model, refine_iters=0)
    assert [(s, label) for s, label, _h, _b in adapter.render_debug(debug_out, page.meta)] == streamed
    stages = {s for s, _ in streamed}
    for stage in ("color_separation", "ocr", "text_removal", "tiles", "primitives", "refine", "merge",
                  "vector_diff"):
        assert stage in stages
    assert all(b[:4] == b"%PDF" for *_x, b in seen)


def test_pale_ink_layer_is_still_vectorized(page_meta):
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    arr[39:42, 10:190] = (242, 236, 109)  # pale yellow: gray ~225
    image = Image(array=arr, bbox=(0.0, 0.0, 200.0, 100.0), dpi=72.0, source="embedded")
    vectors, _ = adapter.extract([image], _page(page_meta), ocr_fns=_NO_OCR, model_fn=_fake_model, refine_iters=0)
    assert len(vectors) == 1
    assert vectors[0].color == pytest.approx((242 / 255, 236 / 255, 109 / 255), abs=0.05)

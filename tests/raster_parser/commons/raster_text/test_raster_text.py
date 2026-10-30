from __future__ import annotations

import numpy as np
import pytest

from raster_parser.models.source import Source
from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.commons.raster_text.color_layers import PAPER
from raster_parser.commons.raster_text.paddle_engine import OcrBox
from raster_parser.commons.raster_text.raster_text import (
    HISTOGRAM_FOLDER,
    prelude_debug_layers,
    prepare_images,
)


def _scene():
    arr = np.full((150, 200, 3), 255, dtype=np.uint8)
    arr[20:32, 20:120] = (200, 20, 20)  # red "text" block
    arr[100:103, 10:190] = 0  # a black line elsewhere
    return arr


def _fns(detect_text: bool = True, text: str = "HELLO"):
    def detect_many(images):
        out = []
        for im in images:
            im = np.asarray(im)
            # BGR in: red ink is high in channel 2 only
            ink = (im[:, :, 2] > 128) & (im[:, :, 0] < 128)
            ys, xs = np.nonzero(ink)
            if not detect_text or not len(xs):
                out.append([])
                continue
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            out.append([np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)])
        return out

    return {
        "detect_many": detect_many,
        "recognize": lambda crops: [OcrBox(text, 0.9) for _ in crops],
        "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
    }


def _page(page_meta) -> Page:
    return Page(
        doc_path="synthetic.pdf", meta=page_meta(width=100.0, height=200.0), pymupdf_page=None
    )


def test_texts_through_a_quarter_turn_placement(page_meta):
    # image x -> page +y, image y -> page -x (a 90-degree placement)
    image = Image(
        array=_scene(),
        bbox=(0.0, 0.0, 75.0, 100.0),
        dpi=144.0,
        source="embedded",
        transform=(0.0, 100.0, -75.0, 0.0, 75.0, 0.0),
    )
    prelude = prepare_images([image], _page(page_meta), ocr_fns=_fns())

    assert [t.text for t in prelude.texts] == ["HELLO"]
    t = prelude.texts[0]
    assert t.source == Source.FROM_IMAGE and t.seqno == 0
    # pixel box (20..120, 20..32) at 0.5 pt/px, rotated a quarter turn
    assert t.bbox == pytest.approx((75 - 16, 10, 75 - 10, 60), abs=1.0)
    assert t.direction == pytest.approx((0.0, 1.0), abs=1e-6)
    r, g, b = (t.color >> 16) & 255, (t.color >> 8) & 255, t.color & 255
    assert r > 150 and g < 60 and b < 60  # the red layer's colour
    assert len(prelude.images) == 1


def test_text_layer_is_erased_line_survives(page_meta):
    image = Image(array=_scene(), bbox=(0.0, 0.0, 200.0, 150.0), dpi=72.0, source="embedded")
    prelude = prepare_images([image], _page(page_meta), ocr_fns=_fns())
    labels = prelude.images[0].layers.labels
    assert (labels[20:32, 20:120] == PAPER).all()
    assert (labels[100:103, 20:180] != PAPER).all()


def test_blank_hit_is_kept_out_of_texts_and_not_erased(page_meta):
    image = Image(array=_scene(), bbox=(0.0, 0.0, 200.0, 150.0), dpi=72.0, source="embedded")
    prelude = prepare_images([image], _page(page_meta), ocr_fns=_fns(text=""))
    assert prelude.texts == []
    assert len(prelude.debug.quads["refined_failed"]) == 1
    assert (prelude.images[0].layers.labels[20:32, 20:120] != PAPER).all()


def test_streamed_layers_match_batch_and_histograms(page_meta):
    image = Image(array=_scene(), bbox=(0.0, 0.0, 200.0, 150.0), dpi=72.0, source="embedded")
    page = _page(page_meta)
    seen: list = []
    images_offered: list = []
    prelude = prepare_images(
        [image],
        page,
        ocr_fns=_fns(),
        on_debug_layer=lambda *layer: seen.append(layer),
        on_debug_image=lambda key, name, make: images_offered.append((key, name, make())),
    )
    streamed = [(s, l) for s, l, _h, _b in seen]
    assert len(set(streamed)) == len(streamed)
    stages = {s for s, _ in streamed}
    assert {"enhance", "ink", "color_layers", "text_removal", "ocr"} <= stages
    assert ("ink", "chroma map") in streamed and ("text_removal", "erased pixels") in streamed
    assert sum(1 for s, l in streamed if s == "color_layers" and l.startswith("layer ")) == 2
    assert all(b[:4] == b"%PDF" for *_x, b in seen)

    kept = prepare_images([image], page, ocr_fns=_fns(), keep_debug=True)
    assert [(s, l) for s, l, _h, _b in prelude_debug_layers(page.meta, kept)] == streamed

    assert len(prelude.debug.histograms) == 1
    name, png = prelude.debug.histograms[0]
    assert name == "p0_img0.png" and png[:4] == b"\x89PNG"
    assert [(k, n) for k, n, _img in images_offered] == [(HISTOGRAM_FOLDER, "img0.png")]
    assert images_offered[0][2].size[0] > 100


def test_no_debug_builds_no_rasters(page_meta):
    image = Image(array=_scene(), bbox=(0.0, 0.0, 200.0, 150.0), dpi=72.0, source="embedded")
    prelude = prepare_images([image], _page(page_meta), ocr_fns=_fns())
    assert prelude.debug.rasters == [] and prelude.debug.histograms == []


def test_consume_images_releases_each_prepared_image():
    from raster_parser.commons.raster_text.raster_text import RasterPrelude

    prelude = RasterPrelude(images=["a", "b"])
    seen = []
    for item in prelude.consume_images():
        seen.append((item, len(prelude.images)))
    assert seen == [("a", 1), ("b", 0)]
    assert prelude.images == []

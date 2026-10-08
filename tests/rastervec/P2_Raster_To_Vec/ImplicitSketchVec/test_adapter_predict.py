from __future__ import annotations

import json

import fitz
import numpy as np
import pytest

from rastervec.commons.models import Image, Page
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import adapter, predict
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import geometry as geo
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import train_data as td
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.paddle_engine import OcrBox

_NO_OCR = {
    "detect_many": lambda images: [[] for _ in images],
    "recognize": lambda crops: [OcrBox("", 0.0) for _ in crops],
    "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
}


def _perfect_model(tiles):
    """The fields a perfect network would predict for each tile's ink: one
    horizontal line (row mean, full x span), through the training targets."""
    out = []
    for t in tiles:
        ys, xs = np.nonzero(t < 128)
        strokes = []
        if len(xs):
            y = ys.mean() + 0.5
            strokes = [geo.line_to_cubic((xs.min(), y), (xs.max() + 1.0, y))[None]]
        tg = td.make_targets(strokes, t.shape[0])
        out.append({"udf": tg["udf"], "edge": tg["edge"].astype(np.uint8), "vertex": tg["vert"]})
    return out


def _image(transform=None) -> Image:
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    arr[39:42, 10:190] = (0, 0, 0)
    return Image(array=arr, bbox=(0.0, 0.0, 200.0, 100.0), dpi=72.0, source="embedded", transform=transform)


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def _points(vectors):
    return np.array([p for v in vectors for it in v.items for p in it[1:]], float)


def test_line_is_one_seamless_vector_in_page_space(page_meta):
    vectors, texts = adapter.extract([_image()], _page(page_meta), ocr_fns=_NO_OCR, model_fn=_perfect_model)
    assert texts == []
    assert len(vectors) == 1  # tile cores stitched into one graph, one stroke
    pts = _points(vectors)
    assert pts[:, 0].min() == pytest.approx(10, abs=1.0) and pts[:, 0].max() == pytest.approx(190, abs=1.0)
    assert np.abs(pts[:, 1] - 40.5).max() < 0.5
    assert vectors[0].width == pytest.approx(3.0, abs=1.0)
    assert vectors[0].color == pytest.approx((0.0, 0.0, 0.0), abs=0.1)


def test_quarter_turn_placement(page_meta):
    transform = (0.0, -200.0, 100.0, 0.0, 0.0, 200.0)
    vectors, _ = adapter.extract([_image(transform)], _page(page_meta), ocr_fns=_NO_OCR, model_fn=_perfect_model)
    pts = _points(vectors)
    assert pts[:, 0].mean() == pytest.approx(40.5, abs=1.0)
    assert pts[:, 1].min() == pytest.approx(10, abs=1.0) and pts[:, 1].max() == pytest.approx(190, abs=1.0)


def test_debug_layers_stream_and_batch_match(page_meta):
    page = _page(page_meta)
    seen: list[tuple] = []
    adapter.extract([_image()], page, on_debug_layer=lambda *layer: seen.append(layer),
                    ocr_fns=_NO_OCR, model_fn=_perfect_model)
    streamed = [(s, label) for s, label, _h, _b in seen]
    assert len(set(streamed)) == len(streamed)
    debug_out: dict = {}
    adapter.extract([_image()], page, debug_out=debug_out, ocr_fns=_NO_OCR, model_fn=_perfect_model)
    assert [(s, label) for s, label, _h, _b in adapter.render_debug(debug_out, page.meta)] == streamed
    stages = {s for s, _ in streamed}
    for stage in ("color_separation", "ocr", "text_removal", "tiles", "dfp", "ndc", "refine", "topology",
                  "group", "fit", "vector_diff"):
        assert stage in stages
    assert all(b[:4] == b"%PDF" for *_x, b in seen)


# ---------------------------------------------------------------------------
# predict.py
# ---------------------------------------------------------------------------
def _pdf(tmp_path, synthetic_pdf_factory, rotation=0):
    doc = synthetic_pdf_factory([{
        "width": 200, "height": 120, "rotation": rotation,
        "drawings": [{"lines": [((20, 40), (180, 40))], "color": (0, 0, 0), "width": 1.5}],
    }])
    path = tmp_path / f"p_r{rotation}.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("rotation", [0, 90])
def test_predict_writes_viewer_folder_without_ocr(tmp_path, synthetic_pdf_factory, monkeypatch, rotation):
    path = _pdf(tmp_path, synthetic_pdf_factory, rotation)
    seen = {}
    real_extract = predict.adapter.extract

    def spy(images, page, **kw):
        seen.update(kw)
        vectors, texts = real_extract(images, page, **kw)
        seen["vectors"] = vectors
        return vectors, texts

    monkeypatch.setattr(predict.adapter, "extract", spy)
    out = tmp_path / "pred"
    manifest = predict.run(path, [0], out, clip=(10, 20, 190, 60), dpi=144, model_fn=_perfect_model,
                           say=lambda _m: None)
    assert seen["ocr_fns"] is predict.NO_OCR
    assert "--ocr" not in predict.build_arg_parser().format_help()
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["engine"] == "ImplicitSketchVec"
    assert {"original", "detected", "dfp", "ndc", "fit"} <= {e["stage"] for e in on_disk["layers"]}
    for entry in manifest["layers"]:
        with fitz.open(out / entry["file"]) as doc:
            assert doc.page_count == 1
    pts = _points(seen["vectors"])
    assert pts[:, 0].min() == pytest.approx(20, abs=1.5) and pts[:, 0].max() == pytest.approx(180, abs=1.5)
    assert np.abs(pts[:, 1] - 40).max() < 1.5

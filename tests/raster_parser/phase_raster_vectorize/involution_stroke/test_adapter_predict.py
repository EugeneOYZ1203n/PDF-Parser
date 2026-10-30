from __future__ import annotations

import json

import pymupdf
import numpy as np
import pytest

pytest.importorskip("torch")  # inference imports torch even with fake models (the `ml` extra)

from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.phase_raster_vectorize.involution_stroke import adapter, predict
from raster_parser.commons.raster_text.paddle_engine import OcrBox

from ._fakes import FAKE_MODELS

_NO_OCR = {
    "detect_many": lambda images: [[] for _ in images],
    "recognize": lambda crops: [OcrBox("", 0.0) for _ in crops],
    "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
}


def _image(transform=None) -> Image:
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    arr[39:42, 10:190] = (0, 0, 0)
    return Image(
        array=arr, bbox=(0.0, 0.0, 200.0, 100.0), dpi=72.0, source="embedded", transform=transform
    )


def _page(page_meta) -> Page:
    return Page(
        doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), pymupdf_page=None
    )


def _points(vectors):
    return np.array([p for v in vectors for it in v.items for p in it[1:]], float)


# The fake centerline is a skeleton, which stops ~half the stroke width short
# of a butt end (the 3 px line is 12.5 px wide at the 300 dpi canonical scale).
_END_TOL = 2.0


def test_registered():
    from raster_parser.core.registry import VECTORIZE_BACKENDS

    assert "involution_stroke" in VECTORIZE_BACKENDS


def test_line_is_one_vector_in_page_space(page_meta):
    vectors, texts = adapter.extract(
        [_image()], _page(page_meta), ocr_fns=_NO_OCR, model_fns=FAKE_MODELS
    )
    assert texts == []
    assert len(vectors) == 1  # 300 dpi canonical: 833 px wide -> one 1024 px tile
    v = vectors[0]
    assert [it[0] for it in v.items] == ["l"]
    pts = _points(vectors)
    assert pts[:, 0].min() == pytest.approx(10, abs=_END_TOL)
    assert pts[:, 0].max() == pytest.approx(190, abs=_END_TOL)
    assert np.abs(pts[:, 1] - 40.5).max() < 0.5
    assert v.width == pytest.approx(3.0, abs=1.0)
    assert v.color == pytest.approx((0.0, 0.0, 0.0), abs=0.1)


def test_quarter_turn_placement(page_meta):
    transform = (0.0, -200.0, 100.0, 0.0, 0.0, 200.0)
    vectors, _ = adapter.extract(
        [_image(transform)], _page(page_meta), ocr_fns=_NO_OCR, model_fns=FAKE_MODELS
    )
    pts = _points(vectors)
    assert pts[:, 0].mean() == pytest.approx(40.5, abs=1.0)
    assert pts[:, 1].min() == pytest.approx(10, abs=_END_TOL)
    assert pts[:, 1].max() == pytest.approx(190, abs=_END_TOL)


def test_debug_layers_stream_and_batch_match(page_meta):
    page = _page(page_meta)
    seen: list[tuple] = []
    adapter.extract(
        [_image()],
        page,
        on_debug_layer=lambda *layer: seen.append(layer),
        ocr_fns=_NO_OCR,
        model_fns=FAKE_MODELS,
    )
    streamed = [(s, label) for s, label, _h, _b in seen]
    assert len(set(streamed)) == len(streamed)
    debug_out: dict = {}
    adapter.extract([_image()], page, debug_out=debug_out, ocr_fns=_NO_OCR, model_fns=FAKE_MODELS)
    assert [
        (s, label) for s, label, _h, _b in adapter.render_debug(debug_out, page.meta)
    ] == streamed
    stages = {s for s, _ in streamed}
    for stage in (
        "enhance",
        "ink",
        "color_layers",
        "ocr",
        "text_removal",
        "tiles",
        "centerline",
        "graph",
        "pairing",
        "merge",
        "fit",
        "vector_diff",
    ):
        assert stage in stages
    assert all(b[:4] == b"%PDF" for *_x, b in seen)


# ---------------------------------------------------------------------------
# predict.py
# ---------------------------------------------------------------------------
def _pdf(tmp_path, synthetic_pdf_factory, rotation=0):
    doc = synthetic_pdf_factory(
        [
            {
                "width": 200,
                "height": 120,
                "rotation": rotation,
                "drawings": [{"lines": [((20, 40), (180, 40))], "color": (0, 0, 0), "width": 1.5}],
            }
        ]
    )
    path = tmp_path / f"p_r{rotation}.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("rotation", [0, 90])
def test_predict_writes_viewer_folder_without_ocr(
    tmp_path, synthetic_pdf_factory, monkeypatch, rotation
):
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
    manifest = predict.run(
        path, [0], out, clip=(10, 20, 190, 60), dpi=144, model_fns=FAKE_MODELS, say=lambda _m: None
    )
    assert seen["ocr_fns"] is predict.NO_OCR
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["engine"] == "InvolutionStroke"
    assert {"original", "detected", "centerline", "graph", "fit"} <= {
        e["stage"] for e in on_disk["layers"]
    }
    for entry in manifest["layers"]:
        with pymupdf.open(out / entry["file"]) as doc:
            assert doc.page_count == 1
    pts = _points(seen["vectors"])
    assert pts[:, 0].min() == pytest.approx(20, abs=1.5) and pts[:, 0].max() == pytest.approx(
        180, abs=1.5
    )
    assert np.abs(pts[:, 1] - 40).max() < 1.5


def test_predict_cli_has_both_weight_flags():
    help_text = predict.build_arg_parser().format_help()
    assert "--centerline-weights" in help_text and "--intersection-weights" in help_text
    assert "--ocr" not in help_text

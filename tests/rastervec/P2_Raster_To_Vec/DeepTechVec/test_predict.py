from __future__ import annotations

import json

import fitz
import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.DeepTechVec import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepTechVec import predict


def _fake_model(patches):
    out = []
    for t in patches:
        ys, xs = np.nonzero(t < 128)
        if not len(xs):
            out.append([])
            continue
        y = ys.mean() + 0.5
        out.append([geo.Prim(np.array([[xs.min(), y], [xs.max() + 1.0, y]], float), 2.0)])
    return out


def _pdf(tmp_path, synthetic_pdf_factory, rotation=0, n_pages=1):
    doc = synthetic_pdf_factory([{
        "width": 200, "height": 120, "rotation": rotation,
        "drawings": [{"lines": [((20, 40), (180, 40))], "color": (0, 0, 0), "width": 1.5}],
    } for _ in range(n_pages)])
    path = tmp_path / f"p_r{rotation}.pdf"
    doc.save(path)
    doc.close()
    return path


def test_run_writes_viewer_folder(tmp_path, synthetic_pdf_factory):
    path = _pdf(tmp_path, synthetic_pdf_factory, n_pages=2)
    out = tmp_path / "pred"
    manifest = predict.run(path, [0, 1], out, clip=(10, 20, 190, 60), dpi=144, refine_iters=0,
                           model_fn=_fake_model, say=lambda _m: None)
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["pages"] == [0, 1] and on_disk["engine"] == "DeepTechVec"
    stages = {e["stage"] for e in on_disk["layers"]}
    assert {"original", "detected", "color_separation", "tiles", "primitives", "merge"} <= stages
    for entry in manifest["layers"]:
        with fitz.open(out / entry["file"]) as doc:
            assert doc.page_count == 2
    assert all(s["detected"] >= 1 and s["original"] == 1 for s in manifest["summary"])


def test_ocr_is_never_run(tmp_path, synthetic_pdf_factory, monkeypatch):
    path = _pdf(tmp_path, synthetic_pdf_factory)
    seen = {}
    real_extract = predict.adapter.extract

    def spy(images, page, **kw):
        seen.update(kw)
        return real_extract(images, page, **kw)

    monkeypatch.setattr(predict.adapter, "extract", spy)
    predict.run(path, [0], tmp_path / "pred", dpi=144, refine_iters=0, model_fn=_fake_model, say=lambda _m: None)
    assert seen["ocr_fns"] is predict.NO_OCR
    assert "--ocr" not in predict.build_arg_parser().format_help()


@pytest.mark.parametrize("rotation", [0, 90])
def test_detected_vectors_land_on_the_line(tmp_path, synthetic_pdf_factory, monkeypatch, rotation):
    path = _pdf(tmp_path, synthetic_pdf_factory, rotation=rotation)
    seen = {}
    real_extract = predict.adapter.extract

    def spy(images, page, **kw):
        vectors, texts = real_extract(images, page, **kw)
        seen["vectors"] = vectors
        return vectors, texts

    monkeypatch.setattr(predict.adapter, "extract", spy)
    clip = (10, 20, 190, 60)
    predict.run(path, [0], tmp_path / "pred", clip=clip, dpi=144, refine_iters=0, model_fn=_fake_model,
                say=lambda _m: None)
    pts = np.array([p for v in seen["vectors"] for it in v.items for p in it[1:]], float)
    assert pts[:, 0].min() == pytest.approx(20, abs=1.5) and pts[:, 0].max() == pytest.approx(180, abs=1.5)
    assert np.abs(pts[:, 1] - 40).max() < 1.5

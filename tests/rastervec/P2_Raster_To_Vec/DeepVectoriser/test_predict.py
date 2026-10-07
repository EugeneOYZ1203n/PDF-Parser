from __future__ import annotations

import json

import fitz
import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.DeepVectoriser import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepVectoriser import predict
from rastervec.P2_Raster_To_Vec.DeepVectoriser.inference import TilePrediction


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


def _pdf(tmp_path, synthetic_pdf_factory, rotation=0, n_pages=1):
    doc = synthetic_pdf_factory([{
        "width": 200, "height": 120, "rotation": rotation,
        "drawings": [{"lines": [((20, 40), (180, 40))], "color": (0, 0, 0), "width": 1.5}],
    } for _ in range(n_pages)])
    path = tmp_path / f"p_r{rotation}.pdf"
    doc.save(path)
    doc.close()
    return path


def _points(vectors):
    return np.array([p for v in vectors for it in v.items for p in it[1:]], float)


def test_run_writes_viewer_folder(tmp_path, synthetic_pdf_factory):
    path = _pdf(tmp_path, synthetic_pdf_factory, n_pages=2)
    out = tmp_path / "pred"
    manifest = predict.run(path, [0, 1], out, clip=(10, 20, 190, 60), dpi=144,
                           model_fn=_fake_model, say=lambda _m: None)
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["pages"] == [0, 1] and on_disk["source_pdf"].endswith(path.name)
    stages = {e["stage"] for e in on_disk["layers"]}
    assert {"original", "detected", "color_separation", "tiles", "merge"} <= stages
    for entry in manifest["layers"]:
        with fitz.open(out / entry["file"]) as doc:
            assert doc.page_count == 2  # page-aligned with `pages`
    assert all(s["detected"] >= 1 and s["original"] == 1 for s in manifest["summary"])


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
    predict.run(path, [0], tmp_path / "pred", clip=clip, dpi=144, model_fn=_fake_model, say=lambda _m: None)
    pts = _points(seen["vectors"])
    # unrotated page space, inside the clip, on the y=40 line from x=20 to 180
    assert pts[:, 0].min() == pytest.approx(20, abs=1.5) and pts[:, 0].max() == pytest.approx(180, abs=1.5)
    assert np.abs(pts[:, 1] - 40).max() < 1.5
    assert (pts[:, 0] >= clip[0] - 1).all() and (pts[:, 0] <= clip[2] + 1).all()


def test_render_region_transform_covers_the_pixmap(tmp_path, synthetic_pdf_factory):
    path = _pdf(tmp_path, synthetic_pdf_factory)
    img = predict.render_region(path, 0, (10.3, 20.7, 100.1, 60.2), dpi=144)
    a, b, c, d, e, f = img.transform
    assert (b, c) == (0.0, 0.0)
    assert a == pytest.approx(img.array.shape[1] / 2.0) and d == pytest.approx(img.array.shape[0] / 2.0)
    # MuPDF rounds the clip outwards to whole pixels
    assert e <= 10.3 and f <= 20.7 and e + a >= 100.1 and f + d >= 60.2


def test_parse_clip_rejects_bad_boxes():
    assert predict.parse_clip("1, 2,3,4") == (1.0, 2.0, 3.0, 4.0)
    assert predict.parse_clip(None) is None
    with pytest.raises(SystemExit):
        predict.parse_clip("5,5,1,1")

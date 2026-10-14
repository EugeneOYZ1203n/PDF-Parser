from __future__ import annotations

import json

import fitz
import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.Junction import predict


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
    manifest = predict.run(path, [0, 1], out, clip=(10, 20, 190, 60), dpi=144, ocr=False,
                           say=lambda _m: None)
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["pages"] == [0, 1] and on_disk["engine"] == "Junction" and on_disk["ocr"] is False
    stages = {e["stage"] for e in on_disk["layers"]}
    assert {"original", "detected", "enhance", "ink", "color_layers", "ocr", "graph_build",
            "vectorize", "vector_diff"} <= stages
    for entry in manifest["layers"]:
        with fitz.open(out / entry["file"]) as doc:
            assert doc.page_count == 2  # page-aligned with `pages`
    assert all(s["detected"] >= 1 and s["original"] == 1 for s in manifest["summary"])
    pngs = sorted(p.name for p in (out / predict.HISTOGRAM_DIR).glob("*.png"))
    assert pngs == ["p0_img0.png", "p1_img0.png"]


@pytest.mark.parametrize("rotation", [0, 90])
def test_detected_vectors_land_on_the_line(tmp_path, synthetic_pdf_factory, monkeypatch, rotation):
    path = _pdf(tmp_path, synthetic_pdf_factory, rotation=rotation)
    seen = {}
    real_extract = predict.adapter.extract

    def spy(images, page, **kw):
        assert kw["ocr_fns"] is predict.NO_OCR
        vectors, texts = real_extract(images, page, **kw)
        seen["vectors"] = vectors
        return vectors, texts

    monkeypatch.setattr(predict.adapter, "extract", spy)
    clip = (10, 20, 190, 60)
    predict.run(path, [0], tmp_path / "pred", clip=clip, dpi=144, ocr=False, say=lambda _m: None)
    pts = _points(seen["vectors"])
    # unrotated page space, inside the clip, on the y=40 line from x=20 to 180
    assert pts[:, 0].min() == pytest.approx(20, abs=1.5) and pts[:, 0].max() == pytest.approx(180, abs=1.5)
    assert np.abs(pts[:, 1] - 40).max() < 1.5
    assert (pts[:, 0] >= clip[0] - 1).all() and (pts[:, 0] <= clip[2] + 1).all()


def test_main_rejects_bad_pages(tmp_path, synthetic_pdf_factory):
    path = _pdf(tmp_path, synthetic_pdf_factory)
    with pytest.raises(SystemExit):
        predict.main(["--pdf", str(path), "--pages", "3", "--out", str(tmp_path / "o")])

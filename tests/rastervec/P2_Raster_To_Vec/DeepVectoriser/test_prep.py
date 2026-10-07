from __future__ import annotations

import json

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.DeepVectoriser import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepVectoriser import prep_dataset
from rastervec.P2_Raster_To_Vec.DeepVectoriser import train_data as td

def _pdf(tmp_path, synthetic_pdf_factory, rotation=0):
    doc = synthetic_pdf_factory([{
        "width": 200, "height": 120, "rotation": rotation,
        "drawings": [
            {"lines": [((20, 30), (180, 30))], "color": (0, 0, 0), "width": 1.5},
            {"rects": [(40, 50, 120, 100)], "color": (0.85, 0.1, 0.1), "width": 1.0},
        ],
    }])
    path = tmp_path / f"synthetic_r{rotation}.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_ground_truth_lands_on_rendered_ink(tmp_path, synthetic_pdf_factory, rotation):
    path = _pdf(tmp_path, synthetic_pdf_factory, rotation)
    rgb, strokes, rot = td.page_ground_truth(str(path), 0, dpi=144)
    assert rot == rotation
    gray = rgb.mean(axis=2)
    assert len(strokes) == 2
    for s, _w in strokes:
        pts = np.floor(geo.sample_stroke(s, 8)).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, gray.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, gray.shape[0] - 1)
        # every sample within 1 px of ink
        dark = np.array([gray[max(0, y - 1):y + 2, max(0, x - 1):x + 2].min() for x, y in pts])
        assert (dark < 200).mean() > 0.95


def test_rectangle_chains_into_one_stroke(tmp_path, synthetic_pdf_factory):
    _rgb, strokes, _ = td.page_ground_truth(str(_pdf(tmp_path, synthetic_pdf_factory)), 0, dpi=72)
    lengths = sorted(len(s) for s, _ in strokes)
    assert lengths == [1, 4]
    line_w = [w for s, w in strokes if len(s) == 1][0]
    assert line_w == pytest.approx(1.5)


def test_cut_by_mask_drops_erased_ink():
    stroke = geo.line_to_cubic((0, 5), (100, 5))[None]
    keep = np.ones((10, 100), bool)
    keep[:, 40:60] = False
    parts = td.cut_by_mask([(stroke, 1.0)], keep)
    assert len(parts) == 2
    assert parts[0][0][-1, 3][0] == pytest.approx(40, abs=0.1)
    assert parts[1][0][0, 0][0] == pytest.approx(60, abs=0.1)


def test_assign_layers_by_pixel_coverage():
    labels = np.zeros((20, 20), int)
    labels[5, :] = 1
    labels[:, 15] = 2
    strokes = [(geo.line_to_cubic((0, 5.5), (12, 5.5))[None], 1.0),
               (geo.line_to_cubic((15.5, 8), (15.5, 19)), 1.0),
               (geo.line_to_cubic((2, 15), (8, 15))[None], 1.0)]
    strokes[1] = (strokes[1][0][None], 1.0)
    assert td.assign_layers(strokes, labels, [1, 2]) == [[1], [2], []]
    # an outline lying on two layers (stroke ink + fill boundary) goes to both
    both = np.zeros((20, 20), int)
    both[5, :] = 1
    both[6, :] = 2
    assert td.assign_layers([(geo.line_to_cubic((0, 6.0), (19, 6.0))[None], 1.0)], both, [1, 2]) == [[1, 2]]


def test_augment_keeps_strokes_on_ink():
    rng = np.random.default_rng(3)
    gray = np.full((64, 64), 255, np.uint8)
    gray[10:12, 5:50] = 0
    crop = td.Crop(gray, [geo.line_to_cubic((5.5, 11), (49.5, 11))[None]], [2.0])
    for _ in range(8):
        aug = td._rot90(crop) if rng.random() < 0.5 else crop
        aug = td._flip(aug) if rng.random() < 0.5 else aug
        pts = np.clip(np.floor(geo.sample_stroke(aug.strokes[0], 8)).astype(int), 0, 63)
        assert (aug.gray[pts[:, 1], pts[:, 0]] < 128).mean() > 0.9


def test_prep_page_writes_layers_and_index_and_resumes(tmp_path, synthetic_pdf_factory):
    path = _pdf(tmp_path, synthetic_pdf_factory)
    out = tmp_path / "prep"
    logs: list[str] = []
    manifest = prep_dataset.prep_page(path, 0, out, dpi=144, log=logs.append)
    # both GT strokes saved; an antialiased line may also sit on a fringe layer
    assert manifest["layers"] and sum(layer["n_strokes"] for layer in manifest["layers"]) >= 2
    by_len = {len(s) for layer in manifest["layers"]
              for s in td.load_layer(out / "layers", layer["key"]).strokes}
    assert by_len == {1, 4}
    assert any("render" in m and "saved" in m for m in logs)
    for layer in manifest["layers"]:
        data = td.load_layer(out / "layers", layer["key"])
        # canonical 300 dpi: a 200 pt page is ~833 px wide
        assert data.gray.shape[1] == pytest.approx(200 * 300 / 72, abs=2)
        assert len(data.strokes) == layer["n_strokes"]
    index = prep_dataset.write_index(out, val_frac=0.1)
    assert index["train"] and not index["val"]
    assert json.loads((out / "index.json").read_text())["train"] == index["train"]

    # resume: the CLI skips a page whose manifest exists
    assert prep_dataset.main(["--pdf", str(path), "--out", str(out)]) == 0
    log_text = (out / "prep_log.txt").read_text()
    assert "1 skipped" in log_text


def test_prep_failure_is_reported_with_stage(tmp_path, synthetic_pdf_factory, monkeypatch):
    path = _pdf(tmp_path, synthetic_pdf_factory)
    out = tmp_path / "prep"

    def boom(*_a, **_k):
        raise ValueError("bad colors")

    monkeypatch.setattr(prep_dataset, "separate_colors", boom)
    assert prep_dataset.main(["--pdf", str(path), "--out", str(out)]) == 1
    log_text = (out / "prep_log.txt").read_text()
    assert "FAILED at stage 'colors'" in log_text and "bad colors" in log_text


def test_pdf_dir_recursive_with_distinct_keys(tmp_path, synthetic_pdf_factory):
    src = tmp_path / "pdfs"
    (src / "sub").mkdir(parents=True)
    a = _pdf(tmp_path, synthetic_pdf_factory)
    a.rename(src / "drawing.pdf")
    b = _pdf(tmp_path, synthetic_pdf_factory, rotation=90)
    b.rename(src / "sub" / "drawing.pdf")  # same stem, different folder
    out = tmp_path / "prep"
    assert prep_dataset.main(["--pdf-dir", str(src), "--out", str(out), "--dpi", "72"]) == 0
    keys = sorted(p.stem for p in (out / "pages").glob("*.json"))
    assert keys == ["drawing_p0", "sub__drawing_p0"]
    index = json.loads((out / "index.json").read_text())
    assert sorted(index["pages"]) == keys
    assert "2 PDF(s)" in (out / "prep_log.txt").read_text()


def test_empty_pdf_dir_exits(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no .pdf files"):
        prep_dataset.main(["--pdf-dir", str(tmp_path / "empty"), "--out", str(tmp_path / "prep")])
    with pytest.raises(SystemExit, match="--pdf-dir"):
        prep_dataset.main(["--out", str(tmp_path / "prep")])

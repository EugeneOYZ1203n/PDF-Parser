from __future__ import annotations

import json

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.DeepVectoriser import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepVectoriser import prep_dataset
from rastervec.P2_Raster_To_Vec.DeepVectoriser import train_data as td
from rastervec.P2_Raster_To_Vec.DeepVectoriser import view_dataset as vd


def _layer(n_lines: int, h: int = 400, w: int = 600):
    gray = np.full((h, w), 255, np.uint8)
    strokes = []
    for k in range(n_lines):
        y = 40 + 12 * k
        gray[y, 50:550] = 0
        strokes.append((geo.line_to_cubic((50.5, y + 0.5), (549.5, y + 0.5))[None], 1.0 + k % 3))
    # one two-piece stroke
    gray[300:340, 100] = 0
    gray[340, 100:140] = 0
    strokes.append((np.stack([geo.line_to_cubic((100.5, 300.5), (100.5, 340.5)),
                              geo.line_to_cubic((100.5, 340.5), (140.5, 340.5))]), 2.0))
    return gray, strokes


@pytest.fixture
def dataset(tmp_path):
    out = tmp_path / "ds"
    pages = []
    for p, n in (("docA_p0", 5), ("docB_p0", 2)):
        gray, strokes = _layer(n)
        key = f"{p}__L1"
        stored = td.save_layer(out / "layers", key, gray, strokes, (200, 30, 30), seed=1)
        layers = [{"key": key, "n_strokes": len(strokes), "color": [200, 30, 30], "saved": True,
                   "page_shape": list(gray.shape), **stored},
                  {"key": f"{p}__L2", "n_strokes": 0, "color": [10, 10, 10], "saved": False}]
        pages.append({"pdf": f"{p}.pdf", "page": 0, "key": p, "dpi": 300.0, "rotation": 0,
                      "px_per_pt": 300 / 72, "layers": layers, "seconds": 0.1})
    (out / "pages").mkdir()
    for m in pages:
        (out / "pages" / f"{m['key']}.json").write_text(json.dumps(m), encoding="utf-8")
    prep_dataset.write_index(out, val_frac=0.5)
    return out


def test_load_dataset_rows_and_splits(dataset):
    info = vd.load_dataset(dataset)
    assert len(info.pages) == 2
    assert [r.key for r in info.rows] == ["docA_p0__L1", "docA_p0__L2", "docB_p0__L1", "docB_p0__L2"]
    saved = [r for r in info.rows if r.saved]
    assert {r.split for r in saved} == {"train", "val"}  # >= 2 pages -> one held out
    unsaved = [r for r in info.rows if not r.saved]
    assert all(r.split == "unsaved" and r.n_strokes == 0 for r in unsaved)
    assert info.rows[0].layer == 1 and info.rows[0].color == (200, 30, 30)


def test_load_dataset_missing_index(tmp_path):
    with pytest.raises(SystemExit):
        vd.load_dataset(tmp_path)


def test_read_layer_round_trip_writes_no_cache(dataset):
    loaded = vd.read_layer(dataset, "docA_p0__L1")
    ref = td.load_layer(dataset / "layers", "docA_p0__L1", cache_dir=dataset.parent / "elsewhere")
    assert np.array_equal(np.asarray(loaded.data.gray), np.asarray(ref.gray))
    assert len(loaded.data.strokes) == len(ref.strokes) == 6
    for a, b in zip(loaded.data.strokes, ref.strokes):
        assert np.allclose(a, b)
    assert loaded.color == (200, 30, 30)
    assert loaded.page_shape == (400, 600)
    assert not (dataset / "cache").exists()


def test_stroke_polylines_end_at_stroke_ends(dataset):
    data = vd.read_layer(dataset, "docA_p0__L1").data
    for s, poly in zip(data.strokes, vd.stroke_polylines(data.strokes)):
        assert np.allclose(poly[0], s[0, 0]) and np.allclose(poly[-1], s[-1, 3])


def test_layer_stats(dataset):
    st = vd.layer_stats(vd.read_layer(dataset, "docA_p0__L1").data, max_prims=1)
    assert st["strokes"] == 6 and st["pieces"] == 7
    assert st["pieces_per_stroke"]["1"] == 5 and st["pieces_per_stroke"]["2"] == 1
    assert st["over_max_prims"] == 1
    assert st["width_px"]["min"] == pytest.approx(1.0) and st["width_px"]["max"] == pytest.approx(3.0)
    assert 0 < st["ink_fraction"] < 1


@pytest.mark.parametrize("augment", [False, True])
def test_sample_training_crop_deterministic(dataset, augment):
    data = vd.read_layer(dataset, "docA_p0__L1").data
    cfg = {"n_stroke": 3, "max_prims": 16}
    a = vd.sample_training_crop(data, np.random.default_rng([0, 4]), 128, augment, cfg)
    b = vd.sample_training_crop(data, np.random.default_rng([0, 4]), 128, augment, cfg)
    assert a.gray.shape == (128, 128)
    assert np.array_equal(a.gray, b.gray)
    assert len(a.strokes) == len(b.strokes) <= 3
    assert a.n_found >= len(a.strokes)
    assert (a.window is None) == augment


def test_crop_window_matches_crop_pixels(dataset):
    data = vd.read_layer(dataset, "docA_p0__L1").data
    for k in range(5):
        crop = vd.sample_training_crop(data, np.random.default_rng([1, k]), 96)
        x0, y0, win = crop.window
        assert np.array_equal(crop.gray[:win, :win], td._cut_window(np.asarray(data.gray), x0, y0, win))

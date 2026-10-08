from __future__ import annotations

import math

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.DeepTechVec import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepTechVec import train_data as td


def _layer(lines, size=200, width=3):
    """A synthetic dataset layer: ink drawn for every line, GT = the lines."""
    import cv2

    gray = np.full((size, size), 255, np.uint8)
    strokes, bboxes = [], []
    for a, b in lines:
        cv2.line(gray, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 0, width)
        st = geo.line_to_cubic(a, b)[None]
        strokes.append(st)
        pts = st.reshape(-1, 2)
        bboxes.append([*pts.min(0), *pts.max(0)])
    ys, xs = np.nonzero(gray < 128)
    return td.LayerData("syn", gray, strokes, np.full(len(lines), float(width), np.float32),
                        np.array(bboxes, np.float32), np.stack([xs, ys], 1).astype(np.int32))


def test_unaugmented_patch_has_the_line_as_one_primitive():
    layer = _layer([((20, 100), (180, 100))])
    p = td.cut_patch(layer, (100, 100), 0.0, 1.0, "line")
    assert p.gray.shape == (64, 64)
    assert len(p.prims) == 1
    assert np.allclose(p.prims[0].pts, [[0, 32], [64, 32]], atol=0.5)
    assert p.prims[0].width == pytest.approx(3.0)


@pytest.mark.parametrize("kind", ["line", "curve"])
def test_rotated_scaled_patch_keeps_ground_truth_on_the_ink(kind):
    layer = _layer([((20, 60), (180, 140)), ((40, 170), (170, 30))])
    rng = np.random.default_rng(3)
    for _ in range(10):
        angle, scale = rng.uniform(0, 2 * math.pi), rng.uniform(0.8, 1.25)
        p = td.cut_patch(layer, (100 + rng.uniform(-5, 5), 100 + rng.uniform(-5, 5)), angle, scale, kind)
        assert p.prims
        for prim in p.prims:
            pts = geo.prim_points(prim, 16)
            ij = np.clip(np.floor(pts).astype(int), 0, 63)
            assert (p.gray[ij[:, 1], ij[:, 0]] < 200).mean() > 0.9
            assert prim.width == pytest.approx(3.0 * scale, rel=1e-6)


def test_overflow_keeps_the_longest_n_prim():
    lines = [((10, 10 + 4 * k), (190, 10 + 4 * k)) for k in range(30)]
    layer = _layer(lines, width=1)
    p = td.sample_patch(layer, np.random.default_rng(0), 5, "line", augment=False)
    assert p.overflow and len(p.prims) == 5


def test_encode_decode_round_trip():
    prims = [geo.Prim(np.array([[40.0, 10.0], [4.0, 10.0]]), 2.0), geo.Prim(np.array([[3.0, 3.0], [3.0, 60.0]]), 1.0)]
    t = td.encode_targets(prims, 10, "line")
    assert t.shape == (10, 6)
    assert t[:2, -1].tolist() == [1.0, 1.0] and not t[2:].any()  # zero placeholders
    out = td.decode_output(t, "line")
    assert [p.pts.tolist() for p in out] == [[[3.0, 3.0], [3.0, 60.0]], [[4.0, 10.0], [40.0, 10.0]]]
    assert [p.width for p in out] == pytest.approx([1.0, 2.0])
    q = td.encode_targets([geo.Prim(np.array([[0.0, 0.0], [32.0, 32.0], [64.0, 0.0]]), 1.0)], 4, "curve")
    assert q.shape == (4, 8)
    assert np.allclose(td.decode_output(q, "curve")[0].pts, [[0, 0], [32, 32], [64, 0]])


def test_gray_path_tolerates_a_locked_cache_file(tmp_path, monkeypatch):
    import os

    import cv2

    layers = tmp_path / "layers"
    layers.mkdir()
    ok, png = cv2.imencode(".png", np.full((8, 8), 255, np.uint8))
    png.tofile(str(layers / "k.gray.png"))
    first = td.gray_path(layers, "k")
    os.utime(layers / "k.gray.png")  # PNG newer -> re-decode

    def locked(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(td.os, "replace", locked)
    assert td.gray_path(layers, "k") == first
    assert not list((tmp_path / "cache").glob("*.tmp"))

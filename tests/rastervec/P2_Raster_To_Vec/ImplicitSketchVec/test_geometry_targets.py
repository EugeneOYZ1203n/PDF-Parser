from __future__ import annotations

import math

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import geometry as geo
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import train_data as td


def _line(a, b):
    return geo.line_to_cubic(a, b)[None]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_schneider_fit_reproduces_a_cubic():
    piece = np.array([[0.0, 0.0], [10.0, 30.0], [40.0, 30.0], [50.0, 0.0]])
    pts = geo.bezier_eval(piece, np.linspace(0, 1, 80))
    fit = geo.fit_cubics(pts, 0.1)
    assert len(fit) == 1
    d = np.linalg.norm(pts[:, None] - geo.bezier_eval(fit[0], np.linspace(0, 1, 2000))[None], axis=2)
    assert d.min(axis=1).max() < 0.2


def test_schneider_splits_a_corner():
    pts = np.concatenate([np.linspace([0, 0], [20, 0], 21), np.linspace([20, 0], [20, 20], 21)[1:]])
    fit = geo.fit_cubics(pts, 0.25)
    assert len(fit) >= 2
    assert np.allclose(fit[0][0], [0, 0]) and np.allclose(fit[-1][3], [20, 20])


def test_rdp_keeps_corners_only():
    pts = np.concatenate([np.linspace([0, 0], [20, 0], 21), np.linspace([20, 0], [20, 20], 21)[1:]])
    assert np.allclose(geo.rdp(pts, 0.1), [[0, 0], [20, 0], [20, 20]])


def test_dense_samples_spacing():
    s = geo.dense_samples(_line((0, 0), (10, 0)), 0.1)
    assert np.diff(s[:, 0]).max() <= 0.1 + 1e-9 and s[0, 0] == 0 and s[-1, 0] == 10


@pytest.mark.parametrize("total,tile,overlap", [(1000, 256, 32), (300, 256, 32), (256, 256, 32)])
def test_tile_cores_partition_the_axis(total, tile, overlap):
    cover = np.zeros(total, int)
    for s, (a, b) in geo.core_spans(total, tile, overlap).items():
        assert s <= a < b <= s + tile
        cover[a:b] += 1
    assert (cover == 1).all()


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def test_horizontal_line_targets():
    t = td.make_targets([_line((2, 10.3), (30, 10.3))], 32)
    assert t["udf"].shape == (6, 65, 65) and t["edge"].shape == (64, 64)
    # centerline UDF is 0 on the line and grows away from it
    assert t["udf"][0, 20, 10] == pytest.approx(0.3, abs=0.01) and t["udf"][0, 30, 10] == pytest.approx(4.7, abs=0.1)
    er, eb = t["edge"] % 2 == 1, t["edge"] >= 2
    assert eb.sum() == 0 and set(np.nonzero(er)[0]) == {20}  # only right-edge flags, one row
    ys, xs = np.nonzero(t["vmask"])
    assert np.allclose((ys + t["vert"][1, ys, xs]) / 2, 10.3, atol=0.06)
    assert not t["usm"].any()
    assert np.allclose(sorted(map(tuple, t["keypoints"]["end"])), [(2, 10.3), (30, 10.3)], atol=0.05)
    assert t["mask"].shape == (1, 65, 65) and t["mask"].sum() > 0


def test_line_on_a_grid_line_has_no_spurious_crossings():
    t = td.make_targets([_line((2, 16), (30, 16)), _line((16, 2), (16, 30))], 32)
    assert t["usm"].sum() <= 1  # only the crossing cell


def test_parallel_lines_closer_than_a_cell_are_under_sampled():
    t = td.make_targets([_line((2, 10.1), (30, 10.1)), _line((2, 10.4), (30, 10.4))], 32)
    assert t["usm"].sum() > 40


def test_keypoint_types():
    plus = td.make_targets([_line((2, 16.2), (30, 16.2)), _line((16.3, 2), (16.3, 30))], 32)["keypoints"]
    assert len(plus["junc"]) == 1 and len(plus["end"]) == 4
    corner = td.make_targets([np.stack([geo.line_to_cubic((4, 4), (20, 4)), geo.line_to_cubic((20, 4), (20, 24))])],
                             32)["keypoints"]
    assert np.allclose(corner["sharp"], [[20, 4]]) and len(corner["end"]) == 2
    tee = td.make_targets([_line((2, 16.2), (30, 16.2)), _line((16.3, 16.2), (16.3, 30))], 32)["keypoints"]
    assert len(tee["junc"]) == 1


def _layer(lines, size=200, width=3):
    import cv2

    gray = np.full((size, size), 255, np.uint8)
    strokes, bboxes = [], []
    for a, b in lines:
        cv2.line(gray, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 0, width)
        strokes.append(_line(a, b))
        bboxes.append([*np.minimum(a, b), *np.maximum(a, b)])
    ys, xs = np.nonzero(gray < 128)
    return td.LayerData("syn", gray, strokes, np.full(len(lines), float(width), np.float32),
                        np.array(bboxes, np.float32), np.stack([xs, ys], 1).astype(np.int32))


@pytest.mark.parametrize("flip", [False, True])
def test_rotated_crop_keeps_ground_truth_on_the_ink(flip):
    layer = _layer([((20, 60), (180, 140)), ((40, 170), (170, 30))])
    for angle in (0.3, 1.7, 4.0):
        c = td.cut_crop(layer, (100, 100), angle, flip, 96)
        assert c.strokes
        for s in c.strokes:
            pts = geo.sample_stroke(s, 16)
            pts = pts[(pts >= 0).all(1) & (pts < 96).all(1)]
            ij = np.floor(pts).astype(int)
            assert (c.gray[ij[:, 1], ij[:, 0]] < 200).mean() > 0.9


def test_build_batch_shapes():
    layer = _layer([((20, 60), (180, 140))])
    rng = np.random.default_rng(0)
    b = td.build_batch([td.sample_crop(layer, 64, rng) for _ in range(3)])
    assert b["gray"].shape == (3, 64, 64) and b["udf"].shape == (3, 6, 129, 129)
    assert b["edge"].shape == (3, 128, 128) and b["vert"].shape == (3, 2, 128, 128) and b["size"] == 64
    assert math.isclose(float(b["udf"].max()), 8.0)

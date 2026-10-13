from __future__ import annotations

import cv2
import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.InvolutionStroke import geometry as geo
from rastervec.P2_Raster_To_Vec.InvolutionStroke import inference as inf

from ._fakes import FAKE_MODELS, fake_centerline, pairing_by_angle


def _mask(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 255, np.uint8)


def _length(p: np.ndarray) -> float:
    return geo.polyline_length(p)


# ---------------------------------------------------------------------------
# the upstream chain on one tile
# ---------------------------------------------------------------------------
def test_tile_line_is_one_polyline_and_one_stroke():
    gray = _mask(96, 160)
    cv2.line(gray, (20, 48), (140, 48), 0, 3)
    tr = inf.run_tile(gray, fake_centerline, pairing_by_angle)
    assert len(tr.polylines) == 1
    assert len(tr.strokes) == 1
    s = tr.strokes[0]
    assert s[:, 0].min() == pytest.approx(20, abs=3) and s[:, 0].max() == pytest.approx(140, abs=3)
    assert np.abs(s[:, 1] - 48.5).max() < 1.5


def test_tile_cross_pairs_straight_through():
    gray = _mask(160, 160)
    cv2.line(gray, (20, 80), (140, 80), 0, 3)
    cv2.line(gray, (80, 20), (80, 140), 0, 3)
    tr = inf.run_tile(gray, fake_centerline, pairing_by_angle)
    assert len(tr.polylines) == 4          # four arms meeting at the junction node
    assert len(tr.strokes) == 2            # paired through it into the two lines
    for s in tr.strokes:
        span = s.max(0) - s.min(0)
        assert max(span) > 110 and min(span) < 4   # each one straight and full length


def test_relation_matrix_links_half_edges_at_one_node():
    half_edges = {1: ("a", None), 2: ("b", None), 3: ("a", None), 4: ("c", None)}
    nodes = {"a": [1, 3], "b": [2], "c": [4]}
    R = inf.relation_matrix(half_edges, nodes)
    assert R[0, 0] == 1
    assert R[1, 3] == R[3, 1] == R[1, 1] == 1
    assert R[1, 2] == 0 and R[2, 4] == 0


def test_preprocess_inverts_and_optionally_normalizes():
    gray = _mask(64, 64)
    gray[30:33, :] = 0
    plain = inf.preprocess(gray, normalize=False)
    assert plain[31, 10] == 255 and plain[0, 0] == 0
    norm = inf.preprocess(gray, normalize=True)
    assert norm.shape == gray.shape and norm[0, 0] == 0


# ---------------------------------------------------------------------------
# clipping + seams
# ---------------------------------------------------------------------------
def test_clip_polyline_interpolates_border_crossings():
    pts = np.array([[0.0, 5.0], [20.0, 5.0], [20.0, 30.0]])
    parts = geo.clip_polyline(pts, (5.0, 0.0, 15.0, 10.0))
    assert len(parts) == 1
    assert np.allclose(parts[0], [[5.0, 5.0], [15.0, 5.0]])


def test_clip_polyline_segment_passing_through():
    parts = geo.clip_polyline(np.array([[0.0, 5.0], [30.0, 5.0]]), (10.0, 0.0, 20.0, 10.0))
    assert len(parts) == 1 and np.allclose(parts[0], [[10.0, 5.0], [20.0, 5.0]])


def test_clip_polyline_leaving_and_reentering_splits():
    pts = np.array([[1.0, 1.0], [8.0, 1.0], [8.0, 20.0], [2.0, 20.0], [2.0, 2.0]])
    parts = geo.clip_polyline(pts, (0.0, 0.0, 10.0, 10.0))
    assert len(parts) == 2


def test_join_seams_joins_only_across_tiles_on_core_borders():
    a = inf._Piece(np.array([[10.0, 50.0], [100.0, 50.0]]), 0, (0.0, 0.0, 100.0, 200.0))
    b = inf._Piece(np.array([[100.0, 50.3], [190.0, 50.0]]), 1, (100.0, 0.0, 200.0, 200.0))
    c = inf._Piece(np.array([[100.0, 150.0], [150.0, 150.0]]), 1, (100.0, 0.0, 200.0, 200.0))
    lines, joins = inf.join_seams([a, b, c], (200, 200))
    assert len(joins) == 1 and joins[0] == pytest.approx((100.0, 50.15))
    assert sorted(len(p) for p in lines) == [2, 3]
    joined = max(lines, key=len)
    assert {joined[0, 0], joined[-1, 0]} == {10.0, 190.0}


def test_join_seams_closes_loops():
    left = inf._Piece(np.array([[100.0, 20.0], [50.0, 20.0], [50.0, 80.0], [100.0, 80.0]]), 0,
                      (0.0, 0.0, 100.0, 100.0))
    right = inf._Piece(np.array([[100.0, 80.0], [150.0, 80.0], [150.0, 20.0], [100.0, 20.0]]), 1,
                       (100.0, 0.0, 200.0, 100.0))
    lines, joins = inf.join_seams([left, right], (200, 100))
    assert len(joins) == 2 and len(lines) == 1
    assert len(lines[0]) == 7


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------
def test_layer_line_across_tiles_is_one_stroke():
    gray = _mask(100, 300)
    cv2.line(gray, (10, 50), (290, 50), 0, 3)
    res = inf.vectorize_layer(gray, model_fns=FAKE_MODELS, tile=128, overlap=32, debug=True)
    assert len(res.tiles) == 3 and res.failed_tiles == []
    assert len(res.joins) == 2
    assert len(res.polylines) == 1 and len(res.strokes) == 1
    assert _length(res.polylines[0]) == pytest.approx(280, abs=6)
    assert len(res.strokes[0]) == 1        # a straight line fits as one piece
    assert res.centerline is not None and res.centerline.shape == gray.shape


def test_layer_rectangle_keeps_its_corners():
    gray = _mask(120, 200)
    cv2.rectangle(gray, (20, 20), (180, 100), 0, 3)
    res = inf.vectorize_layer(gray, model_fns=FAKE_MODELS)
    assert len(res.strokes) == 1
    stroke = res.strokes[0]
    assert len(stroke) == 4
    assert all(geo.piece_is_flat(p, 0.75) for p in stroke)


def test_failing_tile_is_skipped(caplog):
    gray = _mask(64, 64)
    gray[30:33, 5:60] = 0

    def boom(_tile):
        raise RuntimeError("graph extraction blew up")

    res = inf.vectorize_layer(gray, model_fns={"centerline": boom, "pairing": pairing_by_angle})
    assert res.tiles == [] and len(res.failed_tiles) == 1 and res.strokes == []


def test_empty_layer_runs_no_tiles():
    res = inf.vectorize_layer(_mask(50, 50), model_fns=FAKE_MODELS)
    assert res.tiles == [] and res.strokes == []

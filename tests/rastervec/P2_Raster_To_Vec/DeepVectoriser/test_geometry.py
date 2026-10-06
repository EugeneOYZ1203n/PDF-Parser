from __future__ import annotations

import numpy as np

from rastervec.P2_Raster_To_Vec.DeepVectoriser import geometry as geo


def _line(a, b) -> np.ndarray:
    return geo.line_to_cubic(a, b)[None]


def test_split_bezier_matches_reparametrized_curve():
    piece = np.array([[0, 0], [10, 30], [40, -10], [50, 20]], float)
    sub = geo.split_bezier(piece, 0.25, 0.75)
    ts = np.linspace(0, 1, 7)
    assert np.allclose(geo.bezier_eval(sub, ts), geo.bezier_eval(piece, 0.25 + 0.5 * ts), atol=1e-9)
    assert np.allclose(geo.split_bezier(piece, 0.0, 1.0), piece)


def test_clip_stroke_cuts_at_the_rect():
    parts = geo.clip_stroke(_line((-10, 50), (110, 50)), (0, 0, 100, 100))
    assert len(parts) == 1
    assert np.allclose(parts[0][0, 0], (0, 50), atol=0.01)
    assert np.allclose(parts[0][-1, 3], (100, 50), atol=0.01)


def test_clip_stroke_splits_when_leaving_and_reentering():
    stroke = np.stack([geo.line_to_cubic((10, 50), (150, 50)), geo.line_to_cubic((150, 50), (10, 80))])
    parts = geo.clip_stroke(stroke, (0, 0, 100, 100))
    assert len(parts) == 2
    assert np.allclose(parts[0][-1, 3], (100, 50), atol=0.01)
    assert np.allclose(parts[1][0, 0][0], 100, atol=0.01)


def test_piece_is_flat():
    assert geo.piece_is_flat(geo.line_to_cubic((0, 0), (10, 5)), 0.5)
    assert not geo.piece_is_flat(np.array([[0, 0], [3, 8], [7, 8], [10, 0]], float), 0.5)
    # controls on the line but far past its end also bend the curve
    assert not geo.piece_is_flat(np.array([[0, 0], [-20, 0], [10, 0], [10, 0]], float), 0.5)


def test_orient_and_sort_strokes():
    a = _line((50, 5), (10, 5))   # reversed: should start at x=10
    b = _line((0, 90), (5, 95))
    out = geo.sort_strokes([a, b])
    assert np.allclose(out[0][0, 0], (0, 90))
    assert np.allclose(out[1][0, 0], (10, 5))


def test_tile_grid_covers_image():
    h, w, t, ov = 300, 250, 128, 16
    covered = np.zeros((h, w), bool)
    for x0, y0 in geo.tile_grid(h, w, t, ov):
        covered[y0:y0 + t, x0:x0 + t] = True
    assert covered.all()


def test_merge_tiles_ownership_snap_and_join():
    core_a = (float("-inf"), float("-inf"), 60.0, float("inf"))
    core_b = (60.0, float("-inf"), float("inf"), float("inf"))
    a = _line((10, 20), (58, 20))
    a_dup_in_b = _line((10.5, 20), (58, 20))     # B's overlap copy: midpoint in A's core
    b = _line((62, 21), (110, 20))
    merged = geo.merge_tiles([
        geo.TileStrokes(0, core_a, [a]),
        geo.TileStrokes(1, core_b, [a_dup_in_b, b]),
    ], snap_px=5.0)
    assert len(merged) == 1
    s = merged[0]
    assert len(s) == 2
    assert np.allclose(s[0, 0], (10, 20)) and np.allclose(s[-1, 3], (110, 20))
    assert np.allclose(s[0, 3], s[1, 0])  # joined at the snapped seam point


def test_merge_tiles_keeps_same_tile_topology():
    core = (float("-inf"), float("-inf"), float("inf"), float("inf"))
    s1, s2 = _line((0, 0), (10, 0)), _line((11, 0), (20, 0))
    merged = geo.merge_tiles([geo.TileStrokes(0, core, [s1, s2])], snap_px=5.0)
    assert len(merged) == 2

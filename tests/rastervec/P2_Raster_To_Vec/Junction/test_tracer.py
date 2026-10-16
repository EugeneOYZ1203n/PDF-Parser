from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.Junction.junction_test.pipeline import Params, binarize, run


def _white(h: int = 120, w: int = 160) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _near(p, q, tol: float = 4.0) -> bool:
    return float(np.hypot(p[0] - q[0], p[1] - q[1])) <= tol


def test_one_px_diagonal_survives_binarize_and_traces_to_one_straight_polyline():
    img = _white()
    cv2.line(img, (10, 10), (140, 100), 0, 1)  # 1 px wide -- a 2x2 open erased this
    ink = binarize(img, Params())
    assert ink.sum() >= 120

    pls = run(img).polylines
    assert len(pls) == 1
    pts = pls[0].points
    ends = (pts[0], pts[-1])
    assert any(_near(e, (10, 10)) for e in ends) and any(_near(e, (140, 100)) for e in ends)
    assert len(pts) <= 4  # Douglas-Peucker: a straight line, not one vertex per pixel
    assert pls[0].width == 1.0


def test_rdp_off_keeps_one_vertex_per_skeleton_pixel():
    img = _white()
    cv2.line(img, (10, 10), (140, 100), 0, 1)
    pts = run(img, Params(rdp_eps_px=0)).polylines[0].points
    assert len(pts) >= 125
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for a, b in zip(pts, pts[1:]))


@pytest.mark.parametrize("merge, expected", [(True, 2), (False, 4)])
def test_plus_crossing_is_two_strokes_with_continuation_and_four_without(merge, expected):
    img = _white(160, 160)
    cv2.line(img, (20, 80), (140, 80), 0, 3)
    cv2.line(img, (80, 20), (80, 140), 0, 3)
    res = run(img, Params(merge_through_junctions=merge))
    assert len(res.graph.nodes) == 1  # the junction pixel cluster is one node
    assert len(res.polylines) == expected
    lengths = [np.hypot(*np.subtract(pl.points[0], pl.points[-1])) for pl in res.polylines]
    assert min(lengths) > 40  # no junction stubs


def test_thick_x_crossing_collapses_to_one_junction():
    img = _white(200, 200)
    cv2.line(img, (20, 30), (180, 170), 0, 9)
    cv2.line(img, (20, 170), (180, 30), 0, 9)
    res = run(img)
    assert len(res.graph.nodes) == 1
    assert len(res.polylines) == 2


def test_pinhole_is_filled_and_makes_no_loop():
    img = _white()
    cv2.line(img, (10, 60), (150, 60), 0, 7)
    img[59:61, 80:82] = 255  # 4 px hole inside the stroke
    ink = binarize(img, Params())
    assert ink[59:61, 80:82].all()
    assert len(run(img).polylines) == 1


def test_large_closed_loop_traces_in_linear_time():
    r = 3500  # ~20k skeleton px; the old list-scan walk took ~15 s here
    img = np.full((2 * r + 20, 2 * r + 20), 255, np.uint8)
    cv2.circle(img, (r + 10, r + 10), r, 0, 3)
    res = run(img)
    assert len(res.polylines) == 1
    assert res.timings["graph"] < 1.0


def test_filled_square_is_one_outline_and_no_strokes():
    img = _white(120, 120)
    cv2.rectangle(img, (40, 40), (79, 79), 0, -1)
    res = run(img, Params(fill_min_width_px=12))
    assert res.polylines == []
    assert len(res.outlines) == 1
    (ring,) = res.outlines[0].rings
    assert ring[0] == ring[-1] and len(ring) == 5  # four corners, closed
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx((40, 40, 79, 79), abs=1)


def test_filled_ring_keeps_its_hole():
    img = _white(160, 160)
    cv2.circle(img, (80, 80), 60, 0, -1)
    cv2.circle(img, (80, 80), 25, 255, -1)
    res = run(img, Params(fill_min_width_px=12))
    assert len(res.outlines) == 1 and len(res.outlines[0].rings) == 2
    assert res.polylines == []


def test_line_attached_to_fill_is_one_stroke_ending_on_the_outline():
    img = _white(120, 200)
    cv2.rectangle(img, (20, 30), (69, 89), 0, -1)
    cv2.line(img, (70, 60), (180, 60), 0, 3)
    res = run(img, Params(fill_min_width_px=12))
    assert len(res.outlines) == 1
    assert len(res.polylines) == 1
    pts = res.polylines[0].points
    near, far = sorted((pts[0], pts[-1]))
    assert far[0] == pytest.approx(180, abs=3)
    # the regrown fill takes the first few px of the line, so the stroke ends
    # on the fill (its outline's small bump), not exactly at the square's edge
    assert 69 <= near[0] <= 76
    x, y = int(round(near[0])), int(round(near[1]))
    assert res.fill[y - 1:y + 2, x - 1:x + 2].any()


def test_crossing_of_thick_lines_is_not_a_fill():
    img = _white(200, 200)
    cv2.line(img, (20, 100), (180, 100), 0, 6)
    cv2.line(img, (100, 20), (100, 180), 0, 6)
    res = run(img, Params(fill_min_width_px=12))
    assert res.outlines == [] and res.fill is None
    assert len(res.polylines) == 2


def test_isolated_speck_is_removed():
    img = _white()
    img[50, 50] = 0
    img[50:52, 80] = 0          # 2 px speck
    img[20:40, 100:103] = 0     # a real stroke
    ink = binarize(img, Params())
    assert not ink[50, 50] and not ink[50, 80]
    assert ink[30, 101]


def test_closed_loop_traces_to_one_closed_polyline():
    img = _white()
    cv2.rectangle(img, (20, 20), (130, 90), 0, 2)
    pls = run(img).polylines
    assert len(pls) == 1
    pts = pls[0].points
    assert pts[0] == pts[-1]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert min(xs) == pytest.approx(20, abs=3) and max(xs) == pytest.approx(130, abs=3)
    assert min(ys) == pytest.approx(20, abs=3) and max(ys) == pytest.approx(90, abs=3)


def test_thick_l_corner_has_no_barb_and_keeps_its_corner():
    img = _white()
    cv2.line(img, (20, 100), (20, 20), 0, 7)
    cv2.line(img, (20, 100), (140, 100), 0, 7)
    pls = run(img).polylines
    assert len(pls) == 1  # no barb off the corner
    assert any(_near(p, (20, 100), 5) for p in pls[0].points)  # corner vertex kept


def test_run_logs_each_stage_at_debug_and_nothing_louder(caplog):
    """`run()` is called once per connected component (thousands of times
    per page), so its per-stage logging must stay at DEBUG -- INFO would
    flood the log at real page scale (see `adapter.py`'s own already-INFO
    per-layer summary for the aggregate view)."""
    img = _white()
    cv2.line(img, (10, 10), (140, 100), 0, 2)

    with caplog.at_level(logging.DEBUG, logger="rastervec.P2.Junction.trace"):
        run(img)

    for stage in ("binarize", "skeleton", "graph", "vectorize"):
        assert stage in caplog.text
    assert all(record.levelno <= logging.DEBUG for record in caplog.records)

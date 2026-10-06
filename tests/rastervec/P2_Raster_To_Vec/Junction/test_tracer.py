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


def test_one_px_diagonal_survives_binarize_and_traces_to_one_raw_polyline():
    img = _white()
    cv2.line(img, (10, 10), (140, 100), 0, 1)  # 1 px wide -- a 2x2 open erased this
    ink = binarize(img, Params())
    assert ink.sum() >= 120

    pls = run(img).polylines
    assert len(pls) == 1
    pts = pls[0].points
    ends = (pts[0], pts[-1])
    assert any(_near(e, (10, 10)) for e in ends) and any(_near(e, (140, 100)) for e in ends)
    # raw: one vertex per skeleton pixel, every step an 8-neighbour move
    assert len(pts) >= 125
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for a, b in zip(pts, pts[1:]))
    assert pls[0].width == 1.0


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

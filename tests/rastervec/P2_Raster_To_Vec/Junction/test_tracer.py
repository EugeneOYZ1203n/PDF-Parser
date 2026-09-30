from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.Junction.junction_test.geom import dist
from rastervec.P2_Raster_To_Vec.Junction.junction_test.pipeline import (
    Params,
    binarize,
    regularize,
    run,
)
from rastervec.P2_Raster_To_Vec.Junction.junction_test.types_ import Segment


def _white(h: int = 120, w: int = 160) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def test_one_px_diagonal_survives_binarize_and_traces_to_one_segment():
    img = _white()
    cv2.line(img, (10, 10), (140, 100), 0, 1)  # 1 px wide -- a 2x2 open erased this
    ink = binarize(img, Params())
    assert ink.sum() >= 120

    segs = run(img).segments
    assert len(segs) == 1
    s = segs[0]
    assert dist(s.p0, s.p1) == pytest.approx(dist((10, 10), (140, 100)), abs=4.0)


def test_isolated_speck_is_removed():
    img = _white()
    img[50, 50] = 0
    img[50:52, 80] = 0          # 2 px speck
    img[20:40, 100:103] = 0     # a real stroke
    ink = binarize(img, Params())
    assert not ink[50, 50] and not ink[50, 80]
    assert ink[30, 101]


def test_rectangle_outline_gives_four_segments():
    img = _white()
    cv2.rectangle(img, (20, 20), (130, 90), 0, 2)
    segs = run(img).segments
    assert len(segs) == 4
    lengths = sorted(dist(s.p0, s.p1) for s in segs)
    assert lengths[0] == pytest.approx(70, abs=5)
    assert lengths[-1] == pytest.approx(110, abs=5)


def test_thick_l_corner_has_no_barb():
    img = _white()
    cv2.line(img, (20, 100), (20, 20), 0, 7)
    cv2.line(img, (20, 100), (140, 100), 0, 7)
    segs = run(img).segments
    assert len(segs) == 2
    assert all(dist(s.p0, s.p1) > 50 for s in segs)
    assert all(s.width == pytest.approx(7, abs=2.5) for s in segs)


def test_collinear_pieces_merge_but_corners_and_foldbacks_do_not():
    p = Params()
    straight = regularize([Segment((0, 0), (50, 1)), Segment((50, 1), (100, 0))], p)
    assert len(straight) == 1
    assert {straight[0].p0, straight[0].p1} == {(0.0, 0.0), (100.0, 0.0)}

    corner = regularize([Segment((0, 0), (50, 0)), Segment((50, 0), (50, 50))], p)
    assert len(corner) == 2

    # folding back on itself is not a straight continuation
    fold = regularize([Segment((0, 0), (50, 0)), Segment((50, 0), (10, 0))], p)
    assert len(fold) == 2


def test_long_chain_of_collinear_pieces_collapses():
    p = Params()
    pieces = [Segment((10.0 * i, 0.0), (10.0 * (i + 1), 0.0)) for i in range(40)]
    out = regularize(pieces, p)
    assert len(out) == 1
    assert dist(out[0].p0, out[0].p1) == pytest.approx(400)


def test_run_logs_each_stage_at_debug_and_nothing_louder(caplog):
    """`run()` is called once per connected component (thousands of times
    per page), so its per-stage logging must stay at DEBUG -- INFO would
    flood the log at real page scale (see `adapter.py`'s own already-INFO
    per-layer summary for the aggregate view)."""
    img = _white()
    cv2.line(img, (10, 10), (140, 100), 0, 2)

    with caplog.at_level(logging.DEBUG, logger="rastervec.P2.Junction.trace"):
        run(img)

    for stage in ("binarize", "skeleton", "graph", "vectorize", "regularize"):
        assert stage in caplog.text
    assert all(record.levelno <= logging.DEBUG for record in caplog.records)

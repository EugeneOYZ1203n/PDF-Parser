"""Unit tests for `raster_parser/notebooks/filter_showcase_helpers.py` -- the
geometry, zoom and step-recomputation helpers behind
`filter_showcase.ipynb`."""

from __future__ import annotations

import numpy as np

from raster_parser.models.page import PageMeta
from notebooks._helpers import filter_showcase_helpers as h
from raster_parser.phase_vector_classify.v2_vector_class.line_geometry import (
    line_crossing_counts,
    line_cubic_crossings,
)


def _line(vector, p0, p1, **kw):
    bbox = (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
    return vector(items=[("l", p0, p1)], bbox=bbox, **kw)


def _meta(w=400.0, h=400.0, rotation=0):
    return PageMeta(index=0, number=1, mediabox=(0, 0, w, h), rotation=rotation, width=w, height=h)


# ---- crossing points -----------------------------------------------------


def test_crossing_points_hash_grid(vector):
    grid = [
        _line(vector, (0, 3), (10, 3)),
        _line(vector, (0, 7), (10, 7)),
        _line(vector, (3, 0), (3, 10)),
        _line(vector, (7, 0), (7, 10)),
    ]
    pts = h.crossing_points(grid)
    assert sorted({(round(x, 6), round(y, 6)) for x, y in pts}) == [(3, 3), (3, 7), (7, 3), (7, 7)]
    # each line's own points match what the step counts
    counts = line_crossing_counts(grid, eps=h.CROSS_EPS_PT)
    for v, n in zip(grid, counts):
        assert len(h.crossing_points(grid, only=[v])) == n


def test_crossing_points_only_filters_targets(vector):
    a = _line(vector, (0, 5), (10, 5))
    b = _line(vector, (5, 0), (5, 10))
    assert h.crossing_points([a, b], only=[a]) == [(5.0, 5.0)]


def test_line_cubic_points_match_count():
    seg = (0.0, 5.0, 30.0, 5.0)
    cubic = (
        (0.0, 0.0),
        (10.0, 20.0),
        (20.0, -10.0),
        (30.0, 10.0),
    )  # S-curve crossing y=5 three times
    pts = h.line_cubic_points(seg, cubic, 0.01)
    assert len(pts) == line_cubic_crossings(seg, cubic, 0.01) == 3
    assert all(abs(y - 5.0) < 1e-6 for _x, y in pts)


def test_crossing_points_line_vs_curve(vector):
    line = _line(vector, (0, 5), (30, 5))
    curve = vector(items=[("c", (0, 0), (10, 20), (20, -10), (30, 10))], bbox=(0, -10, 30, 20))
    assert (
        len(h.crossing_points([line, curve]))
        == line_crossing_counts([line, curve], eps=0.01)[0]
        == 3
    )


# ---- zoom window -----------------------------------------------------------


def _dot(vector, x, y):
    return vector(kind="re", bbox=(x, y, x + 1, y + 1))


def test_zoom_window_prefers_mixed_region(vector):
    dropped = [_dot(vector, 300 + i, 300 + i) for i in range(20)]  # dropped only
    dropped += [_dot(vector, 50 + i, 50) for i in range(5)]  # mixed region
    kept = [_dot(vector, 55 + i, 60) for i in range(5)]
    x0, y0, x1, y1 = h.zoom_window(dropped, kept, _meta(), 0.25)
    assert x1 - x0 == y1 - y0 == 100
    assert x0 <= 50 and x1 >= 60 and y0 <= 50 and y1 >= 61


def test_zoom_window_none_without_both(vector):
    assert h.zoom_window([_dot(vector, 5, 5)], [_dot(vector, 300, 300)], _meta(), 0.25) is None
    assert h.zoom_window([], [_dot(vector, 5, 5)], _meta(), 0.25) is None


# ---- recomputation matches the step's own categories -----------------------


def _page_pdf(synthetic_pdf_factory, tmp_path, drawings, **page):
    doc = synthetic_pdf_factory([{"width": 400, "height": 400, "drawings": drawings, **page}])
    path = tmp_path / "page.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def _dash_line():
    return [{"lines": [((10 + 4 * i, 50), (12 + 4 * i, 50))]} for i in range(60)]


def _rect_lattice():
    return [
        {"rects": [(100 + 10 * i, 150 + 10 * j, 102 + 10 * i, 152 + 10 * j)]}
        for i in range(5)
        for j in range(5)
    ]


def test_collinear_data_matches_drawing_category(synthetic_pdf_factory, tmp_path):
    pdf = _page_pdf(
        synthetic_pdf_factory, tmp_path, _dash_line() + [{"lines": [((10, 300), (200, 310))]}]
    )
    r = h.run(pdf, 0)
    data = h.collinear_data(r)
    assert len(data.drawing) == 1 and len(data.drawing[0]) == 60
    group_ids = [h.ids(g) for g in data.groups]
    assert h.ids(data.drawing[0]) in group_ids
    assert len(data.kept) == 1


def test_pattern_data_matches_pattern_category(synthetic_pdf_factory, tmp_path):
    pdf = _page_pdf(
        synthetic_pdf_factory, tmp_path, _rect_lattice() + [{"lines": [((10, 300), (200, 310))]}]
    )
    r = h.run(pdf, 0)
    data = h.pattern_data(r)
    assert len(data.pattern) == 1 and len(data.pattern[0]) == 25
    largest = max(data.groups, key=lambda g: len(g.vectors))
    assert h.ids(largest.vectors) == h.ids(data.pattern[0])
    assert len(largest.links) == 40  # 5x5 lattice: 2 * 5 * 4 neighbour links
    assert len(data.similar) == 1 and len(data.kept) == 1


# ---- rendering -------------------------------------------------------------


def _dark_pixels(img) -> int:
    return int((np.asarray(img.convert("L")) < 128).sum())


def test_rotated_page_crops_onto_the_vectors(synthetic_pdf_factory, tmp_path):
    pdf = _page_pdf(
        synthetic_pdf_factory,
        tmp_path,
        [{"rects": [(10, 10, 30, 30)], "fill": (0, 0, 0)}],
        width=200,
        height=100,
        rotation=90,
    )
    r = h.run(pdf, 0)
    assert r.page_meta.rotation == 90
    on = h.Frame(r.page_meta, (5, 5, 35, 35), 200)
    off = h.Frame(r.page_meta, (100, 50, 130, 80), 200)
    assert _dark_pixels(h.output(on, r.vectors)) > 1000
    assert _dark_pixels(h.original(pdf, on)) > 1000
    assert _dark_pixels(h.highlight(on, [], [h.Layer(r.vectors, h.RED)])) > 100
    assert _dark_pixels(h.output(off, r.vectors)) == 0


def test_titled_adds_strip():
    from PIL import Image

    img = Image.new("RGB", (300, 100), "white")
    out = h.titled(img, "a fairly long title " * 5)
    assert out.width == 300 and out.height > 100

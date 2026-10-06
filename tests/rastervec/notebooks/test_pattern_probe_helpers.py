"""Unit tests for `rastervec/notebooks/_pattern_probe_helpers.py` -- the
stipple / hatching signals behind `pattern_texture_lab.ipynb`."""
from __future__ import annotations

import math
import random

from rastervec.notebooks import _pattern_probe_helpers as p


def _drawing(vector, items):
    return vector(items=items, bbox=(0, 0, 1, 1))


def _hatch(vector, n, spacing, length=50.0, angle_deg=0.0, x0=0.0, y0=0.0):
    th = math.radians(angle_deg)
    ux, uy, nx, ny = math.cos(th), math.sin(th), -math.sin(th), math.cos(th)
    items = []
    for i in range(n):
        ox, oy = x0 + i * spacing * nx, y0 + i * spacing * ny
        items.append(("l", (ox, oy), (ox + length * ux, oy + length * uy)))
    return _drawing(vector, items)


def _runs(elements, **kw):
    runs = p.hatch_runs(elements, **kw)
    p.pair_cross_hatch(runs)
    p.score_dominance(runs, elements)
    p.judge_hatch(runs)
    return runs


def _regions(elements):
    cands = p.stipple_candidates(elements)
    return p.grow_regions(cands, p.neighbourhood_stats(cands))


# ---- elements ----------------------------------------------------------

def test_explode_splits_on_pen_lift_and_closed_items(vector):
    v = _drawing(vector, [
        ("l", (0, 0), (1, 0)), ("l", (1, 0), (1, 1)),  # one connected polyline
        ("l", (5, 5), (6, 5)),                          # pen lift -> new element
        ("re", (10, 10, 11, 11)),                       # closed -> own element
        ("l", (6, 5), (7, 5)),                          # after a re -> new element
    ])
    els = p.explode_elements(v)
    assert [len(e.vector.items) for e in els] == [2, 1, 1, 1]
    assert els[0].bbox == (0, 0, 1, 1)
    assert els[1].line is not None and els[0].line is None


def test_zero_length_dot_is_size_zero(vector):
    (e,) = p.explode_elements(_drawing(vector, [("l", (3, 3), (3, 3))]))
    assert e.size == 0 and e.ink_len == 0 and e.line is None


def test_shape_key_is_translation_invariant(vector):
    a, b = p.explode_elements(_drawing(vector, [("re", (0, 0, 1, 1)), ("re", (7, 9, 8, 10))]))
    assert a.shape == b.shape


# ---- stipple -----------------------------------------------------------

def test_regular_dot_grid_is_stipple(vector):
    grid = _drawing(vector, [("re", (x * 5, y * 5, x * 5 + 1, y * 5 + 1)) for x in range(6) for y in range(6)])
    (region,) = _regions(p.explode_elements(grid))
    assert region.count == 36
    assert region.clark_evans > 1.5
    assert region.isotropy > 0.3
    assert region.is_stipple


def test_dotted_line_is_not_stipple(vector):
    row = _drawing(vector, [("re", (x * 4, 0, x * 4 + 1, 1)) for x in range(30)])
    (region,) = _regions(p.explode_elements(row))
    assert region.isotropy < 0.05
    assert region.area == 0 and not region.is_stipple


def test_mixed_sizes_fail_size_cv(vector):
    rng = random.Random(0)
    items = []
    for x in range(6):
        for y in range(6):
            s = rng.choice([0.2, 1.9])
            items.append(("re", (x * 5, y * 5, x * 5 + s, y * 5 + s)))
    (region,) = _regions(p.explode_elements(_drawing(vector, items)))
    assert region.size_cv > 0.3 and not region.is_stipple


def test_separate_fields_do_not_link(vector):
    a = [("re", (x * 2, y * 2, x * 2 + 0.5, y * 2 + 0.5)) for x in range(5) for y in range(5)]
    b = [("re", (200 + x * 2, y * 2, 200 + x * 2 + 0.5, y * 2 + 0.5)) for x in range(5) for y in range(5)]
    regions = _regions(p.explode_elements(_drawing(vector, a + b)))
    assert sorted(r.count for r in regions) == [25, 25]


# ---- hatching ----------------------------------------------------------

def test_even_parallel_lines_are_hatch(vector):
    (run,) = _runs(p.explode_elements(_hatch(vector, 10, 3.0)))
    assert run.count == 10
    assert run.spacing_cv < 1e-9
    assert run.density_ratio < 0.1
    assert run.dominance == 1.0
    assert run.is_hatch


def test_irregular_spacing_fails(vector):
    ys = [0, 1, 5, 6, 6.5, 11, 12, 16, 16.4, 21]
    v = _drawing(vector, [("l", (0, y), (50, y)) for y in ys])
    (run,) = _runs(p.explode_elements(v))
    assert run.spacing_cv > 0.25 and not run.is_hatch


def test_sparse_parallel_lines_are_not_a_run(vector):
    runs = _runs(p.explode_elements(_hatch(vector, 5, 30.0)))
    assert all(r.count == 1 for r in runs)


def test_dashed_line_is_one_track_not_a_run(vector):
    dash = _drawing(vector, [("l", (x * 4, 0), (x * 4 + 2, 0)) for x in range(20)])
    (run,) = _runs(p.explode_elements(dash))
    assert run.count == 1 and len(run.tracks[0].members) == 20


def test_dashed_hatch_counts_each_dashed_line_once(vector):
    items = [("l", (x * 4, y * 2), (x * 4 + 2, y * 2)) for y in range(8) for x in range(10)]
    (run,) = _runs(p.explode_elements(_drawing(vector, items)))
    assert run.count == 8


def test_offset_lines_without_along_overlap_do_not_chain(vector):
    # Staircase: each line starts where the previous one ended.
    v = _drawing(vector, [("l", (i * 10, i * 2), (i * 10 + 10, i * 2)) for i in range(6)])
    assert all(r.count == 1 for r in _runs(p.explode_elements(v)))


def test_same_angle_hatches_elsewhere_stay_separate(vector):
    a = _hatch(vector, 6, 2.0, x0=0)
    b = _hatch(vector, 6, 2.0, x0=500, y0=1.0)  # interleaved rho, no along-overlap
    runs = _runs(p.explode_all([a, b]))
    assert sorted(r.count for r in runs) == [6, 6]


def test_cross_hatch_pairs_and_keeps_dominance(vector):
    a = _hatch(vector, 10, 3.0, length=30.0, angle_deg=0.0)
    b = _hatch(vector, 10, 3.0, length=30.0, angle_deg=90.0, x0=30.0)
    runs = _runs(p.explode_all([a, b]))
    big = [r for r in runs if r.count == 10]
    assert len(big) == 2
    assert all(r.partners for r in big)
    assert all(r.dominance == 1.0 and r.is_hatch for r in big)


def test_clutter_lowers_dominance(vector):
    hatch = _hatch(vector, 10, 3.0, length=30.0)
    # Long strokes at all-different angles: no parallel family, so no cross-hatch partner.
    clutter = _drawing(vector, [
        ("l", (15 - 12 * math.cos(math.radians(a)), 13 - 12 * math.sin(math.radians(a))),
              (15 + 12 * math.cos(math.radians(a)), 13 + 12 * math.sin(math.radians(a))))
        for a in range(20, 170, 15)
    ])
    run = max(_runs(p.explode_all([hatch, clutter])), key=lambda r: r.count)
    assert run.count == 10
    assert run.dominance < 0.7 and not run.is_hatch

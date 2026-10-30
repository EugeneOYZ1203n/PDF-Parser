"""Liu & Dori (1997) protocol (`liu_dori.py`).

`paper_mode` tests reproduce the paper's Table 2 (single-bar detection
cases, pixel units, ground truth (10,20)-(90,20) width 8). The rest cover
the production config's deviations (width floor, angle gate, 3-factor
Q_v) and the edge cases the plan accepted (dashed GT, duplicate GT,
curves -> bars)."""

from __future__ import annotations

import math

import pytest

from raster_parser.evaluation.evaluate import liu_dori
from raster_parser.evaluation.evaluate.liu_dori import LiuDoriConfig
from raster_parser.evaluation.evaluate.vector_metrics import GeometryEntry

PAPER = LiuDoriConfig(paper_mode=True)
PROD = LiuDoriConfig()


def _line(p0, p1, *, width=8.0, dashes=None, group=0, vtype="vector_to_raster"):
    return GeometryEntry(
        kind="l", points=(p0, p1), width=width, dashes=dashes, group=group, vector_type=vtype
    )


def _run(gt, pred, cfg=PAPER, dpi=72.0, **kw):
    return liu_dori.evaluate(gt, pred, dpi, cfg, **kw)


GT = [_line((10, 20), (90, 20))]

# (name, preds, extra GT or None, expected {metric: value}, tolerance)
TABLE_2 = [
    (
        "good",
        [_line((10, 21), (90, 21))],
        None,
        dict(qvc=1.0, qb=1.0, qfr=1.0, qvg=1.0, dv=1.0, fv=0.0, vri=1.0),
        0.01,
    ),
    (
        "short",
        [_line((30, 20), (70, 20))],
        None,
        dict(qvc=1.0, qb=0.5, qfr=1.0, qvg=0.5, dv=0.5, fv=0.0, vri=0.75),
        0.01,
    ),
    (
        "long",
        [_line((5, 20), (95, 20))],
        None,
        dict(qvc=1.0, qb=1.0, qfr=1.0, qvg=1.0, dv=1.0, fv=0.11, vri=0.94),
        0.01,
    ),
    (
        "skew",
        [_line((10, 17), (90, 23))],
        None,
        dict(qvc=0.74, qb=0.74, qfr=1.0, qvg=0.74, dv=0.74, fv=0.26, vri=0.74),
        0.01,
    ),
    (
        "narrow",
        [_line((10, 20), (90, 20), width=6.0)],
        None,
        dict(qvc=0.95, qb=0.95, qfr=1.0, qvg=0.95, dv=0.95, fv=0.05, vri=0.95),
        0.01,
    ),
    (
        "style error",
        [_line((10, 20), (90, 20), dashes="[6 3] 0")],
        None,
        dict(qvc=0.82, qb=0.82, qfr=1.0, qvg=0.82, dv=0.82, fv=0.18, vri=0.82),
        0.01,
    ),
    # The paper works in integer pixels (its d2 rounds to 3); ours is 2.54.
    (
        "errors",
        [_line((30, 17), (95, 23), width=6.0)],
        None,
        dict(qvc=0.70, qb=0.53, qfr=1.0, qvg=0.53, dv=0.53, fv=0.35, vri=0.59),
        0.02,
    ),
    (
        "fragmentary",
        [_line((10, 20), (35, 20)), _line((50, 20), (90, 20), group=1)],
        None,
        dict(qb=0.81, qfr=0.73, qvg=0.59, dv=0.59, fv=0.0, vri=0.80),
        0.01,
    ),
    # The paper prints VRI 0.71 here, inconsistent with its own Eq. 27 for
    # its printed D_v 1 / F_v 0.41 (= 0.795; every other column satisfies
    # Eq. 27) -- so VRI is checked against Eq. 27.
    (
        "merged",
        [_line((10, 20), (90, 20))],
        [_line((10, 20), (35, 20)), _line((50, 20), (90, 20), group=1)],
        dict(dv=1.0, fv=0.41, vri=(1.0 + 0.59) / 2),
        0.01,
    ),
]


@pytest.mark.parametrize("name,preds,gt,expected,tol", TABLE_2, ids=[c[0] for c in TABLE_2])
def test_paper_table_2(name, preds, gt, expected, tol):
    _g, _p, overlaps, res = _run(gt or GT, preds)
    got = {
        "dv": res.d_v,
        "fv": res.f_v,
        "vri": res.vri,
    }
    if gt is None:
        g = res.gt_scores[0]
        got.update(qb=g.q_b, qfr=g.q_fr, qvg=g.q_v)
        if len(overlaps) == 1:
            got["qvc"] = overlaps[0].q_v
    for k, v in expected.items():
        assert got[k] == pytest.approx(v, abs=tol), (name, k, got[k])


def test_no_overlap_is_a_miss_and_false_alarm():
    _g, _p, overlaps, res = _run(GT, [_line((10, 60), (90, 60))])
    assert len(overlaps) == 0
    assert res.d_v == 0.0 and res.f_v == 1.0 and res.vri == 0.0
    assert res.gt.histogram["0"] == 1 and res.pred.histogram["0"] == 1


# ---- production config -----------------------------------------------------
def test_hairline_gt_matches_through_width_floor():
    # 300 dpi, 2 px floor = 0.48 pt; a 0.1 pt offset is well inside it.
    gt = [_line((0, 0), (100, 0), width=0.0)]
    pred = [_line((0, 0.1), (100, 0.1), width=0.0)]
    _g, _p, overlaps, res = _run(gt, pred, PROD, dpi=300.0)
    assert len(overlaps) == 1
    assert res.d_v == pytest.approx(math.exp(-(0.2 / 0.48 + 0.2 / 0.48) / 3), abs=1e-3)


def test_hairline_without_floor_never_matches():
    gt = [_line((0, 0), (100, 0), width=0.0)]
    pred = [_line((0, 0.1), (100, 0.1), width=0.0)]
    cfg = LiuDoriConfig(floor_px=0.0)
    _g, _p, overlaps, _res = _run(gt, pred, cfg, dpi=300.0)
    assert len(overlaps) == 0


def test_angle_gate_rejects_perpendicular_short_bar():
    # A 0.3 pt pred bar lying across a 1 pt-wide GT line: both its ends sit
    # inside the GT area, so the paper's rule alone would match it.
    gt = [_line((0, 0), (50, 0), width=1.0)]
    pred = [_line((25, -0.15), (25, 0.15), width=1.0)]
    assert len(_run(gt, pred, PAPER)[2]) == 1
    assert len(_run(gt, pred, PROD, dpi=300.0)[2]) == 0


def test_production_qv_is_cube_root_of_three_factors():
    gt = [_line((0, 0), (100, 0), width=2.0)]
    pred = [_line((0, 0.5), (100, 0.5), width=1.0)]
    _g, _p, overlaps, _res = _run(gt, pred, PROD, dpi=300.0)
    o = overlaps[0]
    assert o.q_v == pytest.approx((o.q_pt * o.q_od * o.q_w) ** (1 / 3))
    assert o.q_w == pytest.approx(math.exp(-0.5))


def test_dashed_gt_is_one_line_fragmented_by_dashes():
    gt = [_line((0, 0), (90, 0), width=1.0, dashes="[6 3] 0")]
    pred = [_line((x, 0), (x + 6, 0), width=1.0, group=i) for i, x in enumerate(range(0, 90, 9))]
    _g, _p, _o, res = _run(gt, pred, PROD, dpi=300.0)
    g = res.gt_scores[0]
    assert g.n_matched == 10
    assert g.q_b == pytest.approx(60 / 90)
    assert g.q_fr == pytest.approx(math.sqrt(10 * 36) / 60)
    assert res.by_style["dashed"].d_v.value == pytest.approx(res.d_v)
    assert not res.by_style["solid"].d_v.applicable


def test_duplicate_gt_halves_fragmentation_quality_of_the_pred():
    gt = [_line((0, 0), (100, 0), width=1.0), _line((0, 0), (100, 0), width=1.0, group=1)]
    pred = [_line((0, 0), (100, 0), width=1.0)]
    _g, _p, _o, res = _run(gt, pred, PROD, dpi=300.0)
    assert res.d_v == pytest.approx(1.0)
    k = res.pred_scores[0]
    assert k.n_matched == 2
    assert k.q_fr == pytest.approx(1 / math.sqrt(2))
    assert res.f_v == pytest.approx(1 - 1 / math.sqrt(2))


def test_curve_becomes_bars_that_match_the_same_curve():
    arc = GeometryEntry(
        kind="c",
        points=((0, 0), (30, 40), (70, 40), (100, 0)),
        width=1.0,
        group=0,
        vector_type="vector_to_raster",
    )
    pred = GeometryEntry(
        kind="c", points=((0, 0), (30, 40), (70, 40), (100, 0)), width=1.0, group=0
    )
    gt_bars, pred_bars, _o, res = _run([arc], [pred], PROD, dpi=300.0)
    assert len(gt_bars) > 3
    assert all(b.width == 1.0 for b in gt_bars)
    assert res.d_v == pytest.approx(1.0, abs=1e-6)
    assert res.f_v == pytest.approx(0.0, abs=1e-6)


def test_text_split():
    gt = [_line((0, 0), (10, 0), width=1.0), _line((50, 0), (100, 0), width=1.0, group=1)]
    pred = [_line((0, 0), (10, 0), width=1.0)]
    _g, _p, _o, res = _run(gt, pred, PROD, dpi=300.0, text_boxes=[(-1, -1, 11, 1)])
    assert res.by_text["text"].d_v.value == pytest.approx(1.0)
    assert res.by_text["non-text"].d_v.value == 0.0
    assert res.by_text["text"].f_v.value == pytest.approx(0.0)
    assert not res.by_text["non-text"].f_v.applicable


def test_empty_prediction_and_empty_gt():
    _g, _p, _o, res = _run(GT, [], PROD, dpi=300.0)
    assert res.d_v == 0.0 and math.isnan(res.f_v) and math.isnan(res.vri)
    _g, _p, _o, res = _run([], GT, PROD, dpi=300.0)
    assert math.isnan(res.d_v) and res.f_v == 1.0


def test_aggregate_micro_averages():
    _g, _p, _o, a = _run(GT, [_line((30, 20), (70, 20))])  # D_v 0.5 over 80
    _g, _p, _o, b = _run([_line((0, 50), (40, 50))], [_line((0, 50), (40, 50))])  # D_v 1 over 40
    agg = liu_dori.aggregate_liu_dori([a, b])
    assert agg.overall.d_v.value == pytest.approx((0.5 * 80 + 40) / 120)
    assert agg.gt.histogram["1"] == 2
    assert agg.n_overlaps == 2


def test_is_dashed():
    assert not liu_dori.is_dashed(None)
    assert not liu_dori.is_dashed("[] 0")
    assert liu_dori.is_dashed("[6 3] 0")


def test_pixel_stats():
    import numpy as np

    gt = np.zeros((4, 4), bool)
    gt[0, :] = True
    pred = np.zeros((4, 4), bool)
    pred[0, :2] = True
    pred[3, :2] = True
    s = liu_dori.pixel_stats(gt, pred)
    assert s.d_p.value == 0.5 and s.f_p.value == 0.5 and s.pri == 0.5

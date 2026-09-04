"""Geometry-accuracy metrics: prediction (PipelineResult) vs GroundTruth."""
from __future__ import annotations

import cv2
import numpy as np

from .geom import dist, seg_midpoint
from .types_ import GroundTruth, PipelineResult, Segment, StaircaseRegion


def _greedy_point_pairs(pred_pts, gt_pts, tol):
    cand = sorted(
        (dist(p, g), pi, gi)
        for pi, p in enumerate(pred_pts) for gi, g in enumerate(gt_pts) if dist(p, g) <= tol
    )
    up, ug, pairs = set(), set(), []
    for _, pi, gi in cand:
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi); pairs.append((pi, gi))
    return pairs


def _render_segments(shape, segs: list[Segment], arcs=None) -> np.ndarray:
    c = np.zeros(shape, np.uint8)
    for s in segs:
        cv2.line(c, _i(s.p0), _i(s.p1), 255, max(1, int(round(s.width))))
    for a in (arcs or []):
        pts = np.round(np.asarray(a.polyline)).astype(np.int32)
        cv2.polylines(c, [pts], a.closed, 255, max(1, int(round(a.width))))
    return c > 0


def _i(p):
    return (int(round(p[0])), int(round(p[1])))


def mask_iou(result: PipelineResult) -> float:
    pred = _render_segments(result.ink.shape, result.segments, result.arcs)
    ink = result.ink.astype(bool)
    inter = (pred & ink).sum()
    union = (pred | ink).sum()
    return float(inter / union) if union else 0.0


def coverage_pct(result: PipelineResult) -> float:
    ink = result.ink.astype(bool).sum()
    rem = result.remainder.astype(bool).sum()
    return float(1.0 - rem / ink) if ink else 0.0


def _match_segments(pred: list[Segment], gt: list[Segment], tol: float):
    cand = []
    for pi, ps in enumerate(pred):
        pm = seg_midpoint(ps.p0, ps.p1)
        for gi, gs in enumerate(gt):
            gm = seg_midpoint(gs.p0, gs.p1)
            d = dist(pm, gm)
            if d <= tol:
                cand.append((d, pi, gi))
    cand.sort()
    used_p, used_g, pairs = set(), set(), []
    for d, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi))
    return pairs


def endpoint_hausdorff(result: PipelineResult, gt: GroundTruth, tol: float = 40.0) -> float:
    pairs = _match_segments(result.segments, gt.segments, tol)
    if not pairs:
        return float("nan")
    errs = []
    for pi, gi in pairs:
        ps, gs = result.segments[pi], gt.segments[gi]
        e1 = max(dist(ps.p0, gs.p0), dist(ps.p1, gs.p1))
        e2 = max(dist(ps.p0, gs.p1), dist(ps.p1, gs.p0))
        errs.append(min(e1, e2))
    return float(np.mean(errs))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def junction_prf(result: PipelineResult, gt: GroundTruth, tol: float = 8.0) -> dict:
    pred = [j.xy for j in result.junctions]
    tgt = [j.xy for j in gt.junctions]
    cand = sorted(
        (dist(p, t), pi, ti)
        for pi, p in enumerate(pred) for ti, t in enumerate(tgt) if dist(p, t) <= tol
    )
    up, ut, tp = set(), set(), 0
    for _, pi, ti in cand:
        if pi in up or ti in ut:
            continue
        up.add(pi)
        ut.add(ti)
        tp += 1
    return _prf(tp, len(pred) - tp, len(tgt) - tp)


def dashed_prf(result: PipelineResult, gt: GroundTruth, tol: float = 30.0) -> dict:
    pred = [s for s in result.segments if s.dashed]
    tgt = [s for s in gt.segments if s.dashed]
    pairs = _match_segments(pred, tgt, tol)
    tp = len(pairs)
    return _prf(tp, len(pred) - tp, len(tgt) - tp)


def arc_count_error(result: PipelineResult, gt: GroundTruth) -> int:
    return len(result.arcs) - len(gt.arcs)


def segment_count_ratio(result: PipelineResult, gt: GroundTruth) -> float:
    g = max(len([s for s in gt.segments if not s.dashed]), 1)
    return len(result.segments) / g


def _match_by_point(pred_pts: list[tuple], gt_pts: list[tuple], tol: float) -> list[tuple[int, int]]:
    """Generic greedy nearest-point bipartite matcher (point-only version of
    `_match_segments`' midpoint matching, used by staircase/symbol metrics
    whose predictions are region/anchor points rather than Segments)."""
    cand = sorted(
        (dist(p, g), pi, gi)
        for pi, p in enumerate(pred_pts) for gi, g in enumerate(gt_pts) if dist(p, g) <= tol
    )
    up, ug, pairs = set(), set(), []
    for _, pi, gi in cand:
        if pi in up or gi in ug:
            continue
        up.add(pi)
        ug.add(gi)
        pairs.append((pi, gi))
    return pairs


def _region_center(r: StaircaseRegion) -> tuple[float, float]:
    return ((r.axis[0][0] + r.axis[1][0]) / 2.0, (r.axis[0][1] + r.axis[1][1]) / 2.0)


def staircase_prf(result: PipelineResult, gt: GroundTruth, tol: float = 40.0) -> dict:
    pred_pts = [_region_center(r) for r in result.staircases]
    gt_pts = [_region_center(r) for r in gt.staircases]
    pairs = _match_by_point(pred_pts, gt_pts, tol)
    tp = len(pairs)
    return _prf(tp, len(pred_pts) - tp, len(gt_pts) - tp)


def staircase_tread_count_err(result: PipelineResult, gt: GroundTruth, tol: float = 40.0) -> float:
    pairs = _match_by_point(
        [_region_center(r) for r in result.staircases],
        [_region_center(r) for r in gt.staircases], tol,
    )
    if not pairs:
        return float("nan")
    errs = [
        abs(result.staircases[pi].n_treads - gt.staircases[gi].n_treads) / max(gt.staircases[gi].n_treads, 1)
        for pi, gi in pairs
    ]
    return float(np.mean(errs))


def symbol_prf(result: PipelineResult, gt: GroundTruth, family: str, tol: float = 25.0) -> dict:
    pred = [s.anchor for s in result.symbols if s.family == family]
    tgt = [s.anchor for s in gt.symbols if s.family == family]
    pairs = _match_by_point(pred, tgt, tol)
    tp = len(pairs)
    return _prf(tp, len(pred) - tp, len(tgt) - tp)


# --------------------------------------------------- JUNCTION_ABLATION.md Sec 5


_JTYPES = ("L", "T", "X", "Y", "star", "endpoint", "coincident_unrelated")


def junction_type_confusion(result: PipelineResult, gt: GroundTruth, tol: float = 10.0) -> dict:
    """{(gt_type, pred_type): count} over greedily matched junction points."""
    pairs = _greedy_point_pairs([j.xy for j in result.junctions],
                                [j.xy for j in gt.junctions], tol)
    conf: dict = {}
    for pi, gi in pairs:
        key = (gt.junctions[gi].jtype or "?", result.junctions[pi].jtype or "?")
        conf[key] = conf.get(key, 0) + 1
    return conf


def junction_type_accuracy(result: PipelineResult, gt: GroundTruth, tol: float = 10.0) -> float:
    conf = junction_type_confusion(result, gt, tol)
    total = sum(conf.values())
    if not total:
        return float("nan")
    return sum(c for (g, p), c in conf.items() if g == p) / total


def x_passthrough_accuracy(result: PipelineResult, gt: GroundTruth, tol: float = 8.0) -> float:
    """At each GT X junction, fraction traced as (>=2) continuous strokes passing
    through, rather than stubs ending on the node."""
    xs = [j for j in gt.junctions if j.jtype == "X"]
    if not xs:
        return float("nan")
    scores = []
    for j in xs:
        through = 0
        for s in result.segments:
            if s.dashed:
                continue
            d = _point_to_seg(j.xy, s.p0, s.p1)
            ends = min(dist(j.xy, s.p0), dist(j.xy, s.p1))
            if d <= tol and ends > 2 * tol:        # node is interior to this segment
                through += 1
        scores.append(min(through, 2) / 2.0)
    return float(np.mean(scores))


def coincident_unrelated_false_merge(result: PipelineResult, gt: GroundTruth, tol: float = 8.0) -> float:
    """Fraction of GT coincident_unrelated points where the pipeline wrongly
    emitted a real junction."""
    cu = [j.xy for j in gt.junctions if j.jtype == "coincident_unrelated" or not j.is_true_connection]
    if not cu:
        return float("nan")
    pred = [j.xy for j in result.junctions]
    bad = sum(1 for c in cu if any(dist(c, p) <= tol for p in pred))
    return bad / len(cu)


def _point_to_seg(p, a, b):
    from .geom import point_to_segment_dist
    return point_to_segment_dist(p, a, b)


def width_mae(result: PipelineResult, gt: GroundTruth, tol: float = 30.0) -> float:
    pairs = _match_segments(result.segments, gt.segments, tol)
    if not pairs:
        return float("nan")
    return float(np.mean([abs(result.segments[pi].width - gt.segments[gi].width)
                          for pi, gi in pairs]))


def dash_style_accuracy(result: PipelineResult, gt: GroundTruth, tol: float = 30.0) -> float:
    pairs = _match_segments(result.segments, gt.segments, tol)
    if not pairs:
        return float("nan")
    ok = 0
    for pi, gi in pairs:
        pred_style = "dashed" if result.segments[pi].dashed else "solid"
        gt_style = "solid" if gt.segments[gi].dash_style == "solid" else "dashed"
        ok += pred_style == gt_style
    return ok / len(pairs)


def arc_radius_rel_err(result: PipelineResult, gt: GroundTruth, tol: float = 40.0) -> float:
    pairs = _greedy_point_pairs([a.center for a in result.arcs],
                                [a.center for a in gt.arcs], tol)
    if not pairs:
        return float("nan")
    return float(np.mean([abs(result.arcs[pi].radius - gt.arcs[gi].radius) / max(gt.arcs[gi].radius, 1.0)
                          for pi, gi in pairs]))


def curve_misfit_count(result: PipelineResult, gt: GroundTruth, tol: float = 6.0) -> int:
    """GT curved primitives (arcs/beziers) emitted as >=3 short straight segments
    with no matching predicted arc."""
    matched_centers = {i for _, i in _greedy_point_pairs(
        [a.center for a in result.arcs], [a.center for a in gt.arcs], 40.0)}
    misfit = 0
    curves = [(a.polyline, i) for i, a in enumerate(gt.arcs)] + \
             [(b.polyline, None) for b in gt.beziers]
    for poly, gi in curves:
        if gi in matched_centers:
            continue
        near = 0
        for s in result.segments:
            m = seg_midpoint(s.p0, s.p1)
            if min(dist(m, q) for q in poly) <= tol:
                near += 1
        if near >= 3:
            misfit += 1
    return misfit


def reconstruction_ssim(result: PipelineResult) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return float("nan")
    pred = _render_segments(result.ink.shape, result.segments, result.arcs).astype(np.uint8) * 255
    ink = result.ink.astype(np.uint8) * 255
    return float(ssim(pred, ink))


def runtime_s(result: PipelineResult) -> float:
    return float(sum(result.timings.values()))


def evaluate(result: PipelineResult, gt: GroundTruth) -> dict:
    stair = staircase_prf(result, gt)
    door = symbol_prf(result, gt, "door")
    window = symbol_prf(result, gt, "window")
    return {
        "mask_iou": mask_iou(result),
        "coverage_pct": coverage_pct(result),
        "endpoint_hausdorff": endpoint_hausdorff(result, gt),
        "junction_precision": junction_prf(result, gt)["precision"],
        "junction_recall": junction_prf(result, gt)["recall"],
        "junction_f1": junction_prf(result, gt)["f1"],
        "dashed_precision": dashed_prf(result, gt)["precision"],
        "dashed_recall": dashed_prf(result, gt)["recall"],
        "arc_count_error": arc_count_error(result, gt),
        "segment_count_ratio": segment_count_ratio(result, gt),
        "n_segments": len(result.segments),
        "n_arcs": len(result.arcs),
        "n_junctions": len(result.junctions),
        "staircase_precision": stair["precision"],
        "staircase_recall": stair["recall"],
        "staircase_f1": stair["f1"],
        "staircase_tread_count_err": staircase_tread_count_err(result, gt),
        "symbol_door_precision": door["precision"],
        "symbol_door_recall": door["recall"],
        "symbol_door_f1": door["f1"],
        "symbol_window_precision": window["precision"],
        "symbol_window_recall": window["recall"],
        "symbol_window_f1": window["f1"],
        "n_staircases": len(result.staircases),
        "n_symbols": len(result.symbols),
        "junction_type_accuracy": junction_type_accuracy(result, gt),
        "x_passthrough_accuracy": x_passthrough_accuracy(result, gt),
        "coincident_unrelated_false_merge": coincident_unrelated_false_merge(result, gt),
        "width_mae": width_mae(result, gt),
        "dash_style_accuracy": dash_style_accuracy(result, gt),
        "arc_radius_rel_err": arc_radius_rel_err(result, gt),
        "curve_misfit_count": curve_misfit_count(result, gt),
        "reconstruction_ssim": reconstruction_ssim(result),
        "runtime_s": runtime_s(result),
    }


def summarize(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float)) and not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out

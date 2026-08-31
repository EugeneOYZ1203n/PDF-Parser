"""Geometry-accuracy metrics: prediction (PipelineResult) vs GroundTruth."""
from __future__ import annotations

import cv2
import numpy as np

from .geom import dist, seg_midpoint
from .types_ import GroundTruth, PipelineResult, Segment


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


def evaluate(result: PipelineResult, gt: GroundTruth) -> dict:
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
    }


def summarize(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float)) and not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out

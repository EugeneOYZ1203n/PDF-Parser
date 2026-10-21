"""Grayscale renders of flattened GT / predicted geometry for
`vector_metrics.image_similarity` (SSIM/PSNR), and ink masks of Liu-Dori
bars at their floored widths for `liu_dori.pixel_stats` (D_p/F_p/PRI).

Both sides go through the identical path: polylines grouped by
`(gray, width)` into `draw.stroke_batch_spec` batches (the fast
content-stream path -- per-shape `Shape` drawing takes minutes on P2 output
with one item per pixel step), drawn on a fresh unrotated page of the page's
own size, lighter batches first so darker ink wins where strokes overlap,
then rasterised to an 8-bit gray pixmap.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import fitz
import numpy as np

from rastervec.Evaluation.Evaluate import liu_dori
from rastervec.Evaluation.Evaluate.vector_metrics import (
    VECTOR_TYPES,
    FlatPolyline,
    GeometryEntry,
    VectorMetricConfig,
    flatten_entries,
    gt_entries_union,
)
from rastervec.commons.renderer.draw import draw_specs, new_page_doc, stroke_batch_spec

if TYPE_CHECKING:
    from rastervec.Evaluation.Evaluate.liu_dori import Bars
    from rastervec.commons.models import PageMeta


def render_gray(polys: "list[FlatPolyline]", page_meta: "PageMeta", dpi: float) -> np.ndarray:
    """(H, W) uint8, white paper, unrotated page space."""
    batches: "dict[tuple[float, float], list]" = defaultdict(list)
    for p in polys:
        batches[(round(p.gray, 4), round(p.width, 4))].append(p.points.tolist())
    specs = [
        stroke_batch_spec(lines, (gray, gray, gray), width=width)
        for (gray, width), lines in sorted(batches.items(), key=lambda kv: -kv[0][0])
    ]
    doc, page = new_page_doc(page_meta)
    try:
        draw_specs(page, specs)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        return img[:, : pix.width].copy()
    finally:
        doc.close()


def render_pair(
    gt_by_type: "dict[str, list[GeometryEntry]]", preds: "list[GeometryEntry]",
    page_meta: "PageMeta", cfg: "VectorMetricConfig | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """`(gt_render, pred_render)` for `evaluate_vector_metrics(images=...)`;
    GT is the union of every `VECTOR_TYPES` type."""
    cfg = cfg or VectorMetricConfig()
    gt_polys = flatten_entries([e for t in VECTOR_TYPES for e in gt_by_type.get(t, [])], cfg)
    pred_polys = flatten_entries(preds, cfg)
    return (
        render_gray(gt_polys, page_meta, cfg.render_dpi),
        render_gray(pred_polys, page_meta, cfg.render_dpi),
    )


def render_bars_ink(bars: "Bars", page_meta: "PageMeta", dpi: float, threshold: int) -> np.ndarray:
    """(H, W) bool ink mask of `liu_dori.Bars` drawn black at their floored
    width (`w_eff`) on white, unrotated page space."""
    widths = np.round(bars.w_eff, 4)
    segs = np.stack([bars.p0, bars.p1], axis=1) if len(bars) else np.zeros((0, 2, 2))
    specs = [
        stroke_batch_spec(segs[widths == w].tolist(), (0.0, 0.0, 0.0), width=float(w))
        for w in np.unique(widths)
    ]
    doc, page = new_page_doc(page_meta)
    try:
        draw_specs(page, specs)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        return img[:, : pix.width] < threshold
    finally:
        doc.close()


def render_ink_pair(
    gt_by_type: "dict[str, list[GeometryEntry]]", preds: "list[GeometryEntry]",
    page_meta: "PageMeta", dpi: float, cfg: "VectorMetricConfig | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """`(gt_ink, pred_ink)` for `evaluate_vector_metrics(ink=...)` --
    `dpi` is the rasterised PDF's resolution (sets the width floor); the
    masks are rendered at `cfg.liu_dori.pixel_render_dpi`."""
    cfg = cfg or VectorMetricConfig()
    ld = cfg.liu_dori
    gt_bars = liu_dori.bars_from_entries(gt_entries_union(gt_by_type), dpi, ld)
    pred_bars = liu_dori.bars_from_entries(preds, dpi, ld)
    return (
        render_bars_ink(gt_bars, page_meta, ld.pixel_render_dpi, ld.ink_threshold),
        render_bars_ink(pred_bars, page_meta, ld.pixel_render_dpi, ld.ink_threshold),
    )

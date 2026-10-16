"""Grayscale renders of flattened GT / predicted geometry for
`vector_metrics.image_similarity` (SSIM/PSNR).

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

from rastervec.Evaluation.Evaluate.vector_metrics import (
    VECTOR_TYPES,
    FlatPolyline,
    GeometryEntry,
    VectorMetricConfig,
    flatten_entries,
)
from rastervec.commons.renderer.draw import draw_specs, new_page_doc, stroke_batch_spec

if TYPE_CHECKING:
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

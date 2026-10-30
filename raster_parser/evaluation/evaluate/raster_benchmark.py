"""Raster benchmark: how well a raster_vectorize backend recovers a page's line work.

The benchmark rasterises each page (`conversion.rasterise_page`) and runs the pipeline on the
image. The ground truth is the original page's own vectors (`raster_ground_truth`: every painted
`Vector`, split into line / curve entries), so no raster labels are drawn by hand. The text labels
(the hand-labelled vector text and the native text) give the boxes that split the scores into
text and non-text lines.

`score_vectors` is the one entry point the report and benchmark scripts call. The scoring itself
is the Liu & Dori (1997) protocol plus pixel, image, length and direction checks
(`vector_metrics.evaluate_vector_metrics`, `liu_dori.py`; `docs/RasterToVectorEvaluation.md`).
Changing the evaluation method means replacing this module's body, not its callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raster_parser.evaluation.evaluate import vector_metrics, vector_render
from raster_parser.evaluation.evaluate.vector_metrics import (
    GeometryEntry,
    VectorMetricConfig,
    VectorMetricSuiteResult,
    geometry_entries_from_vectors,
)

if TYPE_CHECKING:
    from raster_parser.models.page import PageMeta
    from raster_parser.models.vector import Vector

VECTOR_TYPE = "vector_to_raster"


def raster_ground_truth(vectors: "list[Vector]") -> list[GeometryEntry]:
    """Every painted vector (a stroke colour or a fill) as `"l"` / `"c"` entries, one group per
    vector, tagged `VECTOR_TYPE`."""
    painted = [v for v in vectors if v.color is not None or v.fill is not None]
    return [
        GeometryEntry(
            kind=e.kind,
            points=e.points,
            color=e.color,
            fill=e.fill,
            width=e.width,
            dashes=e.dashes,
            group=e.group,
            vector_type=VECTOR_TYPE,
        )
        for e in geometry_entries_from_vectors(painted)
    ]


def score_vectors(
    pred: "list[Vector]",
    gt: "list[Vector]",
    page_meta: "PageMeta",
    *,
    dpi: float,
    text_boxes: "list[tuple[float, float, float, float]] | None" = None,
    render: bool = True,
    cfg: VectorMetricConfig = VectorMetricConfig(),
) -> VectorMetricSuiteResult:
    """Score a raster run's vectors `pred` against the page's original vectors `gt`. `dpi` is the
    rasterised page's resolution. `render=False` skips the pixel and image comparisons."""
    gt_by_type = {VECTOR_TYPE: raster_ground_truth(gt)}
    preds = geometry_entries_from_vectors(pred)
    images = ink = None
    if render:
        images = vector_render.render_pair(gt_by_type, preds, page_meta, cfg)
        ink = vector_render.render_ink_pair(gt_by_type, preds, page_meta, dpi, cfg)
    result = vector_metrics.evaluate_vector_metrics(
        gt_by_type,
        preds,
        {},
        images=images,
        ink=ink,
        dpi=dpi,
        text_boxes=text_boxes,
        cfg=cfg,
    )
    # one ground-truth type here; count source vectors, not their line pieces
    result.label_counts = {VECTOR_TYPE: len({e.group for e in gt_by_type[VECTOR_TYPE]})}
    return result


aggregate = vector_metrics.aggregate_vector_metrics

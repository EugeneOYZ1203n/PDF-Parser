"""Bridge from a real pipeline run to the pure `metrics.py` inputs.

`metrics.py` is deliberately pipeline-agnostic (plain `GtRegion` /
`Prediction` / bbox lists). This module -- the only one in
`Evaluation/Evaluate/` that imports `rastervec.pipeline` -- turns a
`LabelSet` and a `PipelineContext` (or the loose dict the benchmark
notebook carries) into those inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rastervec.Evaluation.Evaluate.metrics import Bbox, GtRegion, Prediction
from rastervec.Evaluation.Labelling.label_schema import LabelSet
from rastervec.helpers.geometry import union_bbox
from rastervec.models import ClusterOcrResult

if TYPE_CHECKING:
    from rastervec.pipeline import ClusteringStageResult, GroupKey, PipelineContext


def gt_regions_from_labelset(labels: LabelSet) -> list[GtRegion]:
    return [
        GtRegion(
            page_index=e.page_index,
            bbox=tuple(e.cluster_bbox),  # type: ignore[arg-type]
            text=e.text,
            expected_rotation=e.expected_rotation,
        )
        for e in labels.entries
    ]


def predictions_from_cluster_ocr(
    results: list[ClusterOcrResult],
) -> list[Prediction]:
    """One `Prediction` per `ClusterOcrResult` -- blank readings kept
    (`ocr_blank=True`) so the classification metrics can still see that the
    cluster reached OCR."""
    preds: list[Prediction] = []
    for r in results:
        resolved = r.resolved
        preds.append(
            Prediction(
                text=resolved.text,
                bbox=tuple(resolved.bbox),  # type: ignore[arg-type]
                rotation=int(resolved.rotation_used),
                reached_ocr=True,
                ocr_blank=not resolved.text.strip(),
                source_cluster_id=id(r.cluster),
            )
        )
    return preds


def text_candidate_boxes(
    regrouped_clusters: list | None,
    cluster_ocr_results: list[ClusterOcrResult] | None,
) -> list[Bbox]:
    """Union bbox per cluster that reached OCR (blank or not). Prefers
    `regrouped_clusters` (exactly what `ocr_compare` ran on); falls back to
    each OCR result's own resolved bbox (the archive legacy path, which has
    no `regrouped_clusters`)."""
    if regrouped_clusters:
        return [
            union_bbox([p.bbox for p in cluster])
            for cluster in regrouped_clusters
            if cluster
        ]
    return [
        tuple(r.resolved.bbox)  # type: ignore[misc]
        for r in (cluster_ocr_results or [])
    ]


@dataclass
class EvalInputs:
    predictions: list[Prediction]
    text_candidate_boxes: list[Bbox]
    clustering: "dict[GroupKey, ClusteringStageResult] | None"
    fast_dropped: list | None
    ocr_failed: list | None


def build_eval_inputs(ctx: "PipelineContext") -> EvalInputs:
    """The pipeline-derived half of the metric inputs (everything except
    the ground truth, which varies by label source)."""
    return EvalInputs(
        predictions=predictions_from_cluster_ocr(ctx.cluster_ocr_results or []),
        text_candidate_boxes=text_candidate_boxes(
            getattr(ctx, "regrouped_clusters", None), ctx.cluster_ocr_results
        ),
        clustering=ctx.clustering,
        fast_dropped=ctx.fast_dropped,
        ocr_failed=ctx.ocr_failed,
    )

"""Bridge from a real pipeline run to the pure `metrics.py` inputs.

`metrics.py` is deliberately pipeline-agnostic (plain `GtRegion` /
`Prediction` / bbox lists). This module turns a `LabelSet` and a
`PipelineResult` into those inputs.

Retargeted for the segment-dedup pipeline: there is no more `cluster_ocr_
results`/`regrouped_clusters` -- every OCR reading is one `Text` (source=
"ocr") in `PipelineResult.texts`/`restored_texts`, one per real segment
occurrence (blank readings included, same as the old design's blank-kept
`ClusterOcrResult`s). `attribute_miss`'s `clustering`/`fast_dropped` need
the pipeline run with `verbose=True` (see `build_eval_inputs`'s call
site in `Reader/Parallel/benchmark_jobs.py`) -- both are verbose-only on
the new `PipelineResult`. There is no longer a distinct "OCR failed"
drop bucket (Phase F's FAST gate is the sole page-content pass/fail
decision; a blank OCR reading still becomes a `Prediction(ocr_blank=
True)`, already handled by the metrics that check that flag directly),
so `ocr_failed` is always `None` here -- `attribute_miss`'s "ocr_blank"
attribution reason is consequently unreachable now, a known, accepted
simplification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rastervec.Evaluation.Evaluate.metrics import Bbox, GtRegion, Prediction
from rastervec.Evaluation.Labelling.label_schema import LabelSet
from rastervec.models import Text

if TYPE_CHECKING:
    from rastervec.pipelines.result import ClusteringStageResult, GroupKey, PipelineResult


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


def predictions_from_texts(texts: list[Text]) -> list[Prediction]:
    """One `Prediction` per OCR `Text` (`source="ocr"`) -- blank readings
    kept (`ocr_blank=True`) so the classification metrics can still see
    that the segment reached OCR. `rotation` is `Text.angle()` snapped to
    the nearest quarter turn, matching `GtRegion.expected_rotation`'s own
    convention (`auto_label.py`'s "most common quarter turn" label)."""
    preds: list[Prediction] = []
    for t in texts:
        rotation = int(round(t.angle() / 90.0) * 90) % 360
        preds.append(
            Prediction(
                text=t.text,
                bbox=tuple(t.bbox),  # type: ignore[arg-type]
                rotation=rotation,
                reached_ocr=True,
                ocr_blank=not t.text.strip(),
                source_cluster_id=id(t),
            )
        )
    return preds


def text_candidate_boxes(texts: list[Text]) -> list[Bbox]:
    """Bbox per OCR `Text` (`source="ocr"`) -- every segment occurrence
    that reached OCR, blank or not."""
    return [tuple(t.bbox) for t in texts]  # type: ignore[misc]


@dataclass
class EvalInputs:
    predictions: list[Prediction]
    text_candidate_boxes: list[Bbox]
    clustering: "dict[GroupKey, ClusteringStageResult] | None"
    fast_dropped: list[list] | None
    ocr_failed: list[list] | None


def build_eval_inputs(res: "PipelineResult") -> EvalInputs:
    """The pipeline-derived half of the metric inputs (everything except
    the ground truth, which varies by label source). Requires `res` from a
    `verbose=True` run for `clustering`/`fast_dropped_vectors` to be
    populated -- see this module's docstring."""
    ocr_texts = [t for t in res.texts if t.source == "ocr"]
    return EvalInputs(
        predictions=predictions_from_texts(ocr_texts),
        text_candidate_boxes=text_candidate_boxes(ocr_texts),
        clustering=res.clustering,
        # `_region_matches_groups` expects list[list[<has .bbox>]] -- each
        # dropped Vector becomes its own singleton "group" now that FAST
        # drops flat Vectors rather than whole candidate clusters.
        fast_dropped=[[v] for v in (res.fast_dropped_vectors or [])],
        ocr_failed=None,
    )

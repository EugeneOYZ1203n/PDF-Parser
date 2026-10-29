"""Independent evaluation metrics for vector-text classification + OCR, built over one shared
many-to-many overlap graph between ground-truth text regions and predicted OCR readings.

- **Text types**, tagged on `GtRegion.text_type` (`metrics_core.TEXT_TYPES`): `native_to_vector`
  (`LabelEntry.source=="native"`) and `original_vector` (`source=="vector"`). Raster label
  sources are not scored yet.
- OCR confusion (category 8) and reading-order (category 7) live in
  `confusion_metrics.py`.

Why this shape: a single greedy 1:1 highest-IoU match reduces *every*
metric out of one assignment, so when one ground-truth line is covered by
several predicted clusters (the common case) all of text / bbox / rotation
accuracy corrupt together. Here each metric is an independent reduction
over `OverlapGraph`, which keeps every (gt, prediction) overlap edge and an
explicit N:1 assignment. One `OverlapGraph` is built per text type (same
flat `predictions` list each time, since the same OCR reading list is
scored against every GT bucket independently) -- `build_overlap_graph`
itself is unchanged and shared across all types.

This module is pure and does not import the pipeline -- callers pass plain
lists (`adapters.py` builds them from an `ExtractResult`).

**This file is a re-export surface.** The actual implementation is split by
concern across `metrics_core.py` (shared types + the `OverlapGraph`
itself), `metrics_text_overlap.py` (categories 1-3), `metrics_distributions.py`
(font-size distribution, category 4, category 5), and `metrics_suite.py`
(category 6, the box overlay data, and the top-level `evaluate_text_metrics`/
`aggregate_text_metrics` orchestration) -- every name below stays importable
from `raster_parser.evaluation.evaluate.metrics` regardless of which of those
files it actually lives in, since several callers (`adapters.py`,
`confusion_metrics.py`, `label_overlays.py`, tests, ...) import specific names straight from
this module.
"""

from __future__ import annotations

from raster_parser.evaluation.evaluate.metrics_core import (  # noqa: F401
    TEXT_TYPES,
    Bbox,
    GtRegion,
    MetricConfig,
    OverlapEdge,
    OverlapGraph,
    Prediction,
    Ratio,
    _NA,
    _multiset_overlap,
    _reading_order_key,
    _region_concat_hyp,
    build_overlap_graph,
    build_overlap_graphs_by_type,
    f1_from,
)
from raster_parser.evaluation.evaluate.metrics_text_overlap import (  # noqa: F401
    OverlapStats,
    TextLabelStats,
    _overlap_stats,
    _word_multiset,
    char_overlap_stats,
    text_label_stats,
    word_overlap_stats,
)
from raster_parser.evaluation.evaluate.metrics_distributions import (  # noqa: F401
    BboxAccuracyStats,
    FontSizeDistribution,
    RotationBucketCounts,
    RotationStats,
    _TEXT_TYPE_SIZE_UNIT,
    _circular_rotation_diff,
    _gt_size,
    _rotation_vote,
    bbox_accuracy_stats,
    bbox_accuracy_unclassified,
    font_size_distribution,
    per_gt_union_pred_iou_mean,
    rotation_outcome,
    rotation_stats,
    truly_unclassified_pred_indices,
)
from raster_parser.evaluation.evaluate.metrics_suite import (  # noqa: F401
    MATCH_BOX_COLOR,
    MISSED_GT_BOX_COLOR,
    ClassificationFunnelStats,
    PerTypeTextResult,
    TextMetricSuiteResult,
    _agg_ratio,
    aggregate_text_metrics,
    classification_funnel_stats,
    evaluate_text_metrics,
)

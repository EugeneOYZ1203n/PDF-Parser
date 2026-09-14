"""Independent evaluation metrics for the Vector_Classification + OCR
pipeline, built over one shared many-to-many overlap graph between
ground-truth text regions and predicted OCR readings.

Reworked for the 4 text-provenance types / 3 vector-provenance types split
(see `docs/EVAL_METRICS.md` and the plan this rework was built from):

- **4 text types**, tagged on `GtRegion.text_type`: `native_to_vector`
  (`LabelEntry.source=="native"`), `original_vector` (`source=="vector"`),
  `vector_to_raster` (`source=="raster"`, `label_id` prefixed `"vecsync:"`),
  `original_raster` (`source=="raster"`, no such prefix).
- **3 vector types** live in `vector_metrics.py` (a separate module -- pure
  geometry/property comparison, no overlap-graph machinery in common with
  text scoring).
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
itself is unchanged and shared across all 4.

This module is pure and does not import the pipeline -- callers pass plain
lists (`adapters.py` builds them from a `PipelineResult`).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from rastervec.Evaluation.Evaluate.text_metrics import (
    char_multiset,
    normalize_text,
    word_tokens,
)
from rastervec.helpers.geometry import (
    bbox_area,
    bbox_intersection_area,
    bbox_iou,
    union_bbox,
)

Bbox = tuple[float, float, float, float]

TEXT_TYPES: tuple[str, ...] = (
    "native_to_vector", "original_vector", "vector_to_raster", "original_raster",
)


# --------------------------------------------------------------------------
# Config + value types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricConfig:
    """Thresholds for the overlap graph. Tune per benchmark run if a
    dataset's granularity needs it."""

    iou_edge_min: float = 0.10
    """Minimum IoU for a (gt, prediction) edge to count as a localisation
    (fallback assignment)."""
    coverage_tau: float = 0.5
    """A prediction is *assigned* to a gt (N:1) when this fraction or more
    of the prediction's area lies inside that gt."""


@dataclass(frozen=True)
class Ratio:
    """An absolute page-level count pair. `value` is the metric; aggregation
    sums numerators and denominators separately (see module docstring)."""

    numerator: float
    denominator: float

    @property
    def value(self) -> float:
        if self.denominator == 0 or math.isnan(self.denominator):
            return math.nan
        return self.numerator / self.denominator

    @property
    def applicable(self) -> bool:
        return not (self.denominator == 0 or math.isnan(self.denominator))


_NA = Ratio(0.0, math.nan)  # "not applicable" -- excluded from aggregates


@dataclass(frozen=True)
class GtRegion:
    page_index: int
    bbox: Bbox
    text: str
    expected_rotation: int = 0
    text_type: str = ""  # one of TEXT_TYPES, set by adapters.gt_regions_by_text_type


@dataclass(frozen=True)
class Prediction:
    text: str
    bbox: Bbox
    rotation: int
    reached_ocr: bool = True
    ocr_blank: bool = False
    source_cluster_id: int = 0


# --------------------------------------------------------------------------
# The overlap graph
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OverlapEdge:
    gt_idx: int
    pred_idx: int  # index into OverlapGraph.preds (non-blank predictions)
    inter_area: float
    iou: float
    gt_coverage: float  # inter / area(gt)
    pred_coverage: float  # inter / area(pred)


@dataclass
class OverlapGraph:
    gt: list[GtRegion]
    preds: list[Prediction]  # non-blank predictions only
    n_pred_total: int
    edges: list[OverlapEdge]
    edges_by_gt: list[list[OverlapEdge]]
    edges_by_pred: list[list[OverlapEdge]]
    assigned_preds_by_gt: list[list[int]]
    overlapping_preds_by_gt: list[list[int]]
    gt_has_overlap: list[bool]
    localized_gt_idxs: list[int]
    missed_gt_idxs: list[int]
    _iou_by_gt_pred: dict[tuple[int, int], float]

    def iou(self, gt_idx: int, pred_idx: int) -> float:
        return self._iou_by_gt_pred.get((gt_idx, pred_idx), 0.0)


def build_overlap_graph(
    gt_regions: list[GtRegion],
    predictions: list[Prediction],
    cfg: MetricConfig = MetricConfig(),
) -> OverlapGraph:
    preds = [p for p in predictions if not p.ocr_blank and p.text.strip()]

    edges: list[OverlapEdge] = []
    iou_by_gt_pred: dict[tuple[int, int], float] = {}
    for gi, g in enumerate(gt_regions):
        g_area = bbox_area(g.bbox)
        for pj, p in enumerate(preds):
            inter = bbox_intersection_area(g.bbox, p.bbox)
            if inter <= 0.0:
                continue
            p_area = bbox_area(p.bbox)
            union = g_area + p_area - inter
            iou = inter / union if union > 0 else 0.0
            edge = OverlapEdge(
                gt_idx=gi,
                pred_idx=pj,
                inter_area=inter,
                iou=iou,
                gt_coverage=(inter / g_area) if g_area > 0 else 0.0,
                pred_coverage=(inter / p_area) if p_area > 0 else 0.0,
            )
            edges.append(edge)
            iou_by_gt_pred[(gi, pj)] = iou

    edges_by_gt: list[list[OverlapEdge]] = [[] for _ in gt_regions]
    edges_by_pred: list[list[OverlapEdge]] = [[] for _ in preds]
    for e in edges:
        edges_by_gt[e.gt_idx].append(e)
        edges_by_pred[e.pred_idx].append(e)
    for lst in edges_by_gt:
        lst.sort(key=lambda e: e.iou, reverse=True)
    for lst in edges_by_pred:
        lst.sort(key=lambda e: e.iou, reverse=True)

    assigned_preds_by_gt: list[list[int]] = []
    overlapping_preds_by_gt: list[list[int]] = []
    for gi in range(len(gt_regions)):
        gt_edges = edges_by_gt[gi]
        overlapping_preds_by_gt.append([e.pred_idx for e in gt_edges])
        assigned = [e.pred_idx for e in gt_edges if e.pred_coverage >= cfg.coverage_tau]
        if not assigned and gt_edges and gt_edges[0].iou >= cfg.iou_edge_min:
            assigned = [gt_edges[0].pred_idx]
        assigned_preds_by_gt.append(assigned)

    gt_has_overlap = [bool(edges_by_gt[gi]) for gi in range(len(gt_regions))]
    localized_gt_idxs = [gi for gi, a in enumerate(assigned_preds_by_gt) if a]
    missed_gt_idxs = [gi for gi, a in enumerate(assigned_preds_by_gt) if not a]

    return OverlapGraph(
        gt=list(gt_regions),
        preds=preds,
        n_pred_total=len(predictions),
        edges=edges,
        edges_by_gt=edges_by_gt,
        edges_by_pred=edges_by_pred,
        assigned_preds_by_gt=assigned_preds_by_gt,
        overlapping_preds_by_gt=overlapping_preds_by_gt,
        gt_has_overlap=gt_has_overlap,
        localized_gt_idxs=localized_gt_idxs,
        missed_gt_idxs=missed_gt_idxs,
        _iou_by_gt_pred=iou_by_gt_pred,
    )


def build_overlap_graphs_by_type(
    gt_by_type: dict[str, list[GtRegion]],
    predictions: list[Prediction],
    cfg: MetricConfig = MetricConfig(),
) -> dict[str, OverlapGraph]:
    """One `OverlapGraph` per text type, all built from the SAME
    `predictions` list -- `build_overlap_graph`'s non-blank filter is the
    same predicate regardless of gt, so `graph.preds` indices line up
    across every type's graph (used by `bbox_accuracy_unclassified` to spot
    predictions unclaimed by ANY text type)."""
    return {
        text_type: build_overlap_graph(gt_by_type.get(text_type, []), predictions, cfg)
        for text_type in TEXT_TYPES
    }


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _multiset_overlap(a: "Counter[str]", b: "Counter[str]") -> int:
    return sum((a & b).values())


def f1_from(recall: Ratio, precision: Ratio) -> float:
    r, p = recall.value, precision.value
    if math.isnan(r) or math.isnan(p) or (r + p) == 0:
        return math.nan
    return 2 * r * p / (r + p)


def _reading_order_key(bbox: Bbox) -> tuple[float, float]:
    """Top-to-bottom, then left-to-right -- default/legacy key, kept for
    `label_overlays.py`. `confusion_metrics._reading_order_sort_key` is the
    rotation-aware key used by category 7's word-order metric."""
    return (bbox[1], bbox[0])


def _region_concat_hyp(graph: OverlapGraph, pred_idxs: list[int]) -> str:
    """The overlapping predictions' text concatenated in reading order."""
    ordered = sorted(pred_idxs, key=lambda pj: _reading_order_key(graph.preds[pj].bbox))
    return " ".join(graph.preds[pj].text for pj in ordered)


# --------------------------------------------------------------------------
# Category 1 -- label description
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TextLabelStats:
    text_type: str
    label_count: int
    char_count: int
    word_count: int
    vector_count: int  # nonzero only for original_vector (sum len(vector_signatures))


def text_label_stats(entries_by_type: dict[str, list]) -> list[TextLabelStats]:
    """`entries_by_type`: `{text_type: [LabelEntry, ...]}` (plain duck-typed
    objects with `.text`/`.vector_signatures`, so this stays pure and does
    not import `label_schema`)."""
    out = []
    for text_type in TEXT_TYPES:
        entries = entries_by_type.get(text_type, [])
        char_count = sum(len(normalize_text(e.text).replace(" ", "")) for e in entries)
        word_count = sum(len(word_tokens(e.text)) for e in entries)
        vector_count = sum(len(getattr(e, "vector_signatures", []) or []) for e in entries)
        out.append(TextLabelStats(
            text_type=text_type, label_count=len(entries),
            char_count=char_count, word_count=word_count, vector_count=vector_count,
        ))
    return out


# --------------------------------------------------------------------------
# Category 2/3 -- character / word overlap
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CharOverlapStats:
    text_type: str
    matched: int
    total_gt: int
    unclassified: int
    missing: int
    precision: Ratio
    recall: Ratio


@dataclass(frozen=True)
class WordOverlapStats:
    text_type: str
    matched: int
    total_gt: int
    unclassified: int
    missing: int
    precision: Ratio
    recall: Ratio


def _overlap_stats(graph: OverlapGraph, text_type: str, tokenizer) -> tuple[int, int, int]:
    """Shared char/word overlap algorithm. `tokenizer(text) -> Counter`.
    Returns `(matched, total_gt, unclassified)`."""
    matched = 0
    total_gt = 0
    for gi, g in enumerate(graph.gt):
        gt_counter = tokenizer(g.text)
        total_gt += sum(gt_counter.values())
        overlap_counter: "Counter[str]" = Counter()
        for pj in graph.overlapping_preds_by_gt[gi]:
            overlap_counter.update(tokenizer(graph.preds[pj].text))
        matched += _multiset_overlap(gt_counter, overlap_counter)
    unclassified = 0
    for pj, p in enumerate(graph.preds):
        if not graph.edges_by_pred[pj]:
            unclassified += sum(tokenizer(p.text).values())
    return matched, total_gt, unclassified


def char_overlap_stats(graph: OverlapGraph, text_type: str) -> CharOverlapStats:
    matched, total_gt, unclassified = _overlap_stats(graph, text_type, char_multiset)
    missing = total_gt - matched
    return CharOverlapStats(
        text_type=text_type, matched=matched, total_gt=total_gt,
        unclassified=unclassified, missing=missing,
        precision=Ratio(float(matched), float(matched + unclassified)),
        recall=Ratio(float(matched), float(total_gt)),
    )


def _word_multiset(text: str) -> "Counter[str]":
    return Counter(word_tokens(text))


def word_overlap_stats(graph: OverlapGraph, text_type: str) -> WordOverlapStats:
    matched, total_gt, unclassified = _overlap_stats(graph, text_type, _word_multiset)
    missing = total_gt - matched
    return WordOverlapStats(
        text_type=text_type, matched=matched, total_gt=total_gt,
        unclassified=unclassified, missing=missing,
        precision=Ratio(float(matched), float(matched + unclassified)),
        recall=Ratio(float(matched), float(total_gt)),
    )


# --------------------------------------------------------------------------
# Category 4 -- text bbox accuracy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BboxAccuracyStats:
    text_type: str
    mean_iou: Ratio = _NA
    n_gt: int = 0
    n_localized: int = 0
    spurious_pred_count: int = 0
    spurious_pred_area_frac: float = math.nan


def per_gt_union_pred_iou_mean(graph: OverlapGraph) -> Ratio:
    if not graph.gt:
        return _NA
    total = 0.0
    for gi, g in enumerate(graph.gt):
        assigned = graph.assigned_preds_by_gt[gi]
        if not assigned:
            continue
        union = union_bbox([graph.preds[pj].bbox for pj in assigned])
        total += bbox_iou(g.bbox, union)
    return Ratio(total, float(len(graph.gt)))


def bbox_accuracy_stats(graph: OverlapGraph, text_type: str) -> BboxAccuracyStats:
    return BboxAccuracyStats(
        text_type=text_type,
        mean_iou=per_gt_union_pred_iou_mean(graph),
        n_gt=len(graph.gt),
        n_localized=len(graph.localized_gt_idxs),
    )


def bbox_accuracy_unclassified(
    graphs_by_type: dict[str, OverlapGraph], *, page_area: float | None = None,
) -> BboxAccuracyStats:
    """Predictions with zero overlap across EVERY text type's graph.
    `graphs_by_type` must come from `build_overlap_graphs_by_type` so
    `graph.preds` indices line up across types."""
    graphs = list(graphs_by_type.values())
    if not graphs:
        return BboxAccuracyStats(text_type="unclassified")
    n_preds = len(graphs[0].preds)
    spurious_idxs = [
        pj for pj in range(n_preds)
        if all(not g.edges_by_pred[pj] for g in graphs)
    ]
    total_area = sum(bbox_area(graphs[0].preds[pj].bbox) for pj in spurious_idxs)
    area_frac = (total_area / page_area) if page_area else math.nan
    return BboxAccuracyStats(
        text_type="unclassified",
        spurious_pred_count=len(spurious_idxs),
        spurious_pred_area_frac=area_frac,
    )


# --------------------------------------------------------------------------
# Category 5 -- rotation accuracy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RotationBucketCounts:
    correct: int = 0
    off_90: int = 0
    off_180: int = 0


@dataclass(frozen=True)
class RotationStats:
    text_type: str
    buckets: RotationBucketCounts = field(default_factory=RotationBucketCounts)
    n_localized: int = 0
    mean_error_deg: float = math.nan
    rmse_deg: float = math.nan


def _rotation_vote(graph: OverlapGraph, gi: int) -> int:
    votes: "Counter[int]" = Counter()
    best_iou: dict[int, float] = {}
    for pj in graph.assigned_preds_by_gt[gi]:
        rot = graph.preds[pj].rotation
        votes[rot] += 1
        best_iou[rot] = max(best_iou.get(rot, -1.0), graph.iou(gi, pj))
    top = max(votes.values())
    winners = [r for r, c in votes.items() if c == top]
    return max(winners, key=lambda r: best_iou[r])


def _circular_rotation_diff(a: int, b: int) -> float:
    raw = abs(a - b) % 360
    return raw if raw <= 180 else 360 - raw


def rotation_stats(graph: OverlapGraph, text_type: str) -> RotationStats:
    localized = graph.localized_gt_idxs
    if not localized:
        return RotationStats(text_type=text_type)
    diffs = []
    for gi in localized:
        predicted = _rotation_vote(graph, gi)
        diffs.append(_circular_rotation_diff(predicted, graph.gt[gi].expected_rotation))
    buckets = RotationBucketCounts(
        correct=sum(1 for d in diffs if round(d / 90) * 90 == 0),
        off_90=sum(1 for d in diffs if round(d / 90) * 90 == 90),
        off_180=sum(1 for d in diffs if round(d / 90) * 90 == 180),
    )
    mean_error = sum(diffs) / len(diffs)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return RotationStats(
        text_type=text_type, buckets=buckets, n_localized=len(localized),
        mean_error_deg=mean_error, rmse_deg=rmse,
    )


# --------------------------------------------------------------------------
# Category 6 -- vector classification funnel (native_to_vector, original_vector only)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationFunnelStats:
    text_type: str
    n_gt_vectors: int
    n_survived: int
    survival_rate: Ratio


FUNNEL_TEXT_TYPES: tuple[str, ...] = ("native_to_vector", "original_vector")


def classification_funnel_stats(
    gt_vector_signatures: "set[str]", survived_signatures: "set[str]", text_type: str,
) -> ClassificationFunnelStats:
    n_gt = len(gt_vector_signatures)
    if n_gt == 0:
        return ClassificationFunnelStats(
            text_type=text_type, n_gt_vectors=0, n_survived=0, survival_rate=_NA,
        )
    survived = len(gt_vector_signatures & survived_signatures)
    return ClassificationFunnelStats(
        text_type=text_type, n_gt_vectors=n_gt, n_survived=survived,
        survival_rate=Ratio(float(survived), float(n_gt)),
    )


# --------------------------------------------------------------------------
# Pred-vs-GT box overlay (for a visual diff PDF -- data only, no rendering)
# --------------------------------------------------------------------------
MATCH_BOX_COLOR = (0.0, 0.7, 0.0)  # green -- gt and pred that overlap
MISSED_GT_BOX_COLOR = (0.85, 0.0, 0.0)  # red -- gt no prediction reached
SPURIOUS_PRED_BOX_COLOR = (0.95, 0.75, 0.0)  # yellow -- pred over no gt


def overlay_boxes(
    graph: OverlapGraph,
) -> list[tuple[Bbox, tuple[float, float, float]]]:
    """`(bbox, rgb)` pairs for a pred-vs-ground-truth visual diff, straight
    off the same overlap graph the metrics score."""
    boxes: list[tuple[Bbox, tuple[float, float, float]]] = []
    for gi, g in enumerate(graph.gt):
        color = MATCH_BOX_COLOR if graph.gt_has_overlap[gi] else MISSED_GT_BOX_COLOR
        boxes.append((g.bbox, color))
    preds_with_overlap = {e.pred_idx for e in graph.edges}
    for pj, p in enumerate(graph.preds):
        color = MATCH_BOX_COLOR if pj in preds_with_overlap else SPURIOUS_PRED_BOX_COLOR
        boxes.append((p.bbox, color))
    return boxes


# One dash style per text type (PyMuPDF dash strings; None = solid).
_TEXT_TYPE_DASHES: dict[str, "str | None"] = {
    "native_to_vector": "[4 3] 0",
    "original_vector": None,
    "vector_to_raster": "[2 2] 0",
    "original_raster": "[6 2 2 2] 0",
}
_PRED_DASHES = "[1 2] 0"


def overlay_boxes_by_type(
    graphs_by_type: dict[str, OverlapGraph],
) -> list[tuple[Bbox, tuple[float, float, float], "str | None"]]:
    """`(bbox, rgb, dashes)` for the combined benchmark box overlay -- one
    dash style per text type's GT boxes, predictions drawn once per type
    (dotted), green if they overlap that type's own GT, yellow otherwise."""
    boxes: list[tuple[Bbox, tuple[float, float, float], "str | None"]] = []
    for text_type, graph in graphs_by_type.items():
        gt_dashes = _TEXT_TYPE_DASHES.get(text_type)
        for gi, g in enumerate(graph.gt):
            color = MATCH_BOX_COLOR if graph.gt_has_overlap[gi] else MISSED_GT_BOX_COLOR
            boxes.append((g.bbox, color, gt_dashes))
        preds_with_overlap = {e.pred_idx for e in graph.edges}
        for pj, p in enumerate(graph.preds):
            color = (
                MATCH_BOX_COLOR if pj in preds_with_overlap
                else SPURIOUS_PRED_BOX_COLOR
            )
            boxes.append((p.bbox, color, _PRED_DASHES))
    return boxes


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
@dataclass
class PerTypeTextResult:
    label_stats: TextLabelStats
    char_overlap: CharOverlapStats
    word_overlap: WordOverlapStats
    bbox_accuracy: BboxAccuracyStats
    rotation: RotationStats
    reading_order: "object"  # confusion_metrics.ReadingOrderStats
    confusion: "dict[str, Counter[str]]"
    funnel: "ClassificationFunnelStats | None" = None


@dataclass
class TextMetricSuiteResult:
    by_type: dict[str, PerTypeTextResult]
    bbox_unclassified: BboxAccuracyStats


def evaluate_text_metrics(
    gt_by_type: dict[str, list[GtRegion]],
    entries_by_type: dict[str, list],
    predictions: list[Prediction],
    *,
    gt_vector_signatures_by_type: "dict[str, set[str]] | None" = None,
    survived_signatures: "set[str] | None" = None,
    page_area: float | None = None,
    cfg: MetricConfig = MetricConfig(),
) -> TextMetricSuiteResult:
    # Local import to avoid a metrics.py <-> confusion_metrics.py import
    # cycle (confusion_metrics imports OverlapGraph/GtRegion from here).
    from rastervec.Evaluation.Evaluate.confusion_metrics import (
        confusion_table,
        reading_order_stats,
    )

    graphs_by_type = build_overlap_graphs_by_type(gt_by_type, predictions, cfg)
    label_stats_by_type = {s.text_type: s for s in text_label_stats(entries_by_type)}
    gt_vector_signatures_by_type = gt_vector_signatures_by_type or {}
    survived_signatures = survived_signatures or set()

    by_type: dict[str, PerTypeTextResult] = {}
    for text_type in TEXT_TYPES:
        graph = graphs_by_type[text_type]
        funnel = None
        if text_type in FUNNEL_TEXT_TYPES:
            funnel = classification_funnel_stats(
                gt_vector_signatures_by_type.get(text_type, set()),
                survived_signatures, text_type,
            )
        by_type[text_type] = PerTypeTextResult(
            label_stats=label_stats_by_type[text_type],
            char_overlap=char_overlap_stats(graph, text_type),
            word_overlap=word_overlap_stats(graph, text_type),
            bbox_accuracy=bbox_accuracy_stats(graph, text_type),
            rotation=rotation_stats(graph, text_type),
            reading_order=reading_order_stats(graph, text_type),
            confusion=confusion_table(graph),
            funnel=funnel,
        )

    return TextMetricSuiteResult(
        by_type=by_type,
        bbox_unclassified=bbox_accuracy_unclassified(graphs_by_type, page_area=page_area),
    )


def _agg_ratio(ratios: list[Ratio]) -> Ratio:
    num = den = 0.0
    any_applicable = False
    for r in ratios:
        if not r.applicable:
            continue
        any_applicable = True
        num += r.numerator
        den += r.denominator
    return Ratio(num, den) if any_applicable else _NA


def aggregate_text_metrics(results: list[TextMetricSuiteResult]) -> "TextMetricSuiteResult | None":
    """Micro-averages every table across pages: counts sum, `Ratio`s
    re-derived from summed numerator/denominator (never averaged directly),
    confusion counters merged, rotation buckets summed and mean/RMSE
    recomputed from the pooled per-page diffs (approximated here by
    weighting each page's mean/RMSE by its own `n_localized`, since only
    the aggregates -- not the raw per-gt diffs -- survive into this
    result)."""
    if not results:
        return None

    by_type: dict[str, PerTypeTextResult] = {}
    for text_type in TEXT_TYPES:
        per_type = [r.by_type[text_type] for r in results]

        label_count = sum(p.label_stats.label_count for p in per_type)
        char_count = sum(p.label_stats.char_count for p in per_type)
        word_count = sum(p.label_stats.word_count for p in per_type)
        vector_count = sum(p.label_stats.vector_count for p in per_type)
        label_stats = TextLabelStats(text_type, label_count, char_count, word_count, vector_count)

        char_overlap = CharOverlapStats(
            text_type=text_type,
            matched=sum(p.char_overlap.matched for p in per_type),
            total_gt=sum(p.char_overlap.total_gt for p in per_type),
            unclassified=sum(p.char_overlap.unclassified for p in per_type),
            missing=sum(p.char_overlap.missing for p in per_type),
            precision=_agg_ratio([p.char_overlap.precision for p in per_type]),
            recall=_agg_ratio([p.char_overlap.recall for p in per_type]),
        )
        word_overlap = WordOverlapStats(
            text_type=text_type,
            matched=sum(p.word_overlap.matched for p in per_type),
            total_gt=sum(p.word_overlap.total_gt for p in per_type),
            unclassified=sum(p.word_overlap.unclassified for p in per_type),
            missing=sum(p.word_overlap.missing for p in per_type),
            precision=_agg_ratio([p.word_overlap.precision for p in per_type]),
            recall=_agg_ratio([p.word_overlap.recall for p in per_type]),
        )
        bbox_accuracy = BboxAccuracyStats(
            text_type=text_type,
            mean_iou=_agg_ratio([p.bbox_accuracy.mean_iou for p in per_type]),
            n_gt=sum(p.bbox_accuracy.n_gt for p in per_type),
            n_localized=sum(p.bbox_accuracy.n_localized for p in per_type),
        )

        n_loc_total = sum(p.rotation.n_localized for p in per_type)
        buckets = RotationBucketCounts(
            correct=sum(p.rotation.buckets.correct for p in per_type),
            off_90=sum(p.rotation.buckets.off_90 for p in per_type),
            off_180=sum(p.rotation.buckets.off_180 for p in per_type),
        )
        if n_loc_total:
            mean_error = sum(
                p.rotation.mean_error_deg * p.rotation.n_localized
                for p in per_type if p.rotation.n_localized
            ) / n_loc_total
            mean_sq = sum(
                (p.rotation.rmse_deg ** 2) * p.rotation.n_localized
                for p in per_type if p.rotation.n_localized
            ) / n_loc_total
            rmse = math.sqrt(mean_sq)
        else:
            mean_error = math.nan
            rmse = math.nan
        rotation = RotationStats(
            text_type=text_type, buckets=buckets, n_localized=n_loc_total,
            mean_error_deg=mean_error, rmse_deg=rmse,
        )

        from rastervec.Evaluation.Evaluate.confusion_metrics import (
            aggregate_reading_order_stats,
        )
        reading_order = aggregate_reading_order_stats([p.reading_order for p in per_type])

        confusion: "Counter[str]" = Counter()
        merged: dict[str, "Counter[str]"] = {}
        for p in per_type:
            for ch, counter in p.confusion.items():
                merged.setdefault(ch, Counter()).update(counter)

        funnel = None
        if text_type in FUNNEL_TEXT_TYPES:
            n_gt_vectors = sum(p.funnel.n_gt_vectors for p in per_type if p.funnel)
            n_survived = sum(p.funnel.n_survived for p in per_type if p.funnel)
            funnel = ClassificationFunnelStats(
                text_type=text_type, n_gt_vectors=n_gt_vectors, n_survived=n_survived,
                survival_rate=(
                    Ratio(float(n_survived), float(n_gt_vectors)) if n_gt_vectors else _NA
                ),
            )

        by_type[text_type] = PerTypeTextResult(
            label_stats=label_stats, char_overlap=char_overlap, word_overlap=word_overlap,
            bbox_accuracy=bbox_accuracy, rotation=rotation, reading_order=reading_order,
            confusion=merged, funnel=funnel,
        )

    bbox_unclassified = BboxAccuracyStats(
        text_type="unclassified",
        spurious_pred_count=sum(r.bbox_unclassified.spurious_pred_count for r in results),
    )
    return TextMetricSuiteResult(by_type=by_type, bbox_unclassified=bbox_unclassified)

"""Independent evaluation metrics for the Vector_Classification + OCR
pipeline, built over one shared many-to-many overlap graph between
ground-truth text regions and predicted OCR readings.

Why this exists (vs `evaluate.py`): the legacy `evaluate_pipeline` reduces
*every* metric out of a single greedy 1:1 highest-IoU match, so when one
ground-truth line is covered by several predicted clusters (the common
case) all of text / bbox / rotation accuracy corrupt together. Here each
metric is an independent reduction over `OverlapGraph`, which keeps every
(gt, prediction) overlap edge and an explicit N:1 assignment.

Two normalisation rules, both documented in full in `EVAL_METRICS.md`:

1. **Text** -- every gt/prediction string goes through
   `text_metrics.normalize_text` (upper-case, trim, collapse whitespace)
   before any comparison. Character metrics compare `char_multiset`s
   (spaces dropped); word metrics compare `word_tokens` multisets; the two
   `region_concat_char_accuracy_*` metrics are position-aware and use
   `text_metrics.levenshtein` (character edit distance) on the normalised
   strings.

2. **Aggregation** -- a page result stores *absolute counts*: every metric
   is a `Ratio(numerator, denominator)`. `aggregate_suite` micro-averages:
   `Ratio(sum numerators, sum denominators)`, NOT the mean of per-page
   ratios. A page with a `nan` denominator (empty gt, no candidates, zero
   misses, `clustering=None`) is "not applicable" and simply excluded from
   that metric's aggregate.

This module is pure and does not import `rastervec.pipeline` -- callers
pass plain lists (`adapters.py` builds them from a `PipelineContext`).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rastervec.Evaluation.Evaluate.text_metrics import (
    char_multiset,
    levenshtein,
    normalize_text,
    word_tokens,
)
from rastervec.helpers.geometry import (
    bbox_area,
    bbox_coverage,
    bbox_intersection_area,
    bbox_iou,
    union_bbox,
)

if TYPE_CHECKING:
    from rastervec.pipeline import ClusteringStageResult, GroupKey

Bbox = tuple[float, float, float, float]


# --------------------------------------------------------------------------
# Config + value types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricConfig:
    """Thresholds for the overlap graph. Tune per benchmark run if a
    dataset's granularity needs it."""

    iou_edge_min: float = 0.10
    """Minimum IoU for a (gt, prediction) edge to count as a localisation
    (fallback assignment, `attribute_miss` group match, metric 43/44
    IoU term)."""
    coverage_tau: float = 0.5
    """A prediction is *assigned* to a gt (N:1) when this fraction or more
    of the prediction's area lies inside that gt. Also the "reached OCR" /
    "candidate is text" coverage threshold."""


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


@dataclass(frozen=True)
class Prediction:
    text: str
    bbox: Bbox
    rotation: int
    reached_ocr: bool = True
    ocr_blank: bool = False
    source_cluster_id: int = 0


@dataclass
class MetricCounts:
    n_gt: int = 0
    n_pred: int = 0
    n_pred_nonblank: int = 0
    n_text_candidates: int = 0
    n_gt_localized: int = 0
    n_gt_missed: int = 0
    n_gt_with_overlap: int = 0

    def __add__(self, other: "MetricCounts") -> "MetricCounts":
        return MetricCounts(
            *(getattr(self, f) + getattr(other, f) for f in _COUNT_FIELDS)
        )


_COUNT_FIELDS = tuple(MetricCounts.__dataclass_fields__)


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


# --------------------------------------------------------------------------
# Metric functions -- each returns a Ratio (absolute page-level counts)
# --------------------------------------------------------------------------
def _multiset_overlap(a: "Counter[str]", b: "Counter[str]") -> int:
    return sum((a & b).values())


def _page_char_counters(
    gt: list[GtRegion], preds: list[Prediction]
) -> tuple["Counter[str]", "Counter[str]"]:
    cg = char_multiset(" ".join(g.text for g in gt))
    cp = char_multiset(" ".join(p.text for p in preds))
    return cg, cp


def _page_word_counters(
    gt: list[GtRegion], preds: list[Prediction]
) -> tuple["Counter[str]", "Counter[str]"]:
    wg: "Counter[str]" = Counter()
    for g in gt:
        wg.update(word_tokens(g.text))
    wp: "Counter[str]" = Counter()
    for p in preds:
        wp.update(word_tokens(p.text))
    return wg, wp


def page_char_multiset_recall(cg: "Counter[str]", cp: "Counter[str]") -> Ratio:
    return Ratio(float(_multiset_overlap(cg, cp)), float(sum(cg.values())))


def page_char_multiset_precision(cg: "Counter[str]", cp: "Counter[str]") -> Ratio:
    return Ratio(float(_multiset_overlap(cg, cp)), float(sum(cp.values())))


def page_word_multiset_recall(wg: "Counter[str]", wp: "Counter[str]") -> Ratio:
    return Ratio(float(_multiset_overlap(wg, wp)), float(sum(wg.values())))


def page_word_multiset_precision(wg: "Counter[str]", wp: "Counter[str]") -> Ratio:
    return Ratio(float(_multiset_overlap(wg, wp)), float(sum(wp.values())))


def f1_from(recall: Ratio, precision: Ratio) -> float:
    r, p = recall.value, precision.value
    if math.isnan(r) or math.isnan(p) or (r + p) == 0:
        return math.nan
    return 2 * r * p / (r + p)


def pred_text_fully_contained_in_overlapping_gt_rate(graph: OverlapGraph) -> Ratio:
    """For each non-blank prediction: is there an overlapping gt whose token
    multiset contains every token of the prediction? Catches hallucinated /
    bled-in predicted text the bbox overlap alone would pass."""
    if not graph.preds:
        return _NA
    gt_tokens = [Counter(word_tokens(g.text)) for g in graph.gt]
    contained = 0
    for pj, p in enumerate(graph.preds):
        p_tokens = Counter(word_tokens(p.text))
        if not p_tokens:
            continue
        for e in graph.edges_by_pred[pj]:
            gc = gt_tokens[e.gt_idx]
            if all(gc[tok] >= cnt for tok, cnt in p_tokens.items()):
                contained += 1
                break
    return Ratio(float(contained), float(len(graph.preds)))


def gt_text_word_coverage_by_overlapping_preds(graph: OverlapGraph) -> Ratio:
    """Sum over gt of (gt tokens also present in the union of overlapping
    predictions' tokens) / sum over gt of (gt token count). Position-aware
    recall: unlike page_word_multiset_recall it gives no credit for the
    right word appearing somewhere unrelated on the page."""
    covered = 0
    total = 0
    for gi, g in enumerate(graph.gt):
        g_tokens = Counter(word_tokens(g.text))
        total += sum(g_tokens.values())
        bag: "Counter[str]" = Counter()
        for pj in graph.overlapping_preds_by_gt[gi]:
            bag.update(word_tokens(graph.preds[pj].text))
        covered += _multiset_overlap(g_tokens, bag)
    if total == 0:
        return _NA
    return Ratio(float(covered), float(total))


def _reading_order_key(bbox: Bbox) -> tuple[float, float]:
    """Top-to-bottom, then left-to-right."""
    return (bbox[1], bbox[0])


def _region_concat_hyp(graph: OverlapGraph, pred_idxs: list[int]) -> str:
    """The overlapping predictions' text concatenated in reading order."""
    ordered = sorted(pred_idxs, key=lambda pj: _reading_order_key(graph.preds[pj].bbox))
    return " ".join(graph.preds[pj].text for pj in ordered)


def _region_char_correct(gt_text: str, hyp_text: str) -> tuple[int, int]:
    """`(chars matched, gt char length)` for one gt region -- character-level
    Levenshtein after `normalize_text`. Matched is `max(0, gt_len - edit)`,
    i.e. `1 - CER` clamped at 0; mirrors archive `native_vs_ocr._cer`."""
    g = normalize_text(gt_text)
    h = normalize_text(hyp_text)
    return max(0, len(g) - levenshtein(g, h)), len(g)


def region_concat_char_accuracy_all_gt(graph: OverlapGraph) -> Ratio:
    """Char accuracy over EVERY gt region: hyp = its overlapping predictions'
    text (empty for a gt nothing overlaps -> counts as a full miss). A global
    page CER built from per-region localised comparisons, like archive's
    `cer_percentage`. `n/a` when the page has no gt characters."""
    num = den = 0
    for gi, g in enumerate(graph.gt):
        hyp = _region_concat_hyp(graph, graph.overlapping_preds_by_gt[gi])
        correct, total = _region_char_correct(g.text, hyp)
        num += correct
        den += total
    if den == 0:
        return _NA
    return Ratio(float(num), float(den))


def region_concat_char_accuracy_overlapping(graph: OverlapGraph) -> Ratio:
    """Same, restricted to gt regions with >=1 overlapping non-blank
    prediction (`gt_has_overlap`). Isolates 'when we found the text, how well
    did we read it' from wholesale misses. `n/a` when no gt was overlapped."""
    num = den = 0
    for gi, g in enumerate(graph.gt):
        if not graph.gt_has_overlap[gi]:
            continue
        hyp = _region_concat_hyp(graph, graph.overlapping_preds_by_gt[gi])
        correct, total = _region_char_correct(g.text, hyp)
        num += correct
        den += total
    if den == 0:
        return _NA
    return Ratio(float(num), float(den))


def per_gt_best_single_pred_iou_mean(graph: OverlapGraph) -> Ratio:
    if not graph.gt:
        return _NA
    total = sum(
        (graph.edges_by_gt[gi][0].iou if graph.edges_by_gt[gi] else 0.0)
        for gi in range(len(graph.gt))
    )
    return Ratio(total, float(len(graph.gt)))


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


def undetected_gt_area_ratio(graph: OverlapGraph) -> Ratio:
    total_area = sum(bbox_area(g.bbox) for g in graph.gt)
    if total_area == 0:
        return _NA
    undetected = sum(
        bbox_area(g.bbox)
        for gi, g in enumerate(graph.gt)
        if not graph.gt_has_overlap[gi]
    )
    return Ratio(undetected, total_area)


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


def rotation_accuracy_localized_gt(graph: OverlapGraph) -> Ratio:
    localized = graph.localized_gt_idxs
    if not localized:
        return _NA
    correct = sum(
        1 for gi in localized
        if _rotation_vote(graph, gi) == graph.gt[gi].expected_rotation
    )
    return Ratio(float(correct), float(len(localized)))


def classification_recall_gt_reached_ocr(
    graph: OverlapGraph, text_candidate_boxes: list[Bbox], cfg: MetricConfig
) -> Ratio:
    if not graph.gt:
        return _NA
    reached = 0
    for g in graph.gt:
        if any(
            bbox_coverage(g.bbox, c) >= cfg.coverage_tau
            or bbox_iou(g.bbox, c) >= cfg.iou_edge_min
            for c in text_candidate_boxes
        ):
            reached += 1
    return Ratio(float(reached), float(len(graph.gt)))


def classification_precision_candidate_is_text(
    graph: OverlapGraph, text_candidate_boxes: list[Bbox], cfg: MetricConfig
) -> Ratio:
    if not text_candidate_boxes:
        return _NA
    is_text = 0
    for c in text_candidate_boxes:
        if any(
            bbox_coverage(c, g.bbox) >= cfg.coverage_tau
            or bbox_coverage(g.bbox, c) >= cfg.coverage_tau
            or bbox_iou(c, g.bbox) >= cfg.iou_edge_min
            for g in graph.gt
        ):
            is_text += 1
    return Ratio(float(is_text), float(len(text_candidate_boxes)))


# --------------------------------------------------------------------------
# Miss attribution (port of evaluate._attribute_miss, region-bbox based)
# --------------------------------------------------------------------------
def _region_matches_groups(
    bbox: Bbox, groups: list[list], cfg: MetricConfig
) -> bool:
    for group in groups:
        if not group:
            continue
        if bbox_iou(bbox, union_bbox([p.bbox for p in group])) >= cfg.iou_edge_min:
            return True
    return False


def attribute_miss(
    gt_bbox: Bbox,
    clustering: "dict[GroupKey, ClusteringStageResult] | None",
    fast_dropped: list[list] | None,
    ocr_failed: list[list] | None,
    cfg: MetricConfig,
) -> str:
    """Which pipeline stage lost this gt region -- checked in pipeline order
    so the *earliest* stage that dropped a group over this bbox is reported.
    Mirrors `evaluate._attribute_miss`."""
    for stage_result in (clustering or {}).values():
        for step in stage_result.steps:
            for category in step.categories.values():
                if category.role != "dropped":
                    continue
                if _region_matches_groups(gt_bbox, category.groups, cfg):
                    return f"classification:{step.label}"
    if _region_matches_groups(gt_bbox, fast_dropped or [], cfg):
        return "fast_text_detect"
    if _region_matches_groups(gt_bbox, ocr_failed or [], cfg):
        return "ocr_blank"
    return "not_found"


# --------------------------------------------------------------------------
# Pred-vs-GT box overlay (for a visual diff PDF -- data only, no rendering)
# --------------------------------------------------------------------------
# (r, g, b) 0..1, matching renderer.render_boxes_pdf's colour param.
MATCH_BOX_COLOR = (0.0, 0.7, 0.0)  # green -- gt and pred that overlap
MISSED_GT_BOX_COLOR = (0.85, 0.0, 0.0)  # red -- gt no prediction reached
SPURIOUS_PRED_BOX_COLOR = (0.95, 0.75, 0.0)  # yellow -- pred over no gt


def overlay_boxes(
    graph: OverlapGraph,
) -> list[tuple[Bbox, tuple[float, float, float]]]:
    """`(bbox, rgb)` pairs for a pred-vs-ground-truth visual diff, straight
    off the same overlap graph the metrics score:

    - **green** -- every gt region that has an overlapping non-blank
      prediction, and every non-blank prediction that overlaps some gt.
    - **red** -- gt regions no prediction touched (wholesale misses).
    - **yellow** -- non-blank predictions sitting over no gt (over-detection).

    Blank OCR predictions are not drawn (they are excluded from `graph.preds`
    and folded into the drawing-vector layer)."""
    boxes: list[tuple[Bbox, tuple[float, float, float]]] = []
    for gi, g in enumerate(graph.gt):
        color = MATCH_BOX_COLOR if graph.gt_has_overlap[gi] else MISSED_GT_BOX_COLOR
        boxes.append((g.bbox, color))
    preds_with_overlap = {e.pred_idx for e in graph.edges}
    for pj, p in enumerate(graph.preds):
        color = MATCH_BOX_COLOR if pj in preds_with_overlap else SPURIOUS_PRED_BOX_COLOR
        boxes.append((p.bbox, color))
    return boxes


_MISS_REASON_FIELDS = {
    "gt_miss_attributed_to_classification_frac": lambda r: r.startswith("classification:"),
    "gt_miss_attributed_to_fast_frac": lambda r: r == "fast_text_detect",
    "gt_miss_attributed_to_ocr_blank_frac": lambda r: r == "ocr_blank",
    "gt_miss_attributed_to_not_found_frac": lambda r: r == "not_found",
}


# --------------------------------------------------------------------------
# Result + entrypoint
# --------------------------------------------------------------------------
_RATIO_FIELDS = (
    "page_char_multiset_recall",
    "page_char_multiset_precision",
    "region_concat_char_accuracy_all_gt",
    "region_concat_char_accuracy_overlapping",
    "page_word_multiset_recall",
    "page_word_multiset_precision",
    "pred_text_fully_contained_in_overlapping_gt_rate",
    "gt_text_word_coverage_by_overlapping_preds",
    "per_gt_best_single_pred_iou_mean",
    "per_gt_union_pred_iou_mean",
    "undetected_gt_area_ratio",
    "rotation_accuracy_localized_gt",
    "classification_recall_gt_reached_ocr",
    "classification_precision_candidate_is_text",
    "gt_miss_attributed_to_classification_frac",
    "gt_miss_attributed_to_fast_frac",
    "gt_miss_attributed_to_ocr_blank_frac",
    "gt_miss_attributed_to_not_found_frac",
)

_DERIVED_F1 = {
    "page_char_multiset_f1": ("page_char_multiset_recall", "page_char_multiset_precision"),
    "page_word_multiset_f1": ("page_word_multiset_recall", "page_word_multiset_precision"),
}
DERIVED_F1_FIELDS = frozenset(_DERIVED_F1)

# Ordered (dimension, (field, ...)) -- the one source of display order for
# benchmark.format_report and the notebook charts.
METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "character",
        (
            "page_char_multiset_recall",
            "page_char_multiset_precision",
            "page_char_multiset_f1",
            "region_concat_char_accuracy_all_gt",
            "region_concat_char_accuracy_overlapping",
        ),
    ),
    (
        "word",
        (
            "page_word_multiset_recall",
            "page_word_multiset_precision",
            "page_word_multiset_f1",
            "pred_text_fully_contained_in_overlapping_gt_rate",
            "gt_text_word_coverage_by_overlapping_preds",
        ),
    ),
    (
        "bbox",
        (
            "per_gt_best_single_pred_iou_mean",
            "per_gt_union_pred_iou_mean",
            "undetected_gt_area_ratio",
        ),
    ),
    ("rotation", ("rotation_accuracy_localized_gt",)),
    (
        "vector_classification",
        (
            "classification_recall_gt_reached_ocr",
            "classification_precision_candidate_is_text",
            "gt_miss_attributed_to_classification_frac",
            "gt_miss_attributed_to_fast_frac",
            "gt_miss_attributed_to_ocr_blank_frac",
            "gt_miss_attributed_to_not_found_frac",
        ),
    ),
)

ALL_METRIC_NAMES: tuple[str, ...] = tuple(
    name for _dimension, names in METRIC_GROUPS for name in names
)

# "lower is better" metrics -- flagged for chart annotation.
LOWER_IS_BETTER = frozenset(
    {
        "undetected_gt_area_ratio",
        "gt_miss_attributed_to_classification_frac",
        "gt_miss_attributed_to_fast_frac",
        "gt_miss_attributed_to_ocr_blank_frac",
        "gt_miss_attributed_to_not_found_frac",
    }
)


@dataclass
class MetricSuiteResult:
    ratios: dict[str, Ratio] = field(default_factory=dict)
    per_stage_miss_counts: dict[str, int] = field(default_factory=dict)
    counts: MetricCounts = field(default_factory=MetricCounts)

    def derived_f1(self, name: str) -> float:
        recall_name, precision_name = _DERIVED_F1[name]
        return f1_from(self.ratios[recall_name], self.ratios[precision_name])

    def get(self, name: str) -> float:
        """Metric value by field name -- Ratio.value for the 16 base
        metrics, computed harmonic mean for the two `*_f1` names."""
        if name in _DERIVED_F1:
            return self.derived_f1(name)
        return self.ratios[name].value


def evaluate_metrics(
    gt_regions: list[GtRegion],
    predictions: list[Prediction],
    text_candidate_boxes: list[Bbox],
    *,
    clustering: "dict[GroupKey, ClusteringStageResult] | None" = None,
    fast_dropped: list[list] | None = None,
    ocr_failed: list[list] | None = None,
    cfg: MetricConfig = MetricConfig(),
) -> MetricSuiteResult:
    graph = build_overlap_graph(gt_regions, predictions, cfg)

    cg, cp = _page_char_counters(gt_regions, graph.preds)
    wg, wp = _page_word_counters(gt_regions, graph.preds)

    ratios: dict[str, Ratio] = {
        "page_char_multiset_recall": page_char_multiset_recall(cg, cp),
        "page_char_multiset_precision": page_char_multiset_precision(cg, cp),
        "region_concat_char_accuracy_all_gt": region_concat_char_accuracy_all_gt(graph),
        "region_concat_char_accuracy_overlapping":
            region_concat_char_accuracy_overlapping(graph),
        "page_word_multiset_recall": page_word_multiset_recall(wg, wp),
        "page_word_multiset_precision": page_word_multiset_precision(wg, wp),
        "pred_text_fully_contained_in_overlapping_gt_rate":
            pred_text_fully_contained_in_overlapping_gt_rate(graph),
        "gt_text_word_coverage_by_overlapping_preds":
            gt_text_word_coverage_by_overlapping_preds(graph),
        "per_gt_best_single_pred_iou_mean": per_gt_best_single_pred_iou_mean(graph),
        "per_gt_union_pred_iou_mean": per_gt_union_pred_iou_mean(graph),
        "undetected_gt_area_ratio": undetected_gt_area_ratio(graph),
        "rotation_accuracy_localized_gt": rotation_accuracy_localized_gt(graph),
        "classification_recall_gt_reached_ocr":
            classification_recall_gt_reached_ocr(graph, text_candidate_boxes, cfg),
        "classification_precision_candidate_is_text":
            classification_precision_candidate_is_text(graph, text_candidate_boxes, cfg),
    }

    # Miss attribution over the missed gt regions.
    per_stage_miss_counts: dict[str, int] = {}
    if clustering is None or not graph.missed_gt_idxs:
        for name in _MISS_REASON_FIELDS:
            ratios[name] = _NA
    else:
        reasons = [
            attribute_miss(
                graph.gt[gi].bbox, clustering, fast_dropped, ocr_failed, cfg
            )
            for gi in graph.missed_gt_idxs
        ]
        per_stage_miss_counts = dict(Counter(reasons))
        n_missed = float(len(reasons))
        for name, pred in _MISS_REASON_FIELDS.items():
            hits = sum(1 for r in reasons if pred(r))
            ratios[name] = Ratio(float(hits), n_missed)

    counts = MetricCounts(
        n_gt=len(gt_regions),
        n_pred=len(predictions),
        n_pred_nonblank=len(graph.preds),
        n_text_candidates=len(text_candidate_boxes),
        n_gt_localized=len(graph.localized_gt_idxs),
        n_gt_missed=len(graph.missed_gt_idxs),
        n_gt_with_overlap=sum(graph.gt_has_overlap),
    )

    return MetricSuiteResult(
        ratios=ratios,
        per_stage_miss_counts=per_stage_miss_counts,
        counts=counts,
    )


def aggregate_suite(results: list[MetricSuiteResult]) -> MetricSuiteResult:
    """Micro-average: for each metric, Ratio(sum numerators, sum
    denominators) over the pages where that metric is applicable (real,
    non-nan denominator). `*_f1` names are recomputed from the aggregated
    recall/precision, never averaged from per-page f1."""
    agg_ratios: dict[str, Ratio] = {}
    for name in _RATIO_FIELDS:
        num = 0.0
        den = 0.0
        any_applicable = False
        for r in results:
            ratio = r.ratios.get(name, _NA)
            if not ratio.applicable:
                continue
            any_applicable = True
            num += ratio.numerator
            den += ratio.denominator
        agg_ratios[name] = Ratio(num, den) if any_applicable else _NA

    merged_miss: "Counter[str]" = Counter()
    for r in results:
        merged_miss.update(r.per_stage_miss_counts)

    total_counts = MetricCounts()
    for r in results:
        total_counts = total_counts + r.counts

    return MetricSuiteResult(
        ratios=agg_ratios,
        per_stage_miss_counts=dict(merged_miss),
        counts=total_counts,
    )

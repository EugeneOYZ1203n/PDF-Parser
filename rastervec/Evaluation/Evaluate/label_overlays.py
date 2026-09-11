"""Ground-truth-vs-prediction visual diff data for the benchmark report.

Pure reductions over the same `metrics.OverlapGraph` the scores come from --
no rendering, no pipeline import. `scripts/generate_pipeline_report.py`
feeds the output straight to `renderer.render_boxes_pdf` /
`renderer.render_reconstructed_pdf`.

Two overlays, both ground-truth-only (predictions are not drawn):

- `gt_bbox_overlay` -- one box per GT region, green if any non-blank
  prediction overlaps it, red otherwise.
- `gt_word_overlay` -- each GT region's text split into per-word sub-boxes,
  each word coloured by how well it was read: green (exact), yellow (char
  edit distance 1-2), red (>=3, or the region had no overlapping
  prediction at all).
"""
from __future__ import annotations

from rastervec.Evaluation.Evaluate.metrics import (
    MATCH_BOX_COLOR,
    MISSED_GT_BOX_COLOR,
    Bbox,
    OverlapGraph,
    _region_concat_hyp,
)
from rastervec.Evaluation.Evaluate.text_metrics import (
    levenshtein,
    normalize_text,
    word_tokens,
)

Rgb = tuple[float, float, float]

WORD_OK_COLOR: Rgb = (0.0, 0.7, 0.0)  # green -- exact
WORD_PARTIAL_COLOR: Rgb = (0.95, 0.75, 0.0)  # yellow -- edit distance 1-2
WORD_WRONG_COLOR: Rgb = (0.85, 0.0, 0.0)  # red -- >=3 or unread

_PARTIAL_MAX_EDITS = 3  # edit distance < this (i.e. 1-2) is "partial"


def gt_bbox_overlay(graph: OverlapGraph) -> list[tuple[Bbox, Rgb]]:
    """`(bbox, rgb)` per GT region: green when it has an overlapping non-blank
    prediction, red when nothing reached it."""
    return [
        (g.bbox, MATCH_BOX_COLOR if graph.gt_has_overlap[gi] else MISSED_GT_BOX_COLOR)
        for gi, g in enumerate(graph.gt)
    ]


def _word_color(gt_word: str, hyp_tokens: list[str]) -> Rgb:
    g = normalize_text(gt_word)
    if not g:
        return WORD_OK_COLOR
    if not hyp_tokens:
        return WORD_WRONG_COLOR
    best = min(levenshtein(g, h) for h in hyp_tokens)
    if best == 0:
        return WORD_OK_COLOR
    if best < _PARTIAL_MAX_EDITS:
        return WORD_PARTIAL_COLOR
    return WORD_WRONG_COLOR


def _is_vertical(rotation: int) -> bool:
    return rotation % 180 == 90


def _split_bbox(bbox: Bbox, weights: list[float], vertical: bool) -> list[Bbox]:
    """Slice `bbox` into `len(weights)` sub-boxes along the reading axis,
    each sized to its weight (min weight 1.0 so a 1-char word still shows)."""
    x0, y0, x1, y1 = bbox
    w = [max(v, 1.0) for v in weights]
    total = sum(w)
    out: list[Bbox] = []
    if vertical:
        cursor = y0
        span = y1 - y0
        for wi in w:
            nxt = cursor + span * wi / total
            out.append((x0, cursor, x1, nxt))
            cursor = nxt
    else:
        cursor = x0
        span = x1 - x0
        for wi in w:
            nxt = cursor + span * wi / total
            out.append((cursor, y0, nxt, y1))
            cursor = nxt
    return out


def gt_word_overlay(
    graph: OverlapGraph,
) -> list[tuple[str, Bbox, float, Rgb]]:
    """`(word, sub_bbox, rotation_deg, rgb)` for every word of every GT
    region. The region bbox is split into per-word slices (proportional to
    each word's normalised char length) along the reading axis inferred from
    `expected_rotation`; each word is coloured by `_word_color` against the
    tokens of that region's overlapping predictions, concatenated in reading
    order (`metrics._region_concat_hyp`)."""
    out: list[tuple[str, Bbox, float, Rgb]] = []
    for gi, g in enumerate(graph.gt):
        words = normalize_text(g.text).split(" ")
        words = [w for w in words if w]
        if not words:
            continue
        hyp_tokens = word_tokens(
            _region_concat_hyp(graph, graph.overlapping_preds_by_gt[gi])
        )
        vertical = _is_vertical(g.expected_rotation)
        slices = _split_bbox(g.bbox, [float(len(w)) for w in words], vertical)
        for word, sub in zip(words, slices):
            out.append((word, sub, float(g.expected_rotation), _word_color(word, hyp_tokens)))
    return out

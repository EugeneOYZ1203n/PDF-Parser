"""Reading-order accuracy (category 7) and OCR confusion-character table
(category 8) -- both need per-word alignment machinery that doesn't belong
in `metrics.py`'s declarative per-table functions.

Pure; imports `OverlapGraph`/`GtRegion` from `metrics.py` (a one-directional
import -- `metrics.py` imports back from here only inside function bodies,
to avoid a cycle).
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field

from rastervec.Evaluation.Evaluate.metrics import Bbox, OverlapGraph, Ratio
from rastervec.Evaluation.Evaluate.text_metrics import char_multiset, levenshtein, word_tokens

_NA_RATIO = Ratio(0.0, math.nan)


# --------------------------------------------------------------------------
# Category 7 -- reading order / word-order accuracy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ReadingOrderStats:
    text_type: str
    n_with_overlap: int = 0
    n_in_order: int = 0
    in_order_rate: Ratio = _NA_RATIO
    edit_distance_min: float = math.nan
    edit_distance_max: float = math.nan
    edit_distance_mean: float = math.nan
    edit_distance_median: float = math.nan
    _edit_distances: "tuple[int, ...]" = field(default_factory=tuple, repr=False, compare=False)


def _reading_order_sort_key(bbox: Bbox, rotation: int):
    """Rotation-aware reading-order key. Rotation follows `Text.angle()`'s
    y-down `dir` convention:

    - 0 deg (left-to-right): primary x0 ascending, secondary y0 ascending.
    - 180 deg (right-to-left): primary x0 descending, secondary y0 ascending.
    - 90 deg (line direction downward, top-to-bottom): primary y0 ascending,
      secondary x0 ascending.
    - 270 deg / -90 deg (line direction upward, bottom-to-top): primary y0
      descending, secondary x0 ascending.
    """
    x0, y0, _x1, _y1 = bbox
    rot = rotation % 360
    if rot == 90:
        return (y0, x0)
    if rot == 270:
        return (-y0, x0)
    if rot == 180:
        return (-x0, y0)
    return (x0, y0)


def reading_order_stats(graph: OverlapGraph, text_type: str) -> ReadingOrderStats:
    from rastervec.Evaluation.Evaluate.text_metrics import word_tokens as _word_tokens

    distances: list[int] = []
    n_in_order = 0
    for gi, g in enumerate(graph.gt):
        overlapping = graph.overlapping_preds_by_gt[gi]
        if not overlapping:
            continue
        ordered = sorted(
            overlapping,
            key=lambda pj: _reading_order_sort_key(graph.preds[pj].bbox, g.expected_rotation),
        )
        gt_words = _word_tokens(g.text)
        pred_words = [w for pj in ordered for w in _word_tokens(graph.preds[pj].text)]
        dist = levenshtein(gt_words, pred_words)
        distances.append(dist)
        if dist == 0:
            n_in_order += 1

    if not distances:
        return ReadingOrderStats(text_type=text_type)
    return ReadingOrderStats(
        text_type=text_type,
        n_with_overlap=len(distances),
        n_in_order=n_in_order,
        in_order_rate=Ratio(float(n_in_order), float(len(distances))),
        edit_distance_min=float(min(distances)),
        edit_distance_max=float(max(distances)),
        edit_distance_mean=statistics.mean(distances),
        edit_distance_median=statistics.median(distances),
        _edit_distances=tuple(distances),
    )


def aggregate_reading_order_stats(results: "list[ReadingOrderStats]") -> ReadingOrderStats:
    if not results:
        return ReadingOrderStats(text_type="")
    text_type = results[0].text_type
    all_distances: list[int] = [d for r in results for d in r._edit_distances]
    n_in_order = sum(r.n_in_order for r in results)
    if not all_distances:
        return ReadingOrderStats(text_type=text_type)
    return ReadingOrderStats(
        text_type=text_type,
        n_with_overlap=len(all_distances),
        n_in_order=n_in_order,
        in_order_rate=Ratio(float(n_in_order), float(len(all_distances))),
        edit_distance_min=float(min(all_distances)),
        edit_distance_max=float(max(all_distances)),
        edit_distance_mean=statistics.mean(all_distances),
        edit_distance_median=statistics.median(all_distances),
        _edit_distances=tuple(all_distances),
    )


# --------------------------------------------------------------------------
# Category 8 -- OCR confusion-character table
# --------------------------------------------------------------------------
def closest_pred_word(gt_word: str, pred_words: list[str]) -> "str | None":
    """Single closest predicted word by char-level Levenshtein among a gt
    region's overlapping predictions' words; `None` if `pred_words` is
    empty. Ties broken by first occurrence (stable `min`)."""
    if not pred_words:
        return None
    return min(pred_words, key=lambda w: levenshtein(gt_word, w))


def align_chars(gt_word: str, pred_word: str) -> list[tuple[str, str]]:
    """Levenshtein DP + backtrace: one `(gt_char, replacement)` pair per
    `gt_word` character. `replacement` is the aligned predicted char on a
    match/substitution, `""` on a deletion (a gt char with no predicted
    counterpart). An insertion (a predicted char with no gt counterpart)
    consumes no gt char and is dropped from this table -- a known, accepted
    limitation (there is no gt character to attribute it to)."""
    n, m = len(gt_word), len(pred_word)
    # dp[i][j] = edit distance between gt_word[:i] and pred_word[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_word[i - 1] == pred_word[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion (gt char consumed)
                dp[i][j - 1] + 1,       # insertion (pred char consumed)
                dp[i - 1][j - 1] + cost,  # match/substitution
            )

    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if gt_word[i - 1] == pred_word[j - 1] else 1):
            pairs.append((gt_word[i - 1], pred_word[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((gt_word[i - 1], ""))
            i -= 1
        else:
            # insertion -- consumes a pred char, no gt char; drop it.
            j -= 1
    pairs.reverse()
    return pairs


def confusion_table(graph: OverlapGraph) -> "dict[str, Counter[str]]":
    """`{gt_char: Counter({replacement: count})}` over every gt word (of
    every gt region with overlaps) that has NO exact verbatim match among
    the set of its region's overlapping predictions' word tokens."""
    table: "dict[str, Counter[str]]" = {}
    for gi, g in enumerate(graph.gt):
        overlapping = graph.overlapping_preds_by_gt[gi]
        if not overlapping:
            continue
        pred_words_all: list[str] = []
        for pj in overlapping:
            pred_words_all.extend(word_tokens(graph.preds[pj].text))
        pred_word_set = set(pred_words_all)
        for gt_word in word_tokens(g.text):
            if gt_word in pred_word_set:
                continue
            closest = closest_pred_word(gt_word, pred_words_all)
            if closest is None:
                for ch in gt_word:
                    table.setdefault(ch, Counter())[""] += 1
                continue
            for gt_char, replacement in align_chars(gt_word, closest):
                table.setdefault(gt_char, Counter())[replacement] += 1
    return table


def extra_predicted_chars(graph: OverlapGraph) -> "Counter[str]":
    """`Counter` of characters belonging to predictions with ZERO
    overlapping GT at all -- the same 'unclassified' whole-prediction set
    `metrics.char_overlap_stats` counts as page-level FP chars, broken down
    per character here instead of just summed. Answers "what did the model
    predict with no ground truth backing it at all"."""
    counter: "Counter[str]" = Counter()
    for pj, p in enumerate(graph.preds):
        if not graph.edges_by_pred[pj]:
            counter.update(char_multiset(p.text))
    return counter

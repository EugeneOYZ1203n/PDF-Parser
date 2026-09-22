"""Categories 1-3: label description (`TextLabelStats`) and character/word
multiset overlap (`CharOverlapStats`/`WordOverlapStats`), scored over an
`OverlapGraph` from `metrics_core.py`."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rastervec.Evaluation.Evaluate.metrics_core import (
    TEXT_TYPES,
    OverlapGraph,
    Ratio,
    _multiset_overlap,
)
from rastervec.Evaluation.Evaluate.text_metrics import (
    char_multiset,
    normalize_text,
    word_tokens,
)


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

"""Native-text-derived labelling: derives ground-truth text labels directly
from a PDF's own native text, independent of any pipeline run. Replaces
`auto_label.py` (`source="auto"` -> `source="native"`, see
`label_schema.LabelSource`).

Ground truth must not depend on what the system under test decided --
an earlier version of this module ran the actual Vector_Classification
chain on the converted page and only emitted a label for a cluster that
*survived* classification, which meant any native word the classification
chain's own filter steps wrongly dropped silently vanished from the
ground-truth set instead of becoming a scored false negative. This version
reads only the *original* PDF's native text and needs no Conversion/
pipeline run at all: `Evaluation/conversion.py`'s
`convert_page_to_vector_text` places its converted content onto a page
sized from the source's own `PageMeta.mediabox` (confirmed exactly
matching, see that module's own docstring), so a native word's bbox on the
original page is already valid ground truth for the converted page too --
no coordinate transform needed.

Native words are grouped by `(block_no, line_no)` (from `native.extract`)
into line-level ground-truth regions -- closer to a vector cluster's
natural granularity than one word each. `expected_rotation` per line is
the median of its words' own quarter-turn-rounded angles, so
`metrics.py`'s rotation-accuracy metric has real ground truth to check
against instead of an always-0 placeholder.

`native_label_pdf` is what the benchmark calls per page for scoring
(`generate_pipeline_report.bench_ground_truth_by_type`). Its entries are never
saved: label files hold hand-labelled vector text only (`label_schema`)."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from raster_parser.commons.helpers.geometry import union_bbox
from raster_parser.commons.logging_setup import get_logger
from raster_parser.models.text import Text
from raster_parser.evaluation.labelling.label_schema import LabelEntry, LabelSet
from raster_parser.phase_native_read.native_text import extract_native_text as extract_native_words
from raster_parser.phase_native_read.reader import Reader

_LOG = get_logger("native_label")


def _quarter_turn(angle: float) -> int:
    return round(angle / 90.0) % 4 * 90


def _expected_rotation(words: list[Text]) -> int:
    """Most common quarter-turn among the line's words (median as a
    tie-break) -- a single stray-angled word no longer mislabels the line."""
    turns = [_quarter_turn(w.angle()) for w in words]
    counts = Counter(turns)
    top = max(counts.values())
    winners = [t for t, c in counts.items() if c == top]
    if len(winners) == 1:
        return winners[0]
    return int(median(sorted(turns)))


def _reading_order_key(word: Text) -> float:
    """Sort key along the line's reading direction: x for horizontal text,
    y for vertical -- so a rotated line's words concatenate in the right
    order."""
    dx, dy = word.direction
    return word.bbox[1] if abs(dy) > abs(dx) else word.bbox[0]


def native_label_pdf(pdf_path: str, page_index: int) -> LabelSet:
    """Labels one page of `pdf_path` from its own native text. Returns a
    `LabelSet` with one `LabelEntry` (source="native") per native-text
    line, independent of any classification/OCR run over that page's
    converted-to-vector counterpart. `vector_signatures` is left empty."""
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        native_words = extract_native_words(page)

    lines: dict[tuple[int, int], list[Text]] = defaultdict(list)
    for word in native_words:
        lines[(word.block_no, word.line_no)].append(word)

    entries: list[LabelEntry] = []
    for (block_no, line_no), words in lines.items():
        words_sorted = sorted(words, key=_reading_order_key)
        bbox = union_bbox([w.bbox for w in words_sorted])
        text = " ".join(w.text for w in words_sorted)
        label_id = f"line:{page_index}:{block_no}:{line_no}"
        entries.append(
            LabelEntry(
                page_index=page_index,
                bbox=bbox,
                label_id=label_id,
                text=text,
                source="native",
                expected_rotation=_expected_rotation(words_sorted),
            )
        )

    _LOG.debug(
        "native_label_pdf: page %d, %d line(s) labelled",
        page_index,
        len(entries),
    )
    return LabelSet(pdf_path=pdf_path, entries=entries)

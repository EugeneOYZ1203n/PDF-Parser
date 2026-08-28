"""Automatic labelling: derives ground-truth text labels directly from a
PDF's own native text, independent of any pipeline run.

Ground truth must not depend on what the system under test decided --
an earlier version of this module ran the actual Vector_Classification
chain on the converted page and only emitted a label for a cluster that
*survived* classification, which meant any native word the classification
chain's own filter steps wrongly dropped silently vanished from the
ground-truth set instead of becoming a scored false negative. This version
reads only the *original* PDF's native text and needs no Conversion/
pipeline run at all: `Evaluation/Conversion/conversion.py`'s
`convert_page_to_vector_text` places its converted content onto a page
sized from the source's own `PageMeta.mediabox` (confirmed exactly
matching, see that module's own docstring), so a native word's bbox on the
original page is already valid ground truth for the converted page too --
no coordinate transform needed.

Native words are grouped by `(block_no, line_no)` (from `Native.
extract_records`, richer than `extract_text`) into line-level ground-truth
regions -- closer to a vector cluster's natural granularity than one word
each. `expected_rotation` per line is each word's own `angle` rounded to
the nearest quarter-turn (same convention `output_types.TextDTO.
from_text_word`'s `rotate` field uses), so `Evaluation/Evaluate/
evaluate.py`'s rotation-accuracy metric has real ground truth to check
against instead of an always-0 placeholder.
"""
from __future__ import annotations

from collections import defaultdict

from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet
from rastervec.helpers.geometry import union_bbox
from rastervec.logging_setup import get_logger
from rastervec.models import TextRecord
from rastervec.Native_Text.native import Native
from rastervec.Reader.reader import Reader

_LOG = get_logger("auto_label")


def _expected_rotation(words: list[TextRecord]) -> int:
    return round(words[0].angle / 90.0) % 4 * 90


def auto_label_pdf(pdf_path: str, page_index: int) -> LabelSet:
    """Auto-labels one page of `pdf_path` from its own native text. Returns
    a `LabelSet` with one `LabelEntry` (source="auto") per native-text
    line, independent of any classification/OCR run over that page's
    converted-to-vector counterpart."""
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        native_records = Native().extract_records(page)

    lines: dict[tuple[int, int], list[TextRecord]] = defaultdict(list)
    for record in native_records:
        lines[(record.block_no, record.line_no)].append(record)

    entries: list[LabelEntry] = []
    for (block_no, line_no), words in lines.items():
        words_sorted = sorted(words, key=lambda w: w.bbox[0])
        bbox = union_bbox([w.bbox for w in words_sorted])
        text = " ".join(w.text for w in words_sorted)
        entries.append(
            LabelEntry(
                page_index=page_index,
                cluster_bbox=bbox,
                cluster_signature=f"line:{page_index}:{block_no}:{line_no}",
                text=text,
                source="auto",
                expected_rotation=_expected_rotation(words_sorted),
            )
        )

    _LOG.debug(
        "auto_label_pdf: page %d, %d line(s) labelled", page_index, len(entries),
    )
    return LabelSet(pdf_path=pdf_path, entries=entries)

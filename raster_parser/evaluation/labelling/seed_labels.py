"""Draft labels for a page from a vector_classify backend, so the labeller corrects text instead
of typing every word (`scripts/label/manual_vector_text_label.py` seeds a page that has no labels
yet).

The backend's `Text`s become `LabelEntry`s. Each is given the vectors it was read from: the
vectors the backend classified as text (not in its returned drawing vectors) whose bbox centre
lies inside the text's bbox, each vector going to the first such text only.
"""

from __future__ import annotations

from raster_parser.commons.helpers.geometry import bbox_contains
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.core.interfaces import VectorClassifier
from raster_parser.core.registry import resolve_classify
from raster_parser.models.settings import RunSettings
from raster_parser.evaluation.labelling.label_schema import LabelEntry
from raster_parser.phase_native_read.native_read import read_and_extract

DEFAULT_SEED_BACKEND = "v2_vector_class"


def seed_entries(vectors: list[Vector], kept: list[Vector], texts: list[Text]) -> list[LabelEntry]:
    """One draft `LabelEntry` per text in `texts`, tied to the classified-as-text vectors inside it
    (`vectors` minus `kept`, the backend's drawing output)."""
    kept_seqnos = {v.seqno for v in kept}
    free = [v for v in vectors if v.seqno not in kept_seqnos]
    entries = []
    for t in texts:
        mine, rest = [], []
        for v in free:
            x0, y0, x1, y1 = v.rect
            (mine if bbox_contains(t.bbox, (x0 + x1) / 2, (y0 + y1) / 2) else rest).append(v)
        free = rest
        entries.append(LabelEntry.from_text(t, mine))
    return entries


def seed_page(
    pdf_path: str,
    page_index: int,
    *,
    classify: VectorClassifier | None = None,
    settings: RunSettings | None = None,
) -> list[LabelEntry]:
    """Draft labels for one page: native_read, then `classify` (default `v2_vector_class`) on the
    page's vectors."""
    classify = classify or resolve_classify(DEFAULT_SEED_BACKEND)
    native = read_and_extract(pdf_path, page_index)
    kept, texts = classify(native.vectors, native.page, settings or RunSettings())
    return seed_entries(native.vectors, kept, texts)

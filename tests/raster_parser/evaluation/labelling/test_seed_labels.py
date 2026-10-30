"""`seed_labels`: draft labels for a page from a vector_classify backend's text, so the labeller
corrects instead of typing everything."""

from __future__ import annotations

from pathlib import Path

from raster_parser.evaluation.labelling.label_schema import vector_signatures_for
from raster_parser.evaluation.labelling.seed_labels import seed_entries, seed_page

PDF = Path(__file__).resolve().parents[3] / "references" / "test_pdfs_1.pdf"


def test_seed_entries_give_each_text_the_classified_vectors_inside_it(text, vector):
    inside = vector(seqno=1, bbox=(10, 10, 12, 12))
    also_inside = vector(seqno=2, bbox=(15, 11, 18, 14))
    outside = vector(seqno=3, bbox=(50, 50, 52, 52))
    kept_drawing = vector(seqno=4, bbox=(11, 11, 13, 13))
    t = text(text="AB", bbox=(9, 9, 20, 15), page_index=0)
    (entry,) = seed_entries([inside, also_inside, outside, kept_drawing], [kept_drawing], [t])
    assert entry.text == "AB" and entry.bbox == (9.0, 9.0, 20.0, 15.0)
    assert entry.vector_signatures == vector_signatures_for([inside, also_inside])


def test_seed_entries_assign_a_vector_to_one_text_only(text, vector):
    v = vector(seqno=1, bbox=(10, 10, 12, 12))
    a = text(text="A", bbox=(9, 9, 20, 15))
    b = text(text="B", bbox=(8, 8, 21, 16))
    first, second = seed_entries([v], [], [a, b])
    assert first.vector_signatures and not second.vector_signatures


def test_seed_page_runs_the_classifier_on_the_page_vectors(text):
    calls = []

    def fake_classify(vectors, page, settings):
        calls.append(len(vectors))
        return vectors[1:], [
            text(text="X", bbox=tuple(vectors[0].rect), page_index=page.meta.index)
        ]

    entries = seed_page(str(PDF), 0, classify=fake_classify)
    assert calls and calls[0] > 0
    assert [e.text for e in entries] == ["X"] and len(entries[0].vector_signatures) >= 1

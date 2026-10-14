"""Phase 4 ordering, id renumbering and the coordinate-space warning."""
from __future__ import annotations

import logging

from raster_parser.commons.models import Page, PageMeta
from raster_parser.P4_Output_Organization.organize import organize_outputs


def _page(width=200.0, height=100.0) -> Page:
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, width, height), rotation=0, width=width, height=height)
    return Page(doc_path="x.pdf", meta=meta, fitz_page=None)


def test_phase_order_and_disjoint_block_numbers(text):
    native = [text(text="n0", block_no=0), text(text="n1", block_no=3)]
    p3 = [text(text="o0", block_no=0, source="ocr"), text(text="o1", block_no=1, source="ocr")]
    p2 = [text(text="r0", block_no=5, source="ocr")]
    texts, _ = organize_outputs(native, p2, p3, [], [], _page())
    assert [t.text for t in texts] == ["n0", "n1", "o0", "o1", "r0"]
    assert [t.block_no for t in texts] == [0, 3, 4, 5, 6]


def test_inputs_are_not_mutated(text, vector):
    p3 = [text(block_no=0)]
    p2v = [vector(seqno=0)]
    organize_outputs([text(block_no=7)], [], p3, p2v, [vector(seqno=9)], _page())
    assert p3[0].block_no == 0 and p2v[0].seqno == 0


def test_phase2_vectors_paint_on_top_with_unique_seqnos(vector):
    p3 = [vector(seqno=2), vector(seqno=7)]
    p2 = [vector(seqno=0), vector(seqno=1)]
    _, vectors = organize_outputs([], [], [], p2, p3, _page())
    assert [v.seqno for v in vectors] == [2, 7, 8, 9]
    assert len({v.seqno for v in vectors}) == len(vectors)


def test_empty_phases(text, vector):
    assert organize_outputs([], [], [], [], [], _page()) == ([], [])
    texts, vectors = organize_outputs([], [text(block_no=4)], [], [vector(seqno=3)], [], _page())
    assert texts[0].block_no == 0 and vectors[0].seqno == 0


def test_out_of_page_items_are_logged_not_dropped(text, vector, caplog):
    inside = text(bbox=(10, 10, 20, 20))
    outside = vector(bbox=(150, 10, 260, 20))  # past the 200-pt-wide page
    with caplog.at_level(logging.WARNING):
        texts, vectors = organize_outputs([inside], [], [], [], [outside], _page())
    assert texts == [inside] and vectors == [outside]
    assert any("outside the unrotated page" in r.message for r in caplog.records)

from __future__ import annotations

import logging

from rastervec.commons.models import Page
from rastervec.P4_Output_Organization.organize import organize_outputs


def _page(page_meta_builder, **overrides) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta_builder(**overrides), fitz_page=None)


def test_organize_outputs_combines_every_source_in_order(page_meta, text, vector):
    page = _page(page_meta, width=200.0, height=100.0)
    t1, t2, t3 = text(text="a"), text(text="b"), text(text="c")
    v1 = vector(bbox=(0, 0, 10, 10))

    texts, vectors = organize_outputs([t1], [t2], [t3], [v1], page)

    assert texts == [t1, t2, t3]
    assert vectors == [v1]


def test_organize_outputs_leaves_in_bounds_geometry_untouched(page_meta, text, vector, caplog):
    page = _page(page_meta, width=200.0, height=100.0)
    t = text(bbox=(5.0, 5.0, 50.0, 50.0))
    v = vector(bbox=(10.0, 10.0, 190.0, 90.0))

    with caplog.at_level(logging.WARNING):
        texts, vectors = organize_outputs([t], [], [], [v], page)

    assert texts == [t]
    assert vectors == [v]
    assert not caplog.records


def test_organize_outputs_warns_on_out_of_bounds_text(page_meta, text, caplog):
    # Shaped like the rotated-space leak this guard exists for: a page
    # that's wide/short in unrotated MediaBox space (200x100) but the bbox
    # falls in what would be *rotated* display space (100x200) instead.
    page = _page(page_meta, width=200.0, height=100.0)
    t = text(bbox=(5.0, 5.0, 50.0, 150.0))

    with caplog.at_level(logging.WARNING):
        organize_outputs([t], [], [], [], page)

    assert any("outside the unrotated MediaBox" in r.message for r in caplog.records)


def test_organize_outputs_warns_on_out_of_bounds_vector(page_meta, vector, caplog):
    page = _page(page_meta, width=200.0, height=100.0)
    v = vector(bbox=(-5.0, 0.0, 10.0, 10.0))

    with caplog.at_level(logging.WARNING):
        organize_outputs([], [], [], [v], page)

    assert any("outside the unrotated MediaBox" in r.message for r in caplog.records)


def test_organize_outputs_tolerates_tiny_float_overrun(page_meta, vector, caplog):
    page = _page(page_meta, width=200.0, height=100.0)
    v = vector(bbox=(0.0, 0.0, 200.0001, 100.0001))

    with caplog.at_level(logging.WARNING):
        organize_outputs([], [], [], [v], page)

    assert not caplog.records

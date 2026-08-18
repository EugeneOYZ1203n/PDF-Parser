from __future__ import annotations

import pytest

from rastervec.reader import Reader


def test_page_count(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}, {}, {}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        assert reader.page_count() == 3


def test_get_page_returns_correct_meta(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{"width": 300, "height": 150}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)

    assert page.meta.index == 0
    assert page.meta.number == 1
    assert page.meta.width == pytest.approx(300)
    assert page.meta.height == pytest.approx(150)
    assert page.meta.mediabox == pytest.approx((0, 0, 300, 150))


def test_get_page_rotation_keeps_mediabox_unrotated(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{"width": 200, "height": 100, "rotation": 90}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(0)

    assert page.meta.rotation == 90
    # mediabox must stay the UNROTATED box (width/height not swapped) --
    # this is the coordinate-space rule the whole pipeline relies on.
    assert page.meta.width == pytest.approx(200)
    assert page.meta.height == pytest.approx(100)


def test_iter_pages_with_indices(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}, {}, {}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        pages = list(reader.iter_pages([0, 2]))

    assert [p.meta.index for p in pages] == [0, 2]


def test_iter_pages_default_yields_all_in_order(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}, {}, {}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        pages = list(reader.iter_pages())

    assert [p.meta.index for p in pages] == [0, 1, 2]


def test_get_page_out_of_range_raises(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        with pytest.raises(IndexError, match="1 page"):
            reader.get_page(5)


def test_context_manager_closes_doc(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        pass

    assert reader._doc.is_closed

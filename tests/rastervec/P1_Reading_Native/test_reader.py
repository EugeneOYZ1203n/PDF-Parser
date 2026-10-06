from __future__ import annotations

import pytest

from rastervec.P1_Reading_Native.reader import Reader


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


def test_get_page_index_is_source_absolute_and_round_trips(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{"width": 100, "height": 100}, {"width": 300, "height": 150}, {}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        page = reader.get_page(1)
        assert page.meta.index == 1
        assert page.meta.number == 2
        assert page.meta.width == pytest.approx(300)
        # meta.index round-trips: get_page(page.meta.index) is the same page
        again = reader.get_page(page.meta.index)
        assert again.meta.width == pytest.approx(300)


def test_open_missing_pdf_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="could not open PDF"):
        Reader(str(tmp_path / "nope.pdf"))


def _cropped_pdf(tmp_path, rotation=0):
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    page.insert_text((100, 100), "HELLO", fontsize=20)
    page.draw_rect(fitz.Rect(100, 200, 200, 250), color=(0, 0, 0), width=2)
    page.set_cropbox(fitz.Rect(50, 30, 550, 380))
    page.set_rotation(rotation)
    path = tmp_path / "cropped.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def test_page_meta_uses_the_cropbox_frame(tmp_path):
    # Extraction coordinates are CropBox-relative, so PageMeta's width/height
    # (the frame every renderer builds) must be the CropBox's, not the
    # MediaBox's -- otherwise a cropped page reconstructs offset and too big.
    with Reader(_cropped_pdf(tmp_path)) as reader:
        meta = reader.get_page(0).meta
    assert (meta.width, meta.height) == pytest.approx((500, 350))
    assert meta.mediabox == pytest.approx((0, 0, 600, 400))
    assert meta.cropbox == pytest.approx((50, 30, 550, 380))


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_cropped_page_reconstructs_at_the_same_display_position(tmp_path, rotation):
    # The source text is Helvetica, so Phase 4's box-fit reconstruction
    # should land (in display space, /Rotate applied) where the source
    # renders it -- a CropBox/MediaBox mix-up shifts it by the crop offset.
    import io

    import numpy as np
    import pymupdf as fitz
    from PIL import Image

    from rastervec.P1_Reading_Native.native_text import extract_native_text
    from rastervec.P1_Reading_Native.vector_extract import extract_vectors
    from rastervec.P4_Output_Organization import render_output_page

    def ink_centroid(img):
        ys, xs = np.nonzero(np.asarray(img.convert("L")) < 128)
        return xs.mean(), ys.mean()

    with Reader(_cropped_pdf(tmp_path, rotation)) as reader:
        page = reader.get_page(0)
        texts, vectors = extract_native_text(page), extract_vectors(page)
        src = Image.open(io.BytesIO(page.fitz_page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")))
        meta = page.meta

    out = render_output_page(meta, texts, vectors, zoom=2.0)
    assert out.size == src.size
    assert ink_centroid(out) == pytest.approx(ink_centroid(src), abs=2.0)  # 1 pt at zoom 2

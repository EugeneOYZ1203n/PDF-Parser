"""v1_vector_class debug layers: rendered only when a callback is set, never change the output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from raster_parser.models.settings import RunSettings
from raster_parser.phase_native_read.native_read import read_and_extract
from raster_parser.phase_vector_classify.v1_vector_class import s4_word_ocr
from raster_parser.phase_vector_classify.v1_vector_class.debug_layers import OCR_INPUT_FOLDER
from raster_parser.phase_vector_classify.v1_vector_class.parse import parse

EXPECTED_LAYERS = [
    ("fill_filter", "glyph candidates"),
    ("fill_filter", "drawing"),
    ("word_grouping", "word groups"),
    ("ocr", "read"),
    ("ocr", "blank"),
    ("ocr", "text"),
    ("orientation", "direction"),
]


class _FakeOcr:
    def __init__(self, use_angle_cls, *, lang="en", drop_score=0.0):
        pass

    def ocr_image(self, image, *, min_confidence=0.5, cls=None):
        return [((0, 0, 1, 1), "ABC", 0.9)]


@pytest.fixture
def fake_ocr(monkeypatch):
    monkeypatch.setattr(s4_word_ocr, "PaddleOcr", _FakeOcr)


def _run(pdf: Path, settings: RunSettings):
    native = read_and_extract(str(pdf), 0)
    return parse(native.vectors, native.page, settings)


def test_layers_and_images_when_callbacks_set(fake_ocr, synthetic_pdf_with_fills: Path) -> None:
    layers: list[tuple[str, str, str, bytes]] = []
    images: list[tuple[str, str, object]] = []
    settings = RunSettings(
        on_debug_layer=lambda *a: layers.append(a),
        on_debug_image=lambda folder, name, make: images.append((folder, name, make())),
    )
    _run(synthetic_pdf_with_fills, settings)

    assert [(stage, label) for stage, label, _hex, _pdf in layers] == EXPECTED_LAYERS
    for _stage, _label, hexc, pdf_bytes in layers:
        assert hexc.startswith("#") and len(hexc) == 7
        with pymupdf.open("pdf", pdf_bytes) as doc:
            assert doc.page_count == 1
            assert doc[0].rect.width == 400
    assert images and all(folder == OCR_INPUT_FOLDER for folder, _n, _i in images)
    assert all(np.asarray(img).ndim in (2, 3) for _f, _n, img in images)


def test_callbacks_do_not_change_the_output(fake_ocr, synthetic_pdf_with_fills: Path) -> None:
    plain = _run(synthetic_pdf_with_fills, RunSettings())
    debugged = _run(
        synthetic_pdf_with_fills,
        RunSettings(
            on_debug_layer=lambda *a: None,
            on_debug_image=lambda f, n, make: make(),
        ),
    )
    assert plain == debugged
    assert plain[1]  # the fake OCR read the fill glyphs

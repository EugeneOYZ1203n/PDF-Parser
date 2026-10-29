"""`core.pipeline.run_pipeline` and the registry, with the no-OCR stub backends."""

from __future__ import annotations

import pymupdf
import pytest

from raster_parser.models import ExtractResult
from raster_parser.core import registry
from raster_parser.core.pipeline import run_pipeline
from raster_parser.models.settings import RunSettings


def _pdf(tmp_pdf_path) -> str:
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((20, 40), "hello world", fontsize=12)
    page.draw_rect(pymupdf.Rect(50, 80, 150, 150), color=(1, 0, 0))
    page.draw_line((10, 180), (290, 180))
    return tmp_pdf_path(doc)


def test_registry_names_and_defaults():
    assert set(registry.VECTORIZE_BACKENDS) == {"stub", "v1_raster_to_vec"}
    assert set(registry.CLASSIFY_BACKENDS) == {"stub", "v1_vector_class"}
    assert (registry.DEFAULT_VECTORIZE, registry.DEFAULT_CLASSIFY) == (
        "v1_raster_to_vec",
        "v1_vector_class",
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [({"p2": "Nope"}, "unknown p2 backend 'Nope'"), ({"p3": "Nope"}, "unknown p3 backend 'Nope'")],
)
def test_unknown_backend_fails_before_reading(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_pipeline("does-not-exist.pdf", 0, **kwargs)


def test_stub_backends_pass_phase1_through(tmp_pdf_path):
    result = run_pipeline(_pdf(tmp_pdf_path), 0, p2="stub", p3="stub")
    assert isinstance(result, ExtractResult)
    assert [t.text for t in result.texts] == ["hello", "world"]
    assert len(result.vectors) == 2
    assert (result.page.index, result.page.width) == (0, 300)
    assert list(result.step_durations) == [
        "native_read",
        "raster_vectorize",
        "vector_classify",
        "output_assemble",
    ]


def test_backends_get_the_phase_inputs_and_settings(tmp_pdf_path, monkeypatch):
    seen = {}

    def fake_p2(images, page, settings):
        seen["p2"] = (images, page.meta.index, settings)
        return [], []

    def fake_p3(vectors, page, settings):
        seen["p3"] = (len(vectors), settings)
        return vectors[:1], []

    monkeypatch.setitem(registry.VECTORIZE_BACKENDS, "Fake", fake_p2)
    monkeypatch.setitem(registry.CLASSIFY_BACKENDS, "Fake", fake_p3)
    settings = RunSettings(ocr_dpi=150, workers=3)
    result = run_pipeline(_pdf(tmp_pdf_path), 0, p2="Fake", p3="Fake", settings=settings)
    assert seen["p2"] == ([], 0, settings)
    assert seen["p3"] == (2, settings)
    assert len(result.vectors) == 1

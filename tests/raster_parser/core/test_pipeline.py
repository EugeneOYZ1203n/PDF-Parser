"""`core.pipeline.run_pipeline` and the registry, with the no-OCR stub backends."""
from __future__ import annotations

import pymupdf as fitz
import pytest

from raster_parser.core import registry
from raster_parser.core.pipeline import run_pipeline
from raster_parser.core.settings import RunSettings


def _pdf(tmp_pdf_path) -> str:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((20, 40), "hello world", fontsize=12)
    page.draw_rect(fitz.Rect(50, 80, 150, 150), color=(1, 0, 0))
    page.draw_line((10, 180), (290, 180))
    return tmp_pdf_path(doc)


def test_registry_names_and_defaults():
    assert set(registry.P2_REGISTRY) == {"Stub", "v1RasterToVec"}
    assert set(registry.P3_REGISTRY) == {"Stub", "v1VectorClass"}
    assert (registry.DEFAULT_P2, registry.DEFAULT_P3) == ("v1RasterToVec", "v1VectorClass")


@pytest.mark.parametrize("kwargs, message", [({"p2": "Nope"}, "unknown p2 backend 'Nope'"),
                                             ({"p3": "Nope"}, "unknown p3 backend 'Nope'")])
def test_unknown_backend_fails_before_reading(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_pipeline("does-not-exist.pdf", 0, **kwargs)


def test_stub_backends_pass_phase1_through(tmp_pdf_path):
    result = run_pipeline(_pdf(tmp_pdf_path), 0, p2="Stub", p3="Stub")
    assert [t.text for t in result.texts] == ["hello", "world"]
    assert len(result.vectors) == 2
    assert (result.p2, result.p3) == ("Stub", "Stub")
    assert set(result.step_durations) == {"phase1", "phase2", "phase3", "phase4"}
    with result.open_page() as page:
        assert page.fitz_page.rect.width == 300


def test_backends_get_the_phase_inputs_and_settings(tmp_pdf_path, monkeypatch):
    seen = {}

    def fake_p2(images, page, settings):
        seen["p2"] = (images, page.meta.index, settings)
        return [], []

    def fake_p3(vectors, page, settings):
        seen["p3"] = (len(vectors), settings)
        return vectors[:1], []

    monkeypatch.setitem(registry.P2_REGISTRY, "Fake", fake_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "Fake", fake_p3)
    settings = RunSettings(ocr_dpi=150, workers=3)
    result = run_pipeline(_pdf(tmp_pdf_path), 0, p2="Fake", p3="Fake", settings=settings)
    assert seen["p2"] == ([], 0, settings)
    assert seen["p3"] == (2, settings)
    assert len(result.vectors) == 1


def _counting_backends(monkeypatch) -> list[str]:
    called: list[str] = []

    def fake_p2(images, page, settings):
        called.append("p2")
        return [], []

    def fake_p3(vectors, page, settings):
        called.append("p3")
        return [], []

    monkeypatch.setitem(registry.P2_REGISTRY, "Count", fake_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "Count", fake_p3)
    return called


@pytest.mark.parametrize(
    "stop_after, backends, steps",
    [
        ("phase1", [], {"phase1", "phase4"}),
        ("phase2", ["p2"], {"phase1", "phase2", "phase4"}),
        ("phase3", ["p2", "p3"], {"phase1", "phase2", "phase3", "phase4"}),
        (None, ["p2", "p3"], {"phase1", "phase2", "phase3", "phase4"}),
    ],
)
def test_stop_after_skips_later_backends(tmp_pdf_path, monkeypatch, stop_after, backends, steps):
    called = _counting_backends(monkeypatch)
    result = run_pipeline(_pdf(tmp_pdf_path), 0, p2="Count", p3="Count", stop_after=stop_after)
    assert called == backends
    assert set(result.step_durations) == steps


def test_stop_before_phase3_outputs_phase1_vectors(tmp_pdf_path, monkeypatch):
    _counting_backends(monkeypatch)
    result = run_pipeline(_pdf(tmp_pdf_path), 0, p2="Count", p3="Count", stop_after="phase2")
    assert len(result.vectors) == 2  # P3 would have returned none
    assert [t.text for t in result.texts] == ["hello", "world"]


def test_bad_stop_after_fails_before_reading():
    with pytest.raises(ValueError, match="stop_after"):
        run_pipeline("does-not-exist.pdf", 0, p2="Stub", p3="Stub", stop_after="phase4")


def test_debug_hooks_default_to_off():
    settings = RunSettings()
    assert settings.on_debug_layer is None and settings.on_debug_image is None

"""`core.pipeline.run_pipeline` and the registry, with the no-OCR stub backends."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from raster_parser.models.result import ExtractResult
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
    assert set(registry.CLASSIFY_BACKENDS) == {"stub", "v1_vector_class", "v2_vector_class"}
    assert (registry.DEFAULT_VECTORIZE, registry.DEFAULT_CLASSIFY) == (
        "v1_raster_to_vec",
        "v2_vector_class",
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


def _counting_backends(monkeypatch) -> list[str]:
    called: list[str] = []

    def fake_p2(images, page, settings):
        called.append("p2")
        return [], []

    def fake_p3(vectors, page, settings):
        called.append("p3")
        return [], []

    monkeypatch.setitem(registry.VECTORIZE_BACKENDS, "Count", fake_p2)
    monkeypatch.setitem(registry.CLASSIFY_BACKENDS, "Count", fake_p3)
    return called


@pytest.mark.parametrize(
    "stop_after, backends, steps",
    [
        ("native_read", [], {"native_read", "output_assemble"}),
        ("raster_vectorize", ["p2"], {"native_read", "raster_vectorize", "output_assemble"}),
        (
            "vector_classify",
            ["p2", "p3"],
            {"native_read", "raster_vectorize", "vector_classify", "output_assemble"},
        ),
        (
            None,
            ["p2", "p3"],
            {"native_read", "raster_vectorize", "vector_classify", "output_assemble"},
        ),
    ],
)
def test_stop_after_skips_later_backends(tmp_pdf_path, monkeypatch, stop_after, backends, steps):
    called = _counting_backends(monkeypatch)
    result = run_pipeline(_pdf(tmp_pdf_path), 0, p2="Count", p3="Count", stop_after=stop_after)
    assert called == backends
    assert set(result.step_durations) == steps


def test_stop_before_phase3_outputs_phase1_vectors(tmp_pdf_path, monkeypatch):
    _counting_backends(monkeypatch)
    result = run_pipeline(
        _pdf(tmp_pdf_path), 0, p2="Count", p3="Count", stop_after="raster_vectorize"
    )
    assert len(result.vectors) == 2  # P3 would have returned none
    assert [t.text for t in result.texts] == ["hello", "world"]


def test_bad_stop_after_fails_before_reading():
    with pytest.raises(ValueError, match="stop_after"):
        run_pipeline("does-not-exist.pdf", 0, p2="stub", p3="stub", stop_after="output_assemble")


def test_debug_hooks_default_to_off():
    settings = RunSettings()
    assert settings.on_debug_layer is None and settings.on_debug_image is None
    assert settings.on_step_time is None


def test_import_stays_light():
    """`import raster_parser` must not load OCR / image / dictionary libraries (v2_vector_class is
    imported on first use)."""
    import subprocess
    import sys

    code = (
        "import sys, raster_parser; "
        "print(sorted(m for m in ('cv2', 'scipy', 'enchant', 'paddle', 'paddleocr') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


def test_v2_vector_class_runs_through_the_registry(tmp_pdf_path, synthetic_pdf_factory):
    """No vectors -> nothing to OCR: the lazily imported v2 backend returns empty output."""
    doc = synthetic_pdf_factory([{"texts": [{"point": (20, 40), "text": "hello"}]}])
    result = run_pipeline(tmp_pdf_path(doc), 0, p2="stub", p3="v2_vector_class")
    assert [t.text for t in result.texts] == ["hello"] and result.vectors == []


@pytest.mark.parametrize(
    "stop_after, skipped", [(None, []), ("native_read", ["raster_vectorize", "vector_classify"])]
)
def test_progress_lines_per_phase_and_same_result(tmp_pdf_path, capsys, stop_after, skipped):
    import logging

    root = logging.getLogger("raster_parser")
    before = root.level
    pdf = _pdf(tmp_pdf_path)
    try:
        root.setLevel(logging.INFO)
        quiet = run_pipeline(pdf, 0, p2="stub", p3="stub", stop_after=stop_after)
        assert capsys.readouterr().err == ""
        root.setLevel(logging.DEBUG)
        loud = run_pipeline(pdf, 0, p2="stub", p3="stub", stop_after=stop_after)
    finally:
        root.setLevel(before)
    err = capsys.readouterr().err
    for phase in ("native_read", "output_assemble"):
        assert f"▶ {Path(pdf).name} p0 › {phase}" in err
        assert f"✔ {Path(pdf).name} p0 › {phase}" in err
    for phase in skipped:
        assert f"› {phase} stub skipped" in err
    assert (loud.texts, loud.vectors) == (quiet.texts, quiet.vectors)
